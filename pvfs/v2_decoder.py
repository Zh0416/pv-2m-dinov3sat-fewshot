from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _DSConvBlock(nn.Module):
    """Depthwise-separable Conv + GroupNorm + SiLU block keeping parameter count minimal."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        groups = 8 if out_ch % 8 == 0 else (4 if out_ch % 4 == 0 else 1)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class SCFBDecoder(nn.Module):
    """Lightweight Support-Conditioned Frequency-Boundary Residual Decoder (~35K params)."""

    def __init__(self, in_ch: int = 16) -> None:
        super().__init__()
        # Multi-scale context path: 512 -> 256 -> 128 -> 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(inplace=True),
            _DSConvBlock(24, 24),
        )
        self.enc2 = nn.Sequential(
            _DSConvBlock(24, 32, stride=2),
            _DSConvBlock(32, 32),
        )
        self.enc3 = nn.Sequential(
            _DSConvBlock(32, 48, stride=2),
            _DSConvBlock(48, 48, dilation=2),
            _DSConvBlock(48, 48, dilation=4),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(48 + 32, 32, kernel_size=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.SiLU(inplace=True),
            _DSConvBlock(32, 32),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 24, 24, kernel_size=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(inplace=True),
            _DSConvBlock(24, 24),
        )
        # Full-resolution 512x512 boundary refinement head fused with fine frequency/edge cues
        self.fine_proj = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
        )
        self.head_512 = nn.Sequential(
            _DSConvBlock(24 + 16, 24),
            nn.Conv2d(24, 1, kernel_size=1),
        )
        nn.init.zeros_(self.head_512[-1].weight)
        nn.init.zeros_(self.head_512[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.50))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.stem(x)   # [B, 24, 256, 256]
        e2 = self.enc2(e1)  # [B, 32, 128, 128]
        e3 = self.enc3(e2)  # [B, 48, 64, 64]

        d2 = self.dec2(torch.cat((F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2), dim=1))
        d1 = self.dec1(torch.cat((F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), dim=1))
        up_512 = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        fine = self.fine_proj(x)
        delta = self.head_512(torch.cat((up_512, fine), dim=1))
        return x[:, 0:1] + self.res_scale * delta


def compute_support_rgb_stats(
    rgb_by_id: dict[str, torch.Tensor],
    mask_by_id: dict[str, np.ndarray],
    support_ids: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute domain-relative RGB foreground/background centroids strictly from support images."""
    fg_pix, bg_pix = [], []
    for sid in support_ids:
        rgb_t = rgb_by_id[sid].to(device).float()  # [3, 512, 512] in [0, 1]
        m = torch.from_numpy(mask_by_id[sid] > 0).to(device)
        if m.any():
            fg_pix.append(rgb_t[:, m].mean(dim=1))
        if (~m).any():
            bg_pix.append(rgb_t[:, ~m].mean(dim=1))
    fg_mean = torch.stack(fg_pix).mean(dim=0) if fg_pix else torch.tensor([0.35, 0.42, 0.48], device=device)
    bg_mean = torch.stack(bg_pix).mean(dim=0) if bg_pix else torch.tensor([0.50, 0.50, 0.46], device=device)
    return fg_mean, bg_mean


def build_decoder_input_512(
    sem_6_64: torch.Tensor,
    z_freq_64: torch.Tensor,
    rgb_512: torch.Tensor,
    freq_512: torch.Tensor,
    fg_rgb_mean: torch.Tensor,
    bg_rgb_mean: torch.Tensor,
    *,
    disable_freq: bool = False,
) -> torch.Tensor:
    """Assemble 16-channel domain-invariant [16, 512, 512] tensor on GPU."""
    sem_512 = F.interpolate(sem_6_64[None], size=(512, 512), mode="bilinear", align_corners=False)[0]  # [6, 512, 512]
    zf_512 = F.interpolate(z_freq_64[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0]  # [1, 512, 512]

    rgb_f = rgb_512.float()
    d_fg = ((rgb_f - fg_rgb_mean[:, None, None]) ** 2).sum(dim=0).sqrt()
    d_bg = ((rgb_f - bg_rgb_mean[:, None, None]) ** 2).sum(dim=0).sqrt()
    color_contrast = ((d_bg - d_fg) / 0.25).clamp(-2.5, 2.5)[None]  # [1, 512, 512]

    f_512 = freq_512.float()
    if disable_freq:
        zf_512 = torch.zeros_like(zf_512)
        f_512 = torch.zeros_like(f_512)
    return torch.cat((sem_512, zf_512, color_contrast, f_512), dim=0)  # [16, 512, 512]


def scfb_episodic_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Joint Dice + Edge-Weighted BCE + Strong Empty-Negative FP Suppression Loss."""
    # targets: [B, 1, H, W] float in {0, 1}
    pos_img = (targets.sum(dim=(1, 2, 3)) > 0.5)
    loss = logits.new_tensor(0.0)
    terms = 0

    if pos_img.any():
        lp = logits[pos_img]
        yp = targets[pos_img]
        probs = torch.sigmoid(lp)
        inter = (probs * yp).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + yp.sum(dim=(1, 2, 3)) + 1e-4
        dice_loss = (1.0 - (2.0 * inter + 1e-4) / union).mean()

        # Highlight 3px morphological edge band around PV ground truth
        dil = F.max_pool2d(yp, kernel_size=5, stride=1, padding=2)
        ero = -F.max_pool2d(-yp, kernel_size=5, stride=1, padding=2)
        edge_band = (dil - ero).clamp(0.0, 1.0)
        weights = 1.0 + 3.0 * edge_band
        bce = F.binary_cross_entropy_with_logits(lp, yp, reduction="none")
        edge_bce = (bce * weights).mean()
        loss = loss + 0.55 * dice_loss + 0.85 * edge_bce
        terms += 1

    if (~pos_img).any():
        ln = logits[~pos_img]
        yn = torch.zeros_like(ln)
        probs_n = torch.sigmoid(ln)
        # Penalize both pixel BCE and peak false-positive mass on negative images
        neg_bce = F.binary_cross_entropy_with_logits(ln, yn)
        neg_peak = (probs_n ** 2).mean() + 0.35 * probs_n.amax(dim=(1, 2, 3)).mean()
        loss = loss + 1.25 * neg_bce + 0.80 * neg_peak
        terms += 1

    return loss / max(1, terms)
