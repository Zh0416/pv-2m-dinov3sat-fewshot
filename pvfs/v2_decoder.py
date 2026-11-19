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
    """Disentangled Support-Conditioned Frequency-Boundary Residual Decoder (~38K params).

    Design:
      - Semantic Context Stream (channels 0..5, 7 = sem_6 + color_contrast) runs 512->64->512
        to clean semantic blobs without overfitting coarse FFT band ratios on dense farms.
      - Fine Frequency-Boundary Stream (channels 0, 6, 8..16 = z_sem + z_freq + luma_hp + freq_512)
        operates strictly at 512x512 resolution to snap edges & suppress non-periodic speckles.
    """

    def __init__(self, in_ch: int = 17) -> None:
        super().__init__()
        ctx_ch = 7   # sem_6 (0..5) + color_contrast (7)
        fine_ch = 11  # z_sem (0) + z_freq (6) + luma_hp (8) + freq_512 (9..16)
        self.stem = nn.Sequential(
            nn.Conv2d(ctx_ch, 24, kernel_size=3, stride=2, padding=1, bias=False),
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
        # Full-resolution 512x512 frequency + signed luma boundary head
        self.fine_proj = nn.Sequential(
            nn.Conv2d(fine_ch, 20, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 20),
            nn.SiLU(inplace=True),
            _DSConvBlock(20, 20, dilation=2),
        )
        self.head_512 = nn.Sequential(
            _DSConvBlock(24 + 20, 24),
            nn.Conv2d(24, 1, kernel_size=1),
        )
        nn.init.zeros_(self.head_512[-1].weight)
        nn.init.zeros_(self.head_512[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.80))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ctx = torch.cat((x[:, :6], x[:, 7:8]), dim=1)  # 7 ch: semantic + support color contrast
        x_fine = torch.cat((x[:, 0:1], x[:, 6:7], x[:, 8:]), dim=1)  # 11 ch: z_sem + z_freq + luma_hp + freq_512

        e1 = self.stem(x_ctx)  # [B, 24, 256, 256]
        e2 = self.enc2(e1)     # [B, 32, 128, 128]
        e3 = self.enc3(e2)     # [B, 48, 64, 64]

        d2 = self.dec2(torch.cat((F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2), dim=1))
        d1 = self.dec1(torch.cat((F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), dim=1))
        up_512 = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        fine = self.fine_proj(x_fine)
        delta = self.head_512(torch.cat((up_512, fine), dim=1))
        # Positive residual only opens near plausible semantic support (prevents open-water hallucinations);
        # negative suppression remains active everywhere to clean isolated confounders.
        sem_anchor = torch.sigmoid((x[:, 0:1] + 2.0) / 0.80)
        bounded_delta = F.relu(delta) * sem_anchor - F.relu(-delta)
        return x[:, 0:1] + self.res_scale * bounded_delta


def compute_support_rgb_stats(
    rgb_by_id: dict[str, torch.Tensor],
    mask_by_id: dict[str, np.ndarray],
    support_ids: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute domain-relative RGB foreground/background centroids strictly from support images."""
    fg_pix, bg_pix = [], []
    for sid in support_ids:
        rgb_t = rgb_by_id[sid].to(device).float()
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
    """Assemble 17-channel domain-invariant [17, 512, 512] tensor on GPU."""
    sem_512 = F.interpolate(sem_6_64[None], size=(512, 512), mode="bilinear", align_corners=False)[0]  # [6, 512, 512]
    zf_512 = F.interpolate(z_freq_64[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0]  # [1, 512, 512]

    rgb_f = rgb_512.float()
    d_fg = ((rgb_f - fg_rgb_mean[:, None, None]) ** 2).sum(dim=0).sqrt()
    d_bg = ((rgb_f - bg_rgb_mean[:, None, None]) ** 2).sum(dim=0).sqrt()
    color_contrast = ((d_bg - d_fg) / 0.25).clamp(-2.5, 2.5)[None]  # [1, 512, 512]

    luma = (0.299 * rgb_f[0] + 0.587 * rgb_f[1] + 0.114 * rgb_f[2])[None, None]
    luma_mean = F.avg_pool2d(luma, kernel_size=31, stride=1, padding=15, count_include_pad=False)
    luma_hp = ((luma - luma_mean) / (luma.std().clamp_min(0.05))).clamp(-3.0, 3.0)[0]  # [1, 512, 512]

    f_512 = freq_512.float()
    if disable_freq:
        zf_512 = torch.zeros_like(zf_512)
        f_512 = torch.zeros_like(f_512)
    return torch.cat((sem_512, zf_512, color_contrast, luma_hp, f_512), dim=0)  # [17, 512, 512]


def scfb_episodic_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Recall-preserving Tversky + Strong Edge-Weighted BCE + Top-1% Hard-Negative Suppression."""
    pos_img = (targets.sum(dim=(1, 2, 3)) > 0.5)
    loss = logits.new_tensor(0.0)
    b_total = max(1, logits.shape[0])

    if pos_img.any():
        lp = logits[pos_img]
        yp = targets[pos_img]
        probs = torch.sigmoid(lp)
        tp = (probs * yp).sum(dim=(1, 2, 3))
        fp = (probs * (1.0 - yp)).sum(dim=(1, 2, 3))
        fn = ((1.0 - probs) * yp).sum(dim=(1, 2, 3))
        tversky = 1.0 - (tp + 1e-4) / (tp + 0.35 * fp + 0.65 * fn + 1e-4)

        dil = F.max_pool2d(yp, kernel_size=5, stride=1, padding=2)
        ero = -F.max_pool2d(-yp, kernel_size=5, stride=1, padding=2)
        edge_band = (dil - ero).clamp(0.0, 1.0)
        weights = 1.0 + 6.0 * edge_band
        bce = F.binary_cross_entropy_with_logits(lp, yp, reduction="none")
        edge_bce = (bce * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3))
        pos_loss = (0.60 * tversky + 0.90 * edge_bce).mean()
        loss = loss + pos_loss * (float(pos_img.sum().item()) / b_total)

    if (~pos_img).any():
        ln = logits[~pos_img].reshape((~pos_img).sum().item(), -1)
        yn = torch.zeros_like(ln)
        neg_bce = F.binary_cross_entropy_with_logits(ln, yn, reduction="none")
        k_hard = max(64, ln.shape[1] // 100)  # top-1% hardest FP pixels on pure-negative scenes
        hard_bce = neg_bce.topk(k_hard, dim=1).values.mean()
        neg_loss = 0.25 * neg_bce.mean() + 0.50 * hard_bce
        loss = loss + neg_loss * (float((~pos_img).sum().item()) / b_total)

    return loss
