from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_smp() -> None:
    if importlib.util.find_spec("segmentation_models_pytorch") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True)


def _bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return bce + dice


def _random_dihedral_aug(x: torch.Tensor, y: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random 90-deg rotation and H/V flip (standard isotropic augmentation for nadir satellite imagery)."""
    k_rot = int(rng.integers(0, 4))
    if k_rot > 0:
        x = torch.rot90(x, k=k_rot, dims=(-2, -1))
        y = torch.rot90(y, k=k_rot, dims=(-2, -1))
    if bool(rng.integers(0, 2)):
        x = torch.flip(x, dims=(-1,))
        y = torch.flip(y, dims=(-1,))
    if bool(rng.integers(0, 2)):
        x = torch.flip(x, dims=(-2,))
        y = torch.flip(y, dims=(-2,))
    return x, y


class _DinoConvDecoder(nn.Module):
    """Lightweight trainable segmentation head on frozen DINOv3-SAT [mid_64 || deep_64] features."""

    def __init__(self, in_ch: int = 2048, mid_ch: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.GroupNorm(16, mid_ch),
            nn.GELU(),
            nn.Dropout2d(0.15),
            nn.Conv2d(mid_ch, mid_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch // 2),
            nn.GELU(),
            nn.Conv2d(mid_ch // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOv3KShotFTRunner:
    """Same frozen DINOv3-SAT ViT-L/16 backbone + K-shot fine-tuned lightweight convolutional decoder."""

    def __init__(self, *, steps: int = 120, lr: float = 3e-3, wd: float = 1e-2, seed: int = 2026) -> None:
        self.steps = int(steps)
        self.lr = float(lr)
        self.wd = float(wd)
        self.seed = int(seed)

    def _build_feat_64(self, tokens_by_key: dict[str, Any], rid: str) -> torch.Tensor:
        d64 = tokens_by_key["deep_64"][rid].T.reshape(-1, 64, 64)
        m64 = tokens_by_key["mid_64"][rid].T.reshape(-1, 64, 64)
        return torch.cat((m64, d64), dim=0).detach().clone()  # [2048, 64, 64]

    def fit_episode(
        self,
        tokens_by_key: dict[str, Any],
        mask_by_id: dict[str, np.ndarray],
        support_ids: list[str],
        *,
        ep_seed_offset: int = 0,
    ) -> nn.Module:
        torch.manual_seed(self.seed + ep_seed_offset)
        rng = np.random.default_rng(self.seed + ep_seed_offset)

        xs, ys = [], []
        for sid in support_ids:
            f64 = self._build_feat_64(tokens_by_key, sid)
            dev = f64.device
            m512 = torch.from_numpy((mask_by_id[sid] > 0).astype(np.float32)).to(dev)[None, None]
            m64 = F.interpolate(m512, size=(64, 64), mode="area")[0]  # [1, 64, 64]
            xs.append(f64)
            ys.append(m64)

        x_sup = torch.stack(xs, dim=0)  # [K, 2048, 64, 64]
        y_sup = torch.stack(ys, dim=0)  # [K, 1, 64, 64]
        dev = x_sup.device

        head = _DinoConvDecoder(in_ch=int(x_sup.shape[1]), mid_ch=256).to(dev)
        head.train()
        opt = torch.optim.AdamW(head.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.steps, eta_min=self.lr * 0.05)

        for _ in range(self.steps):
            x_aug, y_aug = _random_dihedral_aug(x_sup, y_sup, rng)
            opt.zero_grad(set_to_none=True)
            logits = head(x_aug)
            loss = _bce_dice_loss(logits, y_aug)
            loss.backward()
            opt.step()
            sched.step()

        head.eval()
        return head

    @torch.inference_mode()
    def infer_queries(
        self,
        head: nn.Module,
        tokens_by_key: dict[str, Any],
        query_ids: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        masks_out, probs_out, s32_out = {}, {}, {}
        for qid in query_ids:
            f64 = self._build_feat_64(tokens_by_key, qid)[None]  # [1, 2048, 64, 64]
            logits_64 = head(f64)  # [1, 1, 64, 64]
            logits_512 = F.interpolate(logits_64, size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
            prob_512 = torch.sigmoid(logits_512).cpu().numpy().astype(np.float32)
            masks_out[qid] = prob_512 >= 0.50
            probs_out[qid] = prob_512
            s32_out[qid] = F.avg_pool2d(logits_64, 2, stride=2)[0, 0].cpu().numpy().astype(np.float32)
        return masks_out, probs_out, s32_out


class EndToEndSegKShotRunner:
    """End-to-end supervised segmentation model (U-Net ResNet50 or SegFormer MiT-B2) K-shot fine-tuned per episode."""

    def __init__(
        self,
        arch: str,
        *,
        steps: int = 80,
        lr: float = 3e-4,
        wd: float = 1e-2,
        batch_infer: int = 16,
        device: str = "cuda",
        seed: int = 2026,
    ) -> None:
        self.arch = arch
        self.steps = int(steps)
        self.lr = float(lr)
        self.wd = float(wd)
        self.batch_infer = int(batch_infer)
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.seed = int(seed)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=torch.float32)[None, :, None, None]
        self._std = torch.tensor(IMAGENET_STD, device=self.device, dtype=torch.float32)[None, :, None, None]

    def _build_fresh_model(self) -> nn.Module:
        ensure_smp()
        import segmentation_models_pytorch as smp

        if self.arch == "unet_resnet50":
            return smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1).to(self.device)
        if self.arch == "segformer_mit_b2":
            return smp.Segformer(encoder_name="mit_b2", encoder_weights="imagenet", in_channels=3, classes=1).to(self.device)
        raise ValueError(f"Unsupported arch: {self.arch}")

    def _prep_rgb(self, rgb_list: list[np.ndarray]) -> torch.Tensor:
        arr = np.stack(rgb_list, axis=0)  # [B, 512, 512, 3] uint8
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.0
        return (x - self._mean) / self._std

    def fit_episode(
        self,
        rgb_512_by_id: dict[str, np.ndarray],
        mask_by_id: dict[str, np.ndarray],
        support_ids: list[str],
        *,
        ep_seed_offset: int = 0,
    ) -> nn.Module:
        torch.manual_seed(self.seed + ep_seed_offset)
        rng = np.random.default_rng(self.seed + ep_seed_offset)

        model = self._build_fresh_model()
        model.train()
        # Freeze BatchNorm running stats when K=1 (batch size 1 breaks BN variance!)
        if len(support_ids) == 1:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

        x_sup = self._prep_rgb([rgb_512_by_id[sid] for sid in support_ids])
        y_sup = torch.from_numpy(
            np.stack([(mask_by_id[sid] > 0).astype(np.float32) for sid in support_ids], axis=0)
        ).to(self.device)[:, None]  # [K, 1, 512, 512]

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.steps, eta_min=self.lr * 0.05)
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for _ in range(self.steps):
            x_aug, y_aug = _random_dihedral_aug(x_sup, y_sup, rng)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x_aug)
                loss = _bce_dice_loss(logits, y_aug)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        model.eval()
        return model

    @torch.inference_mode()
    def infer_queries(
        self,
        model: nn.Module,
        rgb_512_by_id: dict[str, np.ndarray],
        query_ids: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        masks_out, probs_out, s32_out = {}, {}, {}
        use_amp = self.device == "cuda"
        for start in range(0, len(query_ids), self.batch_infer):
            batch_ids = query_ids[start : start + self.batch_infer]
            xb = self._prep_rgb([rgb_512_by_id[qid] for qid in batch_ids])
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb).float()  # [B, 1, 512, 512]
            probs = torch.sigmoid(logits)[:, 0].cpu().numpy().astype(np.float32)
            s32 = F.avg_pool2d(logits, 16, stride=16)[:, 0].cpu().numpy().astype(np.float32)
            for i, qid in enumerate(batch_ids):
                probs_out[qid] = probs[i]
                masks_out[qid] = probs[i] >= 0.50
                s32_out[qid] = s32[i]
        return masks_out, probs_out, s32_out
