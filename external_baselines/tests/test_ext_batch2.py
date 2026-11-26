from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from external_baselines.finetune_models import DINOv3KShotFTRunner


def test_dinov3_kshot_ft_learns_synthetic_target() -> None:
    runner = DINOv3KShotFTRunner(steps=25, lr=1e-2, wd=1e-3, seed=42)

    tok_d = torch.zeros((4096, 32), dtype=torch.float32)
    tok_m = torch.zeros((4096, 32), dtype=torch.float32)
    tok_d[:, 0] = 1.0
    tok_m[:, 0] = 1.0

    m512 = np.zeros((512, 512), dtype=np.uint8)
    m512[160:352, 160:352] = 1
    fg_grid = m512.reshape(64, 8, 64, 8).mean(axis=(1, 3)).ravel() >= 0.5
    tok_d[fg_grid, 0] = 0.0
    tok_d[fg_grid, 1] = 1.0
    tok_m[fg_grid, 0] = 0.0
    tok_m[fg_grid, 1] = 1.0
    tok_d = F.normalize(tok_d, dim=1)
    tok_m = F.normalize(tok_m, dim=1)

    tokens_by_key = {"deep_64": {"s1": tok_d}, "mid_64": {"s1": tok_m}}
    head = runner.fit_episode(tokens_by_key, {"s1": m512}, ["s1"], ep_seed_offset=1)
    masks, probs, s32 = runner.infer_queries(head, tokens_by_key, ["s1"])

    assert masks["s1"].shape == (512, 512) and masks["s1"].dtype == bool
    assert probs["s1"].shape == (512, 512) and s32["s1"].shape == (32, 32)
    inter = (masks["s1"] & (m512 > 0)).sum()
    dice = 2.0 * inter / max(1, masks["s1"].sum() + (m512 > 0).sum())
    assert dice > 0.90, f"Expected synthetic K-shot FT Dice > 0.90, got {dice:.3f}"
