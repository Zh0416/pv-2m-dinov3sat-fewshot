from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from external_baselines.matcher_dinov3sat import MatcherDINOv3Runner, bidirectional_matching_dinov3
from external_baselines.persam_f import PerSAMFRunner


class _MockPromptEncoder:
    def get_dense_pe(self) -> torch.Tensor:
        return torch.zeros((1, 256, 64, 64), dtype=torch.float32)

    def __call__(self, points=None, boxes=None, masks=None):
        return points, boxes


class _MockMaskDecoder:
    def __call__(self, *, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=True):
        # Produce deterministic 3-scale synthetic logits centered on positive points
        dev = image_embeddings.device
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1024, 256, device=dev),
            torch.linspace(0, 1024, 256, device=dev),
            indexing="ij",
        )
        logits = torch.full((1, 3, 256, 256), -3.0, device=dev)
        if sparse_prompt_embeddings is not None:
            coords, labels = sparse_prompt_embeddings
            pos = coords[0][labels[0] == 1]
            if len(pos) > 0:
                cx, cy = float(pos[0, 0].item()), float(pos[0, 1].item())
                d2 = (xx - cx) ** 2 + (yy - cy) ** 2
                logits[0, 0] = 4.0 * (d2 < 80.0 ** 2).float() - 2.0
                logits[0, 1] = 4.0 * (d2 < 140.0 ** 2).float() - 2.0
                logits[0, 2] = 4.0 * (d2 < 220.0 ** 2).float() - 2.0
        iou = torch.tensor([[0.80, 0.90, 0.75]], device=dev)
        return logits, iou


class _MockSAM:
    def __init__(self) -> None:
        self.prompt_encoder = _MockPromptEncoder()
        self.mask_decoder = _MockMaskDecoder()


def test_persam_f_synthetic_pipeline() -> None:
    sam = _MockSAM()
    runner = PerSAMFRunner(sam, ft_steps=15, lr=0.1)

    emb_s = torch.zeros((1, 256, 64, 64), dtype=torch.float32)
    emb_s[:, 0, :, :] = 1.0  # background direction
    emb_s[:, 0, 20:40, 20:40] = 0.0
    emb_s[:, 1, 20:40, 20:40] = 1.0  # PV direction

    mask_s = np.zeros((512, 512), dtype=np.uint8)
    mask_s[160:320, 160:320] = 1

    state = runner.fit_episode({"s1": emb_s}, {"s1": mask_s}, ["s1"])
    assert state["target_proto"].shape == (256,)
    assert abs(float(state["scale_weights"].sum().item()) - 1.0) < 1e-5

    m, p, s32 = runner.infer_query(emb_s, state)
    assert m.shape == (512, 512) and m.dtype == bool and m.any()
    assert p.shape == (512, 512) and s32.shape == (32, 32)


def test_matcher_dinov3sat_bidirectional_and_pipeline() -> None:
    sam = _MockSAM()
    runner = MatcherDINOv3Runner(sam, num_clusters=4, merge_margin=0.05)

    tok_s = torch.zeros((4096, 32), dtype=torch.float32)
    tok_s[:, 0] = 1.0
    m_s = np.zeros((512, 512), dtype=np.uint8)
    m_s[128:256, 128:256] = 1
    fg_grid = m_s.reshape(64, 8, 64, 8).mean(axis=(1, 3)).ravel() >= 0.5
    tok_s[fg_grid, 0] = 0.0
    tok_s[fg_grid, 1] = 1.0
    tok_s = F.normalize(tok_s, dim=1)

    state = runner.fit_episode({"s1": tok_s}, {"s1": m_s}, ["s1"])
    matched_idx, matched_sc, fg_sim = bidirectional_matching_dinov3(
        tok_s, state["sup_fg"], state["sup_all"], state["sup_is_fg"]
    )
    assert len(matched_idx) > 0 and fg_sim.shape == (64, 64)
    assert np.all(fg_grid[matched_idx])

    emb_q = torch.zeros((1, 256, 64, 64), dtype=torch.float32)
    m, p, s32 = runner.infer_query(tok_s, emb_q, state)
    assert m.shape == (512, 512) and m.dtype == bool and m.any()
    assert p.shape == (512, 512) and s32.shape == (32, 32)
