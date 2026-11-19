from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F


def select_fg_bg_indices(
    mask_512: np.ndarray,
    grid_hw: int,
    core_purity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select support foreground tokens (purity >= core_purity with fallback) and pure bg tokens."""
    patch_px = 512 // grid_hw
    y_grid = mask_512.astype(np.float32).reshape(grid_hw, patch_px, grid_hw, patch_px).mean(axis=(1, 3)).ravel()
    fg_mask = y_grid >= core_purity
    if int(fg_mask.sum()) < 8:
        fg_mask = y_grid >= 0.50
    if not fg_mask.any():
        cands = np.flatnonzero(y_grid > 0.0)
        if len(cands) == 0:
            raise ValueError("Support image has no positive PV pixels")
        count = min(len(cands), max(1, int(math.ceil(float(y_grid.sum())))))
        fg_mask[cands[np.argsort(-y_grid[cands], kind="stable")[:count]]] = True
    bg_mask = y_grid == 0.0
    return fg_mask, bg_mask, y_grid


@torch.inference_mode()
def build_pairwise_topk_cache(
    tokens_by_id: dict[str, torch.Tensor],
    mask_by_id: dict[str, np.ndarray],
    all_ids: list[str],
    pos_ids: list[str],
    *,
    grid_hw: int,
    core_purity: float = 0.85,
    top_m: int = 3,
) -> dict:
    """Precompute exact per-image-pair top-m similarities for fg_core, fg_all, and bg.

    For any set of support indices S, pooling top-m over the union equals taking top-m
    over the concatenation of cached per-image top-m slices, yielding ~100x faster sweeps
    with identical values up to fp16 rounding.
    """
    n_q = grid_hw * grid_hw
    n_pos = len(pos_ids)
    n_all = len(all_ids)
    id_to_all = {rid: i for i, rid in enumerate(all_ids)}
    id_to_pos = {rid: i for i, rid in enumerate(pos_ids)}

    first_tok = tokens_by_id[all_ids[0]]
    dev = first_tok.device

    fg_core_top = torch.full((n_pos, n_all, n_q, top_m), -2.0, dtype=torch.float16, device=dev)
    fg_all_top = torch.full((n_pos, n_all, n_q, top_m), -2.0, dtype=torch.float16, device=dev)
    bg_top = torch.full((n_pos, n_all, n_q, top_m), -2.0, dtype=torch.float16, device=dev)
    fg_core_masks: list[np.ndarray] = []
    fg_all_masks: list[np.ndarray] = []
    bg_masks: list[np.ndarray] = []

    q_stack = torch.stack([tokens_by_id[rid] for rid in all_ids], dim=0)  # [N_all, N_q, C]
    if dev.type == "cuda":
        q_stack_mm = q_stack.to(torch.float16).reshape(n_all * n_q, -1)
    else:
        q_stack_mm = q_stack.float().reshape(n_all * n_q, -1)

    for s_i, sid in enumerate(pos_ids):
        tok_s = tokens_by_id[sid]
        tok_s_mm = tok_s.to(q_stack_mm.dtype)
        m512 = mask_by_id[sid]
        fg_c, bg_m, y_grid = select_fg_bg_indices(m512, grid_hw, core_purity)
        fg_a, _, _ = select_fg_bg_indices(m512, grid_hw, 0.50)
        fg_core_masks.append(fg_c)
        fg_all_masks.append(fg_a)
        bg_masks.append(bg_m)

        def _fill(mask_arr: np.ndarray, dst: torch.Tensor) -> None:
            idx = np.flatnonzero(mask_arr)
            if len(idx) == 0:
                return
            t_idx = torch.from_numpy(idx).to(dev)
            bank = tok_s_mm[t_idx]  # [M, C]
            m_take = min(top_m, bank.shape[0])
            chunk_imgs = 8
            for start in range(0, n_all, chunk_imgs):
                end = min(n_all, start + chunk_imgs)
                q_chunk = q_stack_mm[start * n_q : end * n_q]
                sim = (q_chunk @ bank.T).float().clamp(-1.0, 1.0)
                vals = sim.topk(m_take, dim=1).values.reshape(end - start, n_q, m_take)
                if m_take < top_m:
                    pad = vals.new_full((end - start, n_q, top_m - m_take), -2.0)
                    vals = torch.cat((vals, pad), dim=2)
                dst[s_i, start:end] = vals.to(torch.float16)

        _fill(fg_c, fg_core_top)
        _fill(fg_a, fg_all_top)
        if bg_m.any():
            _fill(bg_m, bg_top)

    return {
        "grid_hw": grid_hw,
        "top_m": top_m,
        "id_to_all": id_to_all,
        "id_to_pos": id_to_pos,
        "fg_core_top": fg_core_top,
        "fg_all_top": fg_all_top,
        "bg_top": bg_top,
        "fg_core_masks": fg_core_masks,
        "fg_all_masks": fg_all_masks,
        "bg_masks": bg_masks,
    }


def score_from_pairwise(
    cache: dict,
    q_id: str,
    support_ids: list[str],
    *,
    use_core: bool = True,
    shot_mode: str = "balanced_70_30",
    alpha_bg: float = 0.50,
    exclude_pos_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (score, s_pos, s_neg) of shape [N_q] using precomputed top-m cache."""
    q_idx = cache["id_to_all"][q_id]
    s_indices = [cache["id_to_pos"][sid] for i, sid in enumerate(support_ids) if i != exclude_pos_idx]
    m = cache["top_m"]
    fg_src = cache["fg_core_top"] if use_core else cache["fg_all_top"]

    def _valid_shot_means(slices: torch.Tensor) -> torch.Tensor:
        valid = slices > -1.5
        denom = valid.sum(dim=2).clamp_min(1).float()
        return (slices * valid).sum(dim=2) / denom

    def _pooled_topm(slices: torch.Tensor) -> torch.Tensor:
        cat = slices.permute(1, 0, 2).reshape(-1, k_eff * m)
        topv = cat.topk(m, dim=1).values
        valid = topv > -1.5
        if not valid.any():
            return torch.zeros(cat.shape[0], dtype=torch.float32, device=cat.device)
        return (topv * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1).float()

    fg_slices = torch.stack([fg_src[si, q_idx] for si in s_indices], dim=0).float()  # [K', N_q, m]
    k_eff = fg_slices.shape[0]
    if k_eff == 1 or shot_mode == "pooled":
        s_pos = _pooled_topm(fg_slices)
    elif shot_mode == "balanced_70_30":
        per_shot = _valid_shot_means(fg_slices)  # [K', N_q]
        s_pos = 0.70 * per_shot.max(dim=0).values + 0.30 * per_shot.mean(dim=0)
    elif shot_mode == "top2_soft":
        per_shot = _valid_shot_means(fg_slices).sort(dim=0, descending=True).values
        if k_eff == 2:
            s_pos = 0.75 * per_shot[0] + 0.25 * per_shot[1]
        else:
            s_pos = 0.65 * per_shot[0] + 0.25 * per_shot[1] + 0.10 * per_shot[2:].mean(dim=0)
    else:
        raise ValueError(f"Unknown shot_mode: {shot_mode}")

    bg_slices = torch.stack([cache["bg_top"][si, q_idx] for si in s_indices], dim=0).float()
    if (bg_slices > -1.5).any():
        s_neg = _pooled_topm(bg_slices)
    else:
        s_neg = torch.zeros_like(s_pos)

    return s_pos - alpha_bg * s_neg, s_pos, s_neg
