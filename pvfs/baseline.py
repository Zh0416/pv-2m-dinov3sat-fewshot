from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from .data import area_target_32


def normalize_dino_tokens(feat_map: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert [C, H, W] feature map into L2-normalized [H*W, C] tokens on target device."""
    t = torch.from_numpy(feat_map).to(device=device, dtype=torch.float32)
    t = F.normalize(t, dim=0)
    return t.flatten(1).T.contiguous()


def build_support_pool(
    tokens_by_id: dict[str, torch.Tensor],
    mask_by_id: dict[str, np.ndarray],
    support_ids: list[str],
    *,
    purity: float = 0.5,
) -> dict:
    """Collect pure PV foreground and pure background tokens from K support images."""
    fg_list, bg_list = [], []
    fg_img_idx, bg_img_idx, bg_coords = [], [], []
    fg_masks_32 = []

    for idx, sid in enumerate(support_ids):
        y32 = area_target_32(mask_by_id[sid]).ravel()
        tok = tokens_by_id[sid]
        fg_mask = y32 >= purity
        if not fg_mask.any():
            candidates = np.flatnonzero(y32 > 0)
            if len(candidates) == 0:
                raise ValueError(f"Support image {sid} has zero PV pixels")
            count = min(len(candidates), max(1, int(math.ceil(float(y32.sum())))))
            order = np.argsort(-y32[candidates], kind="stable")[:count]
            fg_mask[candidates[order]] = True

        bg_mask = y32 == 0.0
        fg_idx_t = torch.from_numpy(np.flatnonzero(fg_mask)).to(tok.device)
        fg_list.append(tok[fg_idx_t])
        fg_img_idx.append(torch.full((fg_idx_t.numel(),), idx, dtype=torch.long, device=tok.device))
        fg_masks_32.append(fg_mask)

        if bg_mask.any():
            bg_flat = np.flatnonzero(bg_mask)
            bg_idx_t = torch.from_numpy(bg_flat).to(tok.device)
            bg_list.append(tok[bg_idx_t])
            bg_img_idx.append(torch.full((bg_idx_t.numel(),), idx, dtype=torch.long, device=tok.device))
            yy = torch.from_numpy(bg_flat // 32).to(tok.device)
            xx = torch.from_numpy(bg_flat % 32).to(tok.device)
            bg_coords.append(torch.stack((yy, xx), dim=1))

    fg = torch.cat(fg_list, dim=0)
    fg_src = torch.cat(fg_img_idx, dim=0)
    if bg_list:
        bg = torch.cat(bg_list, dim=0)
        bg_src = torch.cat(bg_img_idx, dim=0)
        bg_yx = torch.cat(bg_coords, dim=0)
    else:
        bg = fg.new_empty((0, fg.shape[1]))
        bg_src = fg_src.new_empty((0,))
        bg_yx = fg_src.new_empty((0, 2))

    return {
        "fg": fg,
        "fg_src": fg_src,
        "bg": bg,
        "bg_src": bg_src,
        "bg_yx": bg_yx,
        "fg_masks_32": fg_masks_32,
    }


def _topk_mean(sim: torch.Tensor, top_m: int) -> torch.Tensor:
    if sim.shape[1] == 0:
        return sim.new_zeros((sim.shape[0],))
    m = min(top_m, sim.shape[1])
    return sim.topk(m, dim=1).values.mean(dim=1)


def score_raw_tokens(
    q_tokens: torch.Tensor,
    fg_bank: torch.Tensor,
    bg_bank: torch.Tensor,
    *,
    top_m: int = 3,
    alpha_bg: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sim_fg = (q_tokens @ fg_bank.T).clamp(-1.0, 1.0)
    s_pos = _topk_mean(sim_fg, top_m)
    if bg_bank.shape[0] > 0 and alpha_bg > 0.0:
        sim_bg = (q_tokens @ bg_bank.T).clamp(-1.0, 1.0)
        s_neg = _topk_mean(sim_bg, top_m)
        raw = s_pos - alpha_bg * s_neg
    else:
        s_neg = q_tokens.new_zeros((q_tokens.shape[0],))
        raw = s_pos
    return raw, s_pos, s_neg


@torch.inference_mode()
def calibrate_b0(
    tokens_by_id: dict[str, torch.Tensor],
    pool: dict,
    support_ids: list[str],
    *,
    top_m: int = 3,
    alpha_bg: float = 0.5,
) -> tuple[float, float]:
    """Zero-leakage support-only calibration for Baseline B0."""
    k = len(support_ids)
    if k >= 2:
        fg_scores_all, bg_scores_all = [], []
        for idx, sid in enumerate(support_ids):
            other_fg = pool["fg"][pool["fg_src"] != idx]
            other_bg = pool["bg"][pool["bg_src"] != idx]
            if other_fg.shape[0] == 0:
                continue
            raw, _, _ = score_raw_tokens(tokens_by_id[sid], other_fg, other_bg, top_m=top_m, alpha_bg=alpha_bg)
            fg_mask = torch.from_numpy(pool["fg_masks_32"][idx]).to(raw.device)
            fg_scores_all.append(raw[fg_mask])
            bg_mask_indices = (pool["bg_src"] == idx)
            if bg_mask_indices.any():
                yx = pool["bg_yx"][bg_mask_indices]
                flat_bg = yx[:, 0] * 32 + yx[:, 1]
                bg_scores_all.append(raw[flat_bg])
        if fg_scores_all and bg_scores_all:
            fg_cat = torch.cat(fg_scores_all)
            bg_cat = torch.cat(bg_scores_all)
            mu_fg = float(torch.median(fg_cat).item())
            mu_bg = float(torch.quantile(bg_cat, 0.90).item())
            if mu_fg > mu_bg + 0.02:
                bias = 0.5 * (mu_fg + mu_bg)
                temp = max((mu_fg - mu_bg) / (2.0 * math.log(4.0)), 0.04)
                return bias, temp
            return mu_bg + 0.05, 0.06

    # K == 1 (or fallback): anchor solely on support background excluding 3x3 local neighborhood
    bg = pool["bg"]
    fg = pool["fg"]
    if bg.shape[0] >= 16:
        s_pos_bg = _topk_mean((bg @ fg.T).clamp(-1.0, 1.0), top_m)
        if alpha_bg > 0.0:
            yx = pool["bg_yx"]
            dy = (yx[:, 0:1] - yx[:, 0:1].T).abs()
            dx = (yx[:, 1:2] - yx[:, 1:2].T).abs()
            near = (dy <= 1) & (dx <= 1)
            sim_bg = (bg @ bg.T).clamp(-1.0, 1.0)
            sim_bg = sim_bg.masked_fill(near, -2.0)
            m = min(top_m, max(1, bg.shape[0] - 9))
            s_neg_bg = sim_bg.topk(m, dim=1).values.clamp(-1.0, 1.0).mean(dim=1)
            contrast_bg = s_pos_bg - alpha_bg * s_neg_bg
        else:
            contrast_bg = s_pos_bg
        q90 = float(torch.quantile(contrast_bg, 0.90).item())
        return q90 + 0.08, 0.06

    return (0.30 if alpha_bg > 0.0 else 0.50), 0.06


@torch.inference_mode()
def run_b0_episode(
    tokens_by_id: dict[str, torch.Tensor],
    mask_by_id: dict[str, np.ndarray],
    support_ids: list[str],
    query_ids: list[str],
    *,
    top_m: int = 3,
    alpha_bg: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Run Baseline B0 on one episode; never touches query labels."""
    pool = build_support_pool(tokens_by_id, mask_by_id, support_ids, purity=0.5)
    bias, temp = calibrate_b0(tokens_by_id, pool, support_ids, top_m=top_m, alpha_bg=alpha_bg)

    probs_512 = {}
    scores_32 = {}
    for qid in query_ids:
        raw, _, _ = score_raw_tokens(
            tokens_by_id[qid], pool["fg"], pool["bg"], top_m=top_m, alpha_bg=alpha_bg
        )
        raw_32 = raw.reshape(32, 32)
        p32 = torch.sigmoid((raw_32 - bias) / temp)
        p512 = F.interpolate(p32[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
        scores_32[qid] = raw_32.cpu().numpy().astype(np.float32)
        probs_512[qid] = p512.cpu().numpy().astype(np.float32)

    return probs_512, scores_32, {
        "bias": float(bias),
        "temp": float(temp),
        "n_fg_patches": int(pool["fg"].shape[0]),
        "n_bg_patches": int(pool["bg"].shape[0]),
    }
