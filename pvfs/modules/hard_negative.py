from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..fast_bank import select_fg_bg_indices


@dataclass(frozen=True)
class HardNegConfig:
    enabled: bool = False
    mode: str = "off"  # "off" | "random_bg" | "hard_neg_H"
    per_support_max_tokens: int = 32
    per_support_min_confusion_sim: float = 0.30
    dedup_min_spatial_dist: int = 2
    dedup_max_cosine_sim: float = 0.95
    min_total_tokens: int = 12
    k1_ex_neigh_radius: int = 2
    k1_min_independent_fg_tokens: int = 16
    k1_min_bg_tokens: int = 64
    top_m_bank: int = 3
    t_h_fg_quantile: float = 0.20
    gamma: float = 0.35
    max_penalty: float = 0.25

    @classmethod
    def from_json(cls, path: Path | str, *, mode: str = "hard_neg_H") -> "HardNegConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        hb = raw.get("hard_negative_bank_H", {})
        sc = raw.get("reverification_scoring", {})
        return cls(
            enabled=(mode != "off"),
            mode=mode,
            per_support_max_tokens=int(hb.get("per_support_max_H_tokens", 32)),
            per_support_min_confusion_sim=float(hb.get("per_support_min_confusion_sim", 0.30)),
            dedup_min_spatial_dist=int(hb.get("dedup_min_spatial_dist_tokens", 2)),
            dedup_max_cosine_sim=float(hb.get("dedup_max_cosine_sim", 0.95)),
            min_total_tokens=int(hb.get("min_total_H_tokens_to_activate", 12)),
            k1_ex_neigh_radius=int(hb.get("k1_ex_neigh_radius_tokens", 2)),
            k1_min_independent_fg_tokens=int(hb.get("k1_min_independent_fg_tokens", 16)),
            k1_min_bg_tokens=int(hb.get("k1_min_bg_tokens", 64)),
            gamma=float(sc.get("gamma", 0.35)),
            max_penalty=float(sc.get("max_penalty", 0.25)),
        )


def _greedy_dedup_select(
    cand_indices: np.ndarray,
    tokens_64: torch.Tensor,
    *,
    max_keep: int,
    min_spatial_dist: int,
    max_cos_sim: float,
) -> list[int]:
    """Greedily pick up to max_keep token indices satisfying spatial and cosine deduplication."""
    if len(cand_indices) == 0 or max_keep <= 0:
        return []
    selected: list[int] = []
    sel_yx: list[tuple[int, int]] = []
    sel_vecs: list[torch.Tensor] = []

    for idx_val in cand_indices:
        idx = int(idx_val)
        y, x = divmod(idx, 64)
        if any(max(abs(y - py), abs(x - px)) < min_spatial_dist for py, px in sel_yx):
            continue
        vec = tokens_64[idx]
        if sel_vecs:
            sims = torch.mv(torch.stack(sel_vecs, dim=0), vec)
            if float(sims.max().item()) > max_cos_sim:
                continue
        selected.append(idx)
        sel_yx.append((y, x))
        sel_vecs.append(vec)
        if len(selected) >= max_keep:
            break
    return selected


def _deterministic_seed_from_id(sid: str) -> int:
    digest = hashlib.sha256(sid.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


@torch.inference_mode()
def build_support_bg_subbank(
    tokens_64_by_id: dict[str, torch.Tensor],
    mask_by_id: dict[str, np.ndarray],
    active_support_ids: list[str],
    cfg: HardNegConfig,
    *,
    cache_store: dict[tuple, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct independent per-episode background subbank (H or equal-budget random_bg).

    Strictly uses ONLY active_support_ids (zero leakage of any held-out support or query image)
    and is strictly invariant to the ordering of active_support_ids.
    """
    sorted_sids = tuple(sorted(active_support_ids))
    cache_key = ("hard_neg_H_deep64_v1", cfg.mode, sorted_sids, cfg.per_support_max_tokens, cfg.per_support_min_confusion_sim)
    if cache_store is not None and cache_key in cache_store:
        return cache_store[cache_key]

    k_act = len(sorted_sids)
    dev = tokens_64_by_id[sorted_sids[0]].device if k_act > 0 else torch.device("cpu")
    empty_out = {
        "tokens": torch.empty((0, 1), device=dev),
        "token_indices_by_sid": {},
        "total_tokens": 0,
        "valid": False,
        "reason": "insufficient_active_support" if k_act == 0 else "below_min_total_tokens",
    }
    if k_act == 0:
        return empty_out

    # Step 1: Always compute the hard-negative counts per support image first,
    # so equal-budget random_bg matches the EXACT per-support budget of hard_neg_H.
    fg_purity = 0.50 if k_act == 1 else 0.85
    fg_vecs_by_sid: dict[str, torch.Tensor] = {}
    fg_idx_by_sid: dict[str, np.ndarray] = {}
    bg_idx_by_sid: dict[str, np.ndarray] = {}

    for sid in sorted_sids:
        fg_m, bg_m, _ = select_fg_bg_indices(mask_by_id[sid], 64, fg_purity)
        fg_idx = np.flatnonzero(fg_m)
        bg_idx = np.flatnonzero(bg_m)
        fg_idx_by_sid[sid] = fg_idx
        bg_idx_by_sid[sid] = bg_idx
        fg_vecs_by_sid[sid] = tokens_64_by_id[sid][torch.from_numpy(fg_idx).to(dev)]

    hard_selected_by_sid: dict[str, list[int]] = {}
    for sid in sorted_sids:
        bg_idx = bg_idx_by_sid[sid]
        if len(bg_idx) == 0:
            hard_selected_by_sid[sid] = []
            continue
        tok_s = tokens_64_by_id[sid]
        bg_vecs = tok_s[torch.from_numpy(bg_idx).to(dev)]

        if k_act >= 2:
            other_fg = torch.cat([fg_vecs_by_sid[o] for o in sorted_sids if o != sid], dim=0)
            if other_fg.shape[0] == 0:
                hard_selected_by_sid[sid] = []
                continue
            m_take = min(cfg.top_m_bank, other_fg.shape[0])
            conf_sim = (bg_vecs @ other_fg.T).clamp(-1.0, 1.0).topk(m_take, dim=1).values.mean(dim=1)
        else:
            # Single active support: require enough independent FG and BG tokens outside neighborhood radius
            fg_idx = fg_idx_by_sid[sid]
            if len(fg_idx) < cfg.k1_min_independent_fg_tokens or len(bg_idx) < cfg.k1_min_bg_tokens:
                hard_selected_by_sid[sid] = []
                continue
            bg_yx = torch.stack((torch.from_numpy(bg_idx // 64), torch.from_numpy(bg_idx % 64)), dim=1).to(dev)
            fg_yx = torch.stack((torch.from_numpy(fg_idx // 64), torch.from_numpy(fg_idx % 64)), dim=1).to(dev)
            near = (
                ((bg_yx[:, 0:1] - fg_yx[:, 0:1].T).abs() <= cfg.k1_ex_neigh_radius)
                & ((bg_yx[:, 1:2] - fg_yx[:, 1:2].T).abs() <= cfg.k1_ex_neigh_radius)
            )
            valid_fg_counts = (~near).sum(dim=1)
            sim_mat = (bg_vecs @ fg_vecs_by_sid[sid].T).clamp(-1.0, 1.0).masked_fill(near, -2.0)
            m_take = min(cfg.top_m_bank, fg_vecs_by_sid[sid].shape[0])
            conf_sim = sim_mat.topk(m_take, dim=1).values.mean(dim=1)
            conf_sim = torch.where(valid_fg_counts >= cfg.k1_min_independent_fg_tokens, conf_sim, torch.full_like(conf_sim, -2.0))

        conf_np = conf_sim.detach().cpu().numpy()
        keep_mask = conf_np >= cfg.per_support_min_confusion_sim
        if not np.any(keep_mask):
            hard_selected_by_sid[sid] = []
            continue
        qual_bg = bg_idx[keep_mask]
        qual_scores = conf_np[keep_mask]
        # Stable descending sort by confusion score, tie-break by token index
        order = np.lexsort((qual_bg, -qual_scores))
        hard_selected_by_sid[sid] = _greedy_dedup_select(
            qual_bg[order],
            tok_s,
            max_keep=cfg.per_support_max_tokens,
            min_spatial_dist=cfg.dedup_min_spatial_dist,
            max_cos_sim=cfg.dedup_max_cosine_sim,
        )

    # Step 2: Build final token bank for requested mode
    final_by_sid: dict[str, list[int]] = {}
    bank_chunks: list[torch.Tensor] = []

    for sid in sorted_sids:
        h_list = hard_selected_by_sid[sid]
        if cfg.mode == "hard_neg_H":
            chosen = h_list
        elif cfg.mode == "random_bg":
            n_budget = len(h_list)
            bg_idx = bg_idx_by_sid[sid]
            if n_budget == 0 or len(bg_idx) == 0:
                chosen = []
            else:
                rng = np.random.default_rng(_deterministic_seed_from_id(sid))
                perm = bg_idx[rng.permutation(len(bg_idx))]
                chosen = _greedy_dedup_select(
                    perm,
                    tokens_64_by_id[sid],
                    max_keep=n_budget,
                    min_spatial_dist=cfg.dedup_min_spatial_dist,
                    max_cos_sim=cfg.dedup_max_cosine_sim,
                )
        else:
            chosen = []

        final_by_sid[sid] = chosen
        if chosen:
            bank_chunks.append(tokens_64_by_id[sid][torch.tensor(chosen, dtype=torch.long, device=dev)])

    if bank_chunks:
        bank_tokens = torch.cat(bank_chunks, dim=0)
    else:
        c_dim = tokens_64_by_id[sorted_sids[0]].shape[1]
        bank_tokens = torch.empty((0, c_dim), device=dev, dtype=tokens_64_by_id[sorted_sids[0]].dtype)

    total_n = int(bank_tokens.shape[0])
    valid = total_n >= cfg.min_total_tokens
    out = {
        "tokens": bank_tokens,
        "token_indices_by_sid": final_by_sid,
        "total_tokens": total_n,
        "valid": valid,
        "reason": "ok" if valid else "below_min_total_tokens",
    }
    if cache_store is not None:
        cache_store[cache_key] = out
    return out


@torch.inference_mode()
def compute_margin_against_bank(
    q_tok_64: torch.Tensor,
    f_pos_64: torch.Tensor,
    bank_tokens: torch.Tensor,
    *,
    top_m: int = 3,
    exclude_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute m(q) = F_64(q) - TopMean_3(cos(q, H))."""
    m_take = min(top_m, int(bank_tokens.shape[0]))
    sim = (q_tok_64 @ bank_tokens.T).clamp(-1.0, 1.0)
    if exclude_mask is not None:
        sim = sim.masked_fill(exclude_mask, -2.0)
    top_h = sim.topk(m_take, dim=1).values.mean(dim=1)
    return f_pos_64 - top_h


@torch.inference_mode()
def apply_hard_neg_penalty(
    s_sem: torch.Tensor,
    margin_q: torch.Tensor,
    t_h: float,
    cfg: HardNegConfig,
) -> torch.Tensor:
    """Apply bounded hard-negative penalty S' = S_sem - gamma * clip(relu(t_H - m(q)), 0, max_penalty)."""
    pen = F.relu(t_h - margin_q).clamp(0.0, cfg.max_penalty)
    return s_sem - cfg.gamma * pen
