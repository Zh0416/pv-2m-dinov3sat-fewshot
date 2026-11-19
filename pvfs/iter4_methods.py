from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from .episodes import load_and_verify_episodes
from .fast_bank import build_pairwise_topk_cache, score_from_pairwise, select_fg_bg_indices
from .io import write_json
from .iter3_methods import (
    _apply_query_self_contrast,
    _ex_neigh_bg_contrast,
    _save_before_after_panels,
    _to_tokens_norm,
    compute_dino_affinity_weights,
    propagate_affinity,
)
from .metrics import aggregate_episodes, compare_methods, evaluate_episode


@dataclass(frozen=True)
class Iter4Config:
    name: str
    use_64: bool = True
    shot_mode: str = "top2_soft"
    k1_use_core: bool = False
    kge2_use_core: bool = True
    calib_v2: bool = True
    w_fg_calib: float = 0.38
    bg_q_calib: float = 0.95
    k1_margin: float = 0.075
    w_s64: float = 0.75
    use_mid_veto: bool = True
    lambda_mid: float = 0.35
    beta_veto: float = 0.40
    w_white: float = 0.35
    w_ridge: float = 0.45
    use_self_rescue: bool = True
    seed_prob_thr: float = 0.68
    min_seeds: int = 5
    min_prominence: float = 0.14
    qsc_weight: float = 0.40
    use_affinity: bool = True
    aff_steps: int = 2
    aff_eta: float = 0.45
    aff_tau: float = 0.12


ITER4_VARIANTS: tuple[Iter4Config, ...] = (
    Iter4Config(
        name="B0_32_Ref",
        use_64=False,
        shot_mode="pooled",
        k1_use_core=False,
        kge2_use_core=False,
        calib_v2=False,
        w_fg_calib=0.50,
        bg_q_calib=0.90,
        k1_margin=0.080,
        use_mid_veto=False,
        w_white=0.0,
        w_ridge=0.0,
        use_self_rescue=False,
        use_affinity=False,
    ),
    Iter4Config(
        name="M1_Full_Ref",
        use_64=True,
        shot_mode="balanced_70_30",
        k1_use_core=True,
        kge2_use_core=True,
        calib_v2=False,
        w_fg_calib=0.45,
        bg_q_calib=0.95,
        k1_margin=0.075,
        w_s64=0.70,
        use_mid_veto=True,
        w_white=0.0,
        w_ridge=0.0,
        use_self_rescue=False,
        use_affinity=False,
    ),
    Iter4Config(
        name="M3_Pruned_Ref",
        w_white=0.0,
        w_ridge=0.0,
        seed_prob_thr=0.66,
        min_seeds=4,
    ),
    Iter4Config(
        name="M4_Whiten",
        w_white=0.45,
        w_ridge=0.0,
    ),
    Iter4Config(
        name="M4_Ridge",
        w_white=0.0,
        w_ridge=0.55,
    ),
    Iter4Config(
        name="M4_Full",
        w_white=0.35,
        w_ridge=0.45,
    ),
    Iter4Config(
        name="M4_NoWhiten_LOO",
        w_white=0.0,
        w_ridge=0.45,
    ),
    Iter4Config(
        name="M4_NoRidge_LOO",
        w_white=0.35,
        w_ridge=0.0,
    ),
    Iter4Config(
        name="M4_NoRescue_LOO",
        w_white=0.35,
        w_ridge=0.45,
        use_self_rescue=False,
    ),
    Iter4Config(
        name="M4_NoAff_LOO",
        w_white=0.35,
        w_ridge=0.45,
        use_affinity=False,
    ),
)


def _fit_whitening_transform(
    fg_per_shot: list[torch.Tensor],
    bg_bank: torch.Tensor,
    *,
    lam_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Fit background-nullspace shrinkage whitener (attenuates clutter variance without re-stretching)."""
    dim = fg_per_shot[0].shape[1]
    dev = fg_per_shot[0].device
    if bg_bank.shape[0] < 16:
        eye = torch.eye(dim, device=dev, dtype=torch.float32)
        mu = torch.zeros((1, dim), device=dev, dtype=torch.float32)
        return mu, eye, fg_per_shot, bg_bank

    mu = bg_bank.mean(dim=0, keepdim=True)
    bc = bg_bank - mu
    cov = (bc.T @ bc) / float(bg_bank.shape[0])
    lam = max(float(cov.trace().item()) * lam_frac, 1e-3)
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(0.0)
    # Orthogonal signal axes keep weight ~1.0; high-variance background axes shrink by sqrt(lam/(eval+lam))
    shrink = torch.sqrt(lam / (evals + lam))
    w_mat = (evecs * shrink.unsqueeze(0)) @ evecs.T
    fg_w = [F.normalize((f - mu) @ w_mat, dim=1) for f in fg_per_shot]
    bg_w = F.normalize(bc @ w_mat, dim=1)
    return mu, w_mat, fg_w, bg_w


def _score_whitened_knn(
    q_tok: torch.Tensor,
    mu: torch.Tensor,
    w_mat: torch.Tensor,
    fg_w_list: list[torch.Tensor],
    bg_w: torch.Tensor,
    *,
    shot_mode: str,
    top_m: int = 3,
    alpha_bg: float = 0.50,
) -> torch.Tensor:
    # Keep projected query norm attenuation along high-variance bg clutter directions
    q_p = ((q_tok - mu) @ w_mat).clamp(-2.0, 2.0)
    if len(fg_w_list) == 1 or shot_mode == "pooled":
        f_all = torch.cat(fg_w_list, dim=0)
        s_pos = (q_p @ f_all.T).clamp(-1.0, 1.0).topk(min(top_m, f_all.shape[0]), dim=1).values.mean(dim=1)
    else:
        per_shot = []
        for fw in fg_w_list:
            m = min(top_m, fw.shape[0])
            per_shot.append((q_p @ fw.T).clamp(-1.0, 1.0).topk(m, dim=1).values.mean(dim=1))
        st = torch.stack(per_shot, dim=0)  # [K', N_q]
        if shot_mode == "balanced_70_30":
            s_pos = 0.70 * st.max(dim=0).values + 0.30 * st.mean(dim=0)
        else:
            st_s = st.sort(dim=0, descending=True).values
            if st_s.shape[0] == 2:
                s_pos = 0.75 * st_s[0] + 0.25 * st_s[1]
            else:
                s_pos = 0.65 * st_s[0] + 0.25 * st_s[1] + 0.10 * st_s[2:].mean(dim=0)

    if bg_w.shape[0] > 0:
        s_neg = (q_p @ bg_w.T).clamp(-1.0, 1.0).topk(min(top_m, bg_w.shape[0]), dim=1).values.mean(dim=1)
        return s_pos - alpha_bg * s_neg
    return s_pos


def _fit_balanced_ridge(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    *,
    lam_frac: float,
) -> torch.Tensor:
    """Closed-form trace-regularized class-balanced Ridge discriminant on support tokens."""
    dim = x_pos.shape[1]
    dev = x_pos.device
    if x_pos.shape[0] == 0 or x_neg.shape[0] == 0:
        return torch.zeros((dim,), device=dev, dtype=torch.float32)
    cov_p = (x_pos.T @ x_pos) / float(2 * x_pos.shape[0])
    cov_n = (x_neg.T @ x_neg) / float(2 * x_neg.shape[0])
    a_mat = cov_p + cov_n
    lam = max(float(a_mat.trace().item()) * lam_frac, 1e-3)
    a_mat.diagonal().add_(lam)
    b_vec = 0.5 * x_pos.mean(dim=0) - 0.5 * x_neg.mean(dim=0)
    w = torch.linalg.solve(a_mat, b_vec)
    return w / w.norm().clamp_min(1e-6)


def _score_ridge_bounded(tokens: torch.Tensor, w_r: torch.Tensor) -> torch.Tensor:
    # Floor extreme negative background response so support calibration doesn't pull threshold below zero
    return (tokens @ w_r).clamp(min=-0.12, max=0.75)


def _k1_spatial_oof_halves(fg_mask: np.ndarray, bg_mask: np.ndarray, grid_hw: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Split a single support image into two spatially disjoint halves containing balanced fg & bg."""
    fg_yx = np.argwhere(fg_mask.reshape(grid_hw, grid_hw))
    yy, xx = np.mgrid[0:grid_hw, 0:grid_hw]
    if len(fg_yx) >= 4:
        span_y = int(fg_yx[:, 0].max() - fg_yx[:, 0].min())
        span_x = int(fg_yx[:, 1].max() - fg_yx[:, 1].min())
        if span_y >= span_x:
            split = float(np.median(fg_yx[:, 0]))
            half0 = (yy <= split).ravel()
        else:
            split = float(np.median(fg_yx[:, 1]))
            half0 = (xx <= split).ravel()
    else:
        half0 = (yy < grid_hw // 2).ravel()
    half1 = ~half0
    return half0, half1


def _joint_tok(tokens_by_key: dict[str, dict[str, torch.Tensor]], rid: str) -> torch.Tensor:
    return F.normalize(torch.cat((tokens_by_key["deep_64"][rid], 0.55 * tokens_by_key["mid_64"][rid]), dim=1), dim=1)


def _compute_episode_component_maps(
    caches: dict[str, dict],
    tokens_by_key: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    support_ids: list[str],
    query_ids: list[str],
    *,
    use_core: bool,
    shot_mode: str,
) -> dict:
    """Compute base/whiten/ridge 64x64 components once per episode (zero query leakage)."""
    k = len(support_ids)
    core_purity = 0.85 if use_core else 0.50

    fg_masks, bg_masks = [], []
    fg_64_list, bg_64_list = [], []
    fg_j_list, bg_j_list = [], []
    for sid in support_ids:
        fg_m, bg_m, _ = select_fg_bg_indices(mask_by_id[sid], 64, core_purity)
        fg_masks.append(fg_m)
        bg_masks.append(bg_m)
        fg_idx = torch.from_numpy(np.flatnonzero(fg_m)).to(tokens_by_key["deep_64"][sid].device)
        bg_idx = torch.from_numpy(np.flatnonzero(bg_m)).to(tokens_by_key["deep_64"][sid].device)
        t64 = tokens_by_key["deep_64"][sid]
        tj = _joint_tok(tokens_by_key, sid)
        fg_64_list.append(t64[fg_idx])
        bg_64_list.append(t64[bg_idx] if bg_idx.numel() > 0 else t64.new_empty((0, t64.shape[1])))
        fg_j_list.append(tj[fg_idx])
        bg_j_list.append(tj[bg_idx] if bg_idx.numel() > 0 else tj.new_empty((0, tj.shape[1])))

    lam_w = 0.06 if k == 1 else (0.03 if k <= 3 else 0.02)
    lam_r = 0.08 if k == 1 else (0.04 if k <= 3 else 0.025)

    def _whiten_ridge_for_subset(sub_indices: list[int], target_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        f64_s = [fg_64_list[i] for i in sub_indices]
        b64_s = torch.cat([bg_64_list[i] for i in sub_indices], dim=0)
        mu, w_mat, fw, bw = _fit_whitening_transform(f64_s, b64_s, lam_frac=lam_w)
        s_w = _score_whitened_knn(tokens_by_key["deep_64"][target_id], mu, w_mat, fw, bw, shot_mode=shot_mode)

        xp = torch.cat([fg_j_list[i] for i in sub_indices], dim=0)
        xn = torch.cat([bg_j_list[i] for i in sub_indices], dim=0)
        w_r = _fit_balanced_ridge(xp, xn, lam_frac=lam_r)
        s_r = _score_ridge_bounded(_joint_tok(tokens_by_key, target_id), w_r)
        return s_w, s_r

    # Full-support whitener & ridge weights for queries
    b64_full = torch.cat(bg_64_list, dim=0)
    mu_full, wmat_full, fw_full, bw_full = _fit_whitening_transform(fg_64_list, b64_full, lam_frac=lam_w)
    w_ridge_full = _fit_balanced_ridge(torch.cat(fg_j_list, dim=0), torch.cat(bg_j_list, dim=0), lam_frac=lam_r)

    q_white, q_ridge = {}, {}
    for qid in query_ids:
        q_white[qid] = _score_whitened_knn(tokens_by_key["deep_64"][qid], mu_full, wmat_full, fw_full, bw_full, shot_mode=shot_mode)
        q_ridge[qid] = _score_ridge_bounded(_joint_tok(tokens_by_key, qid), w_ridge_full)

    # Support OOF scores for zero-leakage calibration
    s_white, s_ridge = {}, {}
    if k >= 2:
        all_idx = list(range(k))
        for idx, sid in enumerate(support_ids):
            sub = [i for i in all_idx if i != idx]
            s_white[sid], s_ridge[sid] = _whiten_ridge_for_subset(sub, sid)
    else:
        sid = support_ids[0]
        h0, h1 = _k1_spatial_oof_halves(fg_masks[0], bg_masks[0], 64)
        t64 = tokens_by_key["deep_64"][sid]
        tj = _joint_tok(tokens_by_key, sid)
        w_out = torch.zeros(4096, dtype=torch.float32, device=t64.device)
        r_out = torch.zeros(4096, dtype=torch.float32, device=t64.device)
        for train_h, test_h in ((h0, h1), (h1, h0)):
            tr_fg = np.flatnonzero(fg_masks[0] & train_h)
            tr_bg = np.flatnonzero(bg_masks[0] & train_h)
            te_all = torch.from_numpy(np.flatnonzero(test_h)).to(t64.device)
            if len(tr_fg) >= 4 and len(tr_bg) >= 32:
                tf_t = torch.from_numpy(tr_fg).to(t64.device)
                tb_t = torch.from_numpy(tr_bg).to(t64.device)
                mu_h, wm_h, fw_h, bw_h = _fit_whitening_transform([t64[tf_t]], t64[tb_t], lam_frac=lam_w)
                w_out[te_all] = _score_whitened_knn(t64[te_all], mu_h, wm_h, fw_h, bw_h, shot_mode="pooled")
                wr_h = _fit_balanced_ridge(tj[tf_t], tj[tb_t], lam_frac=lam_r)
                r_out[te_all] = _score_ridge_bounded(tj[te_all], wr_h)
            else:
                w_out[te_all] = _score_whitened_knn(t64[te_all], mu_full, wmat_full, fw_full, bw_full, shot_mode="pooled")
                r_out[te_all] = _score_ridge_bounded(tj[te_all], w_ridge_full)
        s_white[sid], s_ridge[sid] = w_out, r_out

    return {
        "q_white": q_white,
        "q_ridge": q_ridge,
        "s_white": s_white,
        "s_ridge": s_ridge,
    }


@torch.inference_mode()
def run_iter4_episode(
    caches: dict[str, dict],
    tokens_by_key: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    aff_by_id: dict[str, torch.Tensor] | None,
    support_ids: list[str],
    query_ids: list[str],
    cfg: Iter4Config,
    precomp_wr: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    k = len(support_ids)
    use_core = cfg.k1_use_core if k == 1 else cfg.kge2_use_core
    core_purity = 0.85 if use_core else 0.50
    use_core_calib = use_core and not cfg.calib_v2
    need_wr = (cfg.w_white != 0.0) or (cfg.w_ridge != 0.0)
    if need_wr and precomp_wr is None:
        precomp_wr = _compute_episode_component_maps(
            caches, tokens_by_key, mask_by_id, support_ids, query_ids, use_core=use_core, shot_mode=cfg.shot_mode
        )

    mid_veto_ref = None
    if cfg.use_64 and cfg.use_mid_veto:
        if k >= 2:
            mid_vals = []
            for idx, sid in enumerate(support_ids):
                sm, _, _ = score_from_pairwise(
                    caches["mid_64"], sid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=idx
                )
                bm = caches["mid_64"]["bg_masks"][caches["mid_64"]["id_to_pos"][sid]]
                if bm.any():
                    mid_vals.append(sm[torch.from_numpy(np.flatnonzero(bm)).to(sm.device)])
            if mid_vals:
                mid_veto_ref = float(torch.quantile(torch.cat(mid_vals), 0.85).item())
        else:
            c_bg_m = _ex_neigh_bg_contrast(tokens_by_key["mid_64"][support_ids[0]], mask_by_id[support_ids[0]], 64, core_purity=core_purity)
            if c_bg_m is not None:
                mid_veto_ref = float(torch.quantile(c_bg_m, 0.85).item())
        if mid_veto_ref is None:
            mid_veto_ref = 0.20

    def raw_base(qid: str, ex_idx: int | None) -> torch.Tensor:
        s32, _, _ = score_from_pairwise(
            caches["deep_32"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx
        )
        if not cfg.use_64:
            return s32
        s64, _, _ = score_from_pairwise(
            caches["deep_64"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx
        )
        s32_up = F.interpolate(s32.reshape(1, 1, 32, 32), size=(64, 64), mode="bilinear", align_corners=False).reshape(-1)
        comb = cfg.w_s64 * s64 + (1.0 - cfg.w_s64) * s32_up
        if cfg.use_mid_veto:
            sm, _, _ = score_from_pairwise(
                caches["mid_64"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx
            )
            comb = comb + cfg.lambda_mid * sm - cfg.beta_veto * F.softplus((mid_veto_ref - sm) * 10.0) / 10.0
        return comb

    if k >= 2:
        ref_key = "deep_64" if cfg.use_64 else "deep_32"
        fg_items, bg_list = [], []
        for idx, sid in enumerate(support_ids):
            comb_s = raw_base(sid, idx)
            if need_wr:
                comb_s = comb_s + cfg.w_white * precomp_wr["s_white"][sid] + cfg.w_ridge * precomp_wr["s_ridge"][sid]
            p_i = caches[ref_key]["id_to_pos"][sid]
            fg_m = caches[ref_key]["fg_core_masks"][p_i] if use_core_calib else caches[ref_key]["fg_all_masks"][p_i]
            bg_m = caches[ref_key]["bg_masks"][p_i]
            fg_idx = torch.from_numpy(np.flatnonzero(fg_m)).to(comb_s.device)
            if cfg.calib_v2 or cfg.name == "M1_Full_Ref":
                fg_items.append(float(torch.median(comb_s[fg_idx]).item()))
            else:
                fg_items.append(comb_s[fg_idx])
            if bg_m.any():
                bg_list.append(comb_s[torch.from_numpy(np.flatnonzero(bg_m)).to(comb_s.device)])
        if bg_list:
            mu_fg = float(np.mean(fg_items)) if isinstance(fg_items[0], float) else float(torch.median(torch.cat(fg_items)).item())
            mu_bg = float(torch.quantile(torch.cat(bg_list), cfg.bg_q_calib).item())
            if mu_fg > mu_bg + 0.02:
                bias = cfg.w_fg_calib * mu_fg + (1.0 - cfg.w_fg_calib) * mu_bg
                temp = max((mu_fg - mu_bg) / (2.0 * math.log(4.0)), 0.04)
            else:
                bias, temp = mu_bg + 0.05, 0.06
        else:
            bias, temp = 0.30, 0.06
    else:
        c32_b = _ex_neigh_bg_contrast(tokens_by_key["deep_32"][support_ids[0]], mask_by_id[support_ids[0]], 32, core_purity=core_purity)
        if not cfg.use_64:
            bias = (float(torch.quantile(c32_b, cfg.bg_q_calib).item()) + cfg.k1_margin) if c32_b is not None else 0.30
        else:
            sid0 = support_ids[0]
            c64_b = _ex_neigh_bg_contrast(tokens_by_key["deep_64"][sid0], mask_by_id[sid0], 64, core_purity=core_purity)
            cm_b = _ex_neigh_bg_contrast(tokens_by_key["mid_64"][sid0], mask_by_id[sid0], 64, core_purity=core_purity) if cfg.use_mid_veto else None
            if c64_b is not None and c32_b is not None:
                _, bg64_m, _ = select_fg_bg_indices(mask_by_id[sid0], 64, core_purity)
                _, bg32_m, _ = select_fg_bg_indices(mask_by_id[sid0], 32, core_purity)
                bg32_flat = np.flatnonzero(bg32_m)
                c32_full = torch.zeros((32, 32), device=c64_b.device)
                c32_full[torch.from_numpy(bg32_flat // 32).to(c64_b.device), torch.from_numpy(bg32_flat % 32).to(c64_b.device)] = c32_b
                bg64_flat = np.flatnonzero(bg64_m)
                bg64_t = torch.from_numpy(bg64_flat).to(c64_b.device)
                c32_at_64 = c32_full[(bg64_t // 64) // 2, (bg64_t % 64) // 2]
                bg_s = cfg.w_s64 * c64_b + (1.0 - cfg.w_s64) * c32_at_64
                if cfg.use_mid_veto and cm_b is not None and mid_veto_ref is not None:
                    bg_s = bg_s + cfg.lambda_mid * cm_b - cfg.beta_veto * F.softplus((mid_veto_ref - cm_b) * 10.0) / 10.0
                if need_wr:
                    bg_s = bg_s + cfg.w_white * precomp_wr["s_white"][sid0][bg64_t] + cfg.w_ridge * precomp_wr["s_ridge"][sid0][bg64_t]
                bias = float(torch.quantile(bg_s, cfg.bg_q_calib).item()) + cfg.k1_margin
            else:
                bias = 0.30
        temp = 0.06

    out_hw = 64 if cfg.use_64 else 32
    probs_512, scores_32 = {}, {}
    for qid in query_ids:
        comb = raw_base(qid, None)
        if need_wr:
            comb = comb + cfg.w_white * precomp_wr["q_white"][qid] + cfg.w_ridge * precomp_wr["q_ridge"][qid]

        p_base = torch.sigmoid((comb - bias) / temp)
        if cfg.use_64 and cfg.use_self_rescue:
            comb_fin = _apply_query_self_contrast(comb, p_base, tokens_by_key["deep_64"][qid], cfg)
            p_grid = torch.sigmoid((comb_fin - bias) / temp)
            s32_out = F.avg_pool2d(comb_fin.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
        else:
            p_grid = p_base
            s32_out = comb.reshape(32, 32) if out_hw == 32 else F.avg_pool2d(comb.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]

        if cfg.use_64 and cfg.use_affinity:
            w_aff = aff_by_id[qid] if aff_by_id is not None else compute_dino_affinity_weights(tokens_by_key["deep_64"][qid], tau=cfg.aff_tau)
            p_grid = propagate_affinity(p_grid.reshape(64, 64), w_aff, steps=cfg.aff_steps, eta=cfg.aff_eta)

        p512 = F.interpolate(p_grid.reshape(1, 1, out_hw, out_hw), size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
        probs_512[qid] = p512.cpu().numpy().astype(np.float32)
        scores_32[qid] = s32_out.cpu().numpy().astype(np.float32)
    return probs_512, scores_32


def run_iter4_ablations(
    *,
    split_dir: Path,
    cache_pt: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    device: str = "cuda",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)

    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    records_by_id, mask_by_id = {}, {}
    for row in dev_records:
        rid = row["id"]
        mask = read_verified_mask(lbl_dir, row)
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}
        mask_by_id[rid] = mask

    caches: dict[str, dict] = {}
    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64)):
        toks = {rid: _to_tokens_norm(feats[rid][key], run_device) for rid in all_ids}
        caches[key] = build_pairwise_topk_cache(toks, mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        tokens_by_key[key] = toks
        if run_device == "cuda":
            torch.cuda.empty_cache()

    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=0.12) for rid in all_ids}

    ep_by_k = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    # Precompute episode Whitening + Ridge component maps once per (K, ep)
    wr_cache: dict[tuple[int, int], dict] = {}
    for k, eps in sorted(ep_by_k.items()):
        use_core = False if k == 1 else True
        for ei, ep in enumerate(eps):
            wr_cache[(k, ei)] = _compute_episode_component_maps(
                caches,
                tokens_by_key,
                mask_by_id,
                ep["support_ids"],
                ep["query_ids"],
                use_core=use_core,
                shot_mode="top2_soft",
            )
    print(f"Precomputed Whitening + Ridge maps for {len(wr_cache)} episodes.", flush=True)

    panel_targets = [
        "正样本/PV-55.tif",
        "正样本/PV-49.tif",
        "正样本/PV-73.tif",
        "正样本/PV-80.tif",
        "负样本/PV-93.tif",
        "负样本/PV-100.tif",
        "负样本/PV-96.tif",
    ]
    panel_store: dict[str, dict[str, list[np.ndarray]]] = {
        v: {t: [] for t in panel_targets} for v in ("B0_32_Ref", "M3_Pruned_Ref", "M4_Full")
    }

    results: dict[str, dict] = {cfg.name: {} for cfg in ITER4_VARIANTS}
    for cfg in ITER4_VARIANTS:
        for k, eps in sorted(ep_by_k.items()):
            ep_evals = []
            for ei, ep in enumerate(eps):
                need_wr = (cfg.w_white != 0.0) or (cfg.w_ridge != 0.0)
                pre_wr = wr_cache[(k, ei)] if need_wr else None
                p512, s32 = run_iter4_episode(
                    caches, tokens_by_key, mask_by_id, aff_by_id, ep["support_ids"], ep["query_ids"], cfg, precomp_wr=pre_wr
                )
                ep_evals.append(evaluate_episode([records_by_id[qid] for qid in ep["query_ids"]], p512, s32, threshold=0.5))
                if k == 5 and cfg.name in panel_store:
                    for t in panel_targets:
                        if t in p512:
                            panel_store[cfg.name][t].append(p512[t])
            agg = aggregate_episodes(ep_evals, episodes_per_fold=episodes_lock["episodes_per_fold"])
            results[cfg.name][str(k)] = agg
            m = agg["macro"]
            sd = agg["strata"].get("small_lt5pct", {}).get("dice", 0.0)
            sr = agg["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            print(
                f"[Iter4] {cfg.name:16s} K={k}: Dice={m['dice']:.4f} AP512={m['ap_512']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} BF1={m['boundary_f1']:.4f} "
                f"NegFP={m['neg_fp']:.4f} PosBg={m['pos_bg_fp']:.4f} SmallD={sd:.4f} SmallR={sr:.4f}",
                flush=True,
            )

    mean_p = {
        v: {t: np.mean(np.stack(lst, axis=0), axis=0) for t, lst in by_t.items() if lst}
        for v, by_t in panel_store.items()
    }
    fn_panel = [t for t in ("正样本/PV-55.tif", "正样本/PV-49.tif", "正样本/PV-73.tif") if t in mean_p["M4_Full"]]
    fp_panel = [t for t in ("正样本/PV-80.tif", "负样本/PV-93.tif", "负样本/PV-100.tif", "负样本/PV-96.tif") if t in mean_p["M4_Full"]]
    if fn_panel:
        _save_before_after_panels(output_dir / "panels", img_dir, records_by_id, mean_p["B0_32_Ref"], mean_p["M3_Pruned_Ref"], mean_p["M4_Full"], fn_panel, "k5_fn_compare.png")
    if fp_panel:
        _save_before_after_panels(output_dir / "panels", img_dir, records_by_id, mean_p["B0_32_Ref"], mean_p["M3_Pruned_Ref"], mean_p["M4_Full"], fp_panel, "k5_fp_compare.png")

    pairs = [
        ("M4_Whiten", "M3_Pruned_Ref"),
        ("M4_Ridge", "M3_Pruned_Ref"),
        ("M4_Full", "M3_Pruned_Ref"),
        ("M4_Full", "M4_NoWhiten_LOO"),
        ("M4_Full", "M4_NoRidge_LOO"),
        ("M4_Full", "M4_NoRescue_LOO"),
        ("M4_Full", "M4_NoAff_LOO"),
        ("M4_Full", "M1_Full_Ref"),
        ("M4_Full", "B0_32_Ref"),
        ("M4_Ridge", "B0_32_Ref"),
        ("M4_Whiten", "B0_32_Ref"),
        ("M3_Pruned_Ref", "B0_32_Ref"),
    ]
    comparisons = {
        f"{cand}_vs_{ref}": {
            str(k): compare_methods(results[cand][str(k)], results[ref][str(k)])
            for k in sorted(ep_by_k)
        }
        for cand, ref in pairs
    }

    error_taxonomy = {}
    for variant in ("M3_Pruned_Ref", "M4_Ridge", "M4_Full"):
        error_taxonomy[variant] = {}
        for k_str in ("1", "3", "5"):
            imgs = list(results[variant][k_str]["images"].values())
            pos_imgs = [v for v in imgs if v["positive"]]
            neg_imgs = [v for v in imgs if not v["positive"]]
            error_taxonomy[variant][k_str] = {
                "worst_fn": sorted(pos_imgs, key=lambda v: v["recall"])[:8],
                "worst_neg_fp": sorted(neg_imgs, key=lambda v: -v["neg_fp"])[:8],
                "worst_pos_bg_fp": sorted(pos_imgs, key=lambda v: -v["pos_bg_fp"])[:8],
            }

    summary = {
        "iteration": 4,
        "methods": results,
        "comparisons": comparisons,
        "error_taxonomy": error_taxonomy,
    }
    write_json(output_dir / "iter4_summary.json", summary)
    return summary
