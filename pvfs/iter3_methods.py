from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
from torch.nn import functional as F

from .data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from .episodes import load_and_verify_episodes
from .fast_bank import build_pairwise_topk_cache, score_from_pairwise, select_fg_bg_indices
from .io import write_json
from .metrics import aggregate_episodes, compare_methods, evaluate_episode


@dataclass(frozen=True)
class Iter3Config:
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
    use_shot_dehub: bool = True
    use_self_rescue: bool = True
    seed_prob_thr: float = 0.66
    min_seeds: int = 4
    min_prominence: float = 0.14
    qsc_weight: float = 0.40
    use_affinity: bool = True
    aff_steps: int = 2
    aff_eta: float = 0.45
    aff_tau: float = 0.12
    use_guided: bool = True
    gf_radius: int = 5
    gf_eps: float = 0.002
    gf_blend: float = 0.75
    speckle_max_area: int = 24
    speckle_max_peak: float = 0.70


ITER3_VARIANTS: tuple[Iter3Config, ...] = (
    Iter3Config(
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
        use_shot_dehub=False,
        use_self_rescue=False,
        use_affinity=False,
        use_guided=False,
    ),
    Iter3Config(
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
        use_shot_dehub=False,
        use_self_rescue=False,
        use_affinity=False,
        use_guided=False,
    ),
    Iter3Config(
        name="M3_Base_CalDehub",
        use_shot_dehub=True,
        use_self_rescue=False,
        use_affinity=False,
        use_guided=False,
    ),
    Iter3Config(
        name="M3_Plus_Rescue",
        use_shot_dehub=True,
        use_self_rescue=True,
        use_affinity=False,
        use_guided=False,
    ),
    Iter3Config(
        name="M3_Plus_Affinity",
        use_shot_dehub=True,
        use_self_rescue=True,
        use_affinity=True,
        use_guided=False,
    ),
    Iter3Config(
        name="M3_Full",
        use_shot_dehub=True,
        use_self_rescue=True,
        use_affinity=True,
        use_guided=True,
    ),
    Iter3Config(
        name="M3_NoDehub_LOO",
        use_shot_dehub=False,
        use_self_rescue=True,
        use_affinity=True,
        use_guided=True,
    ),
    Iter3Config(
        name="M3_NoRescue_LOO",
        use_shot_dehub=True,
        use_self_rescue=False,
        use_affinity=True,
        use_guided=True,
    ),
    Iter3Config(
        name="M3_NoAff_LOO",
        use_shot_dehub=True,
        use_self_rescue=True,
        use_affinity=False,
        use_guided=True,
    ),
    Iter3Config(
        name="M3_Cal28_LOO",
        w_fg_calib=0.28,
        use_shot_dehub=True,
        use_self_rescue=True,
        use_affinity=True,
        use_guided=True,
    ),
)


def _to_tokens_norm(feat_map: torch.Tensor, device: str) -> torch.Tensor:
    t = feat_map.to(device=device, dtype=torch.float32)
    t = F.normalize(t, dim=0)
    return t.flatten(1).T.contiguous()


def _ex_neigh_bg_contrast(
    tok: torch.Tensor,
    mask_512: np.ndarray,
    grid_hw: int,
    *,
    core_purity: float,
    alpha_bg: float = 0.50,
    top_m: int = 3,
) -> torch.Tensor | None:
    radius = 2 if grid_hw == 64 else 1
    min_bg = 64 if grid_hw == 64 else 16
    fg_c, bg_m, _ = select_fg_bg_indices(mask_512, grid_hw, core_purity)
    bg_flat = np.flatnonzero(bg_m)
    if len(bg_flat) < min_bg:
        return None
    fg_t = tok[torch.from_numpy(np.flatnonzero(fg_c)).to(tok.device)]
    bg_idx = torch.from_numpy(bg_flat).to(tok.device)
    bg_t = tok[bg_idx]
    yx = torch.stack((bg_idx // grid_hw, bg_idx % grid_hw), dim=1)
    s_pos = (bg_t @ fg_t.T).clamp(-1.0, 1.0).topk(min(top_m, fg_t.shape[0]), dim=1).values.mean(dim=1)
    near = ((yx[:, 0:1] - yx[:, 0:1].T).abs() <= radius) & ((yx[:, 1:2] - yx[:, 1:2].T).abs() <= radius)
    vc = int((~near).sum(dim=1).min().item())
    if vc < 1:
        return None
    s_neg = (bg_t @ bg_t.T).clamp(-1.0, 1.0).masked_fill(near, -2.0).topk(min(top_m, vc), dim=1).values.mean(dim=1)
    return s_pos - alpha_bg * s_neg


def _compute_shot_offsets(
    cache: dict,
    support_ids: list[str],
    *,
    use_core: bool,
    exclude_pos_idx: int | None = None,
) -> torch.Tensor | None:
    """Zero-leakage hubness baseline per support shot from other episode support backgrounds."""
    active = [(i, sid) for i, sid in enumerate(support_ids) if i != exclude_pos_idx]
    if len(active) <= 1:
        return None
    fg_src = cache["fg_core_top"] if use_core else cache["fg_all_top"]
    offsets = []
    for i, sid in active:
        si = cache["id_to_pos"][sid]
        bg_vals = []
        for j, other_id in active:
            if i == j:
                continue
            oi_pos = cache["id_to_pos"][other_id]
            bm = cache["bg_masks"][oi_pos]
            if bm.any():
                oq_all = cache["id_to_all"][other_id]
                bg_idx = torch.from_numpy(np.flatnonzero(bm)).to(fg_src.device)
                bg_vals.append(fg_src[si, oq_all, bg_idx].float().mean(dim=1).mean())
        offsets.append(torch.stack(bg_vals).mean() if bg_vals else fg_src.new_tensor(0.0, dtype=torch.float32))
    off_t = torch.stack(offsets)
    return off_t - off_t.mean()


def _score_dehubbed(
    cache: dict,
    q_id: str,
    support_ids: list[str],
    *,
    use_core: bool,
    shot_mode: str,
    shot_offsets: torch.Tensor | None,
    exclude_pos_idx: int | None = None,
) -> torch.Tensor:
    if shot_offsets is None or shot_mode != "top2_soft":
        sc, _, _ = score_from_pairwise(
            cache,
            q_id,
            support_ids,
            use_core=use_core,
            shot_mode=shot_mode,
            alpha_bg=0.50,
            exclude_pos_idx=exclude_pos_idx,
        )
        return sc

    q_idx = cache["id_to_all"][q_id]
    s_indices = [cache["id_to_pos"][sid] for i, sid in enumerate(support_ids) if i != exclude_pos_idx]
    m = cache["top_m"]
    fg_src = cache["fg_core_top"] if use_core else cache["fg_all_top"]
    fg_slices = torch.stack([fg_src[si, q_idx] for si in s_indices], dim=0).float()  # [K', N_q, m]
    valid = fg_slices > -1.5
    per_shot = (fg_slices * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1).float()  # [K', N_q]
    per_shot = per_shot - shot_offsets[:, None]
    per_sorted = per_shot.sort(dim=0, descending=True).values
    k_eff = per_sorted.shape[0]
    if k_eff == 2:
        s_pos = 0.75 * per_sorted[0] + 0.25 * per_sorted[1]
    else:
        s_pos = 0.65 * per_sorted[0] + 0.25 * per_sorted[1] + 0.10 * per_sorted[2:].mean(dim=0)

    bg_slices = torch.stack([cache["bg_top"][si, q_idx] for si in s_indices], dim=0).float()
    if (bg_slices > -1.5).any():
        cat = bg_slices.permute(1, 0, 2).reshape(-1, k_eff * m)
        topv = cat.topk(m, dim=1).values
        bv = topv > -1.5
        s_neg = torch.where(bv.any(dim=1), (topv * bv).sum(dim=1) / bv.sum(dim=1).clamp_min(1).float(), 0.0)
    else:
        s_neg = torch.zeros_like(s_pos)
    return s_pos - 0.50 * s_neg


def _apply_query_self_contrast(
    base_comb: torch.Tensor,
    p_base: torch.Tensor,
    q_tok_64: torch.Tensor,
    cfg: Iter3Config,
) -> torch.Tensor:
    prominence = float((torch.quantile(base_comb, 0.99) - torch.quantile(base_comb, 0.50)).item())
    if prominence < cfg.min_prominence:
        return base_comb

    seed_mask = p_base >= cfg.seed_prob_thr
    if int(seed_mask.sum().item()) < cfg.min_seeds:
        return base_comb

    bg_mask = p_base <= 0.30
    if int(bg_mask.sum().item()) < 32:
        return base_comb

    seed_idx = torch.nonzero(seed_mask, as_tuple=False).squeeze(1)
    if seed_idx.numel() > 256:
        seed_idx = seed_idx[base_comb[seed_idx].topk(256).indices]

    bg_idx = torch.nonzero(bg_mask, as_tuple=False).squeeze(1)
    if bg_idx.numel() > 1024:
        bg_idx = bg_idx[:: max(1, bg_idx.numel() // 1024)]

    q_fg = q_tok_64[seed_idx]
    q_bg = q_tok_64[bg_idx]
    m_fg = min(3, q_fg.shape[0])
    m_bg = min(3, q_bg.shape[0])

    top_fg = (q_tok_64 @ q_fg.T).clamp(-1.0, 1.0).topk(m_fg, dim=1).values.mean(dim=1)
    top_bg = (q_tok_64 @ q_bg.T).clamp(-1.0, 1.0).topk(m_bg, dim=1).values.mean(dim=1)
    s_self = top_fg - 0.85 * top_bg

    seed_yx = torch.stack((seed_idx // 64, seed_idx % 64), dim=1)
    near_ss = ((seed_yx[:, 0:1] - seed_yx[:, 0:1].T).abs() <= 2) & ((seed_yx[:, 1:2] - seed_yx[:, 1:2].T).abs() <= 2)
    vc_ss = int((~near_ss).sum(dim=1).min().item())
    if vc_ss >= 1:
        fg_loo = (q_fg @ q_fg.T).clamp(-1.0, 1.0).masked_fill(near_ss, -2.0).topk(min(m_fg, vc_ss), dim=1).values.mean(dim=1)
        fg_bg = (q_fg @ q_bg.T).clamp(-1.0, 1.0).topk(m_bg, dim=1).values.mean(dim=1)
        fg_anchor = float(torch.quantile(fg_loo - 0.85 * fg_bg, 0.25).item())
        fg_sim_floor = max(0.55, 0.78 * float(torch.quantile(fg_loo, 0.20).item()))
    else:
        fg_anchor = float(torch.quantile(s_self[seed_idx], 0.25).item())
        fg_sim_floor = max(0.55, 0.78 * float(torch.quantile(top_fg[seed_idx], 0.20).item()))
    bg_anchor = float(torch.quantile(s_self[bg_idx], 0.95).item())
    mid_anchor = 0.45 * fg_anchor + 0.55 * bg_anchor

    delta = (s_self - mid_anchor).clamp(-0.15, 0.35)
    fg_gate = torch.sigmoid((top_fg - fg_sim_floor) / 0.04) * torch.sigmoid(((top_fg - top_bg) - 0.05) / 0.04)
    boost = torch.where(delta > 0.0, cfg.qsc_weight * delta * fg_gate, 0.20 * cfg.qsc_weight * delta)
    return base_comb + boost


def compute_dino_affinity_weights(q_tok_64: torch.Tensor, tau: float = 0.12) -> torch.Tensor:
    """Compute normalized 5x5 local DINO cosine affinity weights [4096, 25] on 64x64 grid."""
    f_map = q_tok_64.T.reshape(1, -1, 64, 64)  # [1, C, 64, 64]
    f_unf = F.unfold(f_map, kernel_size=5, padding=2).reshape(-1, 25, 4096)  # [C, 25, 4096]
    cos_sim = (f_map.reshape(-1, 1, 4096) * f_unf).sum(dim=0).T  # [4096, 25]

    # Spatial Gaussian prior + zero outside valid padded support
    coords = torch.arange(-2, 3, device=q_tok_64.device, dtype=torch.float32)
    dy, dx = torch.meshgrid(coords, coords, indexing="ij")
    sp_w = torch.exp(-(dy * dy + dx * dx) / (2.0 * 1.5 * 1.5)).reshape(1, 25)
    valid = (f_unf.abs().sum(dim=0).T > 0.0)
    log_w = (cos_sim.clamp(0.0, 1.0) - 1.0) / tau
    w = torch.exp(log_w) * sp_w * valid.float()
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)


def propagate_affinity(prob_64: torch.Tensor, weights: torch.Tensor, *, steps: int = 2, eta: float = 0.45) -> torch.Tensor:
    p = prob_64.reshape(1, 1, 64, 64)
    for _ in range(steps):
        p_unf = F.unfold(p, kernel_size=5, padding=2).reshape(25, 4096).T  # [4096, 25]
        p_nbr = (weights * p_unf).sum(dim=1).reshape(1, 1, 64, 64)
        p = (1.0 - eta) * p + eta * p_nbr
    return p.reshape(64, 64).clamp(0.0, 1.0)


def fast_guided_filter_512(
    guide_luma_512: torch.Tensor,
    src_prob_512: torch.Tensor,
    *,
    radius: int = 5,
    eps: float = 0.002,
    blend: float = 0.75,
) -> torch.Tensor:
    """Zero-training RGB luminance-guided filter aligning upsampled edges to 2m image boundaries."""
    k = 2 * radius + 1
    pad = radius
    i_t = guide_luma_512.reshape(1, 1, 512, 512)
    p_t = src_prob_512.reshape(1, 1, 512, 512)

    def box(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad, count_include_pad=False)

    mean_i = box(i_t)
    mean_p = box(p_t)
    corr_ip = box(i_t * p_t)
    var_i = (box(i_t * i_t) - mean_i * mean_i).clamp_min(0.0)
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    q_gf = (box(a) * i_t + box(b)).clamp(0.0, 1.0)
    return (blend * q_gf + (1.0 - blend) * p_t).reshape(512, 512).clamp(0.0, 1.0)


def remove_weak_tiny_speckles(
    prob_512_np: np.ndarray,
    *,
    max_area: int = 24,
    max_peak: float = 0.70,
    threshold: float = 0.50,
) -> np.ndarray:
    if max_area <= 0:
        return prob_512_np
    fg = prob_512_np >= threshold
    if not fg.any():
        return prob_512_np
    labels, n_cc = ndimage.label(fg, structure=np.ones((3, 3), dtype=np.int32))
    if n_cc == 0:
        return prob_512_np
    areas = np.bincount(labels.ravel())
    out = prob_512_np
    copied = False
    for cc_id in range(1, n_cc + 1):
        if areas[cc_id] < max_area:
            mask_cc = labels == cc_id
            if float(prob_512_np[mask_cc].max()) < max_peak:
                if not copied:
                    out = prob_512_np.copy()
                    copied = True
                out[mask_cc] = min(0.48, threshold - 0.02)
    return out


@torch.inference_mode()
def run_iter3_episode(
    caches: dict[str, dict],
    tokens_by_key: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    luma_by_id: dict[str, torch.Tensor] | None,
    aff_by_id: dict[str, torch.Tensor] | None,
    support_ids: list[str],
    query_ids: list[str],
    cfg: Iter3Config,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    k = len(support_ids)
    use_core = cfg.k1_use_core if k == 1 else cfg.kge2_use_core
    core_purity = 0.85 if use_core else 0.50
    use_core_calib = use_core and not cfg.calib_v2

    mid_veto_ref = None
    if cfg.use_64 and cfg.use_mid_veto:
        if k >= 2:
            mid_vals = []
            for idx, sid in enumerate(support_ids):
                off_m = _compute_shot_offsets(caches["mid_64"], support_ids, use_core=use_core, exclude_pos_idx=idx) if cfg.use_shot_dehub else None
                sm = _score_dehubbed(caches["mid_64"], sid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, shot_offsets=off_m, exclude_pos_idx=idx)
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

    # Precompute full-support shot dehub offsets for query inference
    full_offs = {}
    if cfg.use_shot_dehub and k >= 2:
        full_offs["deep_32"] = _compute_shot_offsets(caches["deep_32"], support_ids, use_core=use_core)
        if cfg.use_64:
            full_offs["deep_64"] = _compute_shot_offsets(caches["deep_64"], support_ids, use_core=use_core)
            if cfg.use_mid_veto:
                full_offs["mid_64"] = _compute_shot_offsets(caches["mid_64"], support_ids, use_core=use_core)

    def raw_score(qid: str, ex_idx: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        if ex_idx is None:
            o32 = full_offs.get("deep_32")
            o64 = full_offs.get("deep_64")
            omid = full_offs.get("mid_64")
        elif cfg.use_shot_dehub and k >= 2:
            o32 = _compute_shot_offsets(caches["deep_32"], support_ids, use_core=use_core, exclude_pos_idx=ex_idx)
            o64 = _compute_shot_offsets(caches["deep_64"], support_ids, use_core=use_core, exclude_pos_idx=ex_idx) if cfg.use_64 else None
            omid = _compute_shot_offsets(caches["mid_64"], support_ids, use_core=use_core, exclude_pos_idx=ex_idx) if (cfg.use_64 and cfg.use_mid_veto) else None
        else:
            o32 = o64 = omid = None

        s32 = _score_dehubbed(caches["deep_32"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, shot_offsets=o32, exclude_pos_idx=ex_idx)
        if not cfg.use_64:
            return s32, s32
        s64 = _score_dehubbed(caches["deep_64"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, shot_offsets=o64, exclude_pos_idx=ex_idx)
        s32_up = F.interpolate(s32.reshape(1, 1, 32, 32), size=(64, 64), mode="bilinear", align_corners=False).reshape(-1)
        comb = cfg.w_s64 * s64 + (1.0 - cfg.w_s64) * s32_up
        if cfg.use_mid_veto:
            sm = _score_dehubbed(caches["mid_64"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, shot_offsets=omid, exclude_pos_idx=ex_idx)
            comb = comb + cfg.lambda_mid * sm - cfg.beta_veto * F.softplus((mid_veto_ref - sm) * 10.0) / 10.0
        comb_32 = F.avg_pool2d(comb.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
        return comb, comb_32

    if k >= 2:
        ref_key = "deep_64" if cfg.use_64 else "deep_32"
        fg_items, bg_list = [], []
        for idx, sid in enumerate(support_ids):
            comb_s, _ = raw_score(sid, idx)
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
            c64_b = _ex_neigh_bg_contrast(tokens_by_key["deep_64"][support_ids[0]], mask_by_id[support_ids[0]], 64, core_purity=core_purity)
            cm_b = (
                _ex_neigh_bg_contrast(tokens_by_key["mid_64"][support_ids[0]], mask_by_id[support_ids[0]], 64, core_purity=core_purity)
                if cfg.use_mid_veto
                else None
            )
            if c64_b is not None and c32_b is not None:
                _, bg64_m, _ = select_fg_bg_indices(mask_by_id[support_ids[0]], 64, core_purity)
                _, bg32_m, _ = select_fg_bg_indices(mask_by_id[support_ids[0]], 32, core_purity)
                bg32_flat = np.flatnonzero(bg32_m)
                c32_full = torch.zeros((32, 32), device=c64_b.device)
                c32_full[torch.from_numpy(bg32_flat // 32).to(c64_b.device), torch.from_numpy(bg32_flat % 32).to(c64_b.device)] = c32_b
                bg64_flat = np.flatnonzero(bg64_m)
                c32_at_64 = c32_full[torch.from_numpy(bg64_flat // 64).to(c64_b.device) // 2, torch.from_numpy(bg64_flat % 64).to(c64_b.device) // 2]
                bg_s = cfg.w_s64 * c64_b + (1.0 - cfg.w_s64) * c32_at_64
                if cfg.use_mid_veto and cm_b is not None and mid_veto_ref is not None:
                    bg_s = bg_s + cfg.lambda_mid * cm_b - cfg.beta_veto * F.softplus((mid_veto_ref - cm_b) * 10.0) / 10.0
                bias = float(torch.quantile(bg_s, cfg.bg_q_calib).item()) + cfg.k1_margin
            else:
                bias = 0.30
        temp = 0.06

    out_hw = 64 if cfg.use_64 else 32
    probs_512, scores_32 = {}, {}
    for qid in query_ids:
        comb, comb_32 = raw_score(qid, None)
        p_base = torch.sigmoid((comb - bias) / temp)
        if cfg.use_64 and cfg.use_self_rescue:
            comb_fin = _apply_query_self_contrast(comb, p_base, tokens_by_key["deep_64"][qid], cfg)
            p_grid = torch.sigmoid((comb_fin - bias) / temp)
            s32_out = F.avg_pool2d(comb_fin.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
        else:
            p_grid = p_base
            s32_out = comb_32.reshape(32, 32)

        if cfg.use_64 and cfg.use_affinity:
            w_aff = aff_by_id[qid] if aff_by_id is not None else compute_dino_affinity_weights(tokens_by_key["deep_64"][qid], tau=cfg.aff_tau)
            p_grid = propagate_affinity(p_grid.reshape(64, 64), w_aff, steps=cfg.aff_steps, eta=cfg.aff_eta)

        p_up = F.interpolate(p_grid.reshape(1, 1, out_hw, out_hw), size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
        if cfg.use_guided and luma_by_id is not None:
            p_up = fast_guided_filter_512(
                luma_by_id[qid], p_up, radius=cfg.gf_radius, eps=cfg.gf_eps, blend=cfg.gf_blend
            )
            p_np = remove_weak_tiny_speckles(
                p_up.cpu().numpy().astype(np.float32),
                max_area=cfg.speckle_max_area,
                max_peak=cfg.speckle_max_peak,
            )
        else:
            p_np = p_up.cpu().numpy().astype(np.float32)

        probs_512[qid] = p_np
        scores_32[qid] = s32_out.cpu().numpy().astype(np.float32)
    return probs_512, scores_32


def _save_before_after_panels(
    panel_dir: Path,
    img_dir: Path,
    records_by_id: dict[str, dict],
    preds_b0: dict[str, np.ndarray],
    preds_m1: dict[str, np.ndarray],
    preds_m3: dict[str, np.ndarray],
    target_ids: list[str],
    out_name: str,
) -> None:
    panel_dir.mkdir(parents=True, exist_ok=True)
    tile = 256
    cols = 4  # RGB+GT | B0 | M1 | M3_Full
    hdr = 22
    rows = len(target_ids)
    canvas = Image.new("RGB", (cols * tile, rows * (tile + hdr)), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)

    def _overlay(rgb_chw: np.ndarray, pred_512: np.ndarray, gt_512: np.ndarray) -> Image.Image:
        img = np.transpose(rgb_chw, (1, 2, 0)).astype(np.float32)
        p = pred_512 >= 0.5
        g = gt_512 > 0
        tp = p & g
        fp = p & (~g)
        fn = (~p) & g
        out = img.copy()
        out[tp] = out[tp] * 0.45 + np.array([40, 220, 90]) * 0.55
        out[fp] = out[fp] * 0.40 + np.array([240, 60, 60]) * 0.60
        out[fn] = out[fn] * 0.40 + np.array([250, 215, 45]) * 0.60
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((tile, tile), Image.Resampling.BILINEAR)

    for r_i, rid in enumerate(target_ids):
        rec = records_by_id[rid]
        rgb = read_verified_rgb(img_dir, rec)
        gt = rec["mask"]
        y0 = r_i * (tile + hdr)
        draw.text((6, y0 + 4), f"{rid} | Green=TP Red=FP Yellow=FN", fill=(235, 235, 240))

        gt_only = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
        g_m = gt > 0
        gt_only[g_m] = gt_only[g_m] * 0.50 + np.array([40, 220, 90]) * 0.50
        p_gt = Image.fromarray(np.clip(gt_only, 0, 255).astype(np.uint8)).resize((tile, tile), Image.Resampling.BILINEAR)

        panels = [p_gt, _overlay(rgb, preds_b0[rid], gt), _overlay(rgb, preds_m1[rid], gt), _overlay(rgb, preds_m3[rid], gt)]
        labels = ["GT", "B0_32", "M1_Full", "M3_Full"]
        for c_i, (pan, lbl) in enumerate(zip(panels, labels)):
            canvas.paste(pan, (c_i * tile, y0 + hdr))
            draw.text((c_i * tile + 6, y0 + hdr + 4), lbl, fill=(255, 255, 255))

    canvas.save(panel_dir / out_name)


def run_iter3_ablations(
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
    records_by_id, mask_by_id, luma_by_id = {}, {}, {}
    for row in dev_records:
        rid = row["id"]
        mask = read_verified_mask(lbl_dir, row)
        rgb = read_verified_rgb(img_dir, row).astype(np.float32) / 255.0
        luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}
        mask_by_id[rid] = mask
        luma_by_id[rid] = torch.from_numpy(luma).to(device=run_device, dtype=torch.float32)

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
        v: {t: [] for t in panel_targets} for v in ("B0_32_Ref", "M1_Full_Ref", "M3_Full")
    }

    results: dict[str, dict] = {cfg.name: {} for cfg in ITER3_VARIANTS}
    for cfg in ITER3_VARIANTS:
        for k, eps in sorted(ep_by_k.items()):
            ep_evals = []
            for ep in eps:
                p512, s32 = run_iter3_episode(
                    caches, tokens_by_key, mask_by_id, luma_by_id, aff_by_id, ep["support_ids"], ep["query_ids"], cfg
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
                f"[Iter3] {cfg.name:17s} K={k}: Dice={m['dice']:.4f} AP512={m['ap_512']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} BF1={m['boundary_f1']:.4f} "
                f"NegFP={m['neg_fp']:.4f} PosBg={m['pos_bg_fp']:.4f} SmallD={sd:.4f} SmallR={sr:.4f}",
                flush=True,
            )

    mean_p = {
        v: {t: np.mean(np.stack(lst, axis=0), axis=0) for t, lst in by_t.items() if lst}
        for v, by_t in panel_store.items()
    }
    fn_panel = [t for t in ("正样本/PV-55.tif", "正样本/PV-49.tif", "正样本/PV-73.tif") if t in mean_p["M3_Full"]]
    fp_panel = [t for t in ("正样本/PV-80.tif", "负样本/PV-93.tif", "负样本/PV-100.tif", "负样本/PV-96.tif") if t in mean_p["M3_Full"]]
    if fn_panel:
        _save_before_after_panels(output_dir / "panels", img_dir, records_by_id, mean_p["B0_32_Ref"], mean_p["M1_Full_Ref"], mean_p["M3_Full"], fn_panel, "k5_fn_compare.png")
    if fp_panel:
        _save_before_after_panels(output_dir / "panels", img_dir, records_by_id, mean_p["B0_32_Ref"], mean_p["M1_Full_Ref"], mean_p["M3_Full"], fp_panel, "k5_fp_compare.png")

    pairs = [
        ("M3_Base_CalDehub", "M1_Full_Ref"),
        ("M3_Plus_Rescue", "M3_Base_CalDehub"),
        ("M3_Plus_Affinity", "M3_Plus_Rescue"),
        ("M3_Full", "M3_Plus_Affinity"),
        ("M3_Full", "M3_NoDehub_LOO"),
        ("M3_Full", "M3_NoRescue_LOO"),
        ("M3_Full", "M3_NoAff_LOO"),
        ("M3_Full", "M3_Cal28_LOO"),
        ("M3_Full", "M1_Full_Ref"),
        ("M3_Full", "B0_32_Ref"),
        ("M3_Plus_Affinity", "B0_32_Ref"),
        ("M3_Plus_Rescue", "B0_32_Ref"),
        ("M3_Cal28_LOO", "B0_32_Ref"),
    ]
    comparisons = {
        f"{cand}_vs_{ref}": {
            str(k): compare_methods(results[cand][str(k)], results[ref][str(k)])
            for k in sorted(ep_by_k)
        }
        for cand, ref in pairs
    }

    error_taxonomy = {}
    for variant in ("M3_Plus_Rescue", "M3_Full", "M3_Cal28_LOO"):
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
        "iteration": 3,
        "methods": results,
        "comparisons": comparisons,
        "error_taxonomy": error_taxonomy,
    }
    write_json(output_dir / "iter3_summary.json", summary)
    return summary
