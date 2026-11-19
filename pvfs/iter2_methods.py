from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import area_target_32, load_dev_records, read_verified_mask, resolve_data_dirs
from .episodes import load_and_verify_episodes
from .fast_bank import build_pairwise_topk_cache, score_from_pairwise, select_fg_bg_indices
from .io import write_json
from .metrics import aggregate_episodes, compare_methods, evaluate_episode


@dataclass(frozen=True)
class Iter2Config:
    name: str
    use_64: bool = True
    shot_mode: str = "top2_soft"
    k1_use_core: bool = False  # K=1 keeps all y>=0.50 fg tokens for stable bg calibration
    kge2_use_core: bool = True  # K>=2 uses core>=0.85 for crisp ranking
    calib_v2: bool = True
    w_fg_calib: float = 0.28
    bg_q_calib: float = 0.95
    k1_margin: float = 0.075
    use_scale_sharp: bool = True
    w_s64: float = 0.85
    scale_drop_penalty: float = 0.35
    use_mid_veto: bool = True
    lambda_mid: float = 0.35
    beta_veto: float = 0.40
    use_layer_comp: bool = True
    alpha_bg_mid: float = 0.65
    early_damp_weight: float = 0.30
    use_self_rescue: bool = True
    seed_prob_thr: float = 0.70
    min_seeds: int = 4
    min_prominence: float = 0.14
    qsc_weight: float = 0.40


ITER2_VARIANTS: tuple[Iter2Config, ...] = (
    Iter2Config(
        name="B0_32_Ref",
        use_64=False,
        shot_mode="pooled",
        k1_use_core=False,
        kge2_use_core=False,
        calib_v2=False,
        w_fg_calib=0.50,
        bg_q_calib=0.90,
        k1_margin=0.080,
        use_scale_sharp=False,
        use_mid_veto=False,
        use_layer_comp=False,
        use_self_rescue=False,
    ),
    Iter2Config(
        name="M1_Full_Ref",
        use_64=True,
        shot_mode="balanced_70_30",
        k1_use_core=True,
        kge2_use_core=True,
        calib_v2=False,
        w_fg_calib=0.45,
        bg_q_calib=0.95,
        k1_margin=0.075,
        use_scale_sharp=False,
        w_s64=0.70,
        use_mid_veto=True,
        use_layer_comp=False,
        use_self_rescue=False,
    ),
    Iter2Config(
        name="M2_ShotCal",
        use_64=True,
        shot_mode="top2_soft",
        k1_use_core=False,
        kge2_use_core=True,
        calib_v2=True,
        w_fg_calib=0.28,
        bg_q_calib=0.95,
        k1_margin=0.075,
        use_scale_sharp=False,
        w_s64=0.70,
        use_mid_veto=True,
        use_layer_comp=False,
        use_self_rescue=False,
    ),
    Iter2Config(
        name="M2_ScaleSharp",
        use_64=True,
        shot_mode="top2_soft",
        k1_use_core=False,
        kge2_use_core=True,
        calib_v2=True,
        w_fg_calib=0.28,
        bg_q_calib=0.95,
        k1_margin=0.075,
        use_scale_sharp=True,
        w_s64=0.85,
        scale_drop_penalty=0.35,
        use_mid_veto=True,
        use_layer_comp=False,
        use_self_rescue=False,
    ),
    Iter2Config(
        name="M2_LayerComp",
        use_64=True,
        shot_mode="top2_soft",
        k1_use_core=False,
        kge2_use_core=True,
        calib_v2=True,
        w_fg_calib=0.28,
        bg_q_calib=0.95,
        k1_margin=0.075,
        use_scale_sharp=True,
        w_s64=0.85,
        scale_drop_penalty=0.35,
        use_mid_veto=True,
        use_layer_comp=True,
        alpha_bg_mid=0.65,
        early_damp_weight=0.30,
        use_self_rescue=False,
    ),
    Iter2Config(
        name="M2_Full",
        use_64=True,
        shot_mode="top2_soft",
        k1_use_core=False,
        kge2_use_core=True,
        calib_v2=True,
        w_fg_calib=0.28,
        bg_q_calib=0.95,
        k1_margin=0.075,
        use_scale_sharp=True,
        w_s64=0.85,
        scale_drop_penalty=0.35,
        use_mid_veto=True,
        use_layer_comp=True,
        alpha_bg_mid=0.65,
        early_damp_weight=0.30,
        use_self_rescue=True,
        seed_prob_thr=0.70,
        min_seeds=4,
        min_prominence=0.14,
        qsc_weight=0.40,
    ),
    Iter2Config(
        name="M2_NoLayer_LOO",
        use_64=True,
        shot_mode="top2_soft",
        k1_use_core=False,
        kge2_use_core=True,
        calib_v2=True,
        w_fg_calib=0.28,
        bg_q_calib=0.95,
        k1_margin=0.075,
        use_scale_sharp=True,
        w_s64=0.85,
        scale_drop_penalty=0.35,
        use_mid_veto=True,
        use_layer_comp=False,
        use_self_rescue=True,
        seed_prob_thr=0.70,
        min_seeds=4,
        min_prominence=0.14,
        qsc_weight=0.40,
    ),
    Iter2Config(
        name="M2_NoScale_LOO",
        use_64=True,
        shot_mode="top2_soft",
        k1_use_core=False,
        kge2_use_core=True,
        calib_v2=True,
        w_fg_calib=0.28,
        bg_q_calib=0.95,
        k1_margin=0.075,
        use_scale_sharp=False,
        w_s64=0.70,
        use_mid_veto=True,
        use_layer_comp=True,
        alpha_bg_mid=0.65,
        early_damp_weight=0.30,
        use_self_rescue=True,
        seed_prob_thr=0.70,
        min_seeds=4,
        min_prominence=0.14,
        qsc_weight=0.40,
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


def _apply_query_self_contrast(
    base_comb: torch.Tensor,
    s64: torch.Tensor,
    s32_up: torch.Tensor,
    p_base: torch.Tensor,
    q_tok_64: torch.Tensor,
    cfg: Iter2Config,
) -> torch.Tensor:
    """Rescue visually coherent missed PV sub-patches via gated intra-query self-contrast.

    Seeds Q+ must (a) exceed seed_prob_thr, (b) not degrade sharply at 16m vs 32m,
    and (c) come from an image with clear spatial prominence (q99 - q50 >= min_prominence).
    """
    prominence = float((torch.quantile(base_comb, 0.99) - torch.quantile(base_comb, 0.50)).item())
    if prominence < cfg.min_prominence:
        return base_comb

    scale_ok = s64 >= (s32_up - 0.02) if cfg.use_scale_sharp else torch.ones_like(p_base, dtype=torch.bool)
    seed_mask = (p_base >= cfg.seed_prob_thr) & scale_ok
    if int(seed_mask.sum().item()) < cfg.min_seeds:
        return base_comb

    bg_mask = p_base <= 0.30
    if int(bg_mask.sum().item()) < 32:
        return base_comb

    seed_idx = torch.nonzero(seed_mask, as_tuple=False).squeeze(1)
    if seed_idx.numel() > 256:
        # Take highest confidence seeds for crisp prototype anchor
        top_local = base_comb[seed_idx].topk(256).indices
        seed_idx = seed_idx[top_local]

    bg_idx = torch.nonzero(bg_mask, as_tuple=False).squeeze(1)
    if bg_idx.numel() > 1024:
        # Subsample evenly across background for speed & coverage
        step = max(1, bg_idx.numel() // 1024)
        bg_idx = bg_idx[::step]

    q_fg = q_tok_64[seed_idx]  # [M_fg, C]
    q_bg = q_tok_64[bg_idx]    # [M_bg, C]
    m_fg = min(3, q_fg.shape[0])
    m_bg = min(3, q_bg.shape[0])

    # Centroid + top-3 intra-query PV similarity vs local background
    sim_all_fg = (q_tok_64 @ q_fg.T).clamp(-1.0, 1.0)
    sim_all_bg = (q_tok_64 @ q_bg.T).clamp(-1.0, 1.0)
    top_fg = sim_all_fg.topk(m_fg, dim=1).values.mean(dim=1)
    top_bg = sim_all_bg.topk(m_bg, dim=1).values.mean(dim=1)
    s_self = top_fg - 0.85 * top_bg

    # Zero-leakage self-anchor on non-seed bg vs seeds (excluding self-match for seed tokens)
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
    # Require genuine similarity to query PV seeds Q+ and clear margin over local Q-
    fg_gate = torch.sigmoid((top_fg - fg_sim_floor) / 0.04) * torch.sigmoid(((top_fg - top_bg) - 0.05) / 0.04)
    boost = torch.where(delta > 0.0, cfg.qsc_weight * delta * fg_gate, 0.20 * cfg.qsc_weight * delta)
    return base_comb + boost


@torch.inference_mode()
def run_iter2_episode(
    caches: dict[str, dict],
    tokens_by_key: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    support_ids: list[str],
    query_ids: list[str],
    cfg: Iter2Config,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    k = len(support_ids)
    use_core = cfg.k1_use_core if k == 1 else cfg.kge2_use_core
    core_purity = 0.85 if use_core else 0.50
    alpha_mid = cfg.alpha_bg_mid if cfg.use_layer_comp else 0.50

    # 1. Support mid-layer veto threshold
    mid_veto_ref = None
    if cfg.use_64 and cfg.use_mid_veto:
        if k >= 2:
            mid_vals = []
            for idx, sid in enumerate(support_ids):
                sm, _, _ = score_from_pairwise(
                    caches["mid_64"],
                    sid,
                    support_ids,
                    use_core=use_core,
                    shot_mode=cfg.shot_mode,
                    alpha_bg=alpha_mid,
                    exclude_pos_idx=idx,
                )
                bm = caches["mid_64"]["bg_masks"][caches["mid_64"]["id_to_pos"][sid]]
                if bm.any():
                    mid_vals.append(sm[torch.from_numpy(np.flatnonzero(bm)).to(sm.device)])
            if mid_vals:
                mid_veto_ref = float(torch.quantile(torch.cat(mid_vals), 0.85).item())
        else:
            c_bg_m = _ex_neigh_bg_contrast(
                tokens_by_key["mid_64"][support_ids[0]],
                mask_by_id[support_ids[0]],
                64,
                core_purity=core_purity,
                alpha_bg=alpha_mid,
            )
            if c_bg_m is not None:
                mid_veto_ref = float(torch.quantile(c_bg_m, 0.85).item())
        if mid_veto_ref is None:
            mid_veto_ref = 0.20

    def raw_score(qid: str, ex_idx: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s32, _, _ = score_from_pairwise(
            caches["deep_32"],
            qid,
            support_ids,
            use_core=use_core,
            shot_mode=cfg.shot_mode,
            alpha_bg=0.50,
            exclude_pos_idx=ex_idx,
        )
        if not cfg.use_64:
            return s32, s32, s32, s32

        s64, _, _ = score_from_pairwise(
            caches["deep_64"],
            qid,
            support_ids,
            use_core=use_core,
            shot_mode=cfg.shot_mode,
            alpha_bg=0.50,
            exclude_pos_idx=ex_idx,
        )
        s32_up = F.interpolate(s32.reshape(1, 1, 32, 32), size=(64, 64), mode="bilinear", align_corners=False).reshape(-1)
        if cfg.use_scale_sharp:
            comb = cfg.w_s64 * s64 + (1.0 - cfg.w_s64) * s32_up - cfg.scale_drop_penalty * F.relu(s32_up - s64)
        else:
            comb = cfg.w_s64 * s64 + (1.0 - cfg.w_s64) * s32_up

        if cfg.use_mid_veto:
            sm, _, _ = score_from_pairwise(
                caches["mid_64"],
                qid,
                support_ids,
                use_core=use_core,
                shot_mode=cfg.shot_mode,
                alpha_bg=alpha_mid,
                exclude_pos_idx=ex_idx,
            )
            comb = comb + cfg.lambda_mid * sm - cfg.beta_veto * F.softplus((mid_veto_ref - sm) * 10.0) / 10.0

        if cfg.use_layer_comp and "early_32" in caches:
            _, sp_e, sn_e = score_from_pairwise(
                caches["early_32"],
                qid,
                support_ids,
                use_core=use_core,
                shot_mode=cfg.shot_mode,
                alpha_bg=0.50,
                exclude_pos_idx=ex_idx,
            )
            e_gap = F.relu(sn_e - sp_e)
            e_gap_64 = F.interpolate(e_gap.reshape(1, 1, 32, 32), size=(64, 64), mode="bilinear", align_corners=False).reshape(-1)
            comb = comb - cfg.early_damp_weight * e_gap_64

        comb_32 = F.avg_pool2d(comb.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
        return comb, comb_32, s64, s32_up

    # 2. Zero-leakage calibration
    if k >= 2:
        ref_key = "deep_64" if cfg.use_64 else "deep_32"
        fg_vals_or_meds, bg_list = [], []
        for idx, sid in enumerate(support_ids):
            comb_s, _, _, _ = raw_score(sid, idx)
            p_i = caches[ref_key]["id_to_pos"][sid]
            fg_m = caches[ref_key]["fg_all_masks"][p_i] if cfg.calib_v2 else caches[ref_key]["fg_core_masks"][p_i]
            bg_m = caches[ref_key]["bg_masks"][p_i]
            fg_idx = torch.from_numpy(np.flatnonzero(fg_m)).to(comb_s.device)
            if cfg.calib_v2 or cfg.name == "M1_Full_Ref":
                fg_vals_or_meds.append(float(torch.median(comb_s[fg_idx]).item()))
            else:
                fg_vals_or_meds.append(comb_s[fg_idx])
            if bg_m.any():
                bg_list.append(comb_s[torch.from_numpy(np.flatnonzero(bg_m)).to(comb_s.device)])
        if bg_list:
            if isinstance(fg_vals_or_meds[0], float):
                mu_fg = float(np.mean(fg_vals_or_meds))
            else:
                mu_fg = float(torch.median(torch.cat(fg_vals_or_meds)).item())
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
                _ex_neigh_bg_contrast(tokens_by_key["mid_64"][support_ids[0]], mask_by_id[support_ids[0]], 64, core_purity=core_purity, alpha_bg=alpha_mid)
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
                if cfg.use_scale_sharp:
                    bg_s = cfg.w_s64 * c64_b + (1.0 - cfg.w_s64) * c32_at_64 - cfg.scale_drop_penalty * F.relu(c32_at_64 - c64_b)
                else:
                    bg_s = cfg.w_s64 * c64_b + (1.0 - cfg.w_s64) * c32_at_64
                if cfg.use_mid_veto and cm_b is not None and mid_veto_ref is not None:
                    bg_s = bg_s + cfg.lambda_mid * cm_b - cfg.beta_veto * F.softplus((mid_veto_ref - cm_b) * 10.0) / 10.0
                if cfg.use_layer_comp and "early_32" in tokens_by_key and support_ids[0] in tokens_by_key["early_32"]:
                    ce_b = _ex_neigh_bg_contrast(
                        tokens_by_key["early_32"][support_ids[0]],
                        mask_by_id[support_ids[0]],
                        32,
                        core_purity=core_purity,
                        alpha_bg=1.0,
                    )
                    if ce_b is not None:
                        ce_full = torch.zeros((32, 32), device=c64_b.device)
                        ce_full[torch.from_numpy(bg32_flat // 32).to(c64_b.device), torch.from_numpy(bg32_flat % 32).to(c64_b.device)] = F.relu(-ce_b)
                        ce_at_64 = ce_full[torch.from_numpy(bg64_flat // 64).to(c64_b.device) // 2, torch.from_numpy(bg64_flat % 64).to(c64_b.device) // 2]
                        bg_s = bg_s - cfg.early_damp_weight * ce_at_64
                bias = float(torch.quantile(bg_s, cfg.bg_q_calib).item()) + cfg.k1_margin
            else:
                bias = 0.30
        temp = 0.06

    # 3. Query inference + optional gated intra-query self-contrast rescue
    out_hw = 64 if cfg.use_64 else 32
    probs_512, scores_32 = {}, {}
    for qid in query_ids:
        comb, comb_32, s64, s32_up = raw_score(qid, None)
        p_base = torch.sigmoid((comb - bias) / temp)
        if cfg.use_64 and cfg.use_self_rescue:
            comb_fin = _apply_query_self_contrast(
                comb, s64, s32_up, p_base, tokens_by_key["deep_64"][qid], cfg
            )
            p_grid = torch.sigmoid((comb_fin - bias) / temp)
            s32_out = F.avg_pool2d(comb_fin.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
        else:
            p_grid = p_base
            s32_out = comb_32.reshape(32, 32)
        p512 = F.interpolate(p_grid.reshape(1, 1, out_hw, out_hw), size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
        probs_512[qid] = p512.cpu().numpy().astype(np.float32)
        scores_32[qid] = s32_out.cpu().numpy().astype(np.float32)
    return probs_512, scores_32


def run_iter2_ablations(
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
    _, lbl_dir = resolve_data_dirs(image_dir, label_dir)

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
    for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64), ("early_32", 32)):
        toks = {rid: _to_tokens_norm(feats[rid][key], run_device) for rid in all_ids}
        caches[key] = build_pairwise_topk_cache(toks, mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        if key in ("deep_32", "deep_64", "mid_64"):
            tokens_by_key[key] = toks
        else:
            tokens_by_key[key] = {rid: toks[rid] for rid in pos_ids}
            del toks
        if run_device == "cuda":
            torch.cuda.empty_cache()

    ep_by_k = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    results: dict[str, dict] = {cfg.name: {} for cfg in ITER2_VARIANTS}
    for cfg in ITER2_VARIANTS:
        for k, eps in sorted(ep_by_k.items()):
            ep_evals = []
            for ep in eps:
                p512, s32 = run_iter2_episode(caches, tokens_by_key, mask_by_id, ep["support_ids"], ep["query_ids"], cfg)
                ep_evals.append(evaluate_episode([records_by_id[qid] for qid in ep["query_ids"]], p512, s32, threshold=0.5))
            agg = aggregate_episodes(ep_evals, episodes_per_fold=episodes_lock["episodes_per_fold"])
            results[cfg.name][str(k)] = agg
            m = agg["macro"]
            sd = agg["strata"].get("small_lt5pct", {}).get("dice", 0.0)
            sr = agg["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            print(
                f"[Iter2] {cfg.name:16s} K={k}: Dice={m['dice']:.4f} AP512={m['ap_512']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} BF1={m['boundary_f1']:.4f} "
                f"NegFP={m['neg_fp']:.4f} PosBg={m['pos_bg_fp']:.4f} SmallD={sd:.4f} SmallR={sr:.4f}",
                flush=True,
            )

    pairs = [
        ("M2_ShotCal", "M1_Full_Ref"),
        ("M2_ScaleSharp", "M2_ShotCal"),
        ("M2_LayerComp", "M2_ScaleSharp"),
        ("M2_Full", "M2_LayerComp"),
        ("M2_Full", "M2_NoLayer_LOO"),
        ("M2_Full", "M2_NoScale_LOO"),
        ("M2_Full", "M1_Full_Ref"),
        ("M2_Full", "B0_32_Ref"),
        ("M2_ShotCal", "B0_32_Ref"),
        ("M2_ScaleSharp", "B0_32_Ref"),
        ("M2_LayerComp", "B0_32_Ref"),
    ]
    comparisons = {
        f"{cand}_vs_{ref}": {
            str(k): compare_methods(results[cand][str(k)], results[ref][str(k)])
            for k in sorted(ep_by_k)
        }
        for cand, ref in pairs
    }

    error_taxonomy = {}
    for variant in ("M2_LayerComp", "M2_Full"):
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
        "iteration": 2,
        "methods": results,
        "comparisons": comparisons,
        "error_taxonomy": error_taxonomy,
    }
    write_json(output_dir / "iter2_summary.json", summary)
    return summary
