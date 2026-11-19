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
from .iter3_methods import (
    _ex_neigh_bg_contrast,
    _save_before_after_panels,
    _to_tokens_norm,
    compute_dino_affinity_weights,
    propagate_affinity,
)
from .metrics import aggregate_episodes, compare_methods, evaluate_episode


@dataclass(frozen=True)
class Iter5Config:
    name: str
    use_64: bool = True
    shot_mode: str = "balanced_70_30"
    k1_use_core: bool = False
    kge2_use_core: bool = True
    calib_v2: bool = True
    w_fg_calib: float = 0.44
    bg_q_calib: float = 0.95
    k1_margin: float = 0.075
    w_s64: float = 0.75
    use_mid_veto: bool = True
    lambda_mid: float = 0.35
    beta_veto: float = 0.40
    use_self_rescue: bool = True
    seed_prob_thr: float = 0.72
    min_seeds: int = 6
    min_prominence: float = 0.14
    seed_mid_margin: float = 0.02
    qsc_weight: float = 0.45
    use_affinity: bool = True
    aff_steps: int = 2
    aff_eta: float = 0.45
    aff_tau: float = 0.12


ITER5_VARIANTS: tuple[Iter5Config, ...] = (
    Iter5Config(
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
        use_self_rescue=False,
        use_affinity=False,
    ),
    Iter5Config(
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
        use_self_rescue=False,
        use_affinity=False,
    ),
    Iter5Config(
        name="M3_Pruned_Ref",
        shot_mode="top2_soft",
        w_fg_calib=0.38,
        seed_prob_thr=0.66,
        min_seeds=4,
        seed_mid_margin=-1.0,
        qsc_weight=0.40,
    ),
    Iter5Config(
        name="M5_BalCal",
        shot_mode="balanced_70_30",
        w_fg_calib=0.44,
        use_self_rescue=False,
        use_affinity=True,
    ),
    Iter5Config(
        name="M5_TightRescue",
        shot_mode="balanced_70_30",
        w_fg_calib=0.44,
        use_self_rescue=True,
        seed_prob_thr=0.72,
        min_seeds=6,
        seed_mid_margin=0.02,
        qsc_weight=0.45,
        use_affinity=True,
    ),
    Iter5Config(
        name="M5_AdaptiveRescue",
        shot_mode="balanced_70_30",
        w_fg_calib=0.43,
        use_self_rescue=True,
        seed_prob_thr=0.70,
        min_seeds=5,
        seed_mid_margin=0.01,
        qsc_weight=0.50,
        use_affinity=True,
    ),
    Iter5Config(
        name="M5_Tight_NoAff_LOO",
        shot_mode="balanced_70_30",
        w_fg_calib=0.44,
        use_self_rescue=True,
        seed_prob_thr=0.72,
        min_seeds=6,
        seed_mid_margin=0.02,
        qsc_weight=0.45,
        use_affinity=False,
    ),
)


def _apply_mid_verified_rescue(
    base_comb: torch.Tensor,
    p_base: torch.Tensor,
    sm_64: torch.Tensor | None,
    mid_veto_ref: float | None,
    q_tok_64: torch.Tensor,
    cfg: Iter5Config,
) -> torch.Tensor:
    """Rescue visually coherent missed PV wings with strict seed-consensus & mid-layer gating."""
    prominence = float((torch.quantile(base_comb, 0.99) - torch.quantile(base_comb, 0.50)).item())
    if prominence < cfg.min_prominence:
        return base_comb

    seed_mask = p_base >= cfg.seed_prob_thr
    if int(seed_mask.sum().item()) < cfg.min_seeds:
        return base_comb

    # Require seeds' mid-64 material score to sit above support background floor (blocks dark greenhouses)
    if sm_64 is not None and mid_veto_ref is not None and cfg.seed_mid_margin > -0.5:
        if float(sm_64[seed_mask].mean().item()) < (mid_veto_ref + cfg.seed_mid_margin):
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


@torch.inference_mode()
def run_iter5_episode(
    caches: dict[str, dict],
    tokens_by_key: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    aff_by_id: dict[str, torch.Tensor] | None,
    support_ids: list[str],
    query_ids: list[str],
    cfg: Iter5Config,
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

    def raw_score(qid: str, ex_idx: int | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        s32, _, _ = score_from_pairwise(
            caches["deep_32"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx
        )
        if not cfg.use_64:
            return s32, None
        s64, _, _ = score_from_pairwise(
            caches["deep_64"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx
        )
        s32_up = F.interpolate(s32.reshape(1, 1, 32, 32), size=(64, 64), mode="bilinear", align_corners=False).reshape(-1)
        comb = cfg.w_s64 * s64 + (1.0 - cfg.w_s64) * s32_up
        sm = None
        if cfg.use_mid_veto:
            sm, _, _ = score_from_pairwise(
                caches["mid_64"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx
            )
            comb = comb + cfg.lambda_mid * sm - cfg.beta_veto * F.softplus((mid_veto_ref - sm) * 10.0) / 10.0
        return comb, sm

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
        sid0 = support_ids[0]
        c32_b = _ex_neigh_bg_contrast(tokens_by_key["deep_32"][sid0], mask_by_id[sid0], 32, core_purity=core_purity)
        if not cfg.use_64:
            bias = (float(torch.quantile(c32_b, cfg.bg_q_calib).item()) + cfg.k1_margin) if c32_b is not None else 0.30
        else:
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
                bias = float(torch.quantile(bg_s, cfg.bg_q_calib).item()) + cfg.k1_margin
            else:
                bias = 0.30
        temp = 0.06

    out_hw = 64 if cfg.use_64 else 32
    probs_512, scores_32 = {}, {}
    for qid in query_ids:
        comb, sm = raw_score(qid, None)
        p_base = torch.sigmoid((comb - bias) / temp)
        if cfg.use_64 and cfg.use_self_rescue:
            comb_fin = _apply_mid_verified_rescue(comb, p_base, sm, mid_veto_ref, tokens_by_key["deep_64"][qid], cfg)
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


def run_iter5_ablations(
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
        v: {t: [] for t in panel_targets} for v in ("B0_32_Ref", "M1_Full_Ref", "M5_TightRescue")
    }

    results: dict[str, dict] = {cfg.name: {} for cfg in ITER5_VARIANTS}
    for cfg in ITER5_VARIANTS:
        for k, eps in sorted(ep_by_k.items()):
            ep_evals = []
            for ep in eps:
                p512, s32 = run_iter5_episode(
                    caches, tokens_by_key, mask_by_id, aff_by_id, ep["support_ids"], ep["query_ids"], cfg
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
                f"[Iter5] {cfg.name:18s} K={k}: Dice={m['dice']:.4f} AP512={m['ap_512']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} BF1={m['boundary_f1']:.4f} "
                f"NegFP={m['neg_fp']:.4f} PosBg={m['pos_bg_fp']:.4f} SmallD={sd:.4f} SmallR={sr:.4f}",
                flush=True,
            )

    mean_p = {
        v: {t: np.mean(np.stack(lst, axis=0), axis=0) for t, lst in by_t.items() if lst}
        for v, by_t in panel_store.items()
    }
    fn_panel = [t for t in ("正样本/PV-55.tif", "正样本/PV-49.tif", "正样本/PV-73.tif") if t in mean_p["M5_TightRescue"]]
    fp_panel = [t for t in ("正样本/PV-80.tif", "负样本/PV-93.tif", "负样本/PV-100.tif", "负样本/PV-96.tif") if t in mean_p["M5_TightRescue"]]
    if fn_panel:
        _save_before_after_panels(output_dir / "panels", img_dir, records_by_id, mean_p["B0_32_Ref"], mean_p["M1_Full_Ref"], mean_p["M5_TightRescue"], fn_panel, "k5_fn_compare.png")
    if fp_panel:
        _save_before_after_panels(output_dir / "panels", img_dir, records_by_id, mean_p["B0_32_Ref"], mean_p["M1_Full_Ref"], mean_p["M5_TightRescue"], fp_panel, "k5_fp_compare.png")

    pairs = [
        ("M5_BalCal", "M1_Full_Ref"),
        ("M5_TightRescue", "M5_BalCal"),
        ("M5_TightRescue", "M5_Tight_NoAff_LOO"),
        ("M5_TightRescue", "M3_Pruned_Ref"),
        ("M5_TightRescue", "M1_Full_Ref"),
        ("M5_TightRescue", "B0_32_Ref"),
        ("M5_AdaptiveRescue", "B0_32_Ref"),
        ("M5_AdaptiveRescue", "M5_BalCal"),
        ("M5_BalCal", "B0_32_Ref"),
    ]
    comparisons = {
        f"{cand}_vs_{ref}": {
            str(k): compare_methods(results[cand][str(k)], results[ref][str(k)])
            for k in sorted(ep_by_k)
        }
        for cand, ref in pairs
    }

    error_taxonomy = {}
    for variant in ("M5_BalCal", "M5_TightRescue", "M5_AdaptiveRescue"):
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
        "iteration": 5,
        "methods": results,
        "comparisons": comparisons,
        "error_taxonomy": error_taxonomy,
    }
    write_json(output_dir / "iter5_summary.json", summary)
    return summary
