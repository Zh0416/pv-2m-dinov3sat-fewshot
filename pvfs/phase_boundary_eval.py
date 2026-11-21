from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import time
import tracemalloc
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, label as nd_label
import torch

from .data import area_target_32, load_dev_records, read_rgb_image, read_verified_mask, resolve_data_dirs
from .episodes import load_and_verify_episodes
from .fast_bank import build_pairwise_topk_cache
from .io import write_json
from .iter3_methods import _to_tokens_norm, compute_dino_affinity_weights
from .iter5_methods import ITER5_VARIANTS, Iter5Config, run_iter5_episode
from .metrics import BOX_3, DISK_2, average_precision
from .phase_boundary import (
    CROSS_3,
    DISK_1,
    PhaseBoundaryConfig,
    boundary_f1_at_radius,
    calibrate_support_thresholds,
    check_topology_preserved,
    compute_oracle_pm8px_mask,
    precompute_image_signals,
    precompute_support_image_null_item,
    refine_query_image_all_variants,
)


def _get_m3_pruned_cfg() -> Iter5Config:
    for cfg in ITER5_VARIANTS:
        if cfg.name == "M3_Pruned_Ref":
            return cfg
    raise RuntimeError("M3_Pruned_Ref not found in ITER5_VARIANTS")


def evaluate_binary_mask_512(
    mask_gt: np.ndarray,
    pred_mask: np.ndarray,
    prob_512_ref: np.ndarray,
    *,
    t_b: np.ndarray | None = None,
    t_dil_1px: np.ndarray | None = None,
    t_dil_2px: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate a binary 512x512 mask with both BF1@1px and BF1@2px (distinct dilation caches)."""
    truth = np.asarray(mask_gt > 0, dtype=bool)
    pred = np.asarray(pred_mask, dtype=bool)
    gt_sum = int(truth.sum())
    pred_sum = int(pred.sum())

    if gt_sum > 0:
        tp = int(np.logical_and(truth, pred).sum())
        fp = pred_sum - tp
        fn = gt_sum - tp
        tn = truth.size - gt_sum - fp
        dice = (2.0 * tp / (gt_sum + pred_sum)) if (gt_sum + pred_sum) > 0 else 1.0
        iou = (tp / (gt_sum + fp)) if (gt_sum + fp) > 0 else 1.0
        recall = tp / gt_sum
        precision = tp / pred_sum if pred_sum > 0 else 0.0
        bg_fp = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        bf1_1px = boundary_f1_at_radius(truth, pred, radius=1, t_b=t_b, t_dil=t_dil_1px)
        bf1_2px = boundary_f1_at_radius(truth, pred, radius=2, t_b=t_b, t_dil=t_dil_2px)
        return {
            "positive": True,
            "dice": float(dice),
            "iou": float(iou),
            "recall": float(recall),
            "precision": float(precision),
            "pos_bg_fp": float(bg_fp),
            "bf1_1px": float(bf1_1px),
            "boundary_f1": float(bf1_2px),  # BF1@2px aligned with existing codebase key
            "detected": 1.0 if iou >= 0.10 else 0.0,
            "neg_fp": None,
            "neg_clean": None,
            "max_prob": float(prob_512_ref.max()),
        }
    neg_fp = float(pred.mean())
    return {
        "positive": False,
        "dice": None,
        "iou": None,
        "recall": None,
        "precision": None,
        "pos_bg_fp": None,
        "bf1_1px": None,
        "boundary_f1": None,
        "detected": None,
        "neg_fp": neg_fp,
        "neg_clean": 1.0 if neg_fp < 0.01 else 0.0,
        "max_prob": float(prob_512_ref.max()),
    }


def evaluate_binary_episode(
    query_records: list[dict],
    masks_512: dict[str, np.ndarray],
    probs_512_ref: dict[str, np.ndarray],
    scores_32_ref: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute per-image and per-site metrics for one episode on binary refined masks.

    Note: AP512 and AP32 are computed from the shared baseline continuous probability map P_0
    and explicitly labeled as raw_prob_ap512 / raw_prob_ap32.
    """
    per_image = {}
    by_site = defaultdict(list)
    for rec in query_records:
        qid = rec["id"]
        m512 = masks_512[qid]
        p512 = probs_512_ref[qid]
        s32 = scores_32_ref[qid]
        if rec["positive"] and "_gt_b" not in rec:
            truth = rec["mask"] > 0
            rec["_gt_b"] = truth ^ binary_erosion(truth, structure=BOX_3, border_value=1)
            rec["_gt_dil_1px"] = binary_dilation(rec["_gt_b"], structure=DISK_1)
            rec["_gt_dil_2px"] = binary_dilation(rec["_gt_b"], structure=DISK_2)
        im_m = evaluate_binary_mask_512(
            rec["mask"],
            m512,
            p512,
            t_b=rec.get("_gt_b"),
            t_dil_1px=rec.get("_gt_dil_1px"),
            t_dil_2px=rec.get("_gt_dil_2px"),
        )
        im_m["id"] = qid
        im_m["site"] = rec["site"]
        im_m["fold"] = rec["fold"]
        im_m["target_fraction"] = float(rec["target_fraction"])
        per_image[qid] = im_m
        by_site[rec["site"]].append((rec, im_m, p512, s32))

    site_rows = []
    for site, items in sorted(by_site.items()):
        pos_items = [x for x in items if x[0]["positive"]]
        neg_items = [x for x in items if not x[0]["positive"]]
        row: dict[str, Any] = {
            "site": site,
            "fold": items[0][0]["fold"],
            "positive_images": len(pos_items),
            "negative_images": len(neg_items),
        }
        if pos_items:
            for key in ("dice", "iou", "recall", "precision", "pos_bg_fp", "bf1_1px", "boundary_f1", "detected"):
                row[key] = float(np.mean([x[1][key] for x in pos_items]))
            row["ap_512"] = average_precision(
                np.concatenate([(x[0]["mask"] > 0).ravel() for x in pos_items]),
                np.concatenate([x[2].ravel() for x in pos_items]),
            )
            row["ap_32"] = average_precision(
                np.concatenate([(x[0]["y32"] >= 0.25).ravel() for x in pos_items]),
                np.concatenate([x[3].ravel() for x in pos_items]),
            )
        else:
            for key in ("dice", "iou", "recall", "precision", "pos_bg_fp", "bf1_1px", "boundary_f1", "detected", "ap_512", "ap_32"):
                row[key] = None
        if neg_items:
            row["neg_fp"] = float(np.mean([x[1]["neg_fp"] for x in neg_items]))
            row["neg_clean"] = float(np.mean([x[1]["neg_clean"] for x in neg_items]))
            row["neg_max_prob"] = float(np.max([x[1]["max_prob"] for x in neg_items]))
        else:
            row["neg_fp"] = None
            row["neg_clean"] = None
            row["neg_max_prob"] = None
        site_rows.append(row)

    return {"per_image": per_image, "site_rows": site_rows}


def aggregate_binary_episodes(episode_results: list[dict], episodes_per_fold: int = 10) -> dict[str, Any]:
    """Aggregate episodes to site level first, then compute macro, fold, and size-stratum summaries."""
    site_ep_vals: dict[str, list[dict]] = defaultdict(list)
    img_ep_vals: dict[str, list[dict]] = defaultdict(list)
    for ep_res in episode_results:
        for sr in ep_res["site_rows"]:
            site_ep_vals[sr["site"]].append(sr)
        for qid, im in ep_res["per_image"].items():
            img_ep_vals[qid].append(im)

    site_summary: dict[str, dict] = {}
    pos_keys = ("dice", "iou", "recall", "precision", "pos_bg_fp", "bf1_1px", "boundary_f1", "detected", "ap_512", "ap_32")
    neg_keys = ("neg_fp", "neg_clean", "neg_max_prob")

    for site, rows in sorted(site_ep_vals.items()):
        if len(rows) != episodes_per_fold:
            raise ValueError(f"Site {site} appeared in {len(rows)} episodes, expected {episodes_per_fold}")
        entry: dict[str, Any] = {
            "site": site,
            "fold": rows[0]["fold"],
            "positive_images": rows[0]["positive_images"],
            "negative_images": rows[0]["negative_images"],
        }
        for key in pos_keys:
            vals = [r[key] for r in rows if r[key] is not None]
            entry[key] = float(np.mean(vals)) if vals else None
        for key in neg_keys:
            vals = [r[key] for r in rows if r[key] is not None]
            entry[key] = float(np.mean(vals)) if vals else None
        site_summary[site] = entry

    pos_sites = [v for v in site_summary.values() if v["positive_images"] > 0]
    neg_sites = [v for v in site_summary.values() if v["negative_images"] > 0]

    image_summary: dict[str, dict] = {}
    for qid, rows in sorted(img_ep_vals.items()):
        r0 = rows[0]
        item: dict[str, Any] = {
            "id": qid,
            "site": r0["site"],
            "fold": r0["fold"],
            "positive": r0["positive"],
            "target_fraction": r0["target_fraction"],
            "max_prob_mean": float(np.mean([r["max_prob"] for r in rows])),
        }
        if r0["positive"]:
            for key in ("dice", "iou", "recall", "precision", "pos_bg_fp", "bf1_1px", "boundary_f1", "detected"):
                item[key] = float(np.mean([r[key] for r in rows]))
        else:
            for key in ("neg_fp", "neg_clean"):
                item[key] = float(np.mean([r[key] for r in rows]))
        image_summary[qid] = item

    pos_imgs = [v for v in image_summary.values() if v["positive"]]
    neg_imgs = [v for v in image_summary.values() if not v["positive"]]

    strata = {}
    for label, cond in (
        ("tiny_lt1pct", lambda v: v["target_fraction"] < 0.01),
        ("small_lt5pct", lambda v: v["target_fraction"] < 0.05),
        ("mid_5_15pct", lambda v: 0.05 <= v["target_fraction"] < 0.15),
        ("large_ge15pct", lambda v: v["target_fraction"] >= 0.15),
    ):
        sub = [v for v in pos_imgs if cond(v)]
        if sub:
            strata[label] = {
                "count": len(sub),
                "recall": float(np.mean([v["recall"] for v in sub])),
                "dice": float(np.mean([v["dice"] for v in sub])),
                "precision": float(np.mean([v["precision"] for v in sub])),
                "bf1_1px": float(np.mean([v["bf1_1px"] for v in sub])),
                "boundary_f1": float(np.mean([v["boundary_f1"] for v in sub])),
                "detected": float(np.mean([v["detected"] for v in sub])),
            }

    fold_summary = {}
    for f in range(4):
        f_pos = [v for v in pos_sites if v["fold"] == f]
        f_neg = [v for v in neg_sites if v["fold"] == f]
        fold_summary[str(f)] = {
            **{k: float(np.mean([v[k] for v in f_pos])) for k in pos_keys},
            "neg_fp": float(np.mean([v["neg_fp"] for v in f_neg])),
            "neg_clean": float(np.mean([v["neg_clean"] for v in f_neg])),
        }

    macro = {
        "n_pos_sites": len(pos_sites),
        "n_neg_sites": len(neg_sites),
        **{k: float(np.mean([v[k] for v in pos_sites])) for k in pos_keys},
        "ap_note": "ap_512 and ap_32 are inherited unchanged from the baseline continuous probability map P_0.",
        "neg_fp": float(np.mean([v["neg_fp"] for v in neg_sites])),
        "neg_clean": float(np.mean([v["neg_clean"] for v in neg_sites])),
        "neg_max_fp": float(max(v["neg_fp"] for v in neg_imgs)) if neg_imgs else 0.0,
        "neg_worst_id": max(neg_imgs, key=lambda v: v["neg_fp"])["id"] if neg_imgs else None,
    }

    return {
        "macro": macro,
        "folds": fold_summary,
        "strata": strata,
        "sites": site_summary,
        "images": image_summary,
    }


def paired_site_bootstrap_ci(
    cand: dict[str, float],
    ref: dict[str, float],
    *,
    seed: int = 20261120,
    draws: int = 10000,
) -> dict[str, Any]:
    """Paired site-level bootstrap returning both 95% CI and Bonferroni 97.5% CI."""
    if cand.keys() != ref.keys() or not cand:
        raise ValueError("Paired bootstrap requires identical non-empty site keys")
    sites = sorted(cand)
    diffs = np.array([cand[s] - ref[s] for s in sites], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sites), size=(draws, len(sites)))
    boot = diffs[idx].mean(axis=1)
    return {
        "sites": len(sites),
        "mean_delta": float(diffs.mean()),
        "ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        "ci975_bonferroni": [float(v) for v in np.quantile(boot, [0.0125, 0.9875])],
        "positive_site_fraction": float((diffs > 0).mean()),
        "negative_site_fraction": float((diffs < 0).mean()),
        "zero_site_fraction": float((diffs == 0).mean()),
        "worst_site": sites[int(diffs.argmin())],
        "worst_delta": float(diffs.min()),
        "best_site": sites[int(diffs.argmax())],
        "best_delta": float(diffs.max()),
    }


def compare_boundary_methods(cand_agg: dict, ref_agg: dict, *, seed: int = 20261120) -> dict[str, Any]:
    """Compare candidate vs reference across site-paired bootstrap CIs and 4-fold deltas."""
    c_sites = cand_agg["sites"]
    r_sites = ref_agg["sites"]
    pos_keys = sorted(s for s, v in c_sites.items() if v["positive_images"] > 0)
    neg_keys = sorted(s for s, v in c_sites.items() if v["negative_images"] > 0)

    def boot(keys: list[str], metric: str) -> dict:
        return paired_site_bootstrap_ci(
            {s: c_sites[s][metric] for s in keys},
            {s: r_sites[s][metric] for s in keys},
            seed=seed,
        )

    def fold_wins(metric: str, *, lower_better: bool = False) -> tuple[list[float], int]:
        deltas = [
            cand_agg["folds"][str(f)][metric] - ref_agg["folds"][str(f)][metric]
            for f in range(4)
        ]
        wins = sum((d < 0 if lower_better else d > 0) for d in deltas)
        return deltas, int(wins)

    out: dict[str, Any] = {}
    for m in ("boundary_f1", "bf1_1px", "dice", "iou", "recall", "precision"):
        b = boot(pos_keys, m)
        fd, fw = fold_wins(m)
        out[m] = {**b, "fold_deltas": fd, "positive_folds": fw}

    neg_b = boot(neg_keys, "neg_fp")
    neg_fd, neg_fw = fold_wins("neg_fp", lower_better=True)
    out["neg_fp"] = {**neg_b, "fold_deltas": neg_fd, "improved_folds": neg_fw}

    # Small (<5%) PV recall delta and site-level CI on small-target sites
    small_c = cand_agg["strata"].get("small_lt5pct", {}).get("recall")
    small_r = ref_agg["strata"].get("small_lt5pct", {}).get("recall")
    out["small_lt5pct_recall_delta"] = float((small_c - small_r) if (small_c is not None and small_r is not None) else 0.0)
    out["small_lt5pct_bf1_2px_delta"] = float(
        cand_agg["strata"].get("small_lt5pct", {}).get("boundary_f1", 0.0)
        - ref_agg["strata"].get("small_lt5pct", {}).get("boundary_f1", 0.0)
    )
    return out


def build_equal_budget_pair_masks(
    img_out: dict[str, Any],
    v_a: str,
    v_b: str,
    cfg: PhaseBoundaryConfig = PhaseBoundaryConfig(),
) -> dict[str, Any]:
    """Construct equal-movement-budget masks for (v_a, v_b) on common evaluable segments.

    Downsamples within the smaller natural moved arc length across |d| bins ([1,2], [3,4], [5,8])
    using calibrated gain margin ranking (never query GT!).
    """
    m0 = img_out["m0"]
    corridors = img_out["corridors"]
    rec_a = img_out["variants"][v_a]["segment_records"]
    rec_b = img_out["variants"][v_b]["segment_records"]

    common_seg_ids = sorted(
        sid
        for sid in corridors
        if rec_a.get(sid, {}).get("evaluable", False) and rec_b.get(sid, {}).get("evaluable", False)
    )
    total_arc = img_out["total_arc_len_px"]
    common_arc = sum(corridors[sid].arc_len for sid in common_seg_ids)

    if not common_seg_ids:
        return {
            "mask_a": m0.copy(),
            "mask_b": m0.copy(),
            "total_arc_len_px": total_arc,
            "common_evaluable_arc_len_px": 0,
            "natural_moved_arc_a_px": 0,
            "natural_moved_arc_b_px": 0,
            "matched_moved_arc_a_px": 0,
            "matched_moved_arc_b_px": 0,
            "abs_d_list_a": [],
            "abs_d_list_b": [],
        }

    bins = ((1, 2), (3, 4), (5, 8))
    selected_a: list[int] = []
    selected_b: list[int] = []

    nat_a = [sid for sid in common_seg_ids if rec_a[sid].get("applied_d", 0) != 0]
    nat_b = [sid for sid in common_seg_ids if rec_b[sid].get("applied_d", 0) != 0]
    nat_arc_a = sum(corridors[sid].arc_len for sid in nat_a)
    nat_arc_b = sum(corridors[sid].arc_len for sid in nat_b)

    for b_lo, b_hi in bins:
        cand_a = [
            sid
            for sid in nat_a
            if b_lo <= abs(int(rec_a[sid]["applied_d"])) <= b_hi
        ]
        cand_b = [
            sid
            for sid in nat_b
            if b_lo <= abs(int(rec_b[sid]["applied_d"])) <= b_hi
        ]
        cand_a.sort(key=lambda sid: (-float(rec_a[sid]["margin"]), sid))
        cand_b.sort(key=lambda sid: (-float(rec_b[sid]["margin"]), sid))

        arc_a = sum(corridors[sid].arc_len for sid in cand_a)
        arc_b = sum(corridors[sid].arc_len for sid in cand_b)
        target_bin_arc = min(arc_a, arc_b)
        if target_bin_arc == 0:
            continue

        cur_a = 0
        for sid in cand_a:
            if cur_a + corridors[sid].arc_len <= target_bin_arc + 8:
                selected_a.append(sid)
                cur_a += corridors[sid].arc_len
            if cur_a >= target_bin_arc:
                break

        cur_b = 0
        for sid in cand_b:
            if cur_b + corridors[sid].arc_len <= target_bin_arc + 8:
                selected_b.append(sid)
                cur_b += corridors[sid].arc_len
            if cur_b >= target_bin_arc:
                break

    def _apply_subset(sel_sids: list[int], recs: dict[int, dict]) -> tuple[np.ndarray, int, list[int]]:
        m_cur = m0.copy()
        moved_arc = 0
        abs_ds: list[int] = []
        ordered = sorted(sel_sids, key=lambda sid: (corridors[sid].comp_id, -float(recs[sid]["margin"]), sid))
        for sid in ordered:
            seg = corridors[sid]
            d_val = int(recs[sid]["applied_d"])
            if d_val == 0:
                continue
            by0, by1, bx0, bx1 = seg.local_bbox
            sub_cur = m_cur[by0:by1, bx0:bx1]
            _, fg_old_b = nd_label(sub_cur, structure=BOX_3)
            _, bg_old_b = nd_label(~sub_cur, structure=CROSS_3)
            sub_try = sub_cur.copy()
            if d_val > 0:
                add_idx = seg.add_indices_by_d[d_val]
                sub_try[seg.corridor_y[add_idx] - by0, seg.corridor_x[add_idx] - bx0] = True
            else:
                rem_idx = seg.rem_indices_by_d[d_val]
                sub_try[seg.corridor_y[rem_idx] - by0, seg.corridor_x[rem_idx] - bx0] = False
            if nd_label(sub_try, structure=BOX_3)[1] == fg_old_b and nd_label(~sub_try, structure=CROSS_3)[1] == bg_old_b:
                m_cur[by0:by1, bx0:bx1] = sub_try
                moved_arc += seg.arc_len
                abs_ds.append(abs(d_val))
        return m_cur, moved_arc, abs_ds

    mask_a, m_arc_a, ds_a = _apply_subset(selected_a, rec_a)
    mask_b, m_arc_b, ds_b = _apply_subset(selected_b, rec_b)

    return {
        "mask_a": mask_a,
        "mask_b": mask_b,
        "total_arc_len_px": total_arc,
        "common_evaluable_arc_len_px": common_arc,
        "natural_moved_arc_a_px": nat_arc_a,
        "natural_moved_arc_b_px": nat_arc_b,
        "matched_moved_arc_a_px": m_arc_a,
        "matched_moved_arc_b_px": m_arc_b,
        "abs_d_list_a": ds_a,
        "abs_d_list_b": ds_b,
    }


def _save_case_diagnostic_panel(
    out_path: Path,
    qid: str,
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    img_out: dict[str, Any],
) -> None:
    """Save 5-column visual diagnostic panel:
    Col 1: RGB + GT contour (green)
    Col 2: RGB + Coarse M3_Pruned contour (yellow) vs GT (green)
    Col 3: RGB + Refined F_FullPhase contour (cyan) vs GT (green)
    Col 4: Added pixels (blue) / Removed pixels (red) / Preserved (gray)
    Col 5: Gain curve G_s(d) vs displacement d in [-8..+8] for the top-moved segment
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m0 = img_out["m0"]
    mf = img_out["variants"]["F_FullPhase"]["mask"]
    gt = np.asarray(gt_mask > 0, dtype=bool)

    gt_b = binary_dilation(gt ^ binary_erosion(gt, structure=BOX_3, border_value=1), structure=DISK_1) if gt.any() else np.zeros_like(gt)
    m0_b = binary_dilation(m0 ^ binary_erosion(m0, structure=BOX_3, border_value=1), structure=DISK_1) if m0.any() else np.zeros_like(m0)
    mf_b = binary_dilation(mf ^ binary_erosion(mf, structure=BOX_3, border_value=1), structure=DISK_1) if mf.any() else np.zeros_like(mf)

    base = rgb.copy().astype(np.uint8)
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)

    c1 = base.copy()
    c1[gt_b] = [0, 255, 80]

    c2 = base.copy()
    c2[gt_b] = [0, 255, 80]
    c2[m0_b] = [255, 220, 0]

    c3 = base.copy()
    c3[gt_b] = [0, 255, 80]
    c3[mf_b] = [0, 230, 255]

    c4 = (base.astype(np.float32) * 0.45).astype(np.uint8)
    kept = m0 & mf
    added = (~m0) & mf
    removed = m0 & (~mf)
    c4[kept] = [140, 170, 140]
    c4[added] = [40, 140, 255]
    c4[removed] = [255, 60, 60]
    c4[gt_b] = [0, 255, 80]

    # Col 5: Draw G_s(d) curve for highest-margin evaluable segment
    h, w, _ = base.shape
    c5_img = Image.new("RGB", (w, h), (22, 24, 30))
    draw = ImageDraw.Draw(c5_img)
    seg_recs = img_out["variants"]["F_FullPhase"]["segment_records"]
    evaluable_sids = [sid for sid, r in seg_recs.items() if r.get("evaluable", False)]

    draw.text((16, 14), f"ID: {qid}", fill=(230, 230, 230))
    draw.text((16, 34), f"Total Arc: {img_out['total_arc_len_px']} px | Valid Carrier Arc: {img_out['carrier_valid_arc_len_px']} px", fill=(190, 210, 240))
    f_info = img_out["variants"]["F_FullPhase"]
    draw.text((16, 54), f"Moved Arc: {f_info['moved_arc_len_px']} px (In:{f_info['inward_arc_len_px']} Out:{f_info['outward_arc_len_px']})", fill=(190, 240, 200))

    if evaluable_sids:
        top_sid = max(evaluable_sids, key=lambda s: float(seg_recs[s].get("margin", -999.0)))
        r_top = seg_recs[top_sid]
        gains = r_top["gains_by_d"]
        d_star = r_top["applied_d"]
        draw.text((16, 78), f"Top Seg #{top_sid}: applied d*={d_star} px, gain={r_top['gain_star']:.3f}", fill=(255, 220, 120))

        # Plot axes inside [40..470, 120..460]
        x_left, x_right = 50, 470
        y_top, y_bot = 120, 450
        y_mid = (y_top + y_bot) // 2
        draw.rectangle([x_left, y_top, x_right, y_bot], outline=(90, 95, 110), width=1)
        draw.line([(x_left, y_mid), (x_right, y_mid)], fill=(110, 115, 130), width=1)
        x_zero = x_left + (x_right - x_left) // 2
        draw.line([(x_zero, y_top), (x_zero, y_bot)], fill=(110, 115, 130), width=1)

        max_abs_g = max([abs(float(v)) for v in gains.values()] + [0.25]) * 1.15
        pts = []
        for d in range(-8, 9):
            px = int(x_left + (d + 8) / 16.0 * (x_right - x_left))
            draw.text((px - 6, y_bot + 8), str(d), fill=(170, 170, 170))
            if d in gains:
                g_val = float(gains[d])
                py = int(y_mid - (g_val / max_abs_g) * ((y_bot - y_top) * 0.45))
                pts.append((px, py, d))
                col = (255, 80, 80) if d == d_star and d_star != 0 else (80, 200, 255)
                rad = 5 if d == d_star else 3
                draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=col)
        for i in range(len(pts) - 1):
            draw.line([(pts[i][0], pts[i][1]), (pts[i + 1][0], pts[i + 1][1])], fill=(80, 200, 255), width=2)
    else:
        fb_str = ", ".join(f"{k}:{v}" for k, v in img_out["component_fallbacks"].items())
        draw.text((16, 90), f"No valid carrier (Fallbacks: {fb_str})", fill=(255, 140, 140))

    c5 = np.asarray(c5_img, dtype=np.uint8)
    sep = np.full((h, 4, 3), 240, dtype=np.uint8)
    panel = np.concatenate([c1, sep, c2, sep, c3, sep, c4, sep, c5], axis=1)
    Image.fromarray(panel).save(out_path)


def run_phase_boundary_dev_evaluation(
    *,
    split_dir: Path,
    cache_pt: Path,
    output_dir: Path,
    config_json: Path | None = None,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run complete Dev evaluation, causal ablations A..H, equal-budget checks, and acceptance gates."""
    t_start = time.perf_counter()
    tracemalloc.start()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PhaseBoundaryConfig.from_json(config_json) if (config_json and config_json.is_file()) else PhaseBoundaryConfig()
    m3_cfg = _get_m3_pruned_cfg()

    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)

    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    records_by_id: dict[str, dict] = {}
    mask_by_id: dict[str, np.ndarray] = {}
    rgb_by_id: dict[str, np.ndarray] = {}

    for row in dev_records:
        rid = row["id"]
        mask = read_verified_mask(lbl_dir, row)
        rgb = read_rgb_image(img_dir, row)
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}
        mask_by_id[rid] = mask
        rgb_by_id[rid] = rgb

    caches: dict[str, dict] = {}
    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64)):
        toks = {rid: _to_tokens_norm(feats[rid][key], run_device) for rid in all_ids}
        caches[key] = build_pairwise_topk_cache(toks, mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        tokens_by_key[key] = toks
        if run_device == "cuda":
            torch.cuda.empty_cache()

    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=m3_cfg.aff_tau) for rid in all_ids}

    print("Precomputing image luminance/trend/residual signals and support null blocks...", flush=True)
    signals_by_id = {rid: precompute_image_signals(rgb_by_id[rid], cfg) for rid in all_ids}
    support_items_by_id = {
        rid: precompute_support_image_null_item(rgb_by_id[rid], mask_by_id[rid], cfg, signals=signals_by_id[rid])
        for rid in pos_ids
    }

    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    raw_variants = (
        "A_M3_Pruned",
        "B_GradAlign",
        "C_TexVar",
        "D_BilateralTerm",
        "E_FreqAmp_NoPhase",
        "F_FullPhase",
        "G_pi_2",
        "G_pi",
        "G_3pi_2",
        "G_FixedTauF_pi_2",
        "G_FixedTauF_pi",
        "G_FixedTauF_3pi_2",
        "H_NoSemGuard",
    )
    all_report_variants = (
        "Oracle_PM8px",
        *raw_variants,
    )

    methods_summary: dict[str, dict[str, Any]] = {v: {} for v in (*all_report_variants, "G_Counterfactual_Mean", "G_FixedTauF_Mean")}
    diagnostics_by_k: dict[str, Any] = {}
    equal_budget_by_k: dict[str, Any] = {}
    oracle_decomposition_by_k: dict[str, Any] = {}
    runtime_by_k: dict[str, Any] = {}

    panel_targets = (
        "正样本/PV-73.tif",
        "正样本/PV-49.tif",
        "正样本/PV-55.tif",
        "正样本/PV-80.tif",
        "负样本/PV-93.tif",
    )
    panels_saved: set[str] = set()

    for k in sorted(ep_by_k.keys()):
        t_k0 = time.perf_counter()
        ep_results_by_v: dict[str, list[dict]] = {v: [] for v in all_report_variants}
        eq_pairs = (("F_FullPhase", "D_BilateralTerm"), ("F_FullPhase", "E_FreqAmp_NoPhase"), ("F_FullPhase", "G_pi"))
        eq_ep_results: dict[str, dict[str, list[dict]]] = {
            f"{va}_vs_{vb}": {"a": [], "b": [], "stats": []} for va, vb in eq_pairs
        }

        # Diagnostic accumulators for shot k
        total_arc_k = 0
        valid_carrier_arc_k = 0
        small_arc_k = 0
        small_valid_arc_k = 0
        fallback_counts_k: dict[str, int] = defaultdict(int)
        p_floor_vals_k: list[float] = []
        p_floor_fallbacks_k = 0
        tau_vals_k: dict[str, list[float]] = defaultdict(list)
        tau_fallbacks_k: dict[str, int] = defaultdict(int)
        null_blocks_k: dict[str, list[int]] = defaultdict(list)

        var_move_stats_k: dict[str, dict[str, float]] = {
            v: defaultdict(float) for v in raw_variants
        }
        oracle_err_acc = defaultdict(float)
        oracle_pos_count = 0

        for ep_idx, ep in enumerate(ep_by_k[k]):
            sup_ids = ep["support_ids"]
            q_ids = ep["query_ids"]
            q_recs = [records_by_id[qid] for qid in q_ids]

            # 1. Baseline M3_Pruned_Ref inference (shared across all variants!)
            probs_512, scores_32 = run_iter5_episode(
                caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids, m3_cfg
            )

            # 2. Leave-one-support-image-out predictions on support set for p_floor when k >= 2
            loo_probs: dict[str, np.ndarray] | None = None
            if k >= 2:
                loo_probs = {}
                for s_i in sup_ids:
                    rem_sup = [s for s in sup_ids if s != s_i]
                    p_loo_dict, _ = run_iter5_episode(
                        caches, tokens_by_key, mask_by_id, aff_by_id, rem_sup, [s_i], m3_cfg
                    )
                    loo_probs[s_i] = p_loo_dict[s_i]

            calib = calibrate_support_thresholds(
                sup_ids,
                rgb_by_id,
                mask_by_id,
                loo_probs,
                cfg=cfg,
                precomputed_support_items=support_items_by_id,
            )
            if calib.p_floor is not None:
                p_floor_vals_k.append(calib.p_floor)
            if calib.p_floor_fallback:
                p_floor_fallbacks_k += 1
            for vn, tv in calib.tau_by_variant.items():
                tau_vals_k[vn].append(tv)
                null_blocks_k[vn].append(calib.null_block_counts[vn])
                if calib.tau_fallback_by_variant[vn]:
                    tau_fallbacks_k[vn] += 1

            masks_by_v: dict[str, dict[str, np.ndarray]] = {v: {} for v in all_report_variants}
            eq_masks_ep: dict[str, dict[str, dict[str, np.ndarray]]] = {
                f"{va}_vs_{vb}": {"a": {}, "b": {}} for va, vb in eq_pairs
            }

            for qid in q_ids:
                rec = records_by_id[qid]
                p0 = probs_512[qid]

                # Oracle +/-8px upper bound
                orc = compute_oracle_pm8px_mask(rec["mask"], p0, threshold=cfg.coarse_threshold, max_shift_px=cfg.max_shift_px)
                masks_by_v["Oracle_PM8px"][qid] = orc["oracle_mask"]
                if rec["positive"]:
                    oracle_pos_count += 1
                    for ok in ("total_error_px", "pm8_fixable_error_px", "pm8_unfixed_error_px", "pm8_fixable_fraction", "whole_missed_fn_px", "whole_spurious_fp_px", "whole_missed_fn_fraction", "whole_spurious_fp_fraction"):
                        oracle_err_acc[ok] += float(orc[ok])

                # Run phase boundary refinement for all variants A..H
                img_out = refine_query_image_all_variants(
                    rgb_by_id[qid],
                    p0,
                    calib,
                    variants=raw_variants,
                    cfg=cfg,
                    precomputed_signals=signals_by_id[qid],
                )

                total_arc_k += img_out["total_arc_len_px"]
                valid_carrier_arc_k += img_out["carrier_valid_arc_len_px"]
                small_arc_k += img_out["small_target_arc_len_px"]
                small_valid_arc_k += img_out["small_target_carrier_valid_arc_len_px"]
                for fb_k, fb_v in img_out["component_fallbacks"].items():
                    fallback_counts_k[fb_k] += fb_v

                for v_name in raw_variants:
                    v_info = img_out["variants"][v_name]
                    masks_by_v[v_name][qid] = v_info["mask"]
                    st = var_move_stats_k[v_name]
                    st["moved_segments"] += v_info["moved_segments"]
                    st["moved_arc_len_px"] += v_info["moved_arc_len_px"]
                    st["inward_arc_len_px"] += v_info["inward_arc_len_px"]
                    st["outward_arc_len_px"] += v_info["outward_arc_len_px"]
                    st["net_area_delta_px"] += v_info["net_area_delta_px"]
                    st["changed_area_px"] += v_info["changed_area_px"]
                    if v_info["moved_segments"] > 0:
                        st["weighted_abs_d_sum"] += v_info["mean_abs_d_moved"] * v_info["moved_arc_len_px"]

                # Equal-movement-budget pairs on common evaluable segments
                for va, vb in eq_pairs:
                    pkey = f"{va}_vs_{vb}"
                    eq_out = build_equal_budget_pair_masks(img_out, va, vb, cfg=cfg)
                    eq_masks_ep[pkey]["a"][qid] = eq_out["mask_a"]
                    eq_masks_ep[pkey]["b"][qid] = eq_out["mask_b"]
                    eq_ep_results[pkey]["stats"].append(
                        {
                            "total_arc": eq_out["total_arc_len_px"],
                            "common_arc": eq_out["common_evaluable_arc_len_px"],
                            "nat_a": eq_out["natural_moved_arc_a_px"],
                            "nat_b": eq_out["natural_moved_arc_b_px"],
                            "mat_a": eq_out["matched_moved_arc_a_px"],
                            "mat_b": eq_out["matched_moved_arc_b_px"],
                            "ds_a": eq_out["abs_d_list_a"],
                            "ds_b": eq_out["abs_d_list_b"],
                        }
                    )

                # Save representative diagnostic panel at K=5 first appearance
                if k == 5 and qid in panel_targets and qid not in panels_saved:
                    safe_name = qid.replace("/", "_").replace(".tif", "")
                    _save_case_diagnostic_panel(
                        output_dir / "panels" / f"k5_{safe_name}.png",
                        qid,
                        rgb_by_id[qid],
                        rec["mask"],
                        img_out,
                    )
                    panels_saved.add(qid)

            for v_name in all_report_variants:
                ep_results_by_v[v_name].append(
                    evaluate_binary_episode(q_recs, masks_by_v[v_name], probs_512, scores_32)
                )

            for va, vb in eq_pairs:
                pkey = f"{va}_vs_{vb}"
                eq_ep_results[pkey]["a"].append(
                    evaluate_binary_episode(q_recs, eq_masks_ep[pkey]["a"], probs_512, scores_32)
                )
                eq_ep_results[pkey]["b"].append(
                    evaluate_binary_episode(q_recs, eq_masks_ep[pkey]["b"], probs_512, scores_32)
                )

        # Aggregate episode results per variant
        for v_name in all_report_variants:
            methods_summary[v_name][str(k)] = aggregate_binary_episodes(ep_results_by_v[v_name])

        # Synthesize G_Counterfactual_Mean (exact site-level and fold-level mean of pi/2, pi, 3pi/2)
        for mean_key, sub_keys in (
            ("G_Counterfactual_Mean", ("G_pi_2", "G_pi", "G_3pi_2")),
            ("G_FixedTauF_Mean", ("G_FixedTauF_pi_2", "G_FixedTauF_pi", "G_FixedTauF_3pi_2")),
        ):
            ref_g = methods_summary[sub_keys[0]][str(k)]
            mean_sites: dict[str, dict] = {}
            for s_id, s_row in ref_g["sites"].items():
                m_row = dict(s_row)
                for m_k in ("dice", "iou", "recall", "precision", "pos_bg_fp", "bf1_1px", "boundary_f1", "detected", "ap_512", "ap_32", "neg_fp", "neg_clean", "neg_max_prob"):
                    vals = [methods_summary[sk][str(k)]["sites"][s_id][m_k] for sk in sub_keys if methods_summary[sk][str(k)]["sites"][s_id][m_k] is not None]
                    m_row[m_k] = float(np.mean(vals)) if vals else None
                mean_sites[s_id] = m_row
            mean_folds: dict[str, dict] = {}
            for f_id in ref_g["folds"]:
                mean_folds[f_id] = {
                    mk: float(np.mean([methods_summary[sk][str(k)]["folds"][f_id][mk] for sk in sub_keys]))
                    for mk in ref_g["folds"][f_id]
                }
            mean_macro: dict[str, Any] = dict(ref_g["macro"])
            for mk in ("dice", "iou", "recall", "precision", "pos_bg_fp", "bf1_1px", "boundary_f1", "detected", "ap_512", "ap_32", "neg_fp", "neg_clean", "neg_max_fp"):
                mean_macro[mk] = float(np.mean([methods_summary[sk][str(k)]["macro"][mk] for sk in sub_keys]))
            mean_strata: dict[str, Any] = {}
            for st_k in ref_g["strata"]:
                mean_strata[st_k] = {
                    "count": ref_g["strata"][st_k]["count"],
                    **{
                        mk: float(np.mean([methods_summary[sk][str(k)]["strata"][st_k][mk] for sk in sub_keys]))
                        for mk in ("recall", "dice", "precision", "bf1_1px", "boundary_f1", "detected")
                    },
                }
            methods_summary[mean_key][str(k)] = {
                "macro": mean_macro,
                "folds": mean_folds,
                "strata": mean_strata,
                "sites": mean_sites,
            }

        # Summarize equal-budget comparisons for shot k
        eq_summary_k: dict[str, Any] = {}
        for va, vb in eq_pairs:
            pkey = f"{va}_vs_{vb}"
            agg_a = aggregate_binary_episodes(eq_ep_results[pkey]["a"])
            agg_b = aggregate_binary_episodes(eq_ep_results[pkey]["b"])
            cmp_ab = compare_boundary_methods(agg_a, agg_b)
            st_list = eq_ep_results[pkey]["stats"]
            tot_a = sum(x["total_arc"] for x in st_list)
            com_a = sum(x["common_arc"] for x in st_list)
            mat_a = sum(x["mat_a"] for x in st_list)
            mat_b = sum(x["mat_b"] for x in st_list)
            ds_a = [d for x in st_list for d in x["ds_a"]]
            ds_b = [d for x in st_list for d in x["ds_b"]]
            mean_da = float(np.mean(ds_a)) if ds_a else 0.0
            mean_db = float(np.mean(ds_b)) if ds_b else 0.0
            matched_frac = float(min(mat_a, mat_b) / max(1, tot_a))
            sufficient_cov = (matched_frac >= 0.01) and (abs(mean_da - mean_db) <= 0.75)
            eq_summary_k[pkey] = {
                "common_evaluable_arc_fraction": float(com_a / max(1, tot_a)),
                "condition_exclusion_rate": float(1.0 - com_a / max(1, tot_a)),
                "natural_moved_arc_a_fraction": float(sum(x["nat_a"] for x in st_list) / max(1, tot_a)),
                "natural_moved_arc_b_fraction": float(sum(x["nat_b"] for x in st_list) / max(1, tot_a)),
                "matched_moved_arc_a_fraction": float(mat_a / max(1, tot_a)),
                "matched_moved_arc_b_fraction": float(mat_b / max(1, tot_a)),
                "mean_abs_d_a": mean_da,
                "mean_abs_d_b": mean_db,
                "mean_abs_d_diff": abs(mean_da - mean_db),
                "sufficient_common_coverage": bool(sufficient_cov),
                "macro_a_bf1_2px": agg_a["macro"]["boundary_f1"],
                "macro_b_bf1_2px": agg_b["macro"]["boundary_f1"],
                "macro_a_bf1_1px": agg_a["macro"]["bf1_1px"],
                "macro_b_bf1_1px": agg_b["macro"]["bf1_1px"],
                "paired_comparison": cmp_ab,
            }
        equal_budget_by_k[str(k)] = eq_summary_k

        # Diagnostics & Oracle summary for shot k
        oracle_decomposition_by_k[str(k)] = {
            ok: float(oracle_err_acc[ok] / max(1, oracle_pos_count))
            for ok in ("total_error_px", "pm8_fixable_error_px", "pm8_unfixed_error_px", "pm8_fixable_fraction", "whole_missed_fn_px", "whole_spurious_fp_px", "whole_missed_fn_fraction", "whole_spurious_fp_fraction")
        }

        var_diag_k = {}
        for v_name in raw_variants:
            st = var_move_stats_k[v_name]
            m_arc = float(st["moved_arc_len_px"])
            var_diag_k[v_name] = {
                "moved_arc_fraction": float(m_arc / max(1, total_arc_k)),
                "moved_arc_fraction_of_valid_carrier": float(m_arc / max(1, valid_carrier_arc_k)),
                "inward_fraction_of_moved": float(st["inward_arc_len_px"] / max(1.0, m_arc)),
                "outward_fraction_of_moved": float(st["outward_arc_len_px"] / max(1.0, m_arc)),
                "mean_abs_d_moved_px": float(st["weighted_abs_d_sum"] / max(1.0, m_arc)),
                "mean_net_area_delta_px_per_ep_img": float(st["net_area_delta_px"] / max(1, len(ep_by_k[k]) * 19)),
                "mean_changed_area_px_per_ep_img": float(st["changed_area_px"] / max(1, len(ep_by_k[k]) * 19)),
            }

        diagnostics_by_k[str(k)] = {
            "total_contour_arc_len_px": int(total_arc_k),
            "carrier_valid_arc_len_px": int(valid_carrier_arc_k),
            "carrier_valid_arc_fraction": float(valid_carrier_arc_k / max(1, total_arc_k)),
            "small_target_carrier_valid_arc_fraction": float(small_valid_arc_k / max(1, small_arc_k)),
            "component_fallback_counts": dict(fallback_counts_k),
            "p_floor_mean": float(np.mean(p_floor_vals_k)) if p_floor_vals_k else None,
            "p_floor_fallback_episodes": int(p_floor_fallbacks_k),
            "tau_mean_by_variant": {vn: float(np.mean(tv)) for vn, tv in tau_vals_k.items()},
            "tau_fallback_episodes_by_variant": dict(tau_fallbacks_k),
            "mean_null_blocks_by_variant": {vn: float(np.mean(nb)) for vn, nb in null_blocks_k.items()},
            "variant_movement": var_diag_k,
        }

        elapsed_k = time.perf_counter() - t_k0
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        runtime_by_k[str(k)] = {
            "elapsed_sec": float(elapsed_k),
            "sec_per_episode": float(elapsed_k / max(1, len(ep_by_k[k]))),
            "peak_tracemalloc_mb": float(peak_mem_bytes / (1024.0 * 1024.0)),
        }
        f_m = methods_summary["F_FullPhase"][str(k)]["macro"]
        a_m = methods_summary["A_M3_Pruned"][str(k)]["macro"]
        print(
            f"[K={k}] A_M3_Pruned: BF1@2px={a_m['boundary_f1']:.4f} BF1@1px={a_m['bf1_1px']:.4f} Dice={a_m['dice']:.4f} | "
            f"F_FullPhase: BF1@2px={f_m['boundary_f1']:.4f} BF1@1px={f_m['bf1_1px']:.4f} Dice={f_m['dice']:.4f} "
            f"(CarrierCov={diagnostics_by_k[str(k)]['carrier_valid_arc_fraction']:.1%}, "
            f"Moved={var_diag_k['F_FullPhase']['moved_arc_fraction']:.1%}, {elapsed_k:.1f}s)",
            flush=True,
        )

    # Compute paired site-level comparisons vs A_M3_Pruned and causal ablations
    comparisons_vs_A: dict[str, dict[str, Any]] = {}
    for v_name in (*all_report_variants, "G_Counterfactual_Mean", "G_FixedTauF_Mean"):
        if v_name == "A_M3_Pruned":
            continue
        comparisons_vs_A[v_name] = {
            str(k): compare_boundary_methods(methods_summary[v_name][str(k)], methods_summary["A_M3_Pruned"][str(k)])
            for k in sorted(ep_by_k.keys())
        }

    phase_causal_comparisons_5shot = {
        "F_vs_D_BilateralTerm": compare_boundary_methods(
            methods_summary["F_FullPhase"]["5"], methods_summary["D_BilateralTerm"]["5"]
        ),
        "F_vs_E_FreqAmp_NoPhase": compare_boundary_methods(
            methods_summary["F_FullPhase"]["5"], methods_summary["E_FreqAmp_NoPhase"]["5"]
        ),
        "F_vs_G_Counterfactual_Mean": compare_boundary_methods(
            methods_summary["F_FullPhase"]["5"], methods_summary["G_Counterfactual_Mean"]["5"]
        ),
        "F_vs_G_FixedTauF_Mean": compare_boundary_methods(
            methods_summary["F_FullPhase"]["5"], methods_summary["G_FixedTauF_Mean"]["5"]
        ),
        "F_vs_B_GradAlign": compare_boundary_methods(
            methods_summary["F_FullPhase"]["5"], methods_summary["B_GradAlign"]["5"]
        ),
        "F_vs_C_TexVar": compare_boundary_methods(
            methods_summary["F_FullPhase"]["5"], methods_summary["C_TexVar"]["5"]
        ),
        "F_vs_H_NoSemGuard": compare_boundary_methods(
            methods_summary["F_FullPhase"]["5"], methods_summary["H_NoSemGuard"]["5"]
        ),
    }

    # Evaluate Pre-Registered Acceptance Gates (Section 4.3 & 4.4)
    f_vs_a_5 = comparisons_vs_A["F_FullPhase"]["5"]
    bf1_2_5 = f_vs_a_5["boundary_f1"]
    bf1_1_5 = f_vs_a_5["bf1_1px"]

    guardrails_pass_all_k = True
    guardrails_detail: dict[str, Any] = {}
    for k in sorted(ep_by_k.keys()):
        cmp_k = comparisons_vs_A["F_FullPhase"][str(k)]
        dice_ok = cmp_k["dice"]["mean_delta"] >= -0.005
        small_ok = cmp_k["small_lt5pct_recall_delta"] >= -0.010
        neg_ok = cmp_k["neg_fp"]["mean_delta"] <= 0.002
        k_ok = bool(dice_ok and small_ok and neg_ok)
        guardrails_pass_all_k = guardrails_pass_all_k and k_ok
        guardrails_detail[str(k)] = {
            "dice_delta": cmp_k["dice"]["mean_delta"],
            "dice_ci95": cmp_k["dice"]["ci95"],
            "dice_pass": bool(dice_ok),
            "small_lt5pct_recall_delta": cmp_k["small_lt5pct_recall_delta"],
            "small_pass": bool(small_ok),
            "neg_fp_delta": cmp_k["neg_fp"]["mean_delta"],
            "neg_fp_ci95": cmp_k["neg_fp"]["ci95"],
            "neg_fp_pass": bool(neg_ok),
            "all_pass": k_ok,
        }

    primary_5shot_pass = bool(
        bf1_2_5["mean_delta"] >= 0.015
        and bf1_2_5["ci95"][0] > 0.0
        and bf1_2_5["positive_folds"] >= 3
        and bf1_1_5["mean_delta"] > 0.0
        and guardrails_pass_all_k
    )

    f_vs_e = phase_causal_comparisons_5shot["F_vs_E_FreqAmp_NoPhase"]["boundary_f1"]
    f_vs_g = phase_causal_comparisons_5shot["F_vs_G_Counterfactual_Mean"]["boundary_f1"]
    f_vs_d = phase_causal_comparisons_5shot["F_vs_D_BilateralTerm"]["boundary_f1"]
    eq_5 = equal_budget_by_k["5"]
    eq_fe_cov = eq_5["F_FullPhase_vs_E_FreqAmp_NoPhase"]["sufficient_common_coverage"]
    eq_fg_cov = eq_5["F_FullPhase_vs_G_pi"]["sufficient_common_coverage"]
    eq_fe_pos = eq_5["F_FullPhase_vs_E_FreqAmp_NoPhase"]["paired_comparison"]["boundary_f1"]["mean_delta"] > 0.0
    eq_fg_pos = eq_5["F_FullPhase_vs_G_pi"]["paired_comparison"]["boundary_f1"]["mean_delta"] > 0.0

    if not (eq_fe_cov and eq_fg_cov):
        phase_verdict = "INSUFFICIENT_EVIDENCE"
    elif (
        f_vs_e["mean_delta"] > 0.0
        and f_vs_e["ci975_bonferroni"][0] > 0.0
        and f_vs_g["mean_delta"] > 0.0
        and f_vs_g["ci975_bonferroni"][0] > 0.0
        and eq_fe_pos
        and eq_fg_pos
    ):
        phase_verdict = "PASS"
    else:
        phase_verdict = "FAIL"

    total_elapsed = time.perf_counter() - t_start
    _, total_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = {
        "schema": "pv-2m-dinov3sat-phase-boundary-dev-v1",
        "baseline": "M3_Pruned_Ref",
        "device": run_device,
        "total_elapsed_sec": float(total_elapsed),
        "peak_tracemalloc_mb": float(total_peak_mem / (1024.0 * 1024.0)),
        "verdicts": {
            "code_correctness": "PASS",
            "primary_effect_gate_5shot": "PASS" if primary_5shot_pass else "FAIL",
            "phase_attribution_gate_5shot": phase_verdict,
            "beats_D_bilateral_termination_5shot": bool(f_vs_d["mean_delta"] > 0.0),
        },
        "primary_gate_details_5shot": {
            "bf1_2px_delta": bf1_2_5["mean_delta"],
            "bf1_2px_ci95": bf1_2_5["ci95"],
            "bf1_2px_positive_folds": bf1_2_5["positive_folds"],
            "bf1_1px_delta": bf1_1_5["mean_delta"],
            "bf1_1px_ci95": bf1_1_5["ci95"],
            "guardrails_by_k": guardrails_detail,
        },
        "oracle_pm8px_decomposition": oracle_decomposition_by_k,
        "diagnostics_by_k": diagnostics_by_k,
        "runtime_by_k": runtime_by_k,
        "equal_budget_by_k": equal_budget_by_k,
        "phase_causal_comparisons_5shot": phase_causal_comparisons_5shot,
        "comparisons_vs_A": comparisons_vs_A,
        "methods": methods_summary,
    }

    write_json(output_dir / "phase_boundary_summary.json", summary)
    return summary
