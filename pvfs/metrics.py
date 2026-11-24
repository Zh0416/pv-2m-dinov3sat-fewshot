from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import torch


def _disk_structure(radius: int) -> np.ndarray:
    grid = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    return (yy * yy + xx * xx) <= radius * radius


DISK_1 = _disk_structure(1)
DISK_2 = _disk_structure(2)
DISK_8 = _disk_structure(8)
BOX_3 = np.ones((3, 3), dtype=bool)


def boundary_f1_at_radius(
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    radius: int = 2,
    t_b: np.ndarray | None = None,
    t_dil: np.ndarray | None = None,
) -> float:
    """Boundary contour F1 at pixel tolerance radius (1px or 2px)."""
    truth_b = np.asarray(truth, dtype=bool)
    pred_b = np.asarray(pred, dtype=bool)
    if t_b is None:
        t_b = truth_b ^ binary_erosion(truth_b, structure=BOX_3, border_value=1)
    p_b = pred_b ^ binary_erosion(pred_b, structure=BOX_3, border_value=1)
    if not t_b.any() and not p_b.any():
        return 1.0
    if not t_b.any() or not p_b.any():
        return 0.0
    disk = DISK_1 if radius == 1 else (DISK_2 if radius == 2 else _disk_structure(radius))
    if t_dil is None:
        t_dil = binary_dilation(t_b, structure=disk)
    p_dil = binary_dilation(p_b, structure=disk)
    prec = float(np.logical_and(p_b, t_dil).sum() / p_b.sum())
    rec = float(np.logical_and(t_b, p_dil).sum() / t_b.sum())
    return (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0


def evaluate_binary_mask_512(
    mask_gt: np.ndarray,
    pred_mask: np.ndarray,
    prob_512_ref: np.ndarray,
    *,
    t_b: np.ndarray | None = None,
    t_dil_1px: np.ndarray | None = None,
    t_dil_2px: np.ndarray | None = None,
) -> dict:
    """Evaluate a binary 512x512 mask with both BF1@1px and BF1@2px."""
    truth = np.asarray(mask_gt > 0, dtype=bool)
    pred = np.asarray(pred_mask, dtype=bool)
    gt_sum = int(truth.sum())
    pred_sum = int(pred.sum())

    if gt_sum > 0:
        tp = int(np.logical_and(truth, pred).sum())
        fp = pred_sum - tp
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
            "boundary_f1": float(bf1_2px),
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


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Exact tie-aware average precision without interpolation."""
    labels = np.asarray(y, dtype=bool).ravel()
    values = np.asarray(score, dtype=np.float32).ravel()
    total_pos = int(labels.sum())
    if total_pos == 0:
        return float("nan")
    if values.size >= 32768 and torch.cuda.is_available():
        t_val = torch.from_numpy(values).cuda()
        t_lbl = torch.from_numpy(labels.astype(np.int32)).cuda()
        order = torch.argsort(t_val, descending=True, stable=True)
        sorted_val = t_val[order]
        sorted_lbl = t_lbl[order]
        diff = torch.nonzero(sorted_val[1:] != sorted_val[:-1], as_tuple=False).squeeze(1)
        last = torch.cat((diff, torch.tensor([sorted_val.numel() - 1], device=t_val.device)))
        tp = torch.cumsum(sorted_lbl, dim=0)[last].double()
        precision = tp / (last.double() + 1.0)
        prev_tp = torch.cat((torch.zeros(1, device=t_val.device, dtype=torch.float64), tp[:-1]))
        recall_step = (tp - prev_tp) / float(total_pos)
        return float((precision * recall_step).sum().item())

    order = np.argsort(-values, kind="stable")
    labels = labels[order]
    values = values[order]
    last = np.r_[np.flatnonzero(np.diff(values)), len(values) - 1]
    tp = np.cumsum(labels, dtype=np.float64)[last]
    precision = tp / (last + 1.0)
    recall_step = np.diff(np.r_[0.0, tp]) / float(total_pos)
    return float(np.sum(precision * recall_step))


def boundary_f1(
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    t_b: np.ndarray | None = None,
    t_dil: np.ndarray | None = None,
) -> float:
    """Boundary contour F1 with 2-pixel (4m at 2m GSD) tolerance."""
    if t_b is None:
        t_b = truth ^ binary_erosion(truth, structure=BOX_3, border_value=1)
    p_b = pred ^ binary_erosion(pred, structure=BOX_3, border_value=1)
    if not t_b.any() and not p_b.any():
        return 1.0
    if not t_b.any() or not p_b.any():
        return 0.0
    if t_dil is None:
        t_dil = binary_dilation(t_b, structure=DISK_2)
    p_dil = binary_dilation(p_b, structure=DISK_2)
    prec = float(np.logical_and(p_b, t_dil).sum() / p_b.sum())
    rec = float(np.logical_and(t_b, p_dil).sum() / t_b.sum())
    return (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0


def evaluate_single_image_512(
    mask: np.ndarray,
    prob_512: np.ndarray,
    threshold: float = 0.5,
    *,
    compute_boundary: bool = True,
    t_b: np.ndarray | None = None,
    t_dil: np.ndarray | None = None,
) -> dict:
    truth = mask > 0
    pred = prob_512 >= threshold
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
        bf1 = boundary_f1(truth, pred, t_b=t_b, t_dil=t_dil) if compute_boundary else 0.0
        return {
            "positive": True,
            "dice": float(dice),
            "iou": float(iou),
            "recall": float(recall),
            "precision": float(precision),
            "pos_bg_fp": float(bg_fp),
            "boundary_f1": float(bf1),
            "detected": 1.0 if iou >= 0.10 else 0.0,
            "neg_fp": None,
            "neg_clean": None,
            "max_prob": float(prob_512.max()),
        }
    neg_fp = float(pred.mean())
    return {
        "positive": False,
        "dice": None,
        "iou": None,
        "recall": None,
        "precision": None,
        "pos_bg_fp": None,
        "boundary_f1": None,
        "detected": None,
        "neg_fp": neg_fp,
        "neg_clean": 1.0 if neg_fp < 0.01 else 0.0,
        "max_prob": float(prob_512.max()),
    }


def evaluate_episode(
    query_records: list[dict],
    probs_512: dict[str, np.ndarray],
    scores_32: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
) -> dict:
    """Compute per-image and per-site metrics for one episode."""
    per_image = {}
    by_site = defaultdict(list)
    for rec in query_records:
        qid = rec["id"]
        p512 = probs_512[qid]
        s32 = scores_32[qid]
        if p512.shape != (512, 512) or s32.shape != (32, 32):
            raise ValueError(f"Unexpected prediction shape for {qid}")
        if not np.isfinite(p512).all() or not np.isfinite(s32).all():
            raise ValueError(f"Non-finite prediction for {qid}")
        if rec["positive"] and "_gt_b" not in rec:
            truth = rec["mask"] > 0
            rec["_gt_b"] = truth ^ binary_erosion(truth, structure=BOX_3, border_value=1)
            rec["_gt_dil"] = binary_dilation(rec["_gt_b"], structure=DISK_2)
        im_m = evaluate_single_image_512(
            rec["mask"],
            p512,
            threshold=threshold,
            t_b=rec.get("_gt_b"),
            t_dil=rec.get("_gt_dil"),
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
        row = {
            "site": site,
            "fold": items[0][0]["fold"],
            "positive_images": len(pos_items),
            "negative_images": len(neg_items),
        }
        if pos_items:
            for key in ("dice", "iou", "recall", "precision", "pos_bg_fp", "boundary_f1", "detected"):
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
            for key in ("dice", "iou", "recall", "precision", "pos_bg_fp", "boundary_f1", "detected", "ap_512", "ap_32"):
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


def aggregate_episodes(episode_results: list[dict], episodes_per_fold: int = 10) -> dict:
    """Average each site across its E episodes first, then macro-average sites."""
    site_ep_vals: dict[str, list[dict]] = defaultdict(list)
    img_ep_vals: dict[str, list[dict]] = defaultdict(list)
    for ep_res in episode_results:
        for sr in ep_res["site_rows"]:
            site_ep_vals[sr["site"]].append(sr)
        for qid, im in ep_res["per_image"].items():
            img_ep_vals[qid].append(im)

    site_summary: dict[str, dict] = {}
    pos_keys = ("dice", "iou", "recall", "precision", "pos_bg_fp", "boundary_f1", "detected", "ap_512", "ap_32")
    neg_keys = ("neg_fp", "neg_clean", "neg_max_prob")

    for site, rows in sorted(site_ep_vals.items()):
        if len(rows) != episodes_per_fold:
            raise ValueError(f"Site {site} appeared in {len(rows)} episodes, expected {episodes_per_fold}")
        entry = {
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

    image_summary = {}
    for qid, rows in sorted(img_ep_vals.items()):
        r0 = rows[0]
        item = {
            "id": qid,
            "site": r0["site"],
            "fold": r0["fold"],
            "positive": r0["positive"],
            "target_fraction": r0["target_fraction"],
            "max_prob_mean": float(np.mean([r["max_prob"] for r in rows])),
        }
        if r0["positive"]:
            for key in ("dice", "iou", "recall", "precision", "pos_bg_fp", "boundary_f1", "detected"):
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


def paired_site_bootstrap(
    cand: dict[str, float],
    ref: dict[str, float],
    *,
    seed: int = 20261120,
    draws: int = 10000,
) -> dict:
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
        "positive_site_fraction": float((diffs > 0).mean()),
        "negative_site_fraction": float((diffs < 0).mean()),
        "worst_site": sites[int(diffs.argmin())],
        "worst_delta": float(diffs.min()),
    }


def compare_methods(cand_agg: dict, ref_agg: dict) -> dict:
    """Compare candidate vs reference and check pre-registered Gate-FP / Gate-FN / Gate-Net."""
    c_sites = cand_agg["sites"]
    r_sites = ref_agg["sites"]
    pos_keys = sorted(s for s, v in c_sites.items() if v["positive_images"] > 0)
    neg_keys = sorted(s for s, v in c_sites.items() if v["negative_images"] > 0)

    def boot(keys: list[str], metric: str) -> dict:
        return paired_site_bootstrap(
            {s: c_sites[s][metric] for s in keys},
            {s: r_sites[s][metric] for s in keys},
        )

    def fold_wins(metric: str, *, lower_better: bool = False) -> tuple[list[float], int]:
        deltas = [
            cand_agg["folds"][str(f)][metric] - ref_agg["folds"][str(f)][metric]
            for f in range(4)
        ]
        wins = sum((d < 0 if lower_better else d > 0) for d in deltas)
        return deltas, int(wins)

    dice_b = boot(pos_keys, "dice")
    rec_b = boot(pos_keys, "recall")
    prec_b = boot(pos_keys, "precision")
    ap512_b = boot(pos_keys, "ap_512")
    ap32_b = boot(pos_keys, "ap_32")
    bf1_b = boot(pos_keys, "boundary_f1")
    neg_fp_b = boot(neg_keys, "neg_fp")

    dice_fd, dice_fw = fold_wins("dice")
    rec_fd, rec_fw = fold_wins("recall")
    prec_fd, prec_fw = fold_wins("precision")
    ap512_fd, ap512_fw = fold_wins("ap_512")
    neg_fd, neg_fw = fold_wins("neg_fp", lower_better=True)

    small_c = cand_agg["strata"].get("small_lt5pct", {}).get("recall")
    small_r = ref_agg["strata"].get("small_lt5pct", {}).get("recall")
    small_rec_delta = (small_c - small_r) if (small_c is not None and small_r is not None) else 0.0

    ref_neg_fp = ref_agg["macro"]["neg_fp"]
    rel_neg_red = (ref_neg_fp - cand_agg["macro"]["neg_fp"]) / max(ref_neg_fp, 1e-9)

    gate_fp = (
        (
            (neg_fp_b["mean_delta"] <= -0.005 and neg_fw >= 3)
            or (ref_neg_fp < 0.02 and rel_neg_red >= 0.25 and neg_fw >= 3)
            or (prec_b["mean_delta"] >= 0.015 and prec_fw >= 3)
        )
        and rec_b["mean_delta"] >= -0.015
        and dice_b["mean_delta"] >= 0.0
    )

    gate_fn = (
        ((rec_b["mean_delta"] >= 0.020 and rec_fw >= 3) or small_rec_delta >= 0.030)
        and neg_fp_b["mean_delta"] <= 0.003
        and dice_b["mean_delta"] > 0.0
    )

    gate_net = (
        dice_b["mean_delta"] >= 0.015
        and dice_fw >= 3
        and ap512_b["mean_delta"] >= 0.015
        and ap512_fw >= 3
        and neg_fp_b["mean_delta"] <= 0.002
    )

    return {
        "dice": {**dice_b, "fold_deltas": dice_fd, "positive_folds": dice_fw},
        "recall": {**rec_b, "fold_deltas": rec_fd, "positive_folds": rec_fw},
        "small_lt5pct_recall_delta": float(small_rec_delta),
        "precision": {**prec_b, "fold_deltas": prec_fd, "positive_folds": prec_fw},
        "ap_512": {**ap512_b, "fold_deltas": ap512_fd, "positive_folds": ap512_fw},
        "ap_32": ap32_b,
        "boundary_f1": bf1_b,
        "neg_fp": {**neg_fp_b, "fold_deltas": neg_fd, "improved_folds": neg_fw, "relative_reduction": float(rel_neg_red)},
        "gates": {
            "Gate_FP": bool(gate_fp),
            "Gate_FN": bool(gate_fn),
            "Gate_Net": bool(gate_net),
        },
    }


def evaluate_binary_episode(
    query_records: list[dict],
    masks_512: dict[str, np.ndarray],
    probs_512_ref: dict[str, np.ndarray],
    scores_32_ref: dict[str, np.ndarray],
) -> dict:
    """Compute per-image and per-site metrics (including BF1@1px & BF1@2px) for binary refined masks."""
    per_image = {}
    by_site = defaultdict(list)
    for rec in query_records:
        qid = rec["id"]
        m512 = masks_512[qid]
        p512 = probs_512_ref[qid]
        s32 = scores_32_ref[qid]
        if rec["positive"] and "_gt_dil_1px" not in rec:
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
        row = {
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


def aggregate_binary_episodes(episode_results: list[dict], episodes_per_fold: int = 10) -> dict:
    """Aggregate binary-mask episode results (preserving bf1_1px alongside boundary_f1/BF1@2px)."""
    agg = aggregate_episodes(episode_results, episodes_per_fold=episodes_per_fold)
    # Add bf1_1px to site, image, strata, fold, and macro summaries
    site_ep_bf1_1: dict[str, list[float]] = defaultdict(list)
    img_ep_bf1_1: dict[str, list[float]] = defaultdict(list)
    for ep_res in episode_results:
        for sr in ep_res["site_rows"]:
            if sr.get("bf1_1px") is not None:
                site_ep_bf1_1[sr["site"]].append(float(sr["bf1_1px"]))
        for qid, im in ep_res["per_image"].items():
            if im.get("bf1_1px") is not None:
                img_ep_bf1_1[qid].append(float(im["bf1_1px"]))

    for s, vals in site_ep_bf1_1.items():
        agg["sites"][s]["bf1_1px"] = float(np.mean(vals))
    for qid, vals in img_ep_bf1_1.items():
        agg["images"][qid]["bf1_1px"] = float(np.mean(vals))
    pos_sites = [v for v in agg["sites"].values() if v["positive_images"] > 0]
    agg["macro"]["bf1_1px"] = float(np.mean([v["bf1_1px"] for v in pos_sites]))
    for f in range(4):
        f_pos = [v for v in pos_sites if v["fold"] == f]
        agg["folds"][str(f)]["bf1_1px"] = float(np.mean([v["bf1_1px"] for v in f_pos]))
    return agg

