from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .baseline import normalize_dino_tokens, run_b0_episode
from .data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from .dinov3_backbone import DinoSATExtractor
from .episodes import load_and_verify_episodes
from .io import write_json
from .metrics import aggregate_episodes, compare_methods, evaluate_episode, evaluate_single_image_512


def _oracle_raw_sweep(
    raw_by_ep_img: list[tuple[dict[str, np.ndarray], list[dict]]],
    thresholds: np.ndarray,
) -> dict:
    """Diagnostic ONLY (never used for tuning/deployment): test upper-bound raw separability."""
    n_t = len(thresholds)
    pos_dices = [[] for _ in range(n_t)]
    pos_recs = [[] for _ in range(n_t)]
    pos_precs = [[] for _ in range(n_t)]
    neg_fps = [[] for _ in range(n_t)]

    for scores32, qrecs in raw_by_ep_img:
        batch_s32 = torch.stack([torch.from_numpy(scores32[rec["id"]]) for rec in qrecs], dim=0)[:, None]
        batch_s512 = F.interpolate(batch_s32, size=(512, 512), mode="bilinear", align_corners=False)[:, 0].numpy()
        for idx, rec in enumerate(qrecs):
            s512 = batch_s512[idx]
            if rec["positive"]:
                truth = rec["mask"] > 0
                gt_sum = int(truth.sum())
                for ti, thr in enumerate(thresholds):
                    pred = s512 >= float(thr)
                    tp = int(np.logical_and(truth, pred).sum())
                    pred_sum = int(pred.sum())
                    pos_dices[ti].append((2.0 * tp / (gt_sum + pred_sum)) if (gt_sum + pred_sum) > 0 else 1.0)
                    pos_recs[ti].append(tp / gt_sum)
                    pos_precs[ti].append(tp / pred_sum if pred_sum > 0 else 0.0)
            else:
                for ti, thr in enumerate(thresholds):
                    neg_fps[ti].append(float((s512 >= float(thr)).mean()))

    best_thr, best_dice, best_rec, best_prec, best_neg_fp = None, -1.0, 0.0, 0.0, 1.0
    sweep = []
    for ti, thr in enumerate(thresholds):
        md = float(np.mean(pos_dices[ti]))
        mr = float(np.mean(pos_recs[ti]))
        mp = float(np.mean(pos_precs[ti]))
        mn = float(np.mean(neg_fps[ti]))
        sweep.append({"thr": float(thr), "img_dice": md, "img_recall": mr, "img_precision": mp, "img_neg_fp": mn})
        if md > best_dice:
            best_thr, best_dice, best_rec, best_prec, best_neg_fp = float(thr), md, mr, mp, mn
    return {
        "best_thr": best_thr,
        "best_img_dice": best_dice,
        "best_img_recall": best_rec,
        "best_img_precision": best_prec,
        "best_img_neg_fp": best_neg_fp,
        "curve": sweep,
    }


def _render_error_grid(
    title: str,
    items: list[dict],
    image_dir: Path,
    records_by_id: dict[str, dict],
    prob_samples: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    if not items:
        return
    cols = 4
    rows = len(items)
    tile_w, tile_h = 256, 256
    hdr = 30
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + hdr)), "white")
    draw = ImageDraw.Draw(canvas)

    for r_idx, item in enumerate(items):
        qid = item["id"]
        rec = records_by_id[qid]
        rgb = read_verified_rgb(image_dir, rec).transpose(1, 2, 0)
        gt = rec["mask"] > 0
        prob = prob_samples[qid]
        pred = prob >= 0.5

        y0 = r_idx * (tile_h + hdr)
        draw.text((6, y0 + 6), f"{qid} | {item['tag']}", fill="black")

        # Col 0: RGB
        canvas.paste(Image.fromarray(rgb).resize((tile_w, tile_h)), (0, y0 + hdr))
        # Col 1: RGB + GT (green)
        ov_gt = rgb.astype(np.float32).copy()
        ov_gt[gt, 1] = np.clip(ov_gt[gt, 1] * 0.5 + 127, 0, 255)
        canvas.paste(Image.fromarray(ov_gt.astype(np.uint8)).resize((tile_w, tile_h)), (tile_w, y0 + hdr))
        # Col 2: Probability heatmap (red-yellow-blue simple ramp)
        p_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        heat = np.stack([p_u8, (128 - np.abs(p_u8.astype(np.int16) - 128)).astype(np.uint8), (255 - p_u8)], axis=-1)
        canvas.paste(Image.fromarray(heat).resize((tile_w, tile_h)), (2 * tile_w, y0 + hdr))
        # Col 3: Error overlay (TP=green, FP=red, FN=yellow)
        err = rgb.astype(np.float32).copy() * 0.55
        tp_m = gt & pred
        fp_m = (~gt) & pred
        fn_m = gt & (~pred)
        err[tp_m, 1] = np.clip(err[tp_m, 1] + 120, 0, 255)
        err[fp_m, 0] = np.clip(err[fp_m, 0] + 150, 0, 255)
        err[fn_m, 0] = np.clip(err[fn_m, 0] + 150, 0, 255)
        err[fn_m, 1] = np.clip(err[fn_m, 1] + 140, 0, 255)
        canvas.paste(Image.fromarray(err.astype(np.uint8)).resize((tile_w, tile_h)), (3 * tile_w, y0 + hdr))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def run_iter0(
    *,
    split_dir: Path,
    output_dir: Path,
    source_dir: Path,
    cache_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    weights_path: Path | None = None,
    device: str = "cuda",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)

    extractor = DinoSATExtractor(
        source_dir=source_dir,
        weights_path=weights_path,
        cache_dir=cache_dir,
        device=device,
    )
    run_device = extractor.device

    records_by_id = {}
    tokens_by_id = {}
    mask_by_id = {}
    for idx, row in enumerate(dev_records, 1):
        feats = extractor.get_features(img_dir, row)
        mask = read_verified_mask(lbl_dir, row)
        rec = {**row, "mask": mask, "y32": area_target_32(mask)}
        records_by_id[row["id"]] = rec
        tokens_by_id[row["id"]] = normalize_dino_tokens(feats["last4_mean"], device=run_device)
        mask_by_id[row["id"]] = mask
        if idx % 20 == 0 or idx == len(dev_records):
            print(f"[Iter0] Cached/loaded DINOv3-SAT features {idx}/{len(dev_records)}", flush=True)
    extractor.close()

    methods = {
        "B0_PosOnly": {"alpha_bg": 0.0, "top_m": 3},
        "B0_Contrast": {"alpha_bg": 0.5, "top_m": 3},
    }

    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    results_by_method: dict[str, dict] = {m: {} for m in methods}
    sample_probs_b0c: dict[int, dict[str, np.ndarray]] = {k: {} for k in sorted(ep_by_k)}
    oracle_diag: dict[str, dict] = {m: {} for m in methods}

    for m_name, m_cfg in methods.items():
        for k, eps in sorted(ep_by_k.items()):
            ep_evals = []
            raw_ep_list = []
            calibs = []
            prob_accum: dict[str, list[np.ndarray]] = defaultdict(list)

            for ep in eps:
                q_recs = [records_by_id[qid] for qid in ep["query_ids"]]
                probs_512, scores_32, info = run_b0_episode(
                    tokens_by_id,
                    mask_by_id,
                    ep["support_ids"],
                    ep["query_ids"],
                    top_m=m_cfg["top_m"],
                    alpha_bg=m_cfg["alpha_bg"],
                )
                calibs.append(info)
                ep_eval = evaluate_episode(q_recs, probs_512, scores_32, threshold=0.5)
                ep_evals.append(ep_eval)
                raw_ep_list.append((scores_32, q_recs))
                if m_name == "B0_Contrast" and k in (1, 5):
                    for qid, p512 in probs_512.items():
                        prob_accum[qid].append(p512)

            agg = aggregate_episodes(ep_evals, episodes_per_fold=episodes_lock["episodes_per_fold"])
            agg["calibration_summary"] = {
                "bias_mean": float(np.mean([c["bias"] for c in calibs])),
                "bias_std": float(np.std([c["bias"] for c in calibs])),
                "temp_mean": float(np.mean([c["temp"] for c in calibs])),
            }
            results_by_method[m_name][str(k)] = agg

            # Diagnostic raw threshold sweep
            thrs = np.linspace(0.05, 0.55, 21) if m_cfg["alpha_bg"] > 0 else np.linspace(0.30, 0.75, 19)
            oracle_diag[m_name][str(k)] = _oracle_raw_sweep(raw_ep_list, thrs)

            if m_name == "B0_Contrast" and k in (1, 5):
                for qid, plist in prob_accum.items():
                    sample_probs_b0c[k][qid] = np.mean(plist, axis=0).astype(np.float32)

            print(
                f"[Iter0] {m_name} K={k}: Dice={agg['macro']['dice']:.4f} "
                f"Recall={agg['macro']['recall']:.4f} Prec={agg['macro']['precision']:.4f} "
                f"AP512={agg['macro']['ap_512']:.4f} NegFP={agg['macro']['neg_fp']:.4f}",
                flush=True,
            )

    comparisons = {}
    for k_str in [str(k) for k in sorted(ep_by_k)]:
        comparisons[k_str] = compare_methods(
            results_by_method["B0_Contrast"][k_str],
            results_by_method["B0_PosOnly"][k_str],
        )

    # Build FN / FP error taxonomy for B0_Contrast
    error_taxonomy = {}
    for k_str in [str(k) for k in sorted(ep_by_k)]:
        imgs = list(results_by_method["B0_Contrast"][k_str]["images"].values())
        pos_imgs = [v for v in imgs if v["positive"]]
        neg_imgs = [v for v in imgs if not v["positive"]]
        worst_fn = sorted(pos_imgs, key=lambda v: v["recall"])[:12]
        worst_pos_bg_fp = sorted(pos_imgs, key=lambda v: -v["pos_bg_fp"])[:12]
        worst_neg_fp = sorted(neg_imgs, key=lambda v: -v["neg_fp"])[:16]
        error_taxonomy[k_str] = {
            "worst_fn_positive_images": worst_fn,
            "worst_pos_bg_fp_images": worst_pos_bg_fp,
            "worst_neg_fp_images": worst_neg_fp,
        }

    # Render diagnostic visual panels for K=1 and K=5
    for k in (1, 5):
        tax = error_taxonomy[str(k)]
        fn_items = [
            {"id": x["id"], "tag": f"FN rec={x['recall']:.2f} dice={x['dice']:.2f} area={x['target_fraction']:.1%}"}
            for x in tax["worst_fn_positive_images"][:6]
        ]
        fp_neg_items = [
            {"id": x["id"], "tag": f"NegFP={x['neg_fp']:.3f} maxP={x['max_prob_mean']:.2f}"}
            for x in tax["worst_neg_fp_images"][:6]
        ]
        fp_bg_items = [
            {"id": x["id"], "tag": f"PosBgFP={x['pos_bg_fp']:.3f} prec={x['precision']:.2f}"}
            for x in tax["worst_pos_bg_fp_images"][:6]
        ]
        _render_error_grid(f"K={k} Worst FN", fn_items, img_dir, records_by_id, sample_probs_b0c[k], output_dir / "panels" / f"worst_fn_k{k}.png")
        _render_error_grid(f"K={k} Worst NegFP", fp_neg_items, img_dir, records_by_id, sample_probs_b0c[k], output_dir / "panels" / f"worst_neg_fp_k{k}.png")
        _render_error_grid(f"K={k} Worst PosBgFP", fp_bg_items, img_dir, records_by_id, sample_probs_b0c[k], output_dir / "panels" / f"worst_pos_bg_fp_k{k}.png")

    summary = {
        "iteration": "iter0_baseline",
        "backbone": "dinov3_vitl16_sat493m",
        "device": run_device,
        "methods": results_by_method,
        "comparisons_contrast_vs_posonly": comparisons,
        "oracle_raw_threshold_diagnostic": oracle_diag,
        "error_taxonomy": error_taxonomy,
    }
    write_json(output_dir / "iter0_summary.json", summary)
    return summary
