from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_baselines.finetune_models import DINOv3KShotFTRunner, EndToEndSegKShotRunner
from external_baselines.matcher_dinov3sat import MatcherDINOv3Runner
from external_baselines.persam_f import PerSAMFRunner
from external_baselines.sam_utils import LazySamEmbedMap, extract_sam_embeddings_to_dir, load_sam_vit_l
from pvfs.data import area_target_32, read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.fast_bank import build_pairwise_topk_cache, compute_dino_affinity_weights, to_tokens_norm
from pvfs.io import read_json, write_json
from pvfs.metrics import (
    BOX_3,
    DISK_1,
    DISK_2,
    average_precision,
    evaluate_binary_mask_512,
    paired_site_bootstrap,
)
from pvfs.modules import PBDCConfig, ThreeModulePipeline, get_b0_32_config, get_clean_core_config, refine_mask_pbdc
from scripts.render_dual_res_visuals import overlay_gt, overlay_tp_fp_fn, pick_zoom_box
from scripts.render_pv100_all100 import _extract_missing_test_features

_FW_RGB_BY_ID: dict[str, np.ndarray] = {}
_FW_PBDC_CFG: PBDCConfig | None = None
_FW_K_SHOTS: int = 5


def _pbdc_worker(args: tuple[str, np.ndarray, np.ndarray]) -> tuple[str, np.ndarray]:
    qid, m0, p0_f16 = args
    assert _FW_PBDC_CFG is not None
    out = refine_mask_pbdc(m0, _FW_RGB_BY_ID[qid], p0_f16.astype(np.float32), k_shots=_FW_K_SHOTS, cfg=_FW_PBDC_CFG)
    return qid, out["mask"]


def evaluate_pv100_ext_episode(
    query_records: list[dict[str, Any]],
    masks_512: dict[str, np.ndarray],
    probs_512: dict[str, np.ndarray],
    scores_32: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate one episode on PV-100 query set (39 pos sites + 20 neg sites)."""
    per_image = {}
    by_site: dict[str, list[tuple[dict, dict, np.ndarray, np.ndarray]]] = defaultdict(list)
    for rec in query_records:
        qid = rec["id"]
        if rec["positive"] and "_gt_b" not in rec:
            truth = rec["mask"] > 0
            rec["_gt_b"] = truth ^ binary_erosion(truth, structure=BOX_3, border_value=1)
            rec["_gt_dil_1px"] = binary_dilation(rec["_gt_b"], structure=DISK_1)
            rec["_gt_dil_2px"] = binary_dilation(rec["_gt_b"], structure=DISK_2)
        im_m = evaluate_binary_mask_512(
            rec["mask"],
            masks_512[qid],
            probs_512[qid],
            t_b=rec.get("_gt_b"),
            t_dil_1px=rec.get("_gt_dil_1px"),
            t_dil_2px=rec.get("_gt_dil_2px"),
        )
        im_m["id"] = qid
        im_m["site"] = rec["site"]
        im_m["target_fraction"] = float(rec["target_fraction"])
        per_image[qid] = im_m
        by_site[rec["site"]].append((rec, im_m, probs_512[qid], scores_32[qid]))

    site_rows = []
    for site, items in sorted(by_site.items()):
        pos_items = [x for x in items if x[0]["positive"]]
        neg_items = [x for x in items if not x[0]["positive"]]
        row: dict[str, Any] = {
            "site": site,
            "positive_images": len(pos_items),
            "negative_images": len(neg_items),
        }
        if pos_items:
            for key in ("dice", "iou", "recall", "precision", "bf1_1px", "boundary_f1", "detected"):
                row[key] = float(np.mean([x[1][key] for x in pos_items]))
            row["ap_512"] = average_precision(
                np.concatenate([(x[0]["mask"] > 0).ravel() for x in pos_items]),
                np.concatenate([x[2].ravel() for x in pos_items]),
            )
        if neg_items:
            row["neg_fp"] = float(np.mean([x[1]["neg_fp"] for x in neg_items]))
            row["neg_clean"] = float(np.mean([x[1]["neg_clean"] for x in neg_items]))
        site_rows.append(row)
    return {"per_image": per_image, "site_rows": site_rows}


def aggregate_pv100_ext_episodes(episode_results: list[dict[str, Any]], expected_episodes: int) -> dict[str, Any]:
    site_ep: dict[str, list[dict]] = defaultdict(list)
    img_ep: dict[str, list[dict]] = defaultdict(list)
    for ep in episode_results:
        for sr in ep["site_rows"]:
            site_ep[sr["site"]].append(sr)
        for qid, im in ep["per_image"].items():
            img_ep[qid].append(im)

    pos_keys = ("dice", "iou", "recall", "precision", "bf1_1px", "boundary_f1", "detected", "ap_512")
    neg_keys = ("neg_fp", "neg_clean")
    sites: dict[str, dict[str, Any]] = {}
    for site, rows in sorted(site_ep.items()):
        assert len(rows) == expected_episodes
        entry: dict[str, Any] = {
            "site": site,
            "positive_images": rows[0]["positive_images"],
            "negative_images": rows[0]["negative_images"],
        }
        for key in pos_keys:
            vals = [r[key] for r in rows if r.get(key) is not None]
            entry[key] = float(np.mean(vals)) if vals else None
        for key in neg_keys:
            vals = [r[key] for r in rows if r.get(key) is not None]
            entry[key] = float(np.mean(vals)) if vals else None
        sites[site] = entry

    pos_sites = [v for v in sites.values() if v["positive_images"] > 0]
    neg_sites = [v for v in sites.values() if v["negative_images"] > 0]
    macro = {key: float(np.mean([s[key] for s in pos_sites])) for key in pos_keys}
    for key in neg_keys:
        macro[key] = float(np.mean([s[key] for s in neg_sites]))

    # Small target (<5%) image recall
    sm_recs = []
    for qid, rows in img_ep.items():
        if rows[0]["positive"] and rows[0]["target_fraction"] < 0.05:
            sm_recs.append(float(np.mean([r["recall"] for r in rows])))
    strata = {"small_lt5pct": {"count": len(sm_recs), "recall": float(np.mean(sm_recs)) if sm_recs else 0.0}}
    return {"macro": macro, "strata": strata, "sites": sites}


def compare_pv100_ext(cand: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pos_sites = sorted(s for s, v in cand["sites"].items() if v["positive_images"] > 0)
    neg_sites = sorted(s for s, v in cand["sites"].items() if v["negative_images"] > 0)
    for key in ("dice", "iou", "bf1_1px", "boundary_f1", "precision", "recall"):
        c_map = {s: float(cand["sites"][s][key]) for s in pos_sites}
        r_map = {s: float(ref["sites"][s][key]) for s in pos_sites}
        b = paired_site_bootstrap(c_map, r_map)
        deltas = np.array([c_map[s] - r_map[s] for s in pos_sites], dtype=np.float64)
        out[key] = {
            "mean_delta": float(b["mean_delta"]),
            "ci95": list(b["ci95"]),
            "improved_sites": int((deltas > 1e-9).sum()),
            "total_sites": len(pos_sites),
        }
    c_neg = {s: float(cand["sites"][s]["neg_fp"]) for s in neg_sites}
    r_neg = {s: float(ref["sites"][s]["neg_fp"]) for s in neg_sites}
    bn = paired_site_bootstrap(c_neg, r_neg)
    neg_deltas = np.array([c_neg[s] - r_neg[s] for s in neg_sites], dtype=np.float64)
    out["neg_fp"] = {
        "mean_delta": float(bn["mean_delta"]),
        "ci95": list(bn["ci95"]),
        "improved_sites": int((neg_deltas < -1e-9).sum()),
        "total_sites": len(neg_sites),
    }
    out["small_lt5pct_recall_delta"] = float(
        cand["strata"]["small_lt5pct"]["recall"] - ref["strata"]["small_lt5pct"]["recall"]
    )
    return out


def _save_ext_pos_zoom_panel(out_path: Path, sc: dict[str, Any]) -> None:
    rgb, gt = sc["rgb"], sc["gt"]
    y0, y1, x0, x1 = pick_zoom_box(gt, sc["masks"]["A0"], sc["masks"]["Ours"], sc["masks"]["Matcher"], crop_sz=128)
    cols = [
        ("RGB Input", rgb),
        ("Ground Truth", overlay_gt(rgb, gt)),
        (
            f"Internal Baseline (A0)\nD={sc['metrics']['A0']['dice']:.3f} B2={sc['metrics']['A0']['boundary_f1']:.3f}",
            overlay_tp_fp_fn(rgb, gt, sc["masks"]["A0"]),
        ),
        (
            f"DINOv3-SAT + SAM (Matcher)\nD={sc['metrics']['Matcher']['dice']:.3f} B2={sc['metrics']['Matcher']['boundary_f1']:.3f}",
            overlay_tp_fp_fn(rgb, gt, sc["masks"]["Matcher"]),
        ),
        (
            f"Ours (v3.1-PBDC-Pure)\nD={sc['metrics']['Ours']['dice']:.3f} B1={sc['metrics']['Ours']['bf1_1px']:.3f} B2={sc['metrics']['Ours']['boundary_f1']:.3f}",
            overlay_tp_fp_fn(rgb, gt, sc["masks"]["Ours"]),
        ),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(18.5, 7.6), dpi=200)
    for c_i, (lbl, im) in enumerate(cols):
        axes[0, c_i].imshow(im)
        axes[0, c_i].set_title(lbl, fontsize=9.5, fontweight=("bold" if c_i == 4 else "normal"), pad=5)
        axes[0, c_i].axis("off")
        axes[0, c_i].add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.6, edgecolor="cyan", facecolor="none", linestyle="--"))
        axes[1, c_i].imshow(im[y0:y1, x0:x1])
        axes[1, c_i].axis("off")
        if c_i == 0:
            axes[1, c_i].set_title("4x Boundary Zoom-in", fontsize=9, color="#006688")
    legend_handles = [
        mpatches.Patch(color="#2ed573", label="TP (Correct PV)"),
        mpatches.Patch(color="#ff4757", label="FP (False Positive)"),
        mpatches.Patch(color="#ffd32a", label="FN (Missed PV)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"PV-100scence (2m, K=5) | {sc['short_id']} | PV={sc['target_pct']:.1f}%", fontsize=12, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0.0, 0.04, 1.0, 0.955])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_ext_grid(out_path: Path, header: str, scenes: list[dict[str, Any]], *, is_neg_only: bool = False) -> None:
    n_rows = len(scenes)
    fig, axes = plt.subplots(n_rows, 5, figsize=(17.0, 3.1 * n_rows), dpi=180)
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = ["RGB Input", "Ground Truth", "Internal Baseline (A0)", "DINOv3-SAT + SAM (Matcher)", "Ours (v3.1-PBDC-Pure)"]
    for r_i, sc in enumerate(scenes):
        rgb, gt = sc["rgb"], sc["gt"]
        imgs = [
            rgb,
            overlay_gt(rgb, gt),
            overlay_tp_fp_fn(rgb, gt, sc["masks"]["A0"]),
            overlay_tp_fp_fn(rgb, gt, sc["masks"]["Matcher"]),
            overlay_tp_fp_fn(rgb, gt, sc["masks"]["Ours"]),
        ]
        if sc["positive"]:
            lbls = [
                sc["short_id"],
                f"PV={sc['target_pct']:.1f}%",
                f"D={sc['metrics']['A0']['dice']:.3f} B2={sc['metrics']['A0']['boundary_f1']:.3f}",
                f"D={sc['metrics']['Matcher']['dice']:.3f} B2={sc['metrics']['Matcher']['boundary_f1']:.3f}",
                f"D={sc['metrics']['Ours']['dice']:.3f} B2={sc['metrics']['Ours']['boundary_f1']:.3f}",
            ]
        else:
            lbls = [
                f"{sc['short_id']} [NEG]",
                "Pure Negative",
                f"FP={sc['fp']['A0']:.2f}%",
                f"FP={sc['fp']['Matcher']:.2f}%",
                f"FP={sc['fp']['Ours']:.2f}%",
            ]
        for c_i in range(5):
            ax = axes[r_i, c_i]
            ax.imshow(imgs[c_i])
            ax.axis("off")
            if r_i == 0:
                ax.set_title(col_titles[c_i], fontsize=10.5, fontweight=("bold" if c_i == 4 else "normal"), pad=5)
            ax.text(
                6,
                18,
                lbls[c_i],
                color="white",
                fontsize=8.2,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.65, pad=2, edgecolor="none"),
            )
    legend_handles = (
        [mpatches.Patch(color="#ff4757", label="FP (False Positive on Pure Negative Scene)")]
        if is_neg_only
        else [
            mpatches.Patch(color="#2ed573", label="TP (Correct PV)"),
            mpatches.Patch(color="#ff4757", label="FP (False Positive)"),
            mpatches.Patch(color="#ffd32a", label="FN (Missed PV)"),
        ]
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.002))
    fig.suptitle(header, fontsize=13, fontweight="bold", y=0.993)
    plt.tight_layout(rect=[0.0, 0.02, 1.0, 0.978])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def format_pv100_ext_report(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    lines = [
        "=== PV-100scence (2m) EXTERNAL MODEL COMPARISON (7 Complete Models, 4-Fold Site-Disjoint CV) ===",
        f"Split: All {meta['total_images']} images ({meta['pos_sites']} pos sites / {meta['pos_images']} imgs + "
        f"{meta['neg_sites']} neg sites / {meta['neg_images']} imgs) | 4 folds x {meta['episodes_per_fold']} eps/fold",
        f"Total Elapsed: {summary['elapsed_sec']/60:.1f} min",
    ]
    order = (
        "A0_B0_32_Ref",
        "PerSAM_F_SAM_ViTL",
        "UNet_ResNet50_FT",
        "SegFormer_MiTB2_FT",
        "DINOv3SAT_ConvHead_FT",
        "Matcher_DINOv3SAT_SAM_ViTL",
        "Ours_v31_PBDC_Pure",
    )
    for k in ("1", "3", "5"):
        lines.append(f"\n--- K={k} ({meta['pos_sites']} pos + {meta['neg_sites']} neg independent sites, 4-fold CV) ---")
        for mname in order:
            res = summary["methods"][mname][k]
            m = res["macro"]
            sm_r = res["strata"]["small_lt5pct"]["recall"]
            lines.append(
                f"  {mname:28s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
                f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} Sm<5R={sm_r:.4f} NegFP={m['neg_fp']:.4f}"
            )

    lines.append("\n--- Paired Site Bootstrap Deltas (Ours vs External / Baseline, 49 Pos + 20 Neg Sites) ---")
    for mname in order[:-1]:
        cname = f"Ours_vs_{mname}"
        lines.append(f"[{cname}]")
        for k in ("1", "3", "5"):
            c = summary["comparisons"][cname][k]
            lines.append(
                f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} "
                f"({c['dice']['improved_sites']}/{c['dice']['total_sites']}) | "
                f"dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x,4) for x in c['boundary_f1']['ci95']]} | "
                f"dSm<5R={c['small_lt5pct_recall_delta']:+.4f} | dNegFP={c['neg_fp']['mean_delta']:+.4f}"
            )
    return "\n".join(lines) + "\n"


def run_pv100_ext_all(
    *,
    ext_split_json: Path,
    dev_cache_pt: Path,
    source_dir: Path,
    sam_cache_dir: Path,
    sam_embed_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    num_workers: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = num_workers if num_workers is not None else min(4, max(1, os.cpu_count() or 1))
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    sp = read_json(ext_split_json)
    all_recs = sp["records"]
    all_ids = [r["id"] for r in all_recs]
    all_pos_ids = [r["id"] for r in all_recs if r["positive"]]
    episodes_per_fold = int(sp["episodes_per_fold"])

    img_dir, lbl_dir = resolve_data_dirs(None, None)
    rgb_chw_by_id: dict[str, np.ndarray] = {}
    rgb_hwc_by_id: dict[str, np.ndarray] = {}
    mask_by_id: dict[str, np.ndarray] = {}
    rec_by_id: dict[str, dict[str, Any]] = {}
    for row in all_recs:
        rid = row["id"]
        rgb_chw = read_verified_rgb(img_dir, row, allow_sealed_test=True)
        rgb_chw_by_id[rid] = rgb_chw
        rgb_hwc_by_id[rid] = np.transpose(rgb_chw, (1, 2, 0)).copy()
        mask = read_verified_mask(lbl_dir, row, allow_sealed_test=True)
        mask_by_id[rid] = mask
        rec_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}

    # 1. Load cached dev DINOv3-SAT features + extract missing (~21 test) features
    bundle = torch.load(dev_cache_pt, map_location="cpu", weights_only=False)
    feats = dict(bundle["features"])
    missing_recs = [r for r in all_recs if r["id"] not in feats]
    print(f"[PV100 Ext 4-Fold] Loaded {len(feats)} cached DINOv3-SAT features; extracting {len(missing_recs)} remaining...", flush=True)
    if missing_recs:
        feats.update(_extract_missing_test_features(missing_recs, rgb_chw_by_id, source_dir=source_dir, device=run_device))

    # 2. Load SAM ViT-L & cache embeddings for all 100 images (~45s)
    sam = load_sam_vit_l(sam_cache_dir, device=run_device)
    extract_sam_embeddings_to_dir(sam, rgb_hwc_by_id, all_ids, sam_embed_dir, device=run_device)
    sam_embeds = LazySamEmbedMap(sam_embed_dir, device=run_device, preload_ids=all_ids)

    # 3. Build DINOv3-SAT token maps, pairwise caches (all 80 pos as potential support), and affinity weights
    tokens_by_key = {
        k: {rid: to_tokens_norm(feats[rid][k], run_device) for rid in all_ids}
        for k in ("deep_32", "deep_64", "mid_64")
    }
    caches = {
        k: build_pairwise_topk_cache(tokens_by_key[k], mask_by_id, all_ids, all_pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        for k, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64))
    }
    core_cfg = get_clean_core_config()
    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=core_cfg.aff_tau) for rid in all_ids}

    pipe_a0 = ThreeModulePipeline(get_b0_32_config(), refiner_cfg=None)
    pipe_a3 = ThreeModulePipeline(core_cfg, refiner_cfg=None)
    pbdc_cfg = PBDCConfig(enabled=True)
    persam_runner = PerSAMFRunner(sam, ft_steps=100, lr=0.05)
    matcher_runner = MatcherDINOv3Runner(sam, num_clusters=8, merge_margin=0.05)
    dino_ft = DINOv3KShotFTRunner(steps=120, lr=3e-3, wd=1e-2)
    unet_ft = EndToEndSegKShotRunner("unet_resnet50", steps=80, lr=3e-4, wd=1e-2, device=run_device)
    segf_ft = EndToEndSegKShotRunner("segformer_mit_b2", steps=80, lr=3e-4, wd=1e-2, device=run_device)

    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    for ep in sp["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    model_names = (
        "A0_B0_32_Ref",
        "PerSAM_F_SAM_ViTL",
        "Matcher_DINOv3SAT_SAM_ViTL",
        "UNet_ResNet50_FT",
        "SegFormer_MiTB2_FT",
        "DINOv3SAT_ConvHead_FT",
        "Ours_v31_PBDC_Pure",
    )
    methods_res: dict[str, dict[str, Any]] = defaultdict(dict)
    vis_k5_ep0: dict[str, dict[str, np.ndarray]] = {"A0": {}, "Matcher": {}, "Ours": {}}
    vis_k5_prob_ours: dict[str, np.ndarray] = {}

    use_fork = (workers > 1) and (sys.platform != "win32") and ("fork" in multiprocessing.get_all_start_methods())
    global _FW_RGB_BY_ID, _FW_PBDC_CFG, _FW_K_SHOTS
    _FW_RGB_BY_ID = rgb_hwc_by_id
    _FW_PBDC_CFG = pbdc_cfg

    for k in sorted(ep_by_k.keys()):
        t_k = time.perf_counter()
        eps = ep_by_k[k]
        ep_evals: dict[str, list[dict]] = {m: [] for m in model_names}

        for ep_idx, ep in enumerate(eps, 1):
            sup_ids = ep["support_ids"]
            q_ids = ep["query_ids"]
            q_recs = [rec_by_id[qid] for qid in q_ids]
            seed_off = k * 1000 + int(ep["fold"]) * 100 + int(ep["episode"])

            # 1. A0 (Internal Baseline)
            out_a0 = pipe_a0.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
            p_a0, s32_a0 = out_a0.to_legacy_outputs()
            m_a0 = out_a0.coarse_masks_512()
            ep_evals["A0_B0_32_Ref"].append(evaluate_pv100_ext_episode(q_recs, m_a0, p_a0, s32_a0))

            # 2. PerSAM-F
            ps_st = persam_runner.fit_episode(sam_embeds, mask_by_id, sup_ids)
            m_ps, p_ps, s32_ps = {}, {}, {}
            for qid in q_ids:
                m_ps[qid], p_ps[qid], s32_ps[qid] = persam_runner.infer_query(sam_embeds[qid], ps_st)
            ep_evals["PerSAM_F_SAM_ViTL"].append(evaluate_pv100_ext_episode(q_recs, m_ps, p_ps, s32_ps))

            # 3. Matcher (DINOv3-SAT + SAM ViT-L)
            mt_st = matcher_runner.fit_episode(tokens_by_key["deep_64"], mask_by_id, sup_ids)
            m_mt, p_mt, s32_mt = {}, {}, {}
            for qid in q_ids:
                m_mt[qid], p_mt[qid], s32_mt[qid] = matcher_runner.infer_query(tokens_by_key["deep_64"][qid], sam_embeds[qid], mt_st)
            ep_evals["Matcher_DINOv3SAT_SAM_ViTL"].append(evaluate_pv100_ext_episode(q_recs, m_mt, p_mt, s32_mt))

            # 4. DINOv3-SAT + ConvHead FT
            head = dino_ft.fit_episode(tokens_by_key, mask_by_id, sup_ids, ep_seed_offset=seed_off)
            m_df, p_df, s32_df = dino_ft.infer_queries(head, tokens_by_key, q_ids)
            ep_evals["DINOv3SAT_ConvHead_FT"].append(evaluate_pv100_ext_episode(q_recs, m_df, p_df, s32_df))
            del head

            # 5. U-Net (ResNet50) FT
            u_mod = unet_ft.fit_episode(rgb_hwc_by_id, mask_by_id, sup_ids, ep_seed_offset=seed_off)
            m_un, p_un, s32_un = unet_ft.infer_queries(u_mod, rgb_hwc_by_id, q_ids)
            ep_evals["UNet_ResNet50_FT"].append(evaluate_pv100_ext_episode(q_recs, m_un, p_un, s32_un))
            del u_mod

            # 6. SegFormer (MiT-B2) FT
            s_mod = segf_ft.fit_episode(rgb_hwc_by_id, mask_by_id, sup_ids, ep_seed_offset=seed_off)
            m_sf, p_sf, s32_sf = segf_ft.infer_queries(s_mod, rgb_hwc_by_id, q_ids)
            ep_evals["SegFormer_MiTB2_FT"].append(evaluate_pv100_ext_episode(q_recs, m_sf, p_sf, s32_sf))
            del s_mod

            # 7. Ours (v3.1-PBDC-Pure)
            out_a3 = pipe_a3.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
            p_a3, s32_a3 = out_a3.to_legacy_outputs()
            m_a3 = out_a3.coarse_masks_512()
            _FW_K_SHOTS = k
            tasks = [(qid, m_a3[qid], p_a3[qid].astype(np.float16)) for qid in q_ids]
            m_ours: dict[str, np.ndarray] = {}
            if use_fork:
                ctx = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                    for qid, m_ref in ex.map(_pbdc_worker, tasks, chunksize=4):
                        m_ours[qid] = m_ref
            else:
                for task in tasks:
                    qid, m_ref = _pbdc_worker(task)
                    m_ours[qid] = m_ref
            ep_evals["Ours_v31_PBDC_Pure"].append(evaluate_pv100_ext_episode(q_recs, m_ours, p_a3, s32_a3))

            if k == 5 and int(ep["episode"]) == 0:
                for qid in q_ids:
                    vis_k5_ep0["A0"][qid] = m_a0[qid]
                    vis_k5_ep0["Matcher"][qid] = m_mt[qid]
                    vis_k5_ep0["Ours"][qid] = m_ours[qid]
                    vis_k5_prob_ours[qid] = p_a3[qid]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if ep_idx % 3 == 0 or ep_idx == len(eps):
                print(f"[PV100 Ext K={k}] Ep {ep_idx}/{len(eps)} done ({(time.perf_counter()-t_k)/60:.1f}m)", flush=True)

        for mname in model_names:
            methods_res[mname][str(k)] = aggregate_pv100_ext_episodes(ep_evals[mname], expected_episodes=episodes_per_fold)
            m = methods_res[mname][str(k)]["macro"]
            print(
                f"  -> [K={k}] {mname:28s}: Dice={m['dice']:.4f} BF1@2={m['boundary_f1']:.4f} NegFP={m['neg_fp']:.4f}",
                flush=True,
            )

    comparisons = {
        f"Ours_vs_{mname}": {
            str(k): compare_pv100_ext(methods_res["Ours_v31_PBDC_Pure"][str(k)], methods_res[mname][str(k)])
            for k in (1, 3, 5)
        }
        for mname in model_names
        if mname != "Ours_v31_PBDC_Pure"
    }

    summary = {
        "schema": "pv100-ext-all7-4fold-v2",
        "elapsed_sec": float(time.perf_counter() - t0),
        "meta": {**sp["summary"], "episodes_per_fold": episodes_per_fold},
        "methods": dict(methods_res),
        "comparisons": comparisons,
    }
    write_json(output_dir / "pv100_ext_all7_summary.json", summary)
    report = format_pv100_ext_report(summary)
    (output_dir / "pv100_ext_all7_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)

    # --- Render Visualizations across ALL 100 query images (K=5 Ep0 per fold: A0 vs DINOv3-SAT+SAM vs Ours) ---
    print("[PV100 Ext] Rendering qualitative comparison panels across all 100 images (A0 vs DINOv3-SAT+SAM vs Ours)...", flush=True)
    vis_dir = output_dir / "visuals"
    zoom_dir = vis_dir / "pos_zoom"
    sheets_dir = vis_dir / "contact_sheets"
    zoom_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    all_vis_items = []
    for rid in all_ids:
        rec = rec_by_id[rid]
        gt = rec["mask"]
        ms = {"A0": vis_k5_ep0["A0"][rid], "Matcher": vis_k5_ep0["Matcher"][rid], "Ours": vis_k5_ep0["Ours"][rid]}
        item: dict[str, Any] = {
            "id": rid,
            "short_id": Path(rid).stem,
            "positive": bool(rec["positive"]),
            "target_pct": float((gt > 0).mean()) * 100.0,
            "rgb": rgb_hwc_by_id[rid],
            "gt": gt,
            "masks": ms,
        }
        if rec["positive"]:
            item["metrics"] = {k_m: evaluate_binary_mask_512(gt, m, vis_k5_prob_ours[rid]) for k_m, m in ms.items()}
            item["gain_vs_matcher"] = item["metrics"]["Ours"]["dice"] - item["metrics"]["Matcher"]["dice"]
        else:
            item["fp"] = {k_m: float((m > 0).mean()) * 100.0 for k_m, m in ms.items()}
        all_vis_items.append(item)

    # Sort query items naturally by PV number for contact sheets
    def _pv_num(it: dict[str, Any]) -> int:
        try:
            return int(it["short_id"].split("-")[-1])
        except ValueError:
            return 9999

    all_vis_items.sort(key=_pv_num)
    pos_items = [x for x in all_vis_items if x["positive"]]
    neg_items = [x for x in all_vis_items if not x["positive"]]

    # 1. 6 diverse positive scenes + 4x zoom-in panels
    sm = sorted([x for x in pos_items if x["target_pct"] < 5.0], key=lambda x: -x["gain_vs_matcher"])[:2]
    md = sorted([x for x in pos_items if 5.0 <= x["target_pct"] < 15.0], key=lambda x: -x["gain_vs_matcher"])[:2]
    lg = sorted([x for x in pos_items if x["target_pct"] >= 15.0], key=lambda x: -x["gain_vs_matcher"])[:2]
    chosen_pos = sm + md + lg
    _save_ext_grid(
        vis_dir / "summary_2m_ext_pos6_grid.png",
        "PV-100scence (2m, K=5) Positive Query Comparison: A0 vs DINOv3-SAT+SAM vs Ours (v3.1-PBDC)",
        chosen_pos,
    )
    for idx, sc in enumerate(chosen_pos, 1):
        _save_ext_pos_zoom_panel(zoom_dir / f"pos_{idx:02d}_{sc['short_id']}.png", sc)

    # 2. 6 diverse pure negative scenes (4 where DINOv3-SAT+SAM has FP while Ours is clean + 1 clean + 1 hardest)
    neg_by_margin = sorted(neg_items, key=lambda x: -(x["fp"]["Matcher"] - x["fp"]["Ours"]))
    chosen_neg = neg_by_margin[:5] + [sorted(neg_items, key=lambda x: -x["fp"]["Ours"])[0]]
    # Deduplicate while preserving order
    seen_n: set[str] = set()
    chosen_neg_u = []
    for x in chosen_neg:
        if x["id"] not in seen_n:
            seen_n.add(x["id"])
            chosen_neg_u.append(x)
    _save_ext_grid(
        vis_dir / "summary_2m_ext_neg6_grid.png",
        "PV-100scence (2m, K=5) Pure Negative Query Comparison: A0 vs DINOv3-SAT+SAM vs Ours (v3.1-PBDC)",
        chosen_neg_u[:6],
        is_neg_only=True,
    )

    # 3. Full query-set contact sheets (10 images/sheet)
    for s_idx in range(0, len(all_vis_items), 10):
        chunk = all_vis_items[s_idx : s_idx + 10]
        snum = s_idx // 10 + 1
        _save_ext_grid(
            sheets_dir / f"query_sheet_{snum:02d}.png",
            f"PV-100scence (2m, K=5) Query Contact Sheet {snum:02d} ({chunk[0]['short_id']} .. {chunk[-1]['short_id']})",
            chunk,
        )

    print(f"[PV100 Ext] All 7 models + visuals complete in {(time.perf_counter()-t0)/60:.1f} min!", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PV-100scence 7-model external comparison + visuals.")
    parser.add_argument("--ext-split-json", type=Path, default=REPO_ROOT / "external_baselines" / "configs" / "pv100_ext_split_lock.json")
    parser.add_argument("--dev-cache-pt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/dinov3-source"))
    parser.add_argument("--sam-cache-dir", type=Path, default=Path("/tmp/sam_weights"))
    parser.add_argument("--sam-embed-dir", type=Path, default=Path("/tmp/pv100_ext_sam_embeds"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "external_baselines" / "results" / "pv100_all7")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    run_pv100_ext_all(
        ext_split_json=args.ext_split_json,
        dev_cache_pt=args.dev_cache_pt,
        source_dir=args.source_dir,
        sam_cache_dir=args.sam_cache_dir,
        sam_embed_dir=args.sam_embed_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
