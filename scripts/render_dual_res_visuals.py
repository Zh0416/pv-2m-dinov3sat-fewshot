from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import binary_erosion
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.data import load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache, compute_dino_affinity_weights, to_tokens_norm
from pvfs.io import read_json
from pvfs.metrics import BOX_3, evaluate_binary_mask_512
from pvfs.modules import (
    DeformableBoundaryConfig,
    PBDCConfig,
    ThreeModulePipeline,
    calibrate_deform_p_floor_loo,
    get_b0_32_config,
    get_clean_core_config,
    refine_mask_deformable_grad,
    refine_mask_pbdc,
)
from pvfs.pv663_eval import (
    LazyDiskTokenMap,
    build_pairwise_topk_cache_streaming,
    extract_pv663_features_to_dir,
    resolve_pv663_dirs,
)


def overlay_tp_fp_fn(rgb_hwc: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Render TP (green), FP (red), FN (yellow) semi-transparent overlay + crisp prediction contour."""
    img = rgb_hwc.astype(np.float32).copy()
    if img.max() <= 1.5:
        img = img * 255.0
    gt_b = gt > 0
    pr_b = pred > 0
    tp = gt_b & pr_b
    fp = (~gt_b) & pr_b
    fn = gt_b & (~pr_b)

    out = img.copy()
    # Green = TP, Red = FP, Yellow = FN
    out[tp] = (1.0 - alpha) * out[tp] + alpha * np.array([46.0, 213.0, 115.0], dtype=np.float32)
    out[fp] = (1.0 - alpha) * out[fp] + alpha * np.array([255.0, 71.0, 87.0], dtype=np.float32)
    out[fn] = (1.0 - alpha) * out[fn] + alpha * np.array([255.0, 211.0, 42.0], dtype=np.float32)

    # Crisp 1px white boundary on predicted mask for sharp visual inspection
    pr_edge = pr_b ^ binary_erosion(pr_b, structure=BOX_3, border_value=0)
    out[pr_edge] = 0.35 * out[pr_edge] + 0.65 * np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def overlay_gt(rgb_hwc: np.ndarray, gt: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    img = rgb_hwc.astype(np.float32).copy()
    if img.max() <= 1.5:
        img = img * 255.0
    gt_b = gt > 0
    out = img.copy()
    out[gt_b] = (1.0 - alpha) * out[gt_b] + alpha * np.array([46.0, 213.0, 115.0], dtype=np.float32)
    edge = gt_b ^ binary_erosion(gt_b, structure=BOX_3, border_value=0)
    out[edge] = np.array([0.0, 255.0, 140.0], dtype=np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def pick_zoom_box(gt: np.ndarray, m_a3: np.ndarray, m_pbdc: np.ndarray, m_old: np.ndarray, crop_sz: int = 128) -> tuple[int, int, int, int]:
    """Center a 128x128 zoom crop around the densest PBDC boundary correction (or GT boundary)."""
    h, w = gt.shape
    diff = (m_pbdc != m_a3) | (m_pbdc != m_old)
    gt_b = gt > 0
    gt_edge = gt_b ^ binary_erosion(gt_b, structure=BOX_3, border_value=0)
    focus = diff & (gt_edge | (m_a3 != gt_b) | (m_old != gt_b))
    ys, xs = np.nonzero(focus)
    if len(ys) < 10:
        ys, xs = np.nonzero(gt_edge)
    if len(ys) == 0:
        cy, cx = h // 2, w // 2
    else:
        # Pick the 128x128 window with maximum diff+edge density on a coarse grid
        best_score = -1
        best_cy, best_cx = int(np.median(ys)), int(np.median(xs))
        half = crop_sz // 2
        for gy in range(half, h - half + 1, 32):
            for gx in range(half, w - half + 1, 32):
                sc = int(focus[gy - half : gy + half, gx - half : gx + half].sum())
                if sc > best_score:
                    best_score = sc
                    best_cy, best_cx = gy, gx
        cy, cx = best_cy, best_cx
    y0 = max(0, min(h - crop_sz, cy - crop_sz // 2))
    x0 = max(0, min(w - crop_sz, cx - crop_sz // 2))
    return y0, y0 + crop_sz, x0, x0 + crop_sz


def save_scene_panel(
    out_path: Path,
    *,
    title: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    masks: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
) -> None:
    """Save 2-row x 6-col publication panel (Row 1: Full 512x512, Row 2: 4x Boundary Zoom-in)."""
    y0, y1, x0, x1 = pick_zoom_box(gt, masks["A3"], masks["PBDC"], masks["Old"], crop_sz=128)
    rgb_u8 = (rgb * 255.0).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.5 else rgb.astype(np.uint8)

    cols = [
        ("RGB Input", rgb_u8),
        ("Ground Truth", overlay_gt(rgb_u8, gt)),
        (
            f"A0 (B0_32)\nDice={metrics['A0']['dice']:.3f} BF2={metrics['A0']['boundary_f1']:.3f}",
            overlay_tp_fp_fn(rgb_u8, gt, masks["A0"]),
        ),
        (
            f"A3 (Before PBDC)\nDice={metrics['A3']['dice']:.3f} BF1={metrics['A3']['bf1_1px']:.3f} BF2={metrics['A3']['boundary_f1']:.3f}",
            overlay_tp_fp_fn(rgb_u8, gt, masks["A3"]),
        ),
        (
            f"A4_Old (Legacy Grad)\nDice={metrics['Old']['dice']:.3f} BF1={metrics['Old']['bf1_1px']:.3f} BF2={metrics['Old']['boundary_f1']:.3f}",
            overlay_tp_fp_fn(rgb_u8, gt, masks["Old"]),
        ),
        (
            f"Ours (v3.1-PBDC-Pure)\nDice={metrics['PBDC']['dice']:.3f} BF1={metrics['PBDC']['bf1_1px']:.3f} BF2={metrics['PBDC']['boundary_f1']:.3f}",
            overlay_tp_fp_fn(rgb_u8, gt, masks["PBDC"]),
        ),
    ]

    fig, axes = plt.subplots(2, 6, figsize=(21, 7.6), dpi=220)
    for col_i, (lbl, im) in enumerate(cols):
        ax0 = axes[0, col_i]
        ax0.imshow(im)
        ax0.set_title(lbl, fontsize=9.5, fontweight=("bold" if col_i == 5 else "normal"), pad=5)
        ax0.axis("off")
        rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.6, edgecolor="cyan", facecolor="none", linestyle="--")
        ax0.add_patch(rect)

        ax1 = axes[1, col_i]
        ax1.imshow(im[y0:y1, x0:x1])
        ax1.axis("off")
        if col_i == 0:
            ax1.set_title("4x Boundary Zoom-in", fontsize=9, color="#006688")

    legend_handles = [
        mpatches.Patch(color="#2ed573", label="TP (Correct PV)"),
        mpatches.Patch(color="#ff4757", label="FP (False Positive)"),
        mpatches.Patch(color="#ffd32a", label="FN (Missed PV)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0.0, 0.04, 1.0, 0.955])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_summary_grid(out_path: Path, *, header: str, scenes: list[dict[str, Any]]) -> None:
    """Save compact multi-scene montage grid (N scenes x 6 columns) for direct paper inclusion."""
    n_rows = len(scenes)
    fig, axes = plt.subplots(n_rows, 6, figsize=(20, 3.35 * n_rows), dpi=200)
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ["RGB Input", "Ground Truth", "A0 (B0_32)", "A3 (Before PBDC)", "A4_Old (Legacy)", "Ours (v3.1-PBDC)"]
    for r_i, sc in enumerate(scenes):
        rgb_u8 = sc["rgb"]
        gt = sc["gt"]
        ms = sc["masks"]
        mt = sc["metrics"]
        imgs = [
            rgb_u8,
            overlay_gt(rgb_u8, gt),
            overlay_tp_fp_fn(rgb_u8, gt, ms["A0"]),
            overlay_tp_fp_fn(rgb_u8, gt, ms["A3"]),
            overlay_tp_fp_fn(rgb_u8, gt, ms["Old"]),
            overlay_tp_fp_fn(rgb_u8, gt, ms["PBDC"]),
        ]
        sub_labels = [
            sc["short_id"],
            f"PV={sc['target_pct']:.1f}%",
            f"D={mt['A0']['dice']:.3f} B2={mt['A0']['boundary_f1']:.3f}",
            f"D={mt['A3']['dice']:.3f} B2={mt['A3']['boundary_f1']:.3f}",
            f"D={mt['Old']['dice']:.3f} B2={mt['Old']['boundary_f1']:.3f}",
            f"D={mt['PBDC']['dice']:.3f} B2={mt['PBDC']['boundary_f1']:.3f}",
        ]
        for c_i in range(6):
            ax = axes[r_i, c_i]
            ax.imshow(imgs[c_i])
            ax.axis("off")
            if r_i == 0:
                ax.set_title(col_titles[c_i], fontsize=11, fontweight=("bold" if c_i == 5 else "normal"), pad=6)
            ax.text(
                6,
                18,
                sub_labels[c_i],
                color="white",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.62, pad=2, edgecolor="none"),
            )

    legend_handles = [
        mpatches.Patch(color="#2ed573", label="TP (Correct PV)"),
        mpatches.Patch(color="#ff4757", label="FP (False Positive)"),
        mpatches.Patch(color="#ffd32a", label="FN (Missed PV)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.002))
    fig.suptitle(header, fontsize=13.5, fontweight="bold", y=0.992)
    plt.tight_layout(rect=[0.0, 0.025, 1.0, 0.975])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_2m_visuals(
    *,
    split_dir: Path,
    dev_cache_pt: Path,
    deform_config_json: Path,
    out_dir: Path,
    device: str = "cuda",
) -> None:
    t0 = time.perf_counter()
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    img_dir, lbl_dir = resolve_data_dirs(None, None)

    bundle = torch.load(dev_cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    mask_by_id = {r["id"]: read_verified_mask(lbl_dir, r) for r in dev_records}
    rgb_by_id = {r["id"]: np.transpose(read_verified_rgb(img_dir, r), (1, 2, 0)).copy() for r in dev_records}
    rec_by_id = {r["id"]: r for r in dev_records}

    tokens_by_key = {k: {rid: to_tokens_norm(feats[rid][k], run_device) for rid in all_ids} for k in ("deep_32", "deep_64", "mid_64")}
    caches = {
        k: build_pairwise_topk_cache(tokens_by_key[k], mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        for k, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64))
    }
    core_cfg = get_clean_core_config()
    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=core_cfg.aff_tau) for rid in all_ids}

    pipe_a0 = ThreeModulePipeline(get_b0_32_config(), refiner_cfg=None)
    pipe_a3 = ThreeModulePipeline(core_cfg, refiner_cfg=None)
    old_cfg = DeformableBoundaryConfig.from_json(deform_config_json, mode="deformable_grad")
    pbdc_cfg = PBDCConfig(enabled=True)

    # First K=5 episode per fold covers all 50 query sites
    eps_k5 = []
    seen_folds = set()
    for ep in episodes_lock["episodes"]:
        if int(ep["k"]) == 5 and int(ep["fold"]) not in seen_folds:
            seen_folds.add(int(ep["fold"]))
            eps_k5.append(ep)

    candidates: list[dict[str, Any]] = []
    for ep in eps_k5:
        sup_ids, q_ids = ep["support_ids"], ep["query_ids"]
        out_a0 = pipe_a0.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
        out_a3 = pipe_a3.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
        p_a3, _ = out_a3.to_legacy_outputs()
        m_a0 = out_a0.coarse_masks_512()
        m_a3 = out_a3.coarse_masks_512()

        loo_probs = {}
        for sid in sup_ids:
            rem = [s for s in sup_ids if s != sid]
            loo_probs[sid] = pipe_a3.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, rem, [sid]).boundary_by_id[sid].p_coarse
        allow_out, p_floor = calibrate_deform_p_floor_loo(sup_ids, mask_by_id, loo_probs)

        for qid in q_ids:
            if not rec_by_id[qid]["positive"]:
                continue
            gt = mask_by_id[qid]
            rgb = rgb_by_id[qid]
            m_old = refine_mask_deformable_grad(m_a3[qid], rgb, p_a3[qid], allow_outward=allow_out, p_floor=p_floor, tau_grad_per_px=0.20, cfg=old_cfg)["mask"]
            m_pbdc = refine_mask_pbdc(m_a3[qid], rgb, p_a3[qid], k_shots=5, cfg=pbdc_cfg)["mask"]
            masks = {"A0": m_a0[qid], "A3": m_a3[qid], "Old": m_old, "PBDC": m_pbdc}
            metrics = {k: evaluate_binary_mask_512(gt, m, p_a3[qid]) for k, m in masks.items()}
            tf = float((gt > 0).mean()) * 100.0
            gain_bf2 = metrics["PBDC"]["boundary_f1"] - metrics["A3"]["boundary_f1"]
            candidates.append(
                {
                    "id": qid,
                    "short_id": Path(qid).stem,
                    "rgb": rgb,
                    "gt": gt,
                    "masks": masks,
                    "metrics": metrics,
                    "target_pct": tf,
                    "gain_bf2": gain_bf2,
                }
            )

    # Select 6 diverse representative 2m scenes (2 small <5%, 2 mid 5-15%, 2 large >=15%)
    smalls = sorted([c for c in candidates if c["target_pct"] < 5.0], key=lambda x: -x["gain_bf2"])[:2]
    mids = sorted([c for c in candidates if 5.0 <= c["target_pct"] < 15.0], key=lambda x: -x["gain_bf2"])[:2]
    larges = sorted([c for c in candidates if c["target_pct"] >= 15.0], key=lambda x: -x["gain_bf2"])[:2]
    chosen = smalls + mids + larges

    sub_dir = out_dir / "2m_pv100"
    sub_dir.mkdir(parents=True, exist_ok=True)
    for idx, sc in enumerate(chosen, 1):
        save_scene_panel(
            sub_dir / f"2m_{idx:02d}_{sc['short_id']}.png",
            title=f"PV-100scence (2m, K=5) | {sc['short_id']} | PV Area={sc['target_pct']:.1f}% | dBF1@2={sc['gain_bf2']:+.3f}",
            rgb=sc["rgb"],
            gt=sc["gt"],
            masks=sc["masks"],
            metrics=sc["metrics"],
        )
    save_summary_grid(
        out_dir / "summary_2m_pv100_grid.png",
        header="PV-100scence (2m Resolution, K=5) Qualitative Query Comparison (v3.1-pbdc-pure)",
        scenes=chosen,
    )
    print(f"[2m Visuals] Saved {len(chosen)} panels + summary grid in {time.perf_counter()-t0:.1f}s", flush=True)


def run_08m_visuals(
    *,
    subset_json: Path,
    deform_config_json: Path,
    source_dir: Path,
    feat_cache_dir: Path,
    out_dir: Path,
    device: str = "cuda",
) -> None:
    t0 = time.perf_counter()
    sub = read_json(subset_json)
    sup_recs = sub["support_records"]
    qry_recs = sub["query_records"]
    used_recs = sup_recs + qry_recs
    all_ids = [r["id"] for r in used_recs]
    sup_pool_ids = [r["id"] for r in sup_recs]
    qry_ids = [r["id"] for r in qry_recs]

    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    img_dir, msk_dir = resolve_pv663_dirs(None, None)
    rgb_by_id, mask_by_id = extract_pv663_features_to_dir(
        used_recs, image_dir=img_dir, mask_dir=msk_dir, feat_dir=feat_cache_dir, source_dir=source_dir, device=run_device
    )

    caches = {
        k: build_pairwise_topk_cache_streaming(feat_cache_dir, k, mask_by_id, all_ids, sup_pool_ids, grid_hw=ghw, core_purity=0.85, top_m=3, device=run_device)
        for k, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64))
    }
    tokens_by_key = {k: LazyDiskTokenMap(feat_cache_dir, k, run_device, preload_ids=sup_pool_ids) for k in ("deep_32", "deep_64", "mid_64")}
    core_cfg = get_clean_core_config()
    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=core_cfg.aff_tau) for rid in all_ids}

    pipe_a0 = ThreeModulePipeline(get_b0_32_config(), refiner_cfg=None)
    pipe_a3 = ThreeModulePipeline(core_cfg, refiner_cfg=None)
    old_cfg = DeformableBoundaryConfig.from_json(deform_config_json, mode="deformable_grad")
    pbdc_cfg = PBDCConfig(enabled=True)

    ep5 = next(ep for ep in sub["episodes"] if int(ep["k"]) == 5)
    sup_ids = ep5["support_ids"]

    out_a0 = pipe_a0.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, qry_ids)
    out_a3 = pipe_a3.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, qry_ids)
    p_a3, _ = out_a3.to_legacy_outputs()
    m_a0 = out_a0.coarse_masks_512()
    m_a3 = out_a3.coarse_masks_512()

    loo_probs = {}
    for sid in sup_ids:
        rem = [s for s in sup_ids if s != sid]
        loo_probs[sid] = pipe_a3.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, rem, [sid]).boundary_by_id[sid].p_coarse
    allow_out, p_floor = calibrate_deform_p_floor_loo(sup_ids, mask_by_id, loo_probs)

    candidates: list[dict[str, Any]] = []
    for qid in qry_ids:
        gt = mask_by_id[qid]
        rgb = rgb_by_id[qid]
        m_old = refine_mask_deformable_grad(m_a3[qid], rgb, p_a3[qid], allow_outward=allow_out, p_floor=p_floor, tau_grad_per_px=0.20, cfg=old_cfg)["mask"]
        m_pbdc = refine_mask_pbdc(m_a3[qid], rgb, p_a3[qid], k_shots=5, cfg=pbdc_cfg)["mask"]
        masks = {"A0": m_a0[qid], "A3": m_a3[qid], "Old": m_old, "PBDC": m_pbdc}
        metrics = {k: evaluate_binary_mask_512(gt, m, p_a3[qid]) for k, m in masks.items()}
        tf = float((gt > 0).mean()) * 100.0
        gain_vs_old = metrics["PBDC"]["boundary_f1"] - metrics["Old"]["boundary_f1"]
        gain_vs_a3 = metrics["PBDC"]["boundary_f1"] - metrics["A3"]["boundary_f1"]
        candidates.append(
            {
                "id": qid,
                "short_id": Path(qid).stem,
                "rgb": rgb,
                "gt": gt,
                "masks": masks,
                "metrics": metrics,
                "target_pct": tf,
                "gain_bf2": gain_vs_a3,
                "score": gain_vs_a3 + 0.5 * gain_vs_old,
            }
        )

    # Select 6 diverse representative 0.8m scenes across area strata (2 small <5%, 2 mid 5-15%, 2 large >=15%)
    smalls = sorted([c for c in candidates if c["target_pct"] < 5.0], key=lambda x: -x["score"])[:2]
    mids = sorted([c for c in candidates if 5.0 <= c["target_pct"] < 15.0], key=lambda x: -x["score"])[:2]
    larges = sorted([c for c in candidates if c["target_pct"] >= 15.0], key=lambda x: -x["score"])[:2]
    chosen = smalls + mids + larges

    sub_dir = out_dir / "08m_pv663"
    sub_dir.mkdir(parents=True, exist_ok=True)
    for idx, sc in enumerate(chosen, 1):
        save_scene_panel(
            sub_dir / f"08m_{idx:02d}_{sc['short_id']}.png",
            title=f"PV663 (0.8m, K=5) | {sc['short_id']} | PV Area={sc['target_pct']:.1f}% | dBF1@2={sc['gain_bf2']:+.3f}",
            rgb=sc["rgb"],
            gt=sc["gt"],
            masks=sc["masks"],
            metrics=sc["metrics"],
        )
    save_summary_grid(
        out_dir / "summary_08m_pv663_grid.png",
        header="PV663 (0.8m Resolution, K=5) Qualitative Query Comparison (v3.1-pbdc-pure)",
        scenes=chosen,
    )
    print(f"[0.8m Visuals] Saved {len(chosen)} panels + summary grid in {time.perf_counter()-t0:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 2m & 0.8m qualitative query panels for v3.1-pbdc-pure.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--dev-cache-pt", type=Path, required=True)
    parser.add_argument("--pv663-subset-json", type=Path, default=REPO_ROOT / "configs" / "pv663_subset_lock.json")
    parser.add_argument("--deform-config-json", type=Path, default=REPO_ROOT / "configs" / "three_module_exp_m3_deformgrad.json")
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/dinov3-source"))
    parser.add_argument("--pv663-feat-dir", type=Path, default=Path("/tmp/pv663_vis_feats"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_2m_visuals(
        split_dir=args.split_dir,
        dev_cache_pt=args.dev_cache_pt,
        deform_config_json=args.deform_config_json,
        out_dir=args.output_dir,
        device=args.device,
    )
    run_08m_visuals(
        subset_json=args.pv663_subset_json,
        deform_config_json=args.deform_config_json,
        source_dir=args.source_dir,
        feat_cache_dir=args.pv663_feat_dir,
        out_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
