from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.data import load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache, compute_dino_affinity_weights, to_tokens_norm
from pvfs.io import read_json
from pvfs.metrics import evaluate_binary_mask_512
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
from scripts.render_dual_res_visuals import overlay_gt, overlay_tp_fp_fn, save_scene_panel, save_summary_grid


def save_neg_grid(out_path: Path, *, header: str, scenes: list[dict[str, Any]]) -> None:
    """Save 5 pure negative samples x 5 columns (RGB, A0, A3, A4_Old, Ours v3.1-PBDC)."""
    n_rows = len(scenes)
    fig, axes = plt.subplots(n_rows, 5, figsize=(17.5, 3.4 * n_rows), dpi=200)
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = ["RGB Input (Pure Neg)", "A0 (B0_32)", "A3 (Before PBDC)", "A4_Old (Legacy)", "Ours (v3.1-PBDC)"]
    for r_i, sc in enumerate(scenes):
        rgb = sc["rgb"]
        gt = sc["gt"]
        ms = sc["masks"]
        fp_pct = {k: float((m > 0).mean()) * 100.0 for k, m in ms.items()}
        imgs = [
            rgb,
            overlay_tp_fp_fn(rgb, gt, ms["A0"]),
            overlay_tp_fp_fn(rgb, gt, ms["A3"]),
            overlay_tp_fp_fn(rgb, gt, ms["Old"]),
            overlay_tp_fp_fn(rgb, gt, ms["PBDC"]),
        ]
        labels = [
            sc["short_id"],
            f"FP={fp_pct['A0']:.2f}%",
            f"FP={fp_pct['A3']:.2f}%",
            f"FP={fp_pct['Old']:.2f}%",
            f"FP={fp_pct['PBDC']:.2f}%",
        ]
        for c_i in range(5):
            ax = axes[r_i, c_i]
            ax.imshow(imgs[c_i])
            ax.axis("off")
            if r_i == 0:
                ax.set_title(col_titles[c_i], fontsize=11, fontweight=("bold" if c_i == 4 else "normal"), pad=6)
            ax.text(
                6,
                18,
                labels[c_i],
                color="white",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.65, pad=2, edgecolor="none"),
            )

    legend_handles = [mpatches.Patch(color="#ff4757", label="FP (False Positive on Pure Negative Scene)")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=1, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.002))
    fig.suptitle(header, fontsize=13.5, fontweight="bold", y=0.992)
    plt.tight_layout(rect=[0.0, 0.025, 1.0, 0.975])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_2m_neg_and_extremes(
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

    eps_k5 = []
    seen_folds = set()
    for ep in episodes_lock["episodes"]:
        if int(ep["k"]) == 5 and int(ep["fold"]) not in seen_folds:
            seen_folds.add(int(ep["fold"]))
            eps_k5.append(ep)

    pos_scenes: list[dict[str, Any]] = []
    neg_scenes: list[dict[str, Any]] = []

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
            gt = mask_by_id[qid]
            rgb = rgb_by_id[qid]
            m_old = refine_mask_deformable_grad(m_a3[qid], rgb, p_a3[qid], allow_outward=allow_out, p_floor=p_floor, tau_grad_per_px=0.20, cfg=old_cfg)["mask"]
            m_pbdc = refine_mask_pbdc(m_a3[qid], rgb, p_a3[qid], k_shots=5, cfg=pbdc_cfg)["mask"]
            masks = {"A0": m_a0[qid], "A3": m_a3[qid], "Old": m_old, "PBDC": m_pbdc}

            if rec_by_id[qid]["positive"]:
                metrics = {k: evaluate_binary_mask_512(gt, m, p_a3[qid]) for k, m in masks.items()}
                pos_scenes.append(
                    {
                        "id": qid,
                        "short_id": Path(qid).stem,
                        "rgb": rgb,
                        "gt": gt,
                        "masks": masks,
                        "metrics": metrics,
                        "target_pct": float((gt > 0).mean()) * 100.0,
                        "dice": metrics["PBDC"]["dice"],
                        "bf2": metrics["PBDC"]["boundary_f1"],
                    }
                )
            else:
                fp_a0 = float((m_a0[qid] > 0).mean())
                fp_pbdc = float((m_pbdc > 0).mean())
                neg_scenes.append(
                    {
                        "id": qid,
                        "short_id": Path(qid).stem,
                        "rgb": rgb,
                        "gt": gt,
                        "masks": masks,
                        "fp_a0": fp_a0,
                        "fp_pbdc": fp_pbdc,
                    }
                )

    # 1. Pick 5 diverse pure negative samples:
    #    - 2 clean/near-zero FP scenes where A0 had FPs (good suppression)
    #    - 2 medium/typical negative scenes
    #    - 1 hardest negative scene (highest FP on PBDC, honest failure visibility!)
    neg_sorted_by_pbdc = sorted(neg_scenes, key=lambda x: x["fp_pbdc"])
    hardest_neg = neg_sorted_by_pbdc[-1]
    remaining_neg = [x for x in neg_scenes if x["id"] != hardest_neg["id"]]
    # Top 2 where PBDC reduced FP most vs A0
    best_suppressed = sorted(remaining_neg, key=lambda x: -(x["fp_a0"] - x["fp_pbdc"]))[:2]
    rem2 = [x for x in remaining_neg if x["id"] not in {s["id"] for s in best_suppressed}]
    # 2 clean zero-FP scenes
    clean_neg = sorted(rem2, key=lambda x: x["fp_pbdc"])[:2]
    chosen_neg = clean_neg + best_suppressed + [hardest_neg]

    save_neg_grid(
        out_dir / "summary_2m_neg5_grid.png",
        header="PV-100scence (2m, K=5) 5 Pure Negative Samples (2 Clean + 2 Suppressed + 1 Hardest)",
        scenes=chosen_neg,
    )

    # 2. Best & Worst positive query image on 2m (by Ours PBDC Dice, tie-break BF2)
    pos_sorted = sorted(pos_scenes, key=lambda x: (x["dice"], x["bf2"]))
    worst_2m = pos_sorted[0]
    best_2m = pos_sorted[-1]

    save_scene_panel(
        out_dir / f"2m_BEST_{best_2m['short_id']}.png",
        title=f"PV-100scence (2m, K=5) BEST Case | {best_2m['short_id']} | Dice={best_2m['dice']:.3f} BF1@2={best_2m['bf2']:.3f}",
        rgb=best_2m["rgb"],
        gt=best_2m["gt"],
        masks=best_2m["masks"],
        metrics=best_2m["metrics"],
    )
    save_scene_panel(
        out_dir / f"2m_WORST_{worst_2m['short_id']}.png",
        title=f"PV-100scence (2m, K=5) WORST Case | {worst_2m['short_id']} | Dice={worst_2m['dice']:.3f} BF1@2={worst_2m['bf2']:.3f}",
        rgb=worst_2m["rgb"],
        gt=worst_2m["gt"],
        masks=worst_2m["masks"],
        metrics=worst_2m["metrics"],
    )
    save_summary_grid(
        out_dir / "summary_2m_best_worst.png",
        header=f"PV-100scence (2m, K=5) Best ({best_2m['short_id']}) vs Worst ({worst_2m['short_id']}) Query Case",
        scenes=[best_2m, worst_2m],
    )
    print(
        f"[2m Neg+Extremes] Best={best_2m['short_id']}(Dice={best_2m['dice']:.3f}) "
        f"Worst={worst_2m['short_id']}(Dice={worst_2m['dice']:.3f}) | 5 Negs saved in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )


def run_08m_extremes(
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

    pos_scenes: list[dict[str, Any]] = []
    for qid in qry_ids:
        gt = mask_by_id[qid]
        rgb = rgb_by_id[qid]
        m_old = refine_mask_deformable_grad(m_a3[qid], rgb, p_a3[qid], allow_outward=allow_out, p_floor=p_floor, tau_grad_per_px=0.20, cfg=old_cfg)["mask"]
        m_pbdc = refine_mask_pbdc(m_a3[qid], rgb, p_a3[qid], k_shots=5, cfg=pbdc_cfg)["mask"]
        masks = {"A0": m_a0[qid], "A3": m_a3[qid], "Old": m_old, "PBDC": m_pbdc}
        metrics = {k: evaluate_binary_mask_512(gt, m, p_a3[qid]) for k, m in masks.items()}
        pos_scenes.append(
            {
                "id": qid,
                "short_id": Path(qid).stem,
                "rgb": rgb,
                "gt": gt,
                "masks": masks,
                "metrics": metrics,
                "target_pct": float((gt > 0).mean()) * 100.0,
                "dice": metrics["PBDC"]["dice"],
                "bf2": metrics["PBDC"]["boundary_f1"],
            }
        )

    pos_sorted = sorted(pos_scenes, key=lambda x: (x["dice"], x["bf2"]))
    worst_08m = pos_sorted[0]
    best_08m = pos_sorted[-1]

    save_scene_panel(
        out_dir / f"08m_BEST_{best_08m['short_id']}.png",
        title=f"PV663 (0.8m, K=5) BEST Case | {best_08m['short_id']} | Dice={best_08m['dice']:.3f} BF1@2={best_08m['bf2']:.3f}",
        rgb=best_08m["rgb"],
        gt=best_08m["gt"],
        masks=best_08m["masks"],
        metrics=best_08m["metrics"],
    )
    save_scene_panel(
        out_dir / f"08m_WORST_{worst_08m['short_id']}.png",
        title=f"PV663 (0.8m, K=5) WORST Case | {worst_08m['short_id']} | Dice={worst_08m['dice']:.3f} BF1@2={worst_08m['bf2']:.3f}",
        rgb=worst_08m["rgb"],
        gt=worst_08m["gt"],
        masks=worst_08m["masks"],
        metrics=worst_08m["metrics"],
    )
    save_summary_grid(
        out_dir / "summary_08m_best_worst.png",
        header=f"PV663 (0.8m, K=5) Best ({best_08m['short_id']}) vs Worst ({worst_08m['short_id']}) Query Case",
        scenes=[best_08m, worst_08m],
    )
    print(
        f"[0.8m Extremes] Best={best_08m['short_id']}(Dice={best_08m['dice']:.3f}) "
        f"Worst={worst_08m['short_id']}(Dice={worst_08m['dice']:.3f}) saved in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )
