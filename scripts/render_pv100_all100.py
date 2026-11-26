from __future__ import annotations

import csv
import importlib
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.data import read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.dinov3_backbone import SAT_MEAN, SAT_STD, prepare_dinov3_source, resolve_dinov3_weights
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache, compute_dino_affinity_weights, to_tokens_norm
from pvfs.io import read_json
from pvfs.metrics import evaluate_binary_mask_512
from pvfs.modules import (
    PBDCConfig,
    ThreeModulePipeline,
    get_b0_32_config,
    get_clean_core_config,
    refine_mask_pbdc,
)
from scripts.render_dual_res_visuals import overlay_gt, overlay_tp_fp_fn


def _load_all_100_records(split_dir: Path) -> list[dict[str, Any]]:
    manifest = read_json(split_dir / "dataset_manifest.json")
    split = read_json(split_dir / "split_lock.json")
    assign = split["assignment"]
    recs: list[dict[str, Any]] = []
    for row in manifest["images"]:
        rid = row["id"]
        a = assign[rid]
        recs.append(
            {
                **row,
                "partition": a["partition"],
                "fold": a["fold"],
                "site": a["site"],
            }
        )
    # Sort naturally by PV number (PV-01 .. PV-100)
    def _sort_key(r: dict[str, Any]) -> tuple[int, str]:
        stem = Path(r["id"]).stem
        try:
            num = int(stem.split("-")[-1])
        except ValueError:
            num = 9999
        return num, r["id"]

    recs.sort(key=_sort_key)
    return recs


@torch.inference_mode()
def _extract_missing_test_features(
    test_recs: list[dict[str, Any]],
    rgb_chw_by_id: dict[str, np.ndarray],
    *,
    source_dir: Path,
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract deep_32 (512), mid_64 (1024), deep_64 (1024) for the 21 Holdout Test images."""
    if not test_recs:
        return {}
    weights = resolve_dinov3_weights(None)
    source = prepare_dinov3_source(source_dir)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    backbones = importlib.import_module("dinov3.hub.backbones")

    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = backbones.dinov3_vitl16(pretrained=True, weights=str(weights.resolve()))
    model.to(run_device).eval().requires_grad_(False)

    mean_t = torch.tensor(SAT_MEAN, device=run_device, dtype=torch.float32)[None, :, None, None]
    std_t = torch.tensor(SAT_STD, device=run_device, dtype=torch.float32)[None, :, None, None]
    layers_32 = [20, 21, 22, 23]
    layers_64 = [12, 13, 14, 15, 20, 21, 22, 23]

    out_feats: dict[str, dict[str, torch.Tensor]] = {}
    for row in test_recs:
        rid = row["id"]
        x512 = (torch.from_numpy(rgb_chw_by_id[rid])[None].to(run_device).float() / 255.0 - mean_t) / std_t
        x1024 = F.interpolate(x512, size=(1024, 1024), mode="bilinear", align_corners=False)

        outs_32 = model.get_intermediate_layers(x512, n=layers_32, reshape=True, norm=True)
        d32 = F.normalize(torch.stack([t[0].float() for t in outs_32], dim=0).mean(dim=0), dim=0)

        outs_64 = model.get_intermediate_layers(x1024, n=layers_64, reshape=True, norm=True)
        m64 = F.normalize(torch.stack([outs_64[i][0].float() for i in range(4)], dim=0).mean(dim=0), dim=0)
        d64 = F.normalize(torch.stack([outs_64[i][0].float() for i in range(4, 8)], dim=0).mean(dim=0), dim=0)

        out_feats[rid] = {"deep_32": d32.half().cpu(), "mid_64": m64.half().cpu(), "deep_64": d64.half().cpu()}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_feats


def _save_single_panel(out_path: Path, item: dict[str, Any]) -> None:
    """Save 1x5 comparison strip for a single PV-100 image."""
    rgb = item["rgb"]
    gt = item["gt"]
    is_pos = item["positive"]
    tag = f"[{item['partition'].upper()}] {'POS' if is_pos else 'NEG'}"
    if is_pos:
        m0, m3, mp = item["m_a0"], item["m_a3"], item["m_pbdc"]
        titles = [
            f"{item['short_id']} {tag}\nPV={item['target_pct']:.1f}%",
            "Ground Truth",
            f"A0 (B0_32)\nD={m0['dice']:.3f} B2={m0['boundary_f1']:.3f}",
            f"A3 (Before PBDC)\nD={m3['dice']:.3f} B2={m3['boundary_f1']:.3f}",
            f"Ours (v3.1-PBDC)\nD={mp['dice']:.3f} B1={mp['bf1_1px']:.3f} B2={mp['boundary_f1']:.3f}",
        ]
    else:
        titles = [
            f"{item['short_id']} {tag}\n(Pure Negative)",
            "Ground Truth (Empty)",
            f"A0 (B0_32)\nFP={item['fp_a0']:.2f}%",
            f"A3 (Before PBDC)\nFP={item['fp_a3']:.2f}%",
            f"Ours (v3.1-PBDC)\nFP={item['fp_pbdc']:.2f}%",
        ]

    imgs = [
        rgb,
        overlay_gt(rgb, gt),
        overlay_tp_fp_fn(rgb, gt, item["mask_a0"]),
        overlay_tp_fp_fn(rgb, gt, item["mask_a3"]),
        overlay_tp_fp_fn(rgb, gt, item["mask_pbdc"]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.6), dpi=150)
    for i in range(5):
        axes[i].imshow(imgs[i])
        axes[i].set_title(titles[i], fontsize=9.5, fontweight=("bold" if i == 4 else "normal"), pad=4)
        axes[i].axis("off")
    plt.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_contact_sheet(out_path: Path, items: list[dict[str, Any]], sheet_idx: int) -> None:
    """Save 10 scenes x 5 columns contact sheet for rapid visual inspection of all 100 images."""
    n_rows = len(items)
    fig, axes = plt.subplots(n_rows, 5, figsize=(16.0, 2.85 * n_rows), dpi=150)
    if n_rows == 1:
        axes = axes[None, :]
    col_headers = ["RGB Input", "Ground Truth", "A0 (B0_32)", "A3 (Before PBDC)", "Ours (v3.1-PBDC)"]
    for r_i, it in enumerate(items):
        rgb, gt = it["rgb"], it["gt"]
        imgs = [
            rgb,
            overlay_gt(rgb, gt),
            overlay_tp_fp_fn(rgb, gt, it["mask_a0"]),
            overlay_tp_fp_fn(rgb, gt, it["mask_a3"]),
            overlay_tp_fp_fn(rgb, gt, it["mask_pbdc"]),
        ]
        tag = "DEV" if it["partition"] == "dev" else "TEST"
        if it["positive"]:
            lbls = [
                f"{it['short_id']} [{tag}]",
                f"PV={it['target_pct']:.1f}%",
                f"D={it['m_a0']['dice']:.3f} B2={it['m_a0']['boundary_f1']:.3f}",
                f"D={it['m_a3']['dice']:.3f} B2={it['m_a3']['boundary_f1']:.3f}",
                f"D={it['m_pbdc']['dice']:.3f} B2={it['m_pbdc']['boundary_f1']:.3f}",
            ]
        else:
            lbls = [
                f"{it['short_id']} [{tag}-NEG]",
                "Pure Negative",
                f"FP={it['fp_a0']:.2f}%",
                f"FP={it['fp_a3']:.2f}%",
                f"FP={it['fp_pbdc']:.2f}%",
            ]
        for c_i in range(5):
            ax = axes[r_i, c_i]
            ax.imshow(imgs[c_i])
            ax.axis("off")
            if r_i == 0:
                ax.set_title(col_headers[c_i], fontsize=10.5, fontweight=("bold" if c_i == 4 else "normal"), pad=5)
            ax.text(
                6,
                18,
                lbls[c_i],
                color="white",
                fontsize=8.0,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.65, pad=2, edgecolor="none"),
            )
    legend_handles = [
        mpatches.Patch(color="#2ed573", label="TP"),
        mpatches.Patch(color="#ff4757", label="FP"),
        mpatches.Patch(color="#ffd32a", label="FN"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.002))
    first_id, last_id = items[0]["short_id"], items[-1]["short_id"]
    fig.suptitle(f"PV-100scence (2m, K=5) Contact Sheet {sheet_idx:02d} ({first_id} .. {last_id})", fontsize=12.5, fontweight="bold", y=0.994)
    plt.tight_layout(rect=[0.0, 0.015, 1.0, 0.982])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_pv100_all_100(
    *,
    split_dir: Path,
    dev_cache_pt: Path,
    source_dir: Path,
    out_dir: Path,
    device: str = "cuda",
) -> None:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_recs = _load_all_100_records(split_dir)
    dev_recs = [r for r in all_recs if r["partition"] == "dev"]
    test_recs = [r for r in all_recs if r["partition"] == "test"]
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_recs)
    img_dir, lbl_dir = resolve_data_dirs(None, None)

    rgb_chw_by_id: dict[str, np.ndarray] = {}
    rgb_hwc_by_id: dict[str, np.ndarray] = {}
    mask_by_id: dict[str, np.ndarray] = {}
    for row in all_recs:
        rid = row["id"]
        rgb_chw = read_verified_rgb(img_dir, row)
        rgb_chw_by_id[rid] = rgb_chw
        rgb_hwc_by_id[rid] = np.transpose(rgb_chw, (1, 2, 0)).copy()
        mask_by_id[rid] = read_verified_mask(lbl_dir, row)

    bundle = torch.load(dev_cache_pt, map_location="cpu", weights_only=False)
    feats = dict(bundle["features"])
    print(f"[PV-100 All] Loaded {len(feats)} dev features; extracting {len(test_recs)} holdout test features...", flush=True)
    test_feats = _extract_missing_test_features(test_recs, rgb_chw_by_id, source_dir=source_dir, device=device)
    feats.update(test_feats)

    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    all_ids = [r["id"] for r in all_recs]
    dev_pos_ids = [r["id"] for r in dev_recs if r["positive"]]

    tokens_by_key = {k: {rid: to_tokens_norm(feats[rid][k], run_device) for rid in all_ids} for k in ("deep_32", "deep_64", "mid_64")}
    caches = {
        k: build_pairwise_topk_cache(tokens_by_key[k], mask_by_id, all_ids, dev_pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        for k, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64))
    }
    core_cfg = get_clean_core_config()
    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=core_cfg.aff_tau) for rid in all_ids}

    pipe_a0 = ThreeModulePipeline(get_b0_32_config(), refiner_cfg=None)
    pipe_a3 = ThreeModulePipeline(core_cfg, refiner_cfg=None)
    pbdc_cfg = PBDCConfig(enabled=True)

    # First K=5 episode per dev fold (zero leakage: supports come strictly from other 3 folds)
    eps_k5 = []
    seen_folds = set()
    for ep in episodes_lock["episodes"]:
        if int(ep["k"]) == 5 and int(ep["fold"]) not in seen_folds:
            seen_folds.add(int(ep["fold"]))
            eps_k5.append(ep)

    # Holdout Test uses 5 distinct dev positive sites from fold-0's K=5 episode (strictly zero leakage!)
    test_sup_ids = eps_k5[0]["support_ids"]
    eval_batches = [(ep["support_ids"], ep["query_ids"]) for ep in eps_k5] + [(test_sup_ids, [r["id"] for r in test_recs])]

    results_by_id: dict[str, dict[str, Any]] = {}
    rec_by_id = {r["id"]: r for r in all_recs}

    for sup_ids, q_ids in eval_batches:
        out_a0 = pipe_a0.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
        out_a3 = pipe_a3.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
        p_a3, _ = out_a3.to_legacy_outputs()
        m_a0 = out_a0.coarse_masks_512()
        m_a3 = out_a3.coarse_masks_512()

        for qid in q_ids:
            row = rec_by_id[qid]
            gt = mask_by_id[qid]
            rgb = rgb_hwc_by_id[qid]
            m_pbdc = refine_mask_pbdc(m_a3[qid], rgb, p_a3[qid], k_shots=5, cfg=pbdc_cfg)["mask"]
            item: dict[str, Any] = {
                "id": qid,
                "short_id": Path(qid).stem,
                "partition": row["partition"],
                "positive": bool(row["positive"]),
                "target_pct": float((gt > 0).mean()) * 100.0,
                "rgb": rgb,
                "gt": gt,
                "mask_a0": m_a0[qid],
                "mask_a3": m_a3[qid],
                "mask_pbdc": m_pbdc,
            }
            if row["positive"]:
                item["m_a0"] = evaluate_binary_mask_512(gt, m_a0[qid], p_a3[qid])
                item["m_a3"] = evaluate_binary_mask_512(gt, m_a3[qid], p_a3[qid])
                item["m_pbdc"] = evaluate_binary_mask_512(gt, m_pbdc, p_a3[qid])
            else:
                item["fp_a0"] = float((m_a0[qid] > 0).mean()) * 100.0
                item["fp_a3"] = float((m_a3[qid] > 0).mean()) * 100.0
                item["fp_pbdc"] = float((m_pbdc > 0).mean()) * 100.0
            results_by_id[qid] = item

    ordered = [results_by_id[r["id"]] for r in all_recs]

    # 1. Render all 100 individual panels into a single fast ZIP archive
    indiv_dir = out_dir / "_tmp_indiv"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "pv100_all_100_visuals.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, it in enumerate(ordered, 1):
            fname = f"{idx:03d}_{it['short_id']}_{it['partition']}.png"
            p = indiv_dir / fname
            _save_single_panel(p, it)
            zf.write(p, arcname=fname)
            p.unlink()
    indiv_dir.rmdir()

    # 2. Render 10 contact sheets (10 images per sheet) directly into out_dir
    for s_idx in range(0, len(ordered), 10):
        chunk = ordered[s_idx : s_idx + 10]
        sheet_num = s_idx // 10 + 1
        _save_contact_sheet(out_dir / f"contact_sheet_{sheet_num:02d}.png", chunk, sheet_num)

    # 3. Save per-image CSV & concise text summary
    csv_path = out_dir / "pv100_all_100_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx", "id", "partition", "type", "pv_pct", "A0_Dice", "A3_Dice", "PBDC_Dice", "A3_BF2", "PBDC_BF1", "PBDC_BF2", "A0_FP_pct", "PBDC_FP_pct"])
        for idx, it in enumerate(ordered, 1):
            if it["positive"]:
                w.writerow([
                    idx, it["short_id"], it["partition"], "POS", f"{it['target_pct']:.2f}",
                    f"{it['m_a0']['dice']:.4f}", f"{it['m_a3']['dice']:.4f}", f"{it['m_pbdc']['dice']:.4f}",
                    f"{it['m_a3']['boundary_f1']:.4f}", f"{it['m_pbdc']['bf1_1px']:.4f}", f"{it['m_pbdc']['boundary_f1']:.4f}",
                    "", "",
                ])
            else:
                w.writerow([
                    idx, it["short_id"], it["partition"], "NEG", "0.00",
                    "", "", "", "", "", "", f"{it['fp_a0']:.3f}", f"{it['fp_pbdc']:.3f}",
                ])

    def _agg(sub: list[dict[str, Any]], label: str) -> str:
        pos = [x for x in sub if x["positive"]]
        neg = [x for x in sub if not x["positive"]]
        return (
            f"{label:12s} (Pos={len(pos):2d}, Neg={len(neg):2d}) | "
            f"A0 Dice={np.mean([x['m_a0']['dice'] for x in pos]):.4f} BF2={np.mean([x['m_a0']['boundary_f1'] for x in pos]):.4f} NegFP={np.mean([x['fp_a0'] for x in neg]):.2f}% | "
            f"A3 Dice={np.mean([x['m_a3']['dice'] for x in pos]):.4f} BF2={np.mean([x['m_a3']['boundary_f1'] for x in pos]):.4f} NegFP={np.mean([x['fp_a3'] for x in neg]):.2f}% | "
            f"PBDC Dice={np.mean([x['m_pbdc']['dice'] for x in pos]):.4f} BF1={np.mean([x['m_pbdc']['bf1_1px'] for x in pos]):.4f} BF2={np.mean([x['m_pbdc']['boundary_f1'] for x in pos]):.4f} NegFP={np.mean([x['fp_pbdc'] for x in neg]):.2f}%"
        )

    lines = [
        f"=== PV-100scence ALL 100 IMAGES SUMMARY (v3.1-pbdc-pure, K=5) | Elapsed={time.perf_counter()-t0:.1f}s ===",
        _agg([x for x in ordered if x["partition"] == "dev"], "Dev (79)"),
        _agg([x for x in ordered if x["partition"] == "test"], "Test (21)"),
        _agg(ordered, "All (100)"),
    ]
    summary_txt = "\n".join(lines) + "\n"
    (out_dir / "pv100_all_100_summary.txt").write_text(summary_txt, encoding="utf-8")
    print("\n" + summary_txt, flush=True)
