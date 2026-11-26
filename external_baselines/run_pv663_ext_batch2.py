from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_baselines.finetune_models import DINOv3KShotFTRunner, EndToEndSegKShotRunner
from pvfs.data import area_target_32
from pvfs.io import read_json, write_json
from pvfs.pv663_eval import (
    LazyDiskTokenMap,
    aggregate_pv663_episodes,
    compare_pv663_methods,
    evaluate_pv663_episode,
    extract_pv663_features_to_dir,
    resolve_pv663_dirs,
)


def _select_episodes(ep_lock: dict[str, Any], episodes_per_k: int) -> dict[int, list[dict[str, Any]]]:
    ep_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ep in ep_lock["episodes"]:
        k = int(ep["k"])
        if len(ep_by_k[k]) < episodes_per_k:
            ep_by_k[k].append(ep)
    return {k: ep_by_k[k] for k in sorted(ep_by_k)}


def format_ext_batch2_report(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    lines = [
        "=== PV663 (0.8m) EXTERNAL MODEL COMPARISON — BATCH 2 (Few-Shot Fine-Tuning Models) ===",
        f"Split: {meta['support_sites']} support sites ({meta['unique_support_used']} used) / "
        f"{meta['query_sites']} query sites ({meta['query_images']} imgs) | min_dist={meta['min_dist_km']:.2f} km",
        f"Episodes/K: {meta['episodes_per_k']} | Total Elapsed: {summary['elapsed_sec']/60:.1f} min",
    ]
    order = ("UNet_ResNet50_FT", "SegFormer_MiTB2_FT", "DINOv3SAT_ConvHead_FT", "Ours_v31_PBDC_Pure")
    for k in ("1", "3", "5"):
        lines.append(f"\n--- K={k} (116 geographic sites, 510 query images x {meta['episodes_per_k']} eps) ---")
        for mname in order:
            if mname not in summary["methods"] or k not in summary["methods"][mname]:
                continue
            res = summary["methods"][mname][k]
            m = res["macro"]
            sm_r = res["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            ft_s = summary["train_sec_per_ep"].get(mname, {}).get(k, 0.0)
            ms = summary["infer_ms_per_img"].get(mname, {}).get(k, 0.0)
            lines.append(
                f"  {mname:24s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
                f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} Sm<5R={sm_r:.4f} "
                f"(train={ft_s:.1f}s/ep, infer={ms:.1f}ms/img)"
            )

    if summary.get("comparisons"):
        lines.append("\n--- Paired Site Bootstrap Deltas (116 Geographic Sites, Ours vs Fine-Tuning) ---")
        for cname in ("Ours_vs_UNet_ResNet50_FT", "Ours_vs_SegFormer_MiTB2_FT", "Ours_vs_DINOv3SAT_ConvHead_FT"):
            if cname not in summary["comparisons"]:
                continue
            lines.append(f"[{cname}]")
            for k in ("1", "3", "5"):
                c = summary["comparisons"][cname][k]
                lines.append(
                    f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} "
                    f"({c['dice']['improved_sites']}/{c['dice']['total_sites']}) | "
                    f"dBF1@1={c['bf1_1px']['mean_delta']:+.4f} CI{[round(x,4) for x in c['bf1_1px']['ci95']]} | "
                    f"dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x,4) for x in c['boundary_f1']['ci95']]} | "
                    f"dSm<5R={c['small_lt5pct_recall_delta']:+.4f}"
                )
    return "\n".join(lines) + "\n"


def run_pv663_ext_batch2(
    *,
    split_json: Path,
    episodes_json: Path,
    ours_ref_json: Path | None,
    source_dir: Path,
    dino_feat_dir: Path,
    output_dir: Path,
    episodes_per_k: int = 5,
    device: str = "cuda",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    sp = read_json(split_json)
    ep_lock = read_json(episodes_json)
    ep_by_k = _select_episodes(ep_lock, episodes_per_k=episodes_per_k)

    rec_by_id = {r["id"]: r for r in sp["records"]}
    query_recs_raw = [r for r in sp["records"] if r["partition"] == "query_set"]
    query_ids = [r["id"] for r in query_recs_raw]
    unique_sup_ids = sorted({sid for eps in ep_by_k.values() for ep in eps for sid in ep["support_ids"]})
    used_recs = [rec_by_id[rid] for rid in unique_sup_ids] + query_recs_raw

    img_dir, msk_dir = resolve_pv663_dirs(None, None)
    print(f"[Ext Batch 2] Extracting frozen DINOv3-SAT features for {len(used_recs)} images...", flush=True)
    rgb_512_by_id, mask_512_by_id = extract_pv663_features_to_dir(
        used_recs,
        image_dir=img_dir,
        mask_dir=msk_dir,
        feat_dir=dino_feat_dir,
        source_dir=source_dir,
        device=run_device,
    )
    query_records = [
        {**r, "mask": mask_512_by_id[r["id"]], "y32": area_target_32(mask_512_by_id[r["id"]])}
        for r in query_recs_raw
    ]
    tokens_by_key = {
        key: LazyDiskTokenMap(dino_feat_dir, key, run_device, preload_ids=unique_sup_ids)
        for key in ("deep_64", "mid_64")
    }

    dino_ft = DINOv3KShotFTRunner(steps=120, lr=3e-3, wd=1e-2)
    unet_ft = EndToEndSegKShotRunner("unet_resnet50", steps=80, lr=3e-4, wd=1e-2, device=run_device)
    segf_ft = EndToEndSegKShotRunner("segformer_mit_b2", steps=80, lr=3e-4, wd=1e-2, device=run_device)

    methods_res: dict[str, dict[str, Any]] = defaultdict(dict)
    train_sec_per_ep: dict[str, dict[str, float]] = defaultdict(dict)
    infer_ms_per_img: dict[str, dict[str, float]] = defaultdict(dict)

    for k in sorted(ep_by_k.keys()):
        t_k = time.perf_counter()
        eps = ep_by_k[k]
        ep_evals: dict[str, list[dict]] = {
            "UNet_ResNet50_FT": [],
            "SegFormer_MiTB2_FT": [],
            "DINOv3SAT_ConvHead_FT": [],
        }
        tr_sec = {m: 0.0 for m in ep_evals}
        inf_sec = {m: 0.0 for m in ep_evals}
        n_q_tot = 0

        for ep_i, ep in enumerate(eps, 1):
            sup_ids = ep["support_ids"]
            n_q_tot += len(query_ids)
            seed_off = k * 100 + ep_i

            # 1. DINOv3-SAT + K-Shot ConvHead FT
            ts = time.perf_counter()
            head = dino_ft.fit_episode(tokens_by_key, mask_512_by_id, sup_ids, ep_seed_offset=seed_off)
            tr_sec["DINOv3SAT_ConvHead_FT"] += time.perf_counter() - ts
            ts = time.perf_counter()
            m_d, p_d, s32_d = dino_ft.infer_queries(head, tokens_by_key, query_ids)
            inf_sec["DINOv3SAT_ConvHead_FT"] += time.perf_counter() - ts
            ep_evals["DINOv3SAT_ConvHead_FT"].append(evaluate_pv663_episode(query_records, m_d, p_d, s32_d))
            del head

            # 2. U-Net (ResNet50) K-Shot E2E FT
            ts = time.perf_counter()
            m_unet = unet_ft.fit_episode(rgb_512_by_id, mask_512_by_id, sup_ids, ep_seed_offset=seed_off)
            tr_sec["UNet_ResNet50_FT"] += time.perf_counter() - ts
            ts = time.perf_counter()
            m_u, p_u, s32_u = unet_ft.infer_queries(m_unet, rgb_512_by_id, query_ids)
            inf_sec["UNet_ResNet50_FT"] += time.perf_counter() - ts
            ep_evals["UNet_ResNet50_FT"].append(evaluate_pv663_episode(query_records, m_u, p_u, s32_u))
            del m_unet

            # 3. SegFormer (MiT-B2) K-Shot E2E FT
            ts = time.perf_counter()
            m_segf = segf_ft.fit_episode(rgb_512_by_id, mask_512_by_id, sup_ids, ep_seed_offset=seed_off)
            tr_sec["SegFormer_MiTB2_FT"] += time.perf_counter() - ts
            ts = time.perf_counter()
            m_s, p_s, s32_s = segf_ft.infer_queries(m_segf, rgb_512_by_id, query_ids)
            inf_sec["SegFormer_MiTB2_FT"] += time.perf_counter() - ts
            ep_evals["SegFormer_MiTB2_FT"].append(evaluate_pv663_episode(query_records, m_s, p_s, s32_s))
            del m_segf

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Ext Batch 2 K={k}] Ep {ep_i}/{len(eps)} done ({(time.perf_counter()-t_k)/60:.1f}m)", flush=True)

        for mname in ep_evals:
            methods_res[mname][str(k)] = aggregate_pv663_episodes(ep_evals[mname], expected_episodes=len(eps))
            train_sec_per_ep[mname][str(k)] = float(tr_sec[mname] / max(1, len(eps)))
            infer_ms_per_img[mname][str(k)] = float(inf_sec[mname] * 1000.0 / max(1, n_q_tot))
            m = methods_res[mname][str(k)]["macro"]
            sm_r = methods_res[mname][str(k)]["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            print(
                f"  -> [K={k}] {mname:24s}: Dice={m['dice']:.4f} BF1@1={m['bf1_1px']:.4f} "
                f"BF1@2={m['boundary_f1']:.4f} Sm<5R={sm_r:.4f}",
                flush=True,
            )

    comparisons: dict[str, Any] = {}
    if ours_ref_json is not None and ours_ref_json.is_file():
        ours_ref = read_json(ours_ref_json)
        methods_res["Ours_v31_PBDC_Pure"] = ours_ref["Ours_v31_PBDC_Pure"]
        for mname in ("UNet_ResNet50_FT", "SegFormer_MiTB2_FT", "DINOv3SAT_ConvHead_FT"):
            comparisons[f"Ours_vs_{mname}"] = {
                str(k): compare_pv663_methods(methods_res["Ours_v31_PBDC_Pure"][str(k)], methods_res[mname][str(k)])
                for k in (1, 3, 5)
            }

    summary = {
        "schema": "pv663-external-batch2-v1",
        "elapsed_sec": float(time.perf_counter() - t0),
        "meta": {
            "total_images": sp["summary"]["total_images"],
            "support_sites": sp["summary"]["support_pool_sites"],
            "unique_support_used": len(unique_sup_ids),
            "query_sites": sp["summary"]["query_set_sites"],
            "query_images": len(query_ids),
            "min_dist_km": float(sp["min_support_query_distance_km"]),
            "episodes_per_k": episodes_per_k,
        },
        "methods": dict(methods_res),
        "comparisons": comparisons,
        "train_sec_per_ep": dict(train_sec_per_ep),
        "infer_ms_per_img": dict(infer_ms_per_img),
    }
    write_json(output_dir / "pv663_ext_batch2_summary.json", summary)
    report = format_ext_batch2_report(summary)
    (output_dir / "pv663_ext_batch2_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PV663 External Model Comparison — Batch 2 (K-Shot Fine-Tuning).")
    parser.add_argument("--split-json", type=Path, default=REPO_ROOT / "configs" / "pv663_split_lock.json")
    parser.add_argument("--episodes-json", type=Path, default=REPO_ROOT / "configs" / "pv663_episodes_lock.json")
    parser.add_argument(
        "--ours-ref-json",
        type=Path,
        default=REPO_ROOT / "external_baselines" / "results" / "pv663_batch1" / "ours_v31_site_ref.json",
    )
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/dinov3-source"))
    parser.add_argument("--dino-feat-dir", type=Path, default=Path("/tmp/pv663_ext2_dino_feats"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "external_baselines" / "results" / "pv663_batch2")
    parser.add_argument("--episodes-per-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_pv663_ext_batch2(
        split_json=args.split_json,
        episodes_json=args.episodes_json,
        ours_ref_json=args.ours_ref_json,
        source_dir=args.source_dir,
        dino_feat_dir=args.dino_feat_dir,
        output_dir=args.output_dir,
        episodes_per_k=args.episodes_per_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()
