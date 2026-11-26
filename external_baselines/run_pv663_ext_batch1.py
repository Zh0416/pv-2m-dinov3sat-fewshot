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

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_baselines.matcher_dinov3sat import MatcherDINOv3Runner
from external_baselines.persam_f import PerSAMFRunner
from external_baselines.sam_utils import LazySamEmbedMap, extract_sam_embeddings_to_dir, load_sam_vit_l
from pvfs.data import area_target_32
from pvfs.fast_bank import compute_dino_affinity_weights
from pvfs.io import read_json, write_json
from pvfs.modules import PBDCConfig, ThreeModulePipeline, get_clean_core_config, refine_mask_pbdc
from pvfs.pv663_eval import (
    LazyDiskTokenMap,
    aggregate_pv663_episodes,
    build_pairwise_topk_cache_streaming,
    compare_pv663_methods,
    evaluate_pv663_episode,
    extract_pv663_features_to_dir,
    resolve_pv663_dirs,
)

# Shared fork worker state for Ours (v3.1-PBDC-Pure)
_FW_RGB_BY_ID: dict[str, np.ndarray] = {}
_FW_PBDC_CFG: PBDCConfig | None = None
_FW_K_SHOTS: int = 5


def _pbdc_worker(args: tuple[str, np.ndarray, np.ndarray]) -> tuple[str, np.ndarray]:
    qid, m0, p0_f16 = args
    assert _FW_PBDC_CFG is not None
    out = refine_mask_pbdc(m0, _FW_RGB_BY_ID[qid], p0_f16.astype(np.float32), k_shots=_FW_K_SHOTS, cfg=_FW_PBDC_CFG)
    return qid, out["mask"]


def _select_episodes(ep_lock: dict[str, Any], episodes_per_k: int) -> dict[int, list[dict[str, Any]]]:
    ep_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ep in ep_lock["episodes"]:
        k = int(ep["k"])
        if len(ep_by_k[k]) < episodes_per_k:
            ep_by_k[k].append(ep)
    return {k: ep_by_k[k] for k in sorted(ep_by_k)}


def format_ext_batch1_report(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    lines = [
        "=== PV663 (0.8m) EXTERNAL MODEL COMPARISON — BATCH 1 (Training-Free Foundation Models) ===",
        f"Split: {meta['support_sites']} support sites ({meta['unique_support_used']} used) / "
        f"{meta['query_sites']} query sites ({meta['query_images']} imgs) | min_dist={meta['min_dist_km']:.2f} km",
        f"Episodes/K: {meta['episodes_per_k']} | Total Elapsed: {summary['elapsed_sec']/60:.1f} min",
    ]
    order = ("PerSAM_F_SAM_ViTL", "Matcher_DINOv3SAT_SAM_ViTL", "Ours_v31_PBDC_Pure")
    for k in ("1", "3", "5"):
        lines.append(f"\n--- K={k} (116 geographic sites, 510 query images x {meta['episodes_per_k']} eps) ---")
        for mname in order:
            res = summary["methods"][mname][k]
            m = res["macro"]
            sm_r = res["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            ms = summary["latency_ms_per_img"].get(mname, {}).get(k, 0.0)
            lines.append(
                f"  {mname:28s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
                f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} Sm<5R={sm_r:.4f} ({ms:.1f} ms/img)"
            )

    lines.append("\n--- Paired Site Bootstrap Deltas (116 Geographic Sites, Ours vs External) ---")
    for cname in ("Ours_vs_PerSAM_F", "Ours_vs_Matcher_DINOv3SAT"):
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


def run_pv663_ext_batch1(
    *,
    split_json: Path,
    episodes_json: Path,
    source_dir: Path,
    dino_feat_dir: Path,
    sam_cache_dir: Path,
    sam_embed_dir: Path,
    output_dir: Path,
    episodes_per_k: int = 5,
    device: str = "cuda",
    num_workers: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = num_workers if num_workers is not None else min(4, max(1, os.cpu_count() or 1))
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    sp = read_json(split_json)
    ep_lock = read_json(episodes_json)
    ep_by_k = _select_episodes(ep_lock, episodes_per_k=episodes_per_k)

    rec_by_id = {r["id"]: r for r in sp["records"]}
    query_recs_raw = [r for r in sp["records"] if r["partition"] == "query_set"]
    query_ids = [r["id"] for r in query_recs_raw]
    unique_sup_ids = sorted({sid for eps in ep_by_k.values() for ep in eps for sid in ep["support_ids"]})
    used_recs = [rec_by_id[rid] for rid in unique_sup_ids] + query_recs_raw
    all_ids = [r["id"] for r in used_recs]

    img_dir, msk_dir = resolve_pv663_dirs(None, None)
    print(f"[Ext Batch 1] 1/3 Extracting frozen DINOv3-SAT features for {len(all_ids)} images...", flush=True)
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

    print(f"[Ext Batch 1] 2/3 Loading SAM ViT-L and extracting cached embeddings for {len(all_ids)} images...", flush=True)
    sam = load_sam_vit_l(sam_cache_dir, device=run_device)
    extract_sam_embeddings_to_dir(sam, rgb_512_by_id, all_ids, sam_embed_dir, device=run_device)
    sam_embeds = LazySamEmbedMap(sam_embed_dir, device=run_device, preload_ids=unique_sup_ids)

    print("[Ext Batch 1] 3/3 Building DINOv3-SAT caches and affinity weights...", flush=True)
    caches = {
        key: build_pairwise_topk_cache_streaming(
            dino_feat_dir, key, mask_512_by_id, all_ids, unique_sup_ids, grid_hw=ghw, core_purity=0.85, top_m=3, device=run_device
        )
        for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64))
    }
    tokens_by_key = {
        key: LazyDiskTokenMap(dino_feat_dir, key, run_device, preload_ids=unique_sup_ids)
        for key in ("deep_32", "deep_64", "mid_64")
    }
    core_cfg = get_clean_core_config()
    aff_cpu_f16 = {
        rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=core_cfg.aff_tau).half().cpu()
        for rid in all_ids
    }

    class _LazyAffMap:
        def __contains__(self, k: object) -> bool:
            return k in aff_cpu_f16

        def __getitem__(self, k: str) -> torch.Tensor:
            return aff_cpu_f16[k].to(device=run_device, dtype=torch.float32)

    aff_map: Any = _LazyAffMap()

    persam_runner = PerSAMFRunner(sam, ft_steps=100, lr=0.05)
    matcher_runner = MatcherDINOv3Runner(sam, num_clusters=8, merge_margin=0.05)
    pipe_ours = ThreeModulePipeline(core_cfg, refiner_cfg=None)
    pbdc_cfg = PBDCConfig(enabled=True)

    methods_res: dict[str, dict[str, Any]] = defaultdict(dict)
    latency_ms: dict[str, dict[str, float]] = defaultdict(dict)

    use_fork = (workers > 1) and (sys.platform != "win32") and ("fork" in multiprocessing.get_all_start_methods())
    global _FW_RGB_BY_ID, _FW_PBDC_CFG, _FW_K_SHOTS
    _FW_RGB_BY_ID = rgb_512_by_id
    _FW_PBDC_CFG = pbdc_cfg

    for k in sorted(ep_by_k.keys()):
        t_k = time.perf_counter()
        eps = ep_by_k[k]
        ep_evals: dict[str, list[dict]] = {
            "PerSAM_F_SAM_ViTL": [],
            "Matcher_DINOv3SAT_SAM_ViTL": [],
            "Ours_v31_PBDC_Pure": [],
        }
        tot_sec = {m: 0.0 for m in ep_evals}
        n_q_tot = 0

        for ep_i, ep in enumerate(eps, 1):
            sup_ids = ep["support_ids"]
            n_q_tot += len(query_ids)

            # --- 1. PerSAM-F ---
            ts = time.perf_counter()
            ps_state = persam_runner.fit_episode(sam_embeds, mask_512_by_id, sup_ids)
            ps_masks, ps_probs, ps_s32 = {}, {}, {}
            for qid in query_ids:
                m, p, s32 = persam_runner.infer_query(sam_embeds[qid], ps_state)
                ps_masks[qid], ps_probs[qid], ps_s32[qid] = m, p, s32
            tot_sec["PerSAM_F_SAM_ViTL"] += time.perf_counter() - ts
            ep_evals["PerSAM_F_SAM_ViTL"].append(evaluate_pv663_episode(query_records, ps_masks, ps_probs, ps_s32))

            # --- 2. Matcher (DINOv3-SAT + SAM ViT-L) ---
            ts = time.perf_counter()
            mt_state = matcher_runner.fit_episode(tokens_by_key["deep_64"], mask_512_by_id, sup_ids)
            mt_masks, mt_probs, mt_s32 = {}, {}, {}
            for qid in query_ids:
                m, p, s32 = matcher_runner.infer_query(tokens_by_key["deep_64"][qid], sam_embeds[qid], mt_state)
                mt_masks[qid], mt_probs[qid], mt_s32[qid] = m, p, s32
            tot_sec["Matcher_DINOv3SAT_SAM_ViTL"] += time.perf_counter() - ts
            ep_evals["Matcher_DINOv3SAT_SAM_ViTL"].append(evaluate_pv663_episode(query_records, mt_masks, mt_probs, mt_s32))

            # --- 3. Ours (v3.1-PBDC-Pure) ---
            ts = time.perf_counter()
            out_a3 = pipe_ours.run_episode(caches, tokens_by_key, mask_512_by_id, aff_map, sup_ids, query_ids)
            p_a3, s32_a3 = out_a3.to_legacy_outputs()
            m_a3 = out_a3.coarse_masks_512()
            _FW_K_SHOTS = k
            tasks = [(qid, m_a3[qid], p_a3[qid].astype(np.float16)) for qid in query_ids]
            ours_masks: dict[str, np.ndarray] = {}
            if use_fork:
                ctx = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                    for qid, m_ref in ex.map(_pbdc_worker, tasks, chunksize=8):
                        ours_masks[qid] = m_ref
            else:
                for task in tasks:
                    qid, m_ref = _pbdc_worker(task)
                    ours_masks[qid] = m_ref
            tot_sec["Ours_v31_PBDC_Pure"] += time.perf_counter() - ts
            ep_evals["Ours_v31_PBDC_Pure"].append(evaluate_pv663_episode(query_records, ours_masks, p_a3, s32_a3))

            print(
                f"[Ext Batch 1 K={k}] Ep {ep_i}/{len(eps)} done ({(time.perf_counter()-t_k)/60:.1f}m for K={k})",
                flush=True,
            )

        for mname in ep_evals:
            methods_res[mname][str(k)] = aggregate_pv663_episodes(ep_evals[mname], expected_episodes=len(eps))
            latency_ms[mname][str(k)] = float(tot_sec[mname] * 1000.0 / max(1, n_q_tot))
            m = methods_res[mname][str(k)]["macro"]
            sm_r = methods_res[mname][str(k)]["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            print(
                f"  -> [K={k}] {mname:28s}: Dice={m['dice']:.4f} BF1@1={m['bf1_1px']:.4f} "
                f"BF1@2={m['boundary_f1']:.4f} Sm<5R={sm_r:.4f} ({latency_ms[mname][str(k)]:.1f} ms/img)",
                flush=True,
            )

    comparisons = {
        "Ours_vs_PerSAM_F": {
            str(k): compare_pv663_methods(methods_res["Ours_v31_PBDC_Pure"][str(k)], methods_res["PerSAM_F_SAM_ViTL"][str(k)])
            for k in (1, 3, 5)
        },
        "Ours_vs_Matcher_DINOv3SAT": {
            str(k): compare_pv663_methods(
                methods_res["Ours_v31_PBDC_Pure"][str(k)], methods_res["Matcher_DINOv3SAT_SAM_ViTL"][str(k)]
            )
            for k in (1, 3, 5)
        },
    }

    summary = {
        "schema": "pv663-external-batch1-v1",
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
        "latency_ms_per_img": dict(latency_ms),
    }
    write_json(output_dir / "pv663_ext_batch1_summary.json", summary)
    report = format_ext_batch1_report(summary)
    (output_dir / "pv663_ext_batch1_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PV663 External Model Comparison — Batch 1 (PerSAM-F & Matcher-DINOv3SAT).")
    parser.add_argument("--split-json", type=Path, default=REPO_ROOT / "configs" / "pv663_split_lock.json")
    parser.add_argument("--episodes-json", type=Path, default=REPO_ROOT / "configs" / "pv663_episodes_lock.json")
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/dinov3-source"))
    parser.add_argument("--dino-feat-dir", type=Path, default=Path("/tmp/pv663_ext_dino_feats"))
    parser.add_argument("--sam-cache-dir", type=Path, default=Path("/tmp/sam_weights"))
    parser.add_argument("--sam-embed-dir", type=Path, default=Path("/tmp/pv663_ext_sam_embeds"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "external_baselines" / "results" / "pv663_batch1")
    parser.add_argument("--episodes-per-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    run_pv663_ext_batch1(
        split_json=args.split_json,
        episodes_json=args.episodes_json,
        source_dir=args.source_dir,
        dino_feat_dir=args.dino_feat_dir,
        sam_cache_dir=args.sam_cache_dir,
        sam_embed_dir=args.sam_embed_dir,
        output_dir=args.output_dir,
        episodes_per_k=args.episodes_per_k,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
