from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
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

from pvfs.data import area_target_32
from pvfs.fast_bank import compute_dino_affinity_weights
from pvfs.io import read_json, write_json
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
    aggregate_pv663_episodes,
    build_pairwise_topk_cache_streaming,
    compare_pv663_methods,
    evaluate_pv663_episode,
    extract_pv663_features_to_dir,
    resolve_pv663_dirs,
)

# Shared read-only Linux fork worker state for fast parallel pixel refinement across 510 query images
_FW_RGB_BY_ID: dict[str, np.ndarray] = {}
_FW_REFINER_SPECS: list[tuple[str, Any]] = []
_FW_ALLOW_OUT: bool = True
_FW_P_FLOOR: float | None = None
_FW_K_SHOTS: int = 5


def _refine_multi_worker(
    args: tuple[str, np.ndarray, np.ndarray],
) -> tuple[str, dict[str, np.ndarray], dict[str, dict[str, float]]]:
    qid, m0, p0_f16 = args
    p0 = p0_f16.astype(np.float32)
    rgb = _FW_RGB_BY_ID[qid]
    masks_out: dict[str, np.ndarray] = {}
    diags_out: dict[str, dict[str, float]] = {}

    for name, cfg in _FW_REFINER_SPECS:
        if isinstance(cfg, PBDCConfig):
            o = refine_mask_pbdc(
                m0,
                rgb,
                p0,
                k_shots=_FW_K_SHOTS,
                p_floor_loo=_FW_P_FLOOR,
                cfg=cfg,
            )
            masks_out[name] = o["mask"]
            diags_out[name] = {
                "total_arc": float(o["total_arc_len_px"]),
                "moved_arc": float(o["moved_arc_len_px"]),
                "abs_d_wsum": float(o["mean_abs_d_moved"] * o["moved_arc_len_px"]),
                "net_area": float(o["net_area_delta_px"]),
                "topo_reverts": float(o["topology_reverts"]),
                "loc_fallbacks": float(o["localized_fallbacks"]),
            }
        else:
            o = refine_mask_deformable_grad(
                m0,
                rgb,
                p0,
                allow_outward=_FW_ALLOW_OUT,
                p_floor=_FW_P_FLOOR,
                tau_grad_per_px=0.20,
                cfg=cfg,
            )
            masks_out[name] = o["mask"]
            diags_out[name] = {
                "total_arc": float(o["total_arc_len_px"]),
                "moved_arc": float(o["moved_arc_len_px"]),
                "abs_d_wsum": float(o["mean_abs_d_moved"] * o["moved_arc_len_px"]),
                "net_area": float(o["net_area_delta_px"]),
                "topo_reverts": float(o["topology_reverts"]),
                "loc_fallbacks": 0.0,
            }
    return qid, masks_out, diags_out


def _summarize_diag(acc: dict[str, float], n_queries: int) -> dict[str, Any]:
    return {
        "moved_arc_fraction": float(acc["moved_arc"] / max(1.0, acc["total_arc"])),
        "mean_abs_d_px": float(acc["abs_d_wsum"] / max(1.0, acc["moved_arc"])),
        "mean_net_area_delta_px": float(acc["net_area"] / max(1, n_queries)),
        "topology_reverts": int(acc["topo_reverts"]),
        "localized_fallbacks": int(acc.get("loc_fallbacks", 0)),
    }


def _select_episodes(ep_lock: dict[str, Any], episodes_per_k: int) -> dict[int, list[dict[str, Any]]]:
    ep_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ep in ep_lock["episodes"]:
        k = int(ep["k"])
        if len(ep_by_k[k]) < episodes_per_k:
            ep_by_k[k].append(ep)
    return {k: ep_by_k[k] for k in sorted(ep_by_k)}


def format_ablation_report(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    lines = [
        "=== PV663 (0.8m) FULL ABLATION REPORT (v3.0-pbdc-final) ===",
        f"Split: {meta['support_sites']} support sites ({meta['support_pool_images']} imgs, {meta['unique_support_used']} used) "
        f"/ {meta['query_sites']} query sites ({meta['query_images']} imgs) | min_dist={meta['min_dist_km']:.2f} km",
        f"Episodes/K: {meta['episodes_per_k']} | Total Elapsed: {summary['elapsed_sec']/60:.1f} min",
    ]

    lines.append("\n================ TABLE 1: MODULE-LEVEL STEPWISE ABLATION (K=1, 3, 5) ================")
    t1_order = [
        "A0_B0_32_Ref",
        "A1_Plus_M1",
        "A2_Plus_M2_SR",
        "A3_Plus_M3A_Aff",
        "A4_Full_PBDC",
        "A4_Old_DeformGrad",
    ]
    for k in ("1", "3", "5"):
        lines.append(f"\n--- K={k} (116 geographic sites, 510 query images x {meta['episodes_per_k']} eps) ---")
        for mname in t1_order:
            if mname not in summary["methods"] or k not in summary["methods"][mname]:
                continue
            res = summary["methods"][mname][k]
            m = res["macro"]
            sm_r = res["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            lines.append(
                f"  {mname:20s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
                f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} Sm<5R={sm_r:.4f}"
            )

    lines.append("\n--- Table 1 Paired Site Bootstrap Stepwise Deltas (116 Sites) ---")
    for cname in (
        "A1_vs_A0",
        "A2_vs_A1",
        "A3_vs_A2",
        "A4_PBDC_vs_A3",
        "A4_PBDC_vs_A0",
        "A4_Old_vs_A3",
        "A4_PBDC_vs_A4_Old",
    ):
        if cname not in summary["comparisons"]:
            continue
        lines.append(f"[{cname}]")
        for k in ("1", "3", "5"):
            if k not in summary["comparisons"][cname]:
                continue
            c = summary["comparisons"][cname][k]
            lines.append(
                f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} "
                f"({c['dice']['improved_sites']}/{c['dice']['total_sites']}) | "
                f"dBF1@1={c['bf1_1px']['mean_delta']:+.4f} CI{[round(x,4) for x in c['bf1_1px']['ci95']]} | "
                f"dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x,4) for x in c['boundary_f1']['ci95']]} | "
                f"dSm<5R={c['small_lt5pct_recall_delta']:+.4f}"
            )

    lines.append("\n================ TABLE 2: CORE MODULE-INTERNAL ABLATION (K=3, 5) ================")
    t2_order = [
        "A3_Plus_M3A_Aff",
        "I1_M1_wo_MidFusion",
        "I2_M2_wo_SR_Gating",
        "A4_Full_PBDC",
        "I3_PBDC_wo_AsymCore",
        "I4_PBDC_wo_UncertLock",
        "I5_PBDC_wo_SectorFB",
    ]
    for k in ("3", "5"):
        lines.append(f"\n--- K={k} ---")
        for mname in t2_order:
            if mname not in summary["methods"] or k not in summary["methods"][mname]:
                continue
            res = summary["methods"][mname][k]
            m = res["macro"]
            sm_r = res["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            lines.append(
                f"  {mname:22s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
                f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} Sm<5R={sm_r:.4f}"
            )

    lines.append("\n--- Table 2 Paired Site Bootstrap Deltas (Full vs Ablated, positive = mechanism helps) ---")
    for cname in (
        "A3_vs_I1_wo_MidFusion",
        "A3_vs_I2_wo_SR_Gating",
        "A4_vs_I3_wo_AsymCore",
        "A4_vs_I4_wo_UncertLock",
        "A4_vs_I5_wo_SectorFB",
    ):
        if cname not in summary["comparisons"]:
            continue
        lines.append(f"[{cname}]")
        for k in ("3", "5"):
            if k not in summary["comparisons"][cname]:
                continue
            c = summary["comparisons"][cname][k]
            lines.append(
                f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} | "
                f"dBF1@1={c['bf1_1px']['mean_delta']:+.4f} CI{[round(x,4) for x in c['bf1_1px']['ci95']]} | "
                f"dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x,4) for x in c['boundary_f1']['ci95']]} | "
                f"dSm<5R={c['small_lt5pct_recall_delta']:+.4f}"
            )

    lines.append("\n--- Pixel Refiner Movement & Topology Diagnostics ---")
    for k in ("1", "3", "5"):
        if k not in summary["diagnostics"]:
            continue
        parts = []
        for rname, d in summary["diagnostics"][k].items():
            parts.append(
                f"{rname}[moved={d['moved_arc_fraction']:.1%} |d|={d['mean_abs_d_px']:.2f}px "
                f"area={d['mean_net_area_delta_px']:+.0f} locFB={d['localized_fallbacks']} rev={d['topology_reverts']}]"
            )
        lines.append(f"  K={k}: " + " | ".join(parts))

    return "\n".join(lines) + "\n"


def run_pv663_full_ablation(
    *,
    split_json: Path,
    episodes_json: Path,
    deform_config_json: Path,
    source_dir: Path,
    feat_cache_dir: Path,
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
    print(
        f"[PV663 Full Ablation] Extracting/loading features for {len(unique_sup_ids)} support + {len(query_ids)} query "
        f"({len(all_ids)} total) on {run_device}...",
        flush=True,
    )
    rgb_512_by_id, mask_512_by_id = extract_pv663_features_to_dir(
        used_recs,
        image_dir=img_dir,
        mask_dir=msk_dir,
        feat_dir=feat_cache_dir,
        source_dir=source_dir,
        device=run_device,
    )

    query_records = [
        {**r, "mask": mask_512_by_id[r["id"]], "y32": area_target_32(mask_512_by_id[r["id"]])}
        for r in query_recs_raw
    ]

    print("[PV663 Full Ablation] Building streaming pairwise top-k caches...", flush=True)
    caches = {
        key: build_pairwise_topk_cache_streaming(
            feat_cache_dir,
            key,
            mask_512_by_id,
            all_ids,
            unique_sup_ids,
            grid_hw=ghw,
            core_purity=0.85,
            top_m=3,
            device=run_device,
        )
        for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64))
    }
    tokens_by_key = {
        key: LazyDiskTokenMap(feat_cache_dir, key, run_device, preload_ids=unique_sup_ids)
        for key in ("deep_32", "deep_64", "mid_64")
    }

    # Precompute affinity weights once for the 510 query + ~35 support images (stored fp16 on CPU to save VRAM)
    core_cfg = get_clean_core_config()
    print(f"[PV663 Full Ablation] Precomputing 5x5 DINO affinity weights (tau={core_cfg.aff_tau})...", flush=True)
    aff_cpu_f16: dict[str, torch.Tensor] = {}
    for rid in all_ids:
        tok64 = tokens_by_key["deep_64"][rid]
        w_aff = compute_dino_affinity_weights(tok64, tau=core_cfg.aff_tau)
        aff_cpu_f16[rid] = w_aff.half().cpu()

    class _LazyAffMap:
        def __contains__(self, k: object) -> bool:
            return k in aff_cpu_f16

        def __getitem__(self, k: str) -> torch.Tensor:
            return aff_cpu_f16[k].to(device=run_device, dtype=torch.float32)

    aff_map: Any = _LazyAffMap()

    # Define non-PBDC pipelines
    pipelines: dict[str, ThreeModulePipeline] = {
        "A0_B0_32_Ref": ThreeModulePipeline(get_b0_32_config(), refiner_cfg=None),
        "A1_Plus_M1": ThreeModulePipeline(
            replace(core_cfg, name="A1_Plus_M1", use_self_rescue=False, use_affinity=False), refiner_cfg=None
        ),
        "A2_Plus_M2_SR": ThreeModulePipeline(
            replace(core_cfg, name="A2_Plus_M2_SR", use_self_rescue=True, use_affinity=False), refiner_cfg=None
        ),
        "A3_Plus_M3A_Aff": ThreeModulePipeline(core_cfg, refiner_cfg=None),
        "I1_M1_wo_MidFusion": ThreeModulePipeline(
            replace(core_cfg, name="I1_M1_wo_MidFusion", use_mid_fusion=False), refiner_cfg=None
        ),
        "I2_M2_wo_SR_Gating": ThreeModulePipeline(
            replace(core_cfg, name="I2_M2_wo_SR_Gating", ablate_no_sr_gating=True), refiner_cfg=None
        ),
    }

    old_cfg = DeformableBoundaryConfig.from_json(deform_config_json, mode="deformable_grad")
    pbdc_full_cfg = PBDCConfig(enabled=True)
    pbdc_i3_cfg = PBDCConfig(enabled=True, ablate_undirected_grad=True)
    pbdc_i4_cfg = PBDCConfig(enabled=True, ablate_no_uncert_lock=True)
    pbdc_i5_cfg = PBDCConfig(enabled=True, ablate_whole_loop_fallback=True)

    methods_res: dict[str, dict[str, Any]] = defaultdict(dict)
    diag_by_k: dict[str, dict[str, Any]] = {}

    use_fork = (workers > 1) and (sys.platform != "win32") and ("fork" in multiprocessing.get_all_start_methods())
    global _FW_RGB_BY_ID, _FW_REFINER_SPECS, _FW_ALLOW_OUT, _FW_P_FLOOR, _FW_K_SHOTS
    _FW_RGB_BY_ID = rgb_512_by_id

    for k in sorted(ep_by_k.keys()):
        t_k = time.perf_counter()
        eps = ep_by_k[k]
        pipe_names = ["A0_B0_32_Ref", "A1_Plus_M1", "A2_Plus_M2_SR", "A3_Plus_M3A_Aff"]
        if k in (3, 5):
            pipe_names += ["I1_M1_wo_MidFusion", "I2_M2_wo_SR_Gating"]

        refiner_specs: list[tuple[str, Any]] = [
            ("A4_Full_PBDC", pbdc_full_cfg),
            ("A4_Old_DeformGrad", old_cfg),
        ]
        if k in (3, 5):
            refiner_specs += [
                ("I3_PBDC_wo_AsymCore", pbdc_i3_cfg),
                ("I4_PBDC_wo_UncertLock", pbdc_i4_cfg),
                ("I5_PBDC_wo_SectorFB", pbdc_i5_cfg),
            ]

        all_row_names = pipe_names + [r[0] for r in refiner_specs]
        ep_evals: dict[str, list[dict]] = {name: [] for name in all_row_names}
        diag_acc: dict[str, dict[str, float]] = {r[0]: defaultdict(float) for r in refiner_specs}
        n_q_tot = 0

        for ep_i, ep in enumerate(eps, 1):
            sup_ids = ep["support_ids"]
            n_q_tot += len(query_ids)

            # 1. Run non-PBDC pipelines (fast GPU/cached inference)
            p_a3: dict[str, np.ndarray] = {}
            s32_a3: dict[str, np.ndarray] = {}
            m_a3: dict[str, np.ndarray] = {}
            for pname in pipe_names:
                out = pipelines[pname].run_episode(caches, tokens_by_key, mask_512_by_id, aff_map, sup_ids, query_ids)
                p_out, s32_out = out.to_legacy_outputs()
                m_out = out.coarse_masks_512()
                ep_evals[pname].append(evaluate_pv663_episode(query_records, m_out, p_out, s32_out))
                if pname == "A3_Plus_M3A_Aff":
                    p_a3, s32_a3, m_a3 = p_out, s32_out, m_out

            # 2. Support-only LOO p_floor calibration for Module ③
            loo_probs = None
            if k >= 2:
                loo_probs = {}
                for sid in sup_ids:
                    rem = [s for s in sup_ids if s != sid]
                    o_l = pipelines["A3_Plus_M3A_Aff"].run_episode(
                        caches, tokens_by_key, mask_512_by_id, aff_map, rem, [sid]
                    )
                    loo_probs[sid] = o_l.boundary_by_id[sid].p_coarse
            allow_out, p_floor = calibrate_deform_p_floor_loo(sup_ids, mask_512_by_id, loo_probs)

            # 3. Run all pixel refiners in parallel on A3's (m_a3, p_a3)
            _FW_REFINER_SPECS = refiner_specs
            _FW_ALLOW_OUT = allow_out
            _FW_P_FLOOR = p_floor
            _FW_K_SHOTS = k
            tasks = [(qid, m_a3[qid], p_a3[qid].astype(np.float16)) for qid in query_ids]
            ref_masks: dict[str, dict[str, np.ndarray]] = {r[0]: {} for r in refiner_specs}

            if use_fork:
                ctx = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                    for qid, m_dict, d_dict in ex.map(_refine_multi_worker, tasks, chunksize=8):
                        for rname in m_dict:
                            ref_masks[rname][qid] = m_dict[rname]
                            for dk, dv in d_dict[rname].items():
                                diag_acc[rname][dk] += dv
            else:
                for task in tasks:
                    qid, m_dict, d_dict = _refine_multi_worker(task)
                    for rname in m_dict:
                        ref_masks[rname][qid] = m_dict[rname]
                        for dk, dv in d_dict[rname].items():
                            diag_acc[rname][dk] += dv

            for rname, _ in refiner_specs:
                ep_evals[rname].append(evaluate_pv663_episode(query_records, ref_masks[rname], p_a3, s32_a3))

            print(
                f"[PV663 K={k}] Ep {ep_i}/{len(eps)} done ({(time.perf_counter()-t_k)/60:.1f}m for K={k})",
                flush=True,
            )

        for name in all_row_names:
            methods_res[name][str(k)] = aggregate_pv663_episodes(ep_evals[name], expected_episodes=len(eps))
            m = methods_res[name][str(k)]["macro"]
            sm_r = methods_res[name][str(k)]["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            print(
                f"  -> [K={k}] {name:22s}: Dice={m['dice']:.4f} BF1@1={m['bf1_1px']:.4f} "
                f"BF1@2={m['boundary_f1']:.4f} Prec={m['precision']:.4f} Sm<5R={sm_r:.4f}",
                flush=True,
            )
        diag_by_k[str(k)] = {rname: _summarize_diag(diag_acc[rname], n_q_tot) for rname, _ in refiner_specs}

    # Compute all paired site bootstrap comparisons across all 116 geographic sites
    comp_pairs = [
        ("A1_vs_A0", "A1_Plus_M1", "A0_B0_32_Ref", ("1", "3", "5")),
        ("A2_vs_A1", "A2_Plus_M2_SR", "A1_Plus_M1", ("1", "3", "5")),
        ("A3_vs_A2", "A3_Plus_M3A_Aff", "A2_Plus_M2_SR", ("1", "3", "5")),
        ("A4_PBDC_vs_A3", "A4_Full_PBDC", "A3_Plus_M3A_Aff", ("1", "3", "5")),
        ("A4_PBDC_vs_A0", "A4_Full_PBDC", "A0_B0_32_Ref", ("1", "3", "5")),
        ("A4_Old_vs_A3", "A4_Old_DeformGrad", "A3_Plus_M3A_Aff", ("1", "3", "5")),
        ("A4_PBDC_vs_A4_Old", "A4_Full_PBDC", "A4_Old_DeformGrad", ("1", "3", "5")),
        ("A3_vs_I1_wo_MidFusion", "A3_Plus_M3A_Aff", "I1_M1_wo_MidFusion", ("3", "5")),
        ("A3_vs_I2_wo_SR_Gating", "A3_Plus_M3A_Aff", "I2_M2_wo_SR_Gating", ("3", "5")),
        ("A4_vs_I3_wo_AsymCore", "A4_Full_PBDC", "I3_PBDC_wo_AsymCore", ("3", "5")),
        ("A4_vs_I4_wo_UncertLock", "A4_Full_PBDC", "I4_PBDC_wo_UncertLock", ("3", "5")),
        ("A4_vs_I5_wo_SectorFB", "A4_Full_PBDC", "I5_PBDC_wo_SectorFB", ("3", "5")),
    ]
    comparisons: dict[str, dict[str, Any]] = {}
    for cname, cand, ref, ks in comp_pairs:
        comparisons[cname] = {
            k: compare_pv663_methods(methods_res[cand][k], methods_res[ref][k])
            for k in ks
            if k in methods_res[cand] and k in methods_res[ref]
        }

    summary = {
        "schema": "pv663-full-ablation-v3.0",
        "elapsed_sec": float(time.perf_counter() - t0),
        "meta": {
            "total_images": sp["summary"]["total_images"],
            "support_sites": sp["summary"]["support_pool_sites"],
            "support_pool_images": sp["summary"]["support_pool"]["total"],
            "unique_support_used": len(unique_sup_ids),
            "query_sites": sp["summary"]["query_set_sites"],
            "query_images": len(query_ids),
            "min_dist_km": float(sp["min_support_query_distance_km"]),
            "episodes_per_k": episodes_per_k,
        },
        "methods": dict(methods_res),
        "comparisons": comparisons,
        "diagnostics": diag_by_k,
    }
    write_json(output_dir / "pv663_full_ablation_summary.json", summary)
    report = format_ablation_report(summary)
    (output_dir / "pv663_full_ablation_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full 663-image PV663 Table 1 + Table 2 ablation study.")
    parser.add_argument("--split-json", type=Path, default=REPO_ROOT / "configs" / "pv663_split_lock.json")
    parser.add_argument("--episodes-json", type=Path, default=REPO_ROOT / "configs" / "pv663_episodes_lock.json")
    parser.add_argument("--deform-config-json", type=Path, default=REPO_ROOT / "configs" / "three_module_exp_m3_deformgrad.json")
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/dinov3-source"))
    parser.add_argument("--feat-cache-dir", type=Path, default=Path("/tmp/pv663_full_feats"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "pv663_full_ablation_v3")
    parser.add_argument("--episodes-per-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    run_pv663_full_ablation(
        split_json=args.split_json,
        episodes_json=args.episodes_json,
        deform_config_json=args.deform_config_json,
        source_dir=args.source_dir,
        feat_cache_dir=args.feat_cache_dir,
        output_dir=args.output_dir,
        episodes_per_k=args.episodes_per_k,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
