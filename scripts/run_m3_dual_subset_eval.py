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

from pvfs.data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache, compute_dino_affinity_weights, to_tokens_norm
from pvfs.io import read_json, write_json
from pvfs.metrics import aggregate_binary_episodes, compare_methods, evaluate_binary_episode
from pvfs.modules import (
    DeformableBoundaryConfig,
    PBDCConfig,
    ThreeModulePipeline,
    calibrate_deform_p_floor_loo,
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

# Shared fork-worker state for fast parallel refinement on Linux Kaggle
_FW_RGB_BY_ID: dict[str, np.ndarray] = {}
_FW_OLD_CFG: DeformableBoundaryConfig | None = None
_FW_PBDC_CFG: PBDCConfig | None = None
_FW_ALLOW_OUT: bool = True
_FW_P_FLOOR: float | None = None
_FW_K_SHOTS: int = 5


def _refine_both_worker(args: tuple[str, np.ndarray, np.ndarray]) -> tuple[str, np.ndarray, dict[str, float], np.ndarray, dict[str, float]]:
    qid, m0, p0_f16 = args
    p0 = p0_f16.astype(np.float32)
    rgb = _FW_RGB_BY_ID[qid]
    assert _FW_OLD_CFG is not None and _FW_PBDC_CFG is not None

    o_old = refine_mask_deformable_grad(
        m0,
        rgb,
        p0,
        allow_outward=_FW_ALLOW_OUT,
        p_floor=_FW_P_FLOOR,
        tau_grad_per_px=0.20,
        cfg=_FW_OLD_CFG,
    )
    o_new = refine_mask_pbdc(
        m0,
        rgb,
        p0,
        k_shots=_FW_K_SHOTS,
        p_floor_loo=_FW_P_FLOOR,
        cfg=_FW_PBDC_CFG,
    )
    d_old = {
        "total_arc": float(o_old["total_arc_len_px"]),
        "moved_arc": float(o_old["moved_arc_len_px"]),
        "abs_d_wsum": float(o_old["mean_abs_d_moved"] * o_old["moved_arc_len_px"]),
        "net_area": float(o_old["net_area_delta_px"]),
        "topo_reverts": float(o_old["topology_reverts"]),
    }
    d_new = {
        "total_arc": float(o_new["total_arc_len_px"]),
        "moved_arc": float(o_new["moved_arc_len_px"]),
        "abs_d_wsum": float(o_new["mean_abs_d_moved"] * o_new["moved_arc_len_px"]),
        "net_area": float(o_new["net_area_delta_px"]),
        "topo_reverts": float(o_new["topology_reverts"]),
        "loc_fallbacks": float(o_new["localized_fallbacks"]),
    }
    return qid, o_old["mask"], d_old, o_new["mask"], d_new


def _summarize_diag(acc: dict[str, float], n_queries: int) -> dict[str, Any]:
    return {
        "moved_arc_fraction": float(acc["moved_arc"] / max(1.0, acc["total_arc"])),
        "mean_abs_d_px": float(acc["abs_d_wsum"] / max(1.0, acc["moved_arc"])),
        "mean_net_area_delta_px": float(acc["net_area"] / max(1, n_queries)),
        "topology_reverts": int(acc["topo_reverts"]),
        "localized_fallbacks": int(acc.get("loc_fallbacks", 0)),
    }


def run_pv100_2m_subset(
    *,
    split_dir: Path,
    cache_pt: Path,
    deform_config_json: Path,
    episodes_per_fold: int = 4,
    device: str = "cuda",
    num_workers: int = 4,
) -> dict[str, Any]:
    """Evaluate M3_Off vs M3_DeformGrad_Old vs M3_PBDC on PV-100scence (2m) 50-site subset."""
    t0 = time.perf_counter()
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    img_dir, lbl_dir = resolve_data_dirs(None, None)

    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    records_by_id: dict[str, dict[str, Any]] = {}
    mask_by_id: dict[str, np.ndarray] = {}
    rgb_hwc_by_id: dict[str, np.ndarray] = {}
    for row in dev_records:
        rid = row["id"]
        rgb_chw = read_verified_rgb(img_dir, row)
        rgb_hwc_by_id[rid] = np.transpose(rgb_chw, (1, 2, 0)).copy()
        mask = read_verified_mask(lbl_dir, row)
        mask_by_id[rid] = mask
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}

    caches: dict[str, dict] = {}
    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64)):
        toks = {rid: to_tokens_norm(feats[rid][key], run_device) for rid in all_ids}
        caches[key] = build_pairwise_topk_cache(toks, mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        tokens_by_key[key] = toks
    core_cfg = get_clean_core_config()
    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=core_cfg.aff_tau) for rid in all_ids}

    pipe_core = ThreeModulePipeline(core_cfg)
    old_cfg = DeformableBoundaryConfig.from_json(deform_config_json, mode="deformable_grad")
    pbdc_cfg = PBDCConfig(enabled=True)

    # Select first `episodes_per_fold` per fold per K (deterministic, balanced across all 4 folds / 50 sites)
    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    by_kf: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        by_kf[(int(ep["k"]), int(ep["fold"]))].append(ep)
    for (k, f), lst in sorted(by_kf.items()):
        ep_by_k[k].extend(lst[:episodes_per_fold])

    variants = ("M3_Off", "M3_DeformGrad_Old", "M3_PBDC")
    methods_res: dict[str, dict[str, Any]] = {v: {} for v in variants}
    diag_by_k: dict[str, dict[str, Any]] = {}

    use_fork = (num_workers > 1) and (sys.platform != "win32") and ("fork" in multiprocessing.get_all_start_methods())
    global _FW_RGB_BY_ID, _FW_OLD_CFG, _FW_PBDC_CFG, _FW_ALLOW_OUT, _FW_P_FLOOR, _FW_K_SHOTS
    _FW_RGB_BY_ID = rgb_hwc_by_id
    _FW_OLD_CFG = old_cfg
    _FW_PBDC_CFG = pbdc_cfg

    for k in sorted(ep_by_k.keys()):
        eps = ep_by_k[k]
        ep_evals: dict[str, list[dict]] = {v: [] for v in variants}
        acc_old: dict[str, float] = defaultdict(float)
        acc_new: dict[str, float] = defaultdict(float)
        n_q_tot = 0

        for ep in eps:
            sup_ids = ep["support_ids"]
            q_ids = ep["query_ids"]
            q_recs = [records_by_id[qid] for qid in q_ids]
            n_q_tot += len(q_ids)

            out_off = pipe_core.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
            p_off, s32_off = out_off.to_legacy_outputs()
            m_off = out_off.coarse_masks_512()

            loo_probs = None
            if k >= 2:
                loo_probs = {}
                for sid in sup_ids:
                    rem = [s for s in sup_ids if s != sid]
                    o_l = pipe_core.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, rem, [sid])
                    loo_probs[sid] = o_l.boundary_by_id[sid].p_coarse
            allow_out, p_floor = calibrate_deform_p_floor_loo(sup_ids, mask_by_id, loo_probs)

            m_old: dict[str, np.ndarray] = {}
            m_new: dict[str, np.ndarray] = {}
            if use_fork:
                _FW_ALLOW_OUT = allow_out
                _FW_P_FLOOR = p_floor
                _FW_K_SHOTS = k
                tasks = [(qid, m_off[qid], p_off[qid].astype(np.float16)) for qid in q_ids]
                ctx = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
                    for qid, mo, do, mn, dn in ex.map(_refine_both_worker, tasks, chunksize=4):
                        m_old[qid] = mo
                        m_new[qid] = mn
                        for dk, dv in do.items():
                            acc_old[dk] += dv
                        for dk, dv in dn.items():
                            acc_new[dk] += dv
            else:
                _FW_ALLOW_OUT = allow_out
                _FW_P_FLOOR = p_floor
                _FW_K_SHOTS = k
                for qid in q_ids:
                    _, mo, do, mn, dn = _refine_both_worker((qid, m_off[qid], p_off[qid].astype(np.float16)))
                    m_old[qid] = mo
                    m_new[qid] = mn
                    for dk, dv in do.items():
                        acc_old[dk] += dv
                    for dk, dv in dn.items():
                        acc_new[dk] += dv

            ep_evals["M3_Off"].append(evaluate_binary_episode(q_recs, m_off, p_off, s32_off))
            ep_evals["M3_DeformGrad_Old"].append(evaluate_binary_episode(q_recs, m_old, p_off, s32_off))
            ep_evals["M3_PBDC"].append(evaluate_binary_episode(q_recs, m_new, p_off, s32_off))

        for v in variants:
            methods_res[v][str(k)] = aggregate_binary_episodes(ep_evals[v], episodes_per_fold=episodes_per_fold)
            m = methods_res[v][str(k)]["macro"]
            print(
                f"[PV100-2m K={k}] {v:18s}: Dice={m['dice']:.4f} BF1@1={m['bf1_1px']:.4f} "
                f"BF1@2={m['boundary_f1']:.4f} Rec={m['recall']:.4f} Prec={m['precision']:.4f} NegFP={m['neg_fp']:.4f}",
                flush=True,
            )
        diag_by_k[str(k)] = {
            "M3_DeformGrad_Old": _summarize_diag(acc_old, n_q_tot),
            "M3_PBDC": _summarize_diag(acc_new, n_q_tot),
        }

    comparisons = {}
    for cname, cand_v, ref_v in (
        ("M3_PBDC_vs_M3_Off", "M3_PBDC", "M3_Off"),
        ("M3_DeformGrad_Old_vs_M3_Off", "M3_DeformGrad_Old", "M3_Off"),
        ("M3_PBDC_vs_M3_DeformGrad_Old", "M3_PBDC", "M3_DeformGrad_Old"),
    ):
        comparisons[cname] = {
            str(k): compare_methods(methods_res[cand_v][str(k)], methods_res[ref_v][str(k)])
            for k in sorted(ep_by_k.keys())
        }
    return {
        "dataset": "PV-100scence (2m, 50 Dev sites, 4 eps/fold)",
        "elapsed_sec": float(time.perf_counter() - t0),
        "methods": methods_res,
        "comparisons": comparisons,
        "diagnostics": diag_by_k,
    }


def run_pv663_08m_subset(
    *,
    subset_json: Path,
    deform_config_json: Path,
    source_dir: Path,
    feat_cache_dir: Path,
    device: str = "cuda",
    num_workers: int = 4,
) -> dict[str, Any]:
    """Evaluate M3_Off vs M3_DeformGrad_Old vs M3_PBDC on PV663 (0.8m) 48-site stratified subset."""
    t0 = time.perf_counter()
    sub = read_json(subset_json)
    img_dir, msk_dir = resolve_pv663_dirs(None, None)
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_recs = list(sub["support_records"]) + list(sub["query_records"])
    rgb_512_by_id, mask_512_by_id = extract_pv663_features_to_dir(
        all_recs,
        image_dir=img_dir,
        mask_dir=msk_dir,
        feat_dir=feat_cache_dir,
        source_dir=source_dir,
        device=run_device,
    )

    unique_sup_ids: list[str] = list(sub["unique_support_ids"])
    query_ids: list[str] = list(sub["query_ids"])
    needed_ids = sorted(set(unique_sup_ids) | set(query_ids))
    rec_by_id = {r["id"]: r for r in all_recs}
    records_eval = {rid: {**rec_by_id[rid], "mask": mask_512_by_id[rid], "y32": area_target_32(mask_512_by_id[rid])} for rid in needed_ids}
    query_recs = [records_eval[qid] for qid in query_ids]

    caches = {
        key: build_pairwise_topk_cache_streaming(
            feat_cache_dir,
            key,
            mask_512_by_id,
            needed_ids,
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

    pipe_core = ThreeModulePipeline(get_clean_core_config())
    old_cfg = DeformableBoundaryConfig.from_json(deform_config_json, mode="deformable_grad")
    pbdc_cfg = PBDCConfig(enabled=True)

    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    for ep in sub["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    variants = ("M3_Off", "M3_DeformGrad_Old", "M3_PBDC")
    methods_res: dict[str, dict[str, Any]] = {v: {} for v in variants}
    diag_by_k: dict[str, dict[str, Any]] = {}

    use_fork = (num_workers > 1) and (sys.platform != "win32") and ("fork" in multiprocessing.get_all_start_methods())
    global _FW_RGB_BY_ID, _FW_OLD_CFG, _FW_PBDC_CFG, _FW_ALLOW_OUT, _FW_P_FLOOR, _FW_K_SHOTS
    _FW_RGB_BY_ID = rgb_512_by_id
    _FW_OLD_CFG = old_cfg
    _FW_PBDC_CFG = pbdc_cfg

    for k in sorted(ep_by_k.keys()):
        eps = ep_by_k[k]
        ep_evals: dict[str, list[dict]] = {v: [] for v in variants}
        acc_old: dict[str, float] = defaultdict(float)
        acc_new: dict[str, float] = defaultdict(float)
        n_q_tot = 0

        for ep in eps:
            sup_ids = ep["support_ids"]
            n_q_tot += len(query_ids)

            out_off = pipe_core.run_episode(caches, tokens_by_key, mask_512_by_id, None, sup_ids, query_ids)
            p_off, s32_off = out_off.to_legacy_outputs()
            m_off = out_off.coarse_masks_512()

            loo_probs = None
            if k >= 2:
                loo_probs = {}
                for sid in sup_ids:
                    rem = [s for s in sup_ids if s != sid]
                    o_l = pipe_core.run_episode(caches, tokens_by_key, mask_512_by_id, None, rem, [sid])
                    loo_probs[sid] = o_l.boundary_by_id[sid].p_coarse
            allow_out, p_floor = calibrate_deform_p_floor_loo(sup_ids, mask_512_by_id, loo_probs)

            m_old: dict[str, np.ndarray] = {}
            m_new: dict[str, np.ndarray] = {}
            if use_fork:
                _FW_ALLOW_OUT = allow_out
                _FW_P_FLOOR = p_floor
                _FW_K_SHOTS = k
                tasks = [(qid, m_off[qid], p_off[qid].astype(np.float16)) for qid in query_ids]
                ctx = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
                    for qid, mo, do, mn, dn in ex.map(_refine_both_worker, tasks, chunksize=6):
                        m_old[qid] = mo
                        m_new[qid] = mn
                        for dk, dv in do.items():
                            acc_old[dk] += dv
                        for dk, dv in dn.items():
                            acc_new[dk] += dv
            else:
                _FW_ALLOW_OUT = allow_out
                _FW_P_FLOOR = p_floor
                _FW_K_SHOTS = k
                for qid in query_ids:
                    _, mo, do, mn, dn = _refine_both_worker((qid, m_off[qid], p_off[qid].astype(np.float16)))
                    m_old[qid] = mo
                    m_new[qid] = mn
                    for dk, dv in do.items():
                        acc_old[dk] += dv
                    for dk, dv in dn.items():
                        acc_new[dk] += dv

            ep_evals["M3_Off"].append(evaluate_pv663_episode(query_recs, m_off, p_off, s32_off))
            ep_evals["M3_DeformGrad_Old"].append(evaluate_pv663_episode(query_recs, m_old, p_off, s32_off))
            ep_evals["M3_PBDC"].append(evaluate_pv663_episode(query_recs, m_new, p_off, s32_off))

        for v in variants:
            methods_res[v][str(k)] = aggregate_pv663_episodes(ep_evals[v], expected_episodes=len(eps))
            m = methods_res[v][str(k)]["macro"]
            print(
                f"[PV663-0.8m K={k}] {v:18s}: Dice={m['dice']:.4f} BF1@1={m['bf1_1px']:.4f} "
                f"BF1@2={m['boundary_f1']:.4f} Rec={m['recall']:.4f} Prec={m['precision']:.4f}",
                flush=True,
            )
        diag_by_k[str(k)] = {
            "M3_DeformGrad_Old": _summarize_diag(acc_old, n_q_tot),
            "M3_PBDC": _summarize_diag(acc_new, n_q_tot),
        }

    comparisons = {}
    for cname, cand_v, ref_v in (
        ("M3_PBDC_vs_M3_Off", "M3_PBDC", "M3_Off"),
        ("M3_DeformGrad_Old_vs_M3_Off", "M3_DeformGrad_Old", "M3_Off"),
        ("M3_PBDC_vs_M3_DeformGrad_Old", "M3_PBDC", "M3_DeformGrad_Old"),
    ):
        comparisons[cname] = {
            str(k): compare_pv663_methods(methods_res[cand_v][str(k)], methods_res[ref_v][str(k)], n_boot=10000)
            for k in sorted(ep_by_k.keys())
        }
    return {
        "dataset": "PV663 (0.8m, 48 stratified 5km-disjoint query sites, 6 eps/K)",
        "elapsed_sec": float(time.perf_counter() - t0),
        "methods": methods_res,
        "comparisons": comparisons,
        "diagnostics": diag_by_k,
    }


def format_dual_report(summary: dict[str, Any]) -> str:
    lines = [
        "=== MODULE 3 CROSS-RESOLUTION BENCHMARK (2m PV-100scence + 0.8m PV663 Subset) ===",
        f"Total Elapsed: {summary['total_elapsed_sec']/60:.1f} min",
    ]
    for ds_key, label in (("pv100_2m", "PV-100scence (2m, 50 Dev Sites)"), ("pv663_08m", "PV663 (0.8m, 48 Stratified Sites)")):
        ds = summary[ds_key]
        lines.append(f"\n================ {label} ({ds['elapsed_sec']/60:.1f} min) ================")
        for v in ("M3_Off", "M3_DeformGrad_Old", "M3_PBDC"):
            lines.append(f"\n--- {v} ---")
            for k in ("1", "3", "5"):
                m = ds["methods"][v][k]["macro"]
                neg_s = f" NegFP={m['neg_fp']:.4f}" if "neg_fp" in m and m["neg_fp"] is not None else ""
                sm_r = ds["methods"][v][k]["strata"].get("small_lt5pct", {}).get("recall", 0.0)
                lines.append(
                    f"  K={k}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} BF1@1={m['bf1_1px']:.4f} "
                    f"BF1@2={m['boundary_f1']:.4f} Rec={m['recall']:.4f} Prec={m['precision']:.4f} "
                    f"Sm<5R={sm_r:.4f}{neg_s}"
                )
        lines.append("\n--- Paired Site Bootstrap Deltas ---")
        for cname in ("M3_PBDC_vs_M3_Off", "M3_DeformGrad_Old_vs_M3_Off", "M3_PBDC_vs_M3_DeformGrad_Old"):
            lines.append(f"[{cname}]")
            for k in ("1", "3", "5"):
                c = ds["comparisons"][cname][k]
                bf1_1_s = f"dBF1@1={c['bf1_1px']['mean_delta']:+.4f} " if "bf1_1px" in c else ""
                lines.append(
                    f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} | "
                    f"{bf1_1_s}dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x,4) for x in c['boundary_f1']['ci95']]} | "
                    f"dSm<5R={c['small_lt5pct_recall_delta']:+.4f}"
                )
        lines.append("\n--- Movement & Topology Diagnostics ---")
        for k in ("1", "3", "5"):
            d_o = ds["diagnostics"][k]["M3_DeformGrad_Old"]
            d_n = ds["diagnostics"][k]["M3_PBDC"]
            lines.append(
                f"  K={k}: Old[moved={d_o['moved_arc_fraction']:.1%} |d|={d_o['mean_abs_d_px']:.2f}px area={d_o['mean_net_area_delta_px']:+.0f} rev={d_o['topology_reverts']}] | "
                f"PBDC[moved={d_n['moved_arc_fraction']:.1%} |d|={d_n['mean_abs_d_px']:.2f}px area={d_n['mean_net_area_delta_px']:+.0f} locFB={d_n['localized_fallbacks']} rev={d_n['topology_reverts']}]"
            )
    return "\n".join(lines) + "\n"


def run_dual_subset_benchmark(
    *,
    split_dir: Path,
    dev_cache_pt: Path,
    pv663_subset_json: Path,
    deform_config_json: Path,
    source_dir: Path,
    pv663_feat_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    num_workers: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = num_workers if num_workers is not None else min(4, max(1, os.cpu_count() or 1))

    res_2m = run_pv100_2m_subset(
        split_dir=split_dir,
        cache_pt=dev_cache_pt,
        deform_config_json=deform_config_json,
        episodes_per_fold=4,
        device=device,
        num_workers=workers,
    )
    res_08m = run_pv663_08m_subset(
        subset_json=pv663_subset_json,
        deform_config_json=deform_config_json,
        source_dir=source_dir,
        feat_cache_dir=pv663_feat_dir,
        device=device,
        num_workers=workers,
    )
    summary = {
        "schema": "pv-dinov3sat-m3-dual-subset-iter1-v1",
        "total_elapsed_sec": float(time.perf_counter() - t0),
        "pbdc_config": PBDCConfig(enabled=True).__dict__,
        "pv100_2m": res_2m,
        "pv663_08m": res_08m,
    }
    write_json(output_dir / "m3_dual_subset_summary.json", summary)
    rep = format_dual_report(summary)
    (output_dir / "m3_dual_subset_report.txt").write_text(rep, encoding="utf-8")
    print("\n" + rep, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 3 cross-resolution (2m + 0.8m) subset benchmark.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--dev-cache-pt", type=Path, required=True)
    parser.add_argument("--pv663-subset-json", type=Path, default=REPO_ROOT / "configs" / "pv663_subset_lock.json")
    parser.add_argument("--deform-config-json", type=Path, default=REPO_ROOT / "configs" / "three_module_exp_m3_deformgrad.json")
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/dinov3-source"))
    parser.add_argument("--pv663-feat-dir", type=Path, default=Path("/tmp/pv663_subset_feats"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "m3_dual_subset_iter1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run_dual_subset_benchmark(
        split_dir=args.split_dir,
        dev_cache_pt=args.dev_cache_pt,
        pv663_subset_json=args.pv663_subset_json,
        deform_config_json=args.deform_config_json,
        source_dir=args.source_dir,
        pv663_feat_dir=args.pv663_feat_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
