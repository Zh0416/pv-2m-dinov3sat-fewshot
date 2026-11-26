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

from pvfs.data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache, compute_dino_affinity_weights, to_tokens_norm
from pvfs.io import write_json
from pvfs.metrics import aggregate_binary_episodes, compare_methods, evaluate_binary_episode
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

# Shared read-only Linux fork worker state for parallel pixel refinement
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


def format_pv100_ablation_report(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    lines = [
        "=== PV-100scence (2m) FULL ABLATION REPORT (v3.0-pbdc-final) ===",
        f"Split: 50 Dev Sites ({meta['pos_sites']} pos + {meta['neg_sites']} neg) | "
        f"4 folds x {meta['episodes_per_fold']} eps/fold | Total Elapsed: {summary['elapsed_sec']/60:.1f} min",
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
        lines.append(f"\n--- K={k} ---")
        for mname in t1_order:
            res = summary["methods"][mname][k]
            m = res["macro"]
            sm_r = res["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            lines.append(
                f"  {mname:20s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
                f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} Sm<5R={sm_r:.4f} NegFP={m['neg_fp']:.4f}"
            )

    lines.append("\n--- Table 1 Paired Site Bootstrap Stepwise Deltas (34 Pos + 16 Neg Sites) ---")
    for cname in (
        "A1_vs_A0",
        "A2_vs_A1",
        "A3_vs_A2",
        "A4_PBDC_vs_A3",
        "A4_PBDC_vs_A0",
        "A4_Old_vs_A3",
        "A4_PBDC_vs_A4_Old",
    ):
        lines.append(f"[{cname}]")
        for k in ("1", "3", "5"):
            c = summary["comparisons"][cname][k]
            lines.append(
                f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} | "
                f"dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x,4) for x in c['boundary_f1']['ci95']]} | "
                f"dSm<5R={c['small_lt5pct_recall_delta']:+.4f} | dNegFP={c['neg_fp']['mean_delta']:+.4f}"
            )

    lines.append("\n================ TABLE 2: CORE MODULE-INTERNAL ABLATION (K=1, 3, 5) ================")
    t2_order = [
        "A3_Plus_M3A_Aff",
        "I1_M1_wo_MidFusion",
        "I2_M2_wo_SR_Gating",
        "A4_Full_PBDC",
        "I3_PBDC_wo_AsymCore",
        "I4_PBDC_wo_UncertLock",
        "I5_PBDC_wo_SectorFB",
    ]
    for k in ("1", "3", "5"):
        lines.append(f"\n--- K={k} ---")
        for mname in t2_order:
            res = summary["methods"][mname][k]
            m = res["macro"]
            sm_r = res["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            lines.append(
                f"  {mname:22s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
                f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} Sm<5R={sm_r:.4f} NegFP={m['neg_fp']:.4f}"
            )

    lines.append("\n--- Table 2 Paired Site Bootstrap Deltas (Full vs Ablated, positive = mechanism helps) ---")
    for cname in (
        "A3_vs_I1_wo_MidFusion",
        "A3_vs_I2_wo_SR_Gating",
        "A4_vs_I3_wo_AsymCore",
        "A4_vs_I4_wo_UncertLock",
        "A4_vs_I5_wo_SectorFB",
    ):
        lines.append(f"[{cname}]")
        for k in ("1", "3", "5"):
            c = summary["comparisons"][cname][k]
            lines.append(
                f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} | "
                f"dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x,4) for x in c['boundary_f1']['ci95']]} | "
                f"dSm<5R={c['small_lt5pct_recall_delta']:+.4f} | dNegFP={c['neg_fp']['mean_delta']:+.4f}"
            )

    lines.append("\n--- Pixel Refiner Movement & Topology Diagnostics ---")
    for k in ("1", "3", "5"):
        parts = []
        for rname, d in summary["diagnostics"][k].items():
            parts.append(
                f"{rname}[moved={d['moved_arc_fraction']:.1%} |d|={d['mean_abs_d_px']:.2f}px "
                f"area={d['mean_net_area_delta_px']:+.0f} locFB={d['localized_fallbacks']} rev={d['topology_reverts']}]"
            )
        lines.append(f"  K={k}: " + " | ".join(parts))

    return "\n".join(lines) + "\n"


def run_pv100_full_ablation(
    *,
    split_dir: Path,
    dev_cache_pt: Path,
    deform_config_json: Path,
    output_dir: Path,
    episodes_per_fold: int = 4,
    device: str = "cuda",
    num_workers: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = num_workers if num_workers is not None else min(4, max(1, os.cpu_count() or 1))
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    img_dir, lbl_dir = resolve_data_dirs(None, None)

    bundle = torch.load(dev_cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]

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
    refiner_specs: list[tuple[str, Any]] = [
        ("A4_Full_PBDC", PBDCConfig(enabled=True)),
        ("A4_Old_DeformGrad", old_cfg),
        ("I3_PBDC_wo_AsymCore", PBDCConfig(enabled=True, ablate_undirected_grad=True)),
        ("I4_PBDC_wo_UncertLock", PBDCConfig(enabled=True, ablate_no_uncert_lock=True)),
        ("I5_PBDC_wo_SectorFB", PBDCConfig(enabled=True, ablate_whole_loop_fallback=True)),
    ]
    pipe_names = list(pipelines.keys())
    all_row_names = pipe_names + [r[0] for r in refiner_specs]

    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    by_kf: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        by_kf[(int(ep["k"]), int(ep["fold"]))].append(ep)
    for (k, f), lst in sorted(by_kf.items()):
        ep_by_k[k].extend(lst[:episodes_per_fold])

    methods_res: dict[str, dict[str, Any]] = defaultdict(dict)
    diag_by_k: dict[str, dict[str, Any]] = {}

    use_fork = (workers > 1) and (sys.platform != "win32") and ("fork" in multiprocessing.get_all_start_methods())
    global _FW_RGB_BY_ID, _FW_REFINER_SPECS, _FW_ALLOW_OUT, _FW_P_FLOOR, _FW_K_SHOTS
    _FW_RGB_BY_ID = rgb_hwc_by_id
    _FW_REFINER_SPECS = refiner_specs

    for k in sorted(ep_by_k.keys()):
        t_k = time.perf_counter()
        eps = ep_by_k[k]
        ep_evals: dict[str, list[dict]] = {name: [] for name in all_row_names}
        diag_acc: dict[str, dict[str, float]] = {r[0]: defaultdict(float) for r in refiner_specs}
        n_q_tot = 0

        for ep in eps:
            sup_ids = ep["support_ids"]
            q_ids = ep["query_ids"]
            q_recs = [records_by_id[qid] for qid in q_ids]
            n_q_tot += len(q_ids)

            p_a3: dict[str, np.ndarray] = {}
            s32_a3: dict[str, np.ndarray] = {}
            m_a3: dict[str, np.ndarray] = {}
            for pname in pipe_names:
                out = pipelines[pname].run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
                p_out, s32_out = out.to_legacy_outputs()
                m_out = out.coarse_masks_512()
                ep_evals[pname].append(evaluate_binary_episode(q_recs, m_out, p_out, s32_out))
                if pname == "A3_Plus_M3A_Aff":
                    p_a3, s32_a3, m_a3 = p_out, s32_out, m_out

            loo_probs = None
            if k >= 2:
                loo_probs = {}
                for sid in sup_ids:
                    rem = [s for s in sup_ids if s != sid]
                    o_l = pipelines["A3_Plus_M3A_Aff"].run_episode(
                        caches, tokens_by_key, mask_by_id, aff_by_id, rem, [sid]
                    )
                    loo_probs[sid] = o_l.boundary_by_id[sid].p_coarse
            allow_out, p_floor = calibrate_deform_p_floor_loo(sup_ids, mask_by_id, loo_probs)

            _FW_ALLOW_OUT = allow_out
            _FW_P_FLOOR = p_floor
            _FW_K_SHOTS = k
            tasks = [(qid, m_a3[qid], p_a3[qid].astype(np.float16)) for qid in q_ids]
            ref_masks: dict[str, dict[str, np.ndarray]] = {r[0]: {} for r in refiner_specs}

            if use_fork:
                ctx = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                    for qid, m_dict, d_dict in ex.map(_refine_multi_worker, tasks, chunksize=4):
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
                ep_evals[rname].append(evaluate_binary_episode(q_recs, ref_masks[rname], p_a3, s32_a3))

        for name in all_row_names:
            methods_res[name][str(k)] = aggregate_binary_episodes(ep_evals[name], episodes_per_fold=episodes_per_fold)
            m = methods_res[name][str(k)]["macro"]
            sm_r = methods_res[name][str(k)]["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            print(
                f"[PV100-2m K={k}] {name:22s}: Dice={m['dice']:.4f} BF1@1={m['bf1_1px']:.4f} "
                f"BF1@2={m['boundary_f1']:.4f} Prec={m['precision']:.4f} Sm<5R={sm_r:.4f} NegFP={m['neg_fp']:.4f}",
                flush=True,
            )
        diag_by_k[str(k)] = {rname: _summarize_diag(diag_acc[rname], n_q_tot) for rname, _ in refiner_specs}
        print(f"[PV100-2m K={k}] finished in {(time.perf_counter()-t_k)/60:.1f}m", flush=True)

    comp_pairs = [
        ("A1_vs_A0", "A1_Plus_M1", "A0_B0_32_Ref"),
        ("A2_vs_A1", "A2_Plus_M2_SR", "A1_Plus_M1"),
        ("A3_vs_A2", "A3_Plus_M3A_Aff", "A2_Plus_M2_SR"),
        ("A4_PBDC_vs_A3", "A4_Full_PBDC", "A3_Plus_M3A_Aff"),
        ("A4_PBDC_vs_A0", "A4_Full_PBDC", "A0_B0_32_Ref"),
        ("A4_Old_vs_A3", "A4_Old_DeformGrad", "A3_Plus_M3A_Aff"),
        ("A4_PBDC_vs_A4_Old", "A4_Full_PBDC", "A4_Old_DeformGrad"),
        ("A3_vs_I1_wo_MidFusion", "A3_Plus_M3A_Aff", "I1_M1_wo_MidFusion"),
        ("A3_vs_I2_wo_SR_Gating", "A3_Plus_M3A_Aff", "I2_M2_wo_SR_Gating"),
        ("A4_vs_I3_wo_AsymCore", "A4_Full_PBDC", "I3_PBDC_wo_AsymCore"),
        ("A4_vs_I4_wo_UncertLock", "A4_Full_PBDC", "I4_PBDC_wo_UncertLock"),
        ("A4_vs_I5_wo_SectorFB", "A4_Full_PBDC", "I5_PBDC_wo_SectorFB"),
    ]
    comparisons = {
        cname: {str(k): compare_methods(methods_res[cand][str(k)], methods_res[ref][str(k)]) for k in (1, 3, 5)}
        for cname, cand, ref in comp_pairs
    }

    summary = {
        "schema": "pv100-2m-full-ablation-v3.0",
        "elapsed_sec": float(time.perf_counter() - t0),
        "meta": {
            "total_sites": len(dev_records),
            "pos_sites": len(pos_ids),
            "neg_sites": len(dev_records) - len(pos_ids),
            "episodes_per_fold": episodes_per_fold,
        },
        "methods": dict(methods_res),
        "comparisons": comparisons,
        "diagnostics": diag_by_k,
    }
    write_json(output_dir / "pv100_full_ablation_summary.json", summary)
    report = format_pv100_ablation_report(summary)
    (output_dir / "pv100_full_ablation_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PV-100scence (2m) Table 1 + Table 2 ablation study.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--dev-cache-pt", type=Path, required=True)
    parser.add_argument("--deform-config-json", type=Path, default=REPO_ROOT / "configs" / "three_module_exp_m3_deformgrad.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "pv100_full_ablation_v3")
    parser.add_argument("--episodes-per-fold", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    run_pv100_full_ablation(
        split_dir=args.split_dir,
        dev_cache_pt=args.dev_cache_pt,
        deform_config_json=args.deform_config_json,
        output_dir=args.output_dir,
        episodes_per_fold=args.episodes_per_fold,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
