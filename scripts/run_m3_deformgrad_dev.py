from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.io import write_json
from pvfs.iter3_methods import _to_tokens_norm, compute_dino_affinity_weights
from pvfs.modules import ThreeModulePipeline, get_m3_pruned_config
from pvfs.modules.deformable_boundary import (
    DeformableBoundaryConfig,
    refine_mask_deformable_grad,
)
from pvfs.phase_boundary import (
    DISK_8,
    PhaseBoundaryConfig,
    SupportCalibrationBundle,
    build_component_geometry_and_corridors,
    compute_oracle_pm8px_mask,
    precompute_image_signals,
    refine_query_image_all_variants,
)
from pvfs.phase_boundary_eval import (
    aggregate_binary_episodes,
    compare_boundary_methods,
    evaluate_binary_episode,
)


def _calibrate_p_floor_loo(
    support_ids: list[str],
    mask_by_id: dict[str, np.ndarray],
    loo_probs: dict[str, np.ndarray] | None,
) -> tuple[bool, float | None]:
    """Shared outward semantic floor p_floor rule: K=1 forbid outward; K>=2 LOO missed-band q25 clamped [0.20, 0.48]."""
    if len(support_ids) == 1 or loo_probs is None:
        return False, None
    missed_vals, edge_vals = [], []
    for sid in support_ids:
        gt = np.asarray(mask_by_id[sid] > 0, dtype=bool)
        if not gt.any() or sid not in loo_probs:
            continue
        p_loo = loo_probs[sid]
        inner = gt & (~binary_erosion(gt, structure=DISK_8, border_value=0))
        if not inner.any():
            inner = gt
        missed = inner & (p_loo < 0.50)
        if int(missed.sum()) >= 16:
            missed_vals.append(p_loo[missed])
        edge_vals.append(p_loo[inner])
    pool = missed_vals if missed_vals else edge_vals
    if not pool:
        return True, 0.35
    q = float(np.quantile(np.concatenate(pool), 0.25))
    return True, float(np.clip(q, 0.20, 0.48))


def _rigid_gt_reachable_ref(m0: np.ndarray, gt_mask: np.ndarray, cfg: PhaseBoundaryConfig) -> np.ndarray:
    """Offline Stage 0 GT reachable reference under old rigid segment translation."""
    m0_b = np.asarray(m0, dtype=bool)
    gt_b = np.asarray(gt_mask > 0, dtype=bool)
    if not m0_b.any():
        return m0_b.copy()
    _, _, corrs, _ = build_component_geometry_and_corridors(m0_b, cfg)
    m_cur = m0_b.copy()
    for segs in corrs.values():
        for seg in segs:
            best_d = 0
            best_u = 0.0
            for d in seg.candidate_shifts:
                if d == 0:
                    continue
                if d > 0:
                    idx = seg.add_indices_by_d[d]
                    u_val = float(np.sum(np.where(gt_b[seg.corridor_y[idx], seg.corridor_x[idx]], 1.0, -1.0)))
                else:
                    idx = seg.rem_indices_by_d[d]
                    u_val = -float(np.sum(np.where(gt_b[seg.corridor_y[idx], seg.corridor_x[idx]], 1.0, -1.0)))
                if u_val > best_u + 1e-9:
                    best_u = u_val
                    best_d = d
            if best_d > 0:
                idx = seg.add_indices_by_d[best_d]
                m_cur[seg.corridor_y[idx], seg.corridor_x[idx]] = True
            elif best_d < 0:
                idx = seg.rem_indices_by_d[best_d]
                m_cur[seg.corridor_y[idx], seg.corridor_x[idx]] = False
    return m_cur


def run_m3_deformgrad_dev_experiment(
    *,
    split_dir: Path,
    cache_pt: Path,
    config_json: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run Module ③ Stage 0 offline geometry reference + Dev ablations (Off vs Rigid_Grad vs Deformable_Grad)."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)

    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    records_by_id, mask_by_id, rgb_by_id = {}, {}, {}
    for row in dev_records:
        rid = row["id"]
        mask = read_verified_mask(lbl_dir, row)
        rgb = np.transpose(read_verified_rgb(img_dir, row), (1, 2, 0))
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}
        mask_by_id[rid] = mask
        rgb_by_id[rid] = rgb

    caches: dict[str, dict] = {}
    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64)):
        toks = {rid: _to_tokens_norm(feats[rid][key], run_device) for rid in all_ids}
        caches[key] = build_pairwise_topk_cache(toks, mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        tokens_by_key[key] = toks
        if run_device == "cuda":
            torch.cuda.empty_cache()

    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=0.12) for rid in all_ids}
    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    m3_cfg = get_m3_pruned_config()
    pipe_r = ThreeModulePipeline(m3_cfg)
    pb_cfg = PhaseBoundaryConfig()
    def_cfg = DeformableBoundaryConfig.from_json(config_json, mode="deformable_grad")
    signals_by_id = {rid: precompute_image_signals(rgb_by_id[rid], pb_cfg) for rid in all_ids}

    variants = ("M3_Off_R", "M3_Rigid_Grad", "M3_Deformable_Grad", "Offline_Rigid_GT_Ref", "Offline_Deform_GT_Ref", "Offline_Loose_PM8px")
    methods_res: dict[str, dict[str, Any]] = {v: {} for v in variants}
    diag_by_k: dict[str, Any] = {}

    for k in sorted(ep_by_k.keys()):
        ep_evals: dict[str, list[dict]] = {v: [] for v in variants}
        move_acc: dict[str, dict[str, float]] = {
            "M3_Rigid_Grad": defaultdict(float),
            "M3_Deformable_Grad": defaultdict(float),
        }

        for ep in ep_by_k[k]:
            sup_ids = ep["support_ids"]
            q_ids = ep["query_ids"]
            q_recs = [records_by_id[qid] for qid in q_ids]

            out_r = pipe_r.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
            p_r, s32_r = out_r.to_legacy_outputs()

            loo_probs = None
            if k >= 2:
                loo_probs = {}
                for s_i in sup_ids:
                    rem = [s for s in sup_ids if s != s_i]
                    o_l = pipe_r.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, rem, [s_i])
                    loo_probs[s_i] = o_l.boundary_by_id[s_i].p_coarse

            allow_out, p_floor = _calibrate_p_floor_loo(sup_ids, mask_by_id, loo_probs)
            # Shared null threshold for gradient scoring (0.60 matches support pure-block q95 scale)
            tau_rigid = 0.63
            tau_deform = 0.20

            rigid_calib = SupportCalibrationBundle(
                k_shots=k,
                allow_outward=allow_out,
                p_floor=p_floor,
                p_floor_fallback=(p_floor is None),
                period_range_px=(2.8, 12.0),
                holdout_r2_floor=0.05,
                tau_by_variant={vn: (tau_rigid if vn == "B_GradAlign" else 1e9) for vn in pb_cfg.fallback_gain_per_px},
                null_block_counts={vn: 10 for vn in pb_cfg.fallback_gain_per_px},
                tau_fallback_by_variant={vn: False for vn in pb_cfg.fallback_gain_per_px},
            )

            masks_v: dict[str, dict[str, np.ndarray]] = {v: {} for v in variants}
            for qid in q_ids:
                m0 = out_r.boundary_by_id[qid].m_coarse
                p0 = p_r[qid]
                gt = records_by_id[qid]["mask"]

                masks_v["M3_Off_R"][qid] = m0

                # Old Rigid + Gradient
                r_out = refine_query_image_all_variants(
                    rgb_by_id[qid], p0, rigid_calib, variants=("B_GradAlign",), cfg=pb_cfg, precomputed_signals=signals_by_id[qid]
                )
                b_info = r_out["variants"]["B_GradAlign"]
                masks_v["M3_Rigid_Grad"][qid] = b_info["mask"]
                move_acc["M3_Rigid_Grad"]["total_arc"] += r_out["total_arc_len_px"]
                move_acc["M3_Rigid_Grad"]["moved_arc"] += b_info["moved_arc_len_px"]
                move_acc["M3_Rigid_Grad"]["net_area"] += b_info["net_area_delta_px"]
                move_acc["M3_Rigid_Grad"]["abs_d_wsum"] += b_info["mean_abs_d_moved"] * b_info["moved_arc_len_px"]

                # New Deformable DP + Gradient
                d_out = refine_mask_deformable_grad(
                    m0,
                    rgb_by_id[qid],
                    p0,
                    allow_outward=allow_out,
                    p_floor=p_floor,
                    tau_grad_per_px=tau_deform,
                    cfg=def_cfg,
                )
                masks_v["M3_Deformable_Grad"][qid] = d_out["mask"]
                move_acc["M3_Deformable_Grad"]["total_arc"] += d_out["total_arc_len_px"]
                move_acc["M3_Deformable_Grad"]["moved_arc"] += d_out["moved_arc_len_px"]
                move_acc["M3_Deformable_Grad"]["net_area"] += d_out["net_area_delta_px"]
                move_acc["M3_Deformable_Grad"]["abs_d_wsum"] += d_out["mean_abs_d_moved"] * d_out["moved_arc_len_px"]
                move_acc["M3_Deformable_Grad"]["topo_reverts"] += d_out["topology_reverts"]

                # Offline Stage 0 GT Reachable References (strictly offline!)
                if k == 5:
                    masks_v["Offline_Rigid_GT_Ref"][qid] = _rigid_gt_reachable_ref(m0, gt, pb_cfg)
                    masks_v["Offline_Deform_GT_Ref"][qid] = refine_mask_deformable_grad(
                        m0, None, p0, allow_outward=True, p_floor=None, gt_mask_offline=(gt > 0), cfg=def_cfg
                    )["mask"]
                    masks_v["Offline_Loose_PM8px"][qid] = compute_oracle_pm8px_mask(gt, p0)["oracle_mask"]
                else:
                    masks_v["Offline_Rigid_GT_Ref"][qid] = m0
                    masks_v["Offline_Deform_GT_Ref"][qid] = m0
                    masks_v["Offline_Loose_PM8px"][qid] = compute_oracle_pm8px_mask(gt, p0)["oracle_mask"]

            for v in variants:
                ep_evals[v].append(evaluate_binary_episode(q_recs, masks_v[v], p_r, s32_r))

        for v in variants:
            methods_res[v][str(k)] = aggregate_binary_episodes(ep_evals[v])

        diag_by_k[str(k)] = {
            vn: {
                "moved_arc_fraction": float(st["moved_arc"] / max(1.0, st["total_arc"])),
                "mean_abs_d_px": float(st["abs_d_wsum"] / max(1.0, st["moved_arc"])),
                "mean_net_area_delta_px": float(st["net_area"] / max(1, len(ep_by_k[k]) * 19)),
                "topology_reverts": int(st.get("topo_reverts", 0)),
            }
            for vn, st in move_acc.items()
        }
        for v in ("M3_Off_R", "M3_Rigid_Grad", "M3_Deformable_Grad", "Offline_Rigid_GT_Ref", "Offline_Deform_GT_Ref"):
            if k != 5 and v.startswith("Offline_"):
                continue
            m = methods_res[v][str(k)]["macro"]
            sm = methods_res[v][str(k)]["strata"]["small_lt5pct"]["recall"]
            print(
                f"[M3 K={k}] {v:22s}: BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                f"Dice={m['dice']:.4f} Sm<5R={sm:.4f} NegFP={m['neg_fp']:.4f}",
                flush=True,
            )

    comparisons = {
        "DeformGrad_vs_OffR": {str(k): compare_boundary_methods(methods_res["M3_Deformable_Grad"][str(k)], methods_res["M3_Off_R"][str(k)]) for k in ("1", "3", "5")},
        "RigidGrad_vs_OffR": {str(k): compare_boundary_methods(methods_res["M3_Rigid_Grad"][str(k)], methods_res["M3_Off_R"][str(k)]) for k in ("1", "3", "5")},
        "DeformGrad_vs_RigidGrad": {str(k): compare_boundary_methods(methods_res["M3_Deformable_Grad"][str(k)], methods_res["M3_Rigid_Grad"][str(k)]) for k in ("1", "3", "5")},
        "Offline_DeformGTRef_vs_RigidGTRef_5shot": compare_boundary_methods(methods_res["Offline_Deform_GT_Ref"]["5"], methods_res["Offline_Rigid_GT_Ref"]["5"]),
    }

    c5 = comparisons["DeformGrad_vs_OffR"]["5"]
    guard_ok = all(
        comparisons["DeformGrad_vs_OffR"][k]["dice"]["mean_delta"] >= -0.005
        and comparisons["DeformGrad_vs_OffR"][k]["small_lt5pct_recall_delta"] >= -0.010
        and comparisons["DeformGrad_vs_OffR"][k]["neg_fp"]["mean_delta"] <= 0.002
        for k in ("1", "3", "5")
    )
    m3_pass = bool(
        c5["boundary_f1"]["mean_delta"] >= 0.015
        and c5["boundary_f1"]["ci95"][0] > 0.0
        and c5["boundary_f1"]["positive_folds"] >= 3
        and c5["bf1_1px"]["mean_delta"] > 0.0
        and guard_ok
    )

    summary = {
        "schema": "pv-2m-dinov3sat-m3-deformgrad-dev-v1",
        "device": run_device,
        "elapsed_sec": float(time.perf_counter() - t0),
        "verdict": {
            "m3_deformable_grad_pass": m3_pass,
            "bf1_2px_delta_5shot": c5["boundary_f1"]["mean_delta"],
            "bf1_2px_ci95_5shot": c5["boundary_f1"]["ci95"],
            "bf1_2px_positive_folds_5shot": c5["boundary_f1"]["positive_folds"],
            "bf1_1px_delta_5shot": c5["bf1_1px"]["mean_delta"],
            "all_k_guardrails_pass": guard_ok,
        },
        "diagnostics_by_k": diag_by_k,
        "comparisons": comparisons,
        "methods": methods_res,
    }
    write_json(output_dir / "m3_deformgrad_summary.json", summary)
    print("M3 Verdict:", summary["verdict"], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module ③ Stage 0 offline geometry ref & Dev ablations.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--cache-pt", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, default=REPO_ROOT / "configs" / "three_module_exp_m3_deformgrad.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "m3_deformgrad_dev")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_m3_deformgrad_dev_experiment(
        split_dir=args.split_dir,
        cache_pt=args.cache_pt,
        config_json=args.config_json,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
