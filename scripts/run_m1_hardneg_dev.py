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

from pvfs.data import area_target_32, load_dev_records, read_verified_mask, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.io import write_json
from pvfs.iter3_methods import _to_tokens_norm, compute_dino_affinity_weights
from pvfs.iter5_methods import run_iter5_episode
from pvfs.metrics import aggregate_episodes, compare_methods, evaluate_episode
from pvfs.modules import ThreeModulePipeline, get_b0_32_config, get_m3_pruned_config
from pvfs.modules.hard_negative import HardNegConfig


def _evaluate_threshold_sweep_for_equal_recall(
    episodes_data: list[tuple[list[dict], dict[str, np.ndarray]]],
    thresholds: np.ndarray,
) -> list[dict[str, float]]:
    """Compute site-macro Pos Pixel Recall, Neg Site FP, and Pos Bg-FP across fixed threshold grid."""
    curve: list[dict[str, float]] = []
    for thr in thresholds:
        t_val = float(thr)
        pos_site_rec: dict[str, list[float]] = defaultdict(list)
        pos_site_bgfp: dict[str, list[float]] = defaultdict(list)
        neg_site_fp: dict[str, list[float]] = defaultdict(list)

        for q_recs, probs_512 in episodes_data:
            by_site_pos_r: dict[str, list[float]] = defaultdict(list)
            by_site_pos_bf: dict[str, list[float]] = defaultdict(list)
            by_site_neg_f: dict[str, list[float]] = defaultdict(list)
            for rec in q_recs:
                p = probs_512[rec["id"]]
                pred = p >= t_val
                if rec["positive"]:
                    gt = rec["mask"] > 0
                    gt_sum = int(gt.sum())
                    tp = int(np.logical_and(gt, pred).sum())
                    fp = int(pred.sum()) - tp
                    bg_tot = gt.size - gt_sum
                    by_site_pos_r[rec["site"]].append(tp / max(1, gt_sum))
                    by_site_pos_bf[rec["site"]].append(fp / max(1, bg_tot))
                else:
                    by_site_neg_f[rec["site"]].append(float(pred.mean()))
            for s, vals in by_site_pos_r.items():
                pos_site_rec[s].append(float(np.mean(vals)))
            for s, vals in by_site_pos_bf.items():
                pos_site_bgfp[s].append(float(np.mean(vals)))
            for s, vals in by_site_neg_f.items():
                neg_site_fp[s].append(float(np.mean(vals)))

        macro_rec = float(np.mean([np.mean(v) for v in pos_site_rec.values()]))
        macro_bgfp = float(np.mean([np.mean(v) for v in pos_site_bgfp.values()]))
        macro_negfp = float(np.mean([np.mean(v) for v in neg_site_fp.values()]))
        curve.append(
            {
                "threshold": t_val,
                "pos_recall": macro_rec,
                "pos_bg_fp": macro_bgfp,
                "neg_site_fp": macro_negfp,
            }
        )
    return curve


def _interp_at_target_recall(curve: list[dict[str, float]], target_recall: float) -> dict[str, float]:
    """Interpolate Neg Site FP and Pos Bg-FP along threshold curve at exact target positive recall."""
    # curve is ordered by threshold ascending -> pos_recall descending
    ordered = sorted(curve, key=lambda r: r["pos_recall"])
    recs = np.array([r["pos_recall"] for r in ordered], dtype=np.float64)
    negfps = np.array([r["neg_site_fp"] for r in ordered], dtype=np.float64)
    bgfps = np.array([r["pos_bg_fp"] for r in ordered], dtype=np.float64)
    thrs = np.array([r["threshold"] for r in ordered], dtype=np.float64)
    return {
        "target_pos_recall": float(target_recall),
        "matched_threshold": float(np.interp(target_recall, recs, thrs)),
        "neg_site_fp_at_recall": float(np.interp(target_recall, recs, negfps)),
        "pos_bg_fp_at_recall": float(np.interp(target_recall, recs, bgfps)),
        "in_range": bool(recs[0] <= target_recall <= recs[-1]),
    }


def run_m1_hardneg_dev_experiment(
    *,
    split_dir: Path,
    cache_pt: Path,
    config_json: Path,
    output_dir: Path,
    label_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    _, lbl_dir = resolve_data_dirs(None, label_dir)

    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    records_by_id, mask_by_id = {}, {}
    for row in dev_records:
        rid = row["id"]
        mask = read_verified_mask(lbl_dir, row)
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}
        mask_by_id[rid] = mask

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
    b0_cfg = get_b0_32_config()

    # Step 1: Full 120-episode L vs R Equivalence Audit on M3_Pruned_Ref and B0_32_Ref
    equiv_audit: dict[str, Any] = {}
    for e_cfg in (m3_cfg, b0_cfg):
        pipe_eq = ThreeModulePipeline(e_cfg)
        e_by_k: dict[str, Any] = {}
        all_ok = True
        for k in sorted(ep_by_k.keys()):
            max_p_diff = 0.0
            max_s_diff = 0.0
            mask_flips = 0
            n_q = 0
            for ep in ep_by_k[k]:
                pL, sL = run_iter5_episode(caches, tokens_by_key, mask_by_id, aff_by_id, ep["support_ids"], ep["query_ids"], e_cfg)
                outR = pipe_eq.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, ep["support_ids"], ep["query_ids"])
                pR, sR = outR.to_legacy_outputs()
                for qid in ep["query_ids"]:
                    n_q += 1
                    max_p_diff = max(max_p_diff, float(np.max(np.abs(pR[qid] - pL[qid]))))
                    max_s_diff = max(max_s_diff, float(np.max(np.abs(sR[qid] - sL[qid]))))
                    mask_flips += int(np.logical_xor(outR.boundary_by_id[qid].m_coarse, pL[qid] >= 0.50).sum())
                    mask_flips += int(np.logical_xor(outR.boundary_by_id[qid].m_final, outR.boundary_by_id[qid].m_coarse).sum())
            ok_k = bool(mask_flips == 0 and max_p_diff == 0.0 and max_s_diff == 0.0)
            all_ok = all_ok and ok_k
            e_by_k[str(k)] = {
                "query_checks": n_q,
                "max_probs_512_diff": max_p_diff,
                "max_scores_32_diff": max_s_diff,
                "mask_flips": mask_flips,
                "exact_pass": ok_k,
            }
        equiv_audit[e_cfg.name] = {"all_passed": all_ok, "by_k": e_by_k}
        print(f"[Equiv L vs R] {e_cfg.name}: all_passed={all_ok} | {e_by_k}", flush=True)

    # Step 2: Module ① Ablation Experiment (M1_Base_R vs M1_Ctrl_EqualBudgetRandomBG vs M1_Cand_HardNegH)
    variant_specs = (
        ("M1_Base_R", HardNegConfig(enabled=False, mode="off")),
        ("M1_Ctrl_EqualBudgetRandomBG", HardNegConfig.from_json(config_json, mode="random_bg")),
        ("M1_Cand_HardNegH", HardNegConfig.from_json(config_json, mode="hard_neg_H")),
    )
    pipes = {name: ThreeModulePipeline(m3_cfg, hard_neg_cfg=hcfg) for name, hcfg in variant_specs}

    methods_res: dict[str, dict[str, Any]] = {name: {} for name, _ in variant_specs}
    diag_by_k: dict[str, dict[str, Any]] = {}
    equal_recall_by_k: dict[str, dict[str, Any]] = {}
    thr_grid = np.linspace(0.05, 0.95, 37)

    for k in sorted(ep_by_k.keys()):
        ep_evals_by_v: dict[str, list[dict]] = {name: [] for name, _ in variant_specs}
        ep_probs_by_v: dict[str, list[tuple[list[dict], dict[str, np.ndarray]]]] = {name: [] for name, _ in variant_specs}
        diag_k: dict[str, dict[str, Any]] = {}

        for name, _ in variant_specs:
            pipe = pipes[name]
            active_eps = 0
            bank_sizes: list[int] = []
            t_hs: list[float] = []
            mean_pens: list[float] = []
            abstain_reasons: dict[str, int] = defaultdict(int)

            for ep in ep_by_k[k]:
                sup_ids = ep["support_ids"]
                q_ids = ep["query_ids"]
                q_recs = [records_by_id[qid] for qid in q_ids]
                out = pipe.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
                p512, s32 = out.to_legacy_outputs()
                ep_evals_by_v[name].append(evaluate_episode(q_recs, p512, s32))
                ep_probs_by_v[name].append((q_recs, p512))

                cd = out.calibration.diagnostics
                if out.calibration.hard_neg_active:
                    active_eps += 1
                    bank_sizes.append(int(cd.get("hard_neg_bank_tokens", 0)))
                    if out.calibration.hard_neg_t_h is not None:
                        t_hs.append(float(out.calibration.hard_neg_t_h))
                else:
                    abstain_reasons[str(cd.get("hard_neg_reason", "disabled"))] += 1
                for qid in q_ids:
                    mean_pens.append(float(out.semantic_by_id[qid].diagnostics.get("hard_neg_mean_penalty", 0.0)))

            methods_res[name][str(k)] = aggregate_episodes(ep_evals_by_v[name])
            diag_k[name] = {
                "active_episodes": active_eps,
                "total_episodes": len(ep_by_k[k]),
                "active_rate": float(active_eps / max(1, len(ep_by_k[k]))),
                "mean_bank_tokens_when_active": float(np.mean(bank_sizes)) if bank_sizes else 0.0,
                "mean_t_h_when_active": float(np.mean(t_hs)) if t_hs else None,
                "mean_query_penalty": float(np.mean(mean_pens)) if mean_pens else 0.0,
                "abstain_reasons": dict(abstain_reasons),
            }
            m = methods_res[name][str(k)]["macro"]
            print(
                f"[M1 K={k}] {name:28s}: Dice={m['dice']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f} "
                f"NegFP={m['neg_fp']:.4f} NegMax={m['neg_max_fp']:.4f} ({m['neg_worst_id']}) "
                f"PosBgFP={m['pos_bg_fp']:.4f} AP512={m['ap_512']:.4f} Active={active_eps}/40",
                flush=True,
            )

        diag_by_k[str(k)] = diag_k

        # Equal-Recall diagnostic curves at baseline R's 0.50 recall operating point
        base_rec_at_05 = methods_res["M1_Base_R"][str(k)]["macro"]["recall"]
        eq_rec_k: dict[str, Any] = {"baseline_R_pos_recall_at_050": base_rec_at_05, "variants": {}}
        for name, _ in variant_specs:
            curve = _evaluate_threshold_sweep_for_equal_recall(ep_probs_by_v[name], thr_grid)
            eq_rec_k["variants"][name] = {
                "at_base_recall": _interp_at_target_recall(curve, base_rec_at_05),
                "at_base_recall_minus_002": _interp_at_target_recall(curve, max(0.05, base_rec_at_05 - 0.02)),
                "curve": curve,
            }
        equal_recall_by_k[str(k)] = eq_rec_k

    # Comparisons: Cand vs Base_R, Ctrl vs Base_R, and Cand vs Ctrl (isolating confusion selection from budget)
    comparisons = {
        "HardNegH_vs_BaseR": {
            str(k): compare_methods(methods_res["M1_Cand_HardNegH"][str(k)], methods_res["M1_Base_R"][str(k)])
            for k in sorted(ep_by_k.keys())
        },
        "RandomBG_vs_BaseR": {
            str(k): compare_methods(methods_res["M1_Ctrl_EqualBudgetRandomBG"][str(k)], methods_res["M1_Base_R"][str(k)])
            for k in sorted(ep_by_k.keys())
        },
        "HardNegH_vs_RandomBG": {
            str(k): compare_methods(methods_res["M1_Cand_HardNegH"][str(k)], methods_res["M1_Ctrl_EqualBudgetRandomBG"][str(k)])
            for k in sorted(ep_by_k.keys())
        },
    }

    gate_fp_pass_ks = [
        k for k in ("1", "3", "5") if comparisons["HardNegH_vs_BaseR"][k]["gates"]["Gate_FP"]
    ]
    gate_fp_overall = len(gate_fp_pass_ks) >= 2 and ("1" in gate_fp_pass_ks or "3" in gate_fp_pass_ks)

    # Check if equal-recall Neg Site FP actually improves (not just global probability lowering)
    eq_recall_improved_ks = []
    for k in ("1", "3", "5"):
        base_nfp = equal_recall_by_k[k]["variants"]["M1_Base_R"]["at_base_recall"]["neg_site_fp_at_recall"]
        cand_nfp = equal_recall_by_k[k]["variants"]["M1_Cand_HardNegH"]["at_base_recall"]["neg_site_fp_at_recall"]
        if cand_nfp < base_nfp - 0.001:
            eq_recall_improved_ks.append(k)

    summary = {
        "schema": "pv-2m-dinov3sat-m1-hardneg-dev-v1",
        "device": run_device,
        "elapsed_sec": float(time.perf_counter() - t0),
        "equivalence_L_vs_R_120ep": equiv_audit,
        "verdict": {
            "equivalence_120ep_pass": bool(all(v["all_passed"] for v in equiv_audit.values())),
            "gate_fp_passed_ks": gate_fp_pass_ks,
            "gate_fp_overall_pass": bool(gate_fp_overall),
            "equal_recall_neg_fp_improved_ks": eq_recall_improved_ks,
            "module_1_mechanism_accepted": bool(gate_fp_overall and len(eq_recall_improved_ks) >= 2),
        },
        "diagnostics_by_k": diag_by_k,
        "equal_recall_by_k": equal_recall_by_k,
        "comparisons": comparisons,
        "methods": methods_res,
    }
    write_json(output_dir / "m1_hardneg_summary.json", summary)
    print("M1 Summary Verdict:", summary["verdict"], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 120-ep L vs R equivalence audit + Module ① hard-negative Dev ablations.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--cache-pt", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, default=REPO_ROOT / "configs" / "three_module_exp_m1_hardneg.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "m1_hardneg_dev")
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_m1_hardneg_dev_experiment(
        split_dir=args.split_dir,
        cache_pt=args.cache_pt,
        config_json=args.config_json,
        output_dir=args.output_dir,
        label_dir=args.label_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
