from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvfs.metrics import compare_methods


def _max_macro_diff(a: dict, b: dict, keys: tuple[str, ...]) -> float:
    return float(max(abs(float(a["macro"][k]) - float(b["macro"][k])) for k in keys))


def _top_image_deltas(
    cand_imgs: dict[str, dict],
    ref_imgs: dict[str, dict],
    metric: str,
    *,
    positive_only: bool = True,
    top_n: int = 5,
) -> dict[str, list[dict]]:
    rows = []
    for qid, c in cand_imgs.items():
        r = ref_imgs[qid]
        if positive_only and not c["positive"]:
            continue
        if (not positive_only) and c["positive"]:
            continue
        delta = float(c[metric]) - float(r[metric])
        row = {
            "id": qid,
            "site": c["site"],
            "fold": int(c["fold"]),
            "target_fraction": round(float(c["target_fraction"]), 5),
            "ref": round(float(r[metric]), 4),
            "cand": round(float(c[metric]), 4),
            "delta": round(delta, 4),
        }
        if positive_only:
            row["dice_ref"] = round(float(r["dice"]), 4)
            row["dice_cand"] = round(float(c["dice"]), 4)
            row["dice_delta"] = round(float(c["dice"]) - float(r["dice"]), 4)
            if "bf1_1px" in c and "bf1_1px" in r:
                row["bf1_1px_ref"] = round(float(r["bf1_1px"]), 4)
                row["bf1_1px_cand"] = round(float(c["bf1_1px"]), 4)
                row["bf1_1px_delta"] = round(float(c["bf1_1px"]) - float(r["bf1_1px"]), 4)
        rows.append(row)
    rows.sort(key=lambda x: x["delta"], reverse=True)
    return {
        "top_gains": rows[:top_n],
        "worst_drops": list(reversed(rows[-top_n:])),
    }


def _worst_negatives(imgs: dict[str, dict], top_n: int = 5) -> list[dict]:
    negs = [v for v in imgs.values() if not v["positive"]]
    negs.sort(key=lambda x: float(x["neg_fp"]), reverse=True)
    return [
        {
            "id": v["id"],
            "site": v["site"],
            "fold": int(v["fold"]),
            "neg_fp": round(float(v["neg_fp"]), 4),
            "neg_clean": round(float(v["neg_clean"]), 4),
            "max_prob_mean": round(float(v.get("max_prob_mean", 0.0)), 4),
        }
        for v in negs[:top_n]
    ]


def _worst_missed_positives(imgs: dict[str, dict], top_n: int = 8) -> list[dict]:
    pos = [v for v in imgs.values() if v["positive"]]
    pos.sort(key=lambda x: (float(x["recall"]), float(x["detected"]), float(x["target_fraction"])))
    return [
        {
            "id": v["id"],
            "site": v["site"],
            "fold": int(v["fold"]),
            "target_fraction": round(float(v["target_fraction"]), 5),
            "recall": round(float(v["recall"]), 4),
            "detected": round(float(v["detected"]), 4),
            "dice": round(float(v["dice"]), 4),
        }
        for v in pos[:top_n]
    ]


def main() -> None:
    stage2_dir = ROOT / "results" / "stage2_independent"
    iter4 = json.loads((ROOT / "results" / "iter4" / "iter4_summary.json").read_text(encoding="utf-8"))
    eq = json.loads((stage2_dir / "equivalence_summary.json").read_text(encoding="utf-8"))
    m1 = json.loads((stage2_dir / "m1_hardneg_summary.json").read_text(encoding="utf-8"))
    m2 = json.loads((stage2_dir / "m2_cropzoom_stage0_summary.json").read_text(encoding="utf-8"))
    m3 = json.loads((stage2_dir / "m3_deformgrad_summary.json").read_text(encoding="utf-8"))
    master = json.loads((stage2_dir / "stage2_master_summary.json").read_text(encoding="utf-8"))

    # 1. Cross-run R baseline integrity check
    common_keys = ("dice", "iou", "recall", "precision", "pos_bg_fp", "boundary_f1", "detected", "ap_512", "ap_32", "neg_fp", "neg_clean", "neg_max_fp")
    baseline_consistency = {}
    for k in ("1", "3", "5"):
        ref_iter4 = iter4["methods"]["M3_Pruned_Ref"][k]
        ref_m1 = m1["methods"]["M1_Base_R"][k]
        ref_m2 = m2["methods"]["M2_SelfRescue_R"][k]
        ref_m3 = m3["methods"]["M3_Off_R"][k]
        baseline_consistency[k] = {
            "max_diff_iter4_vs_m1": _max_macro_diff(ref_iter4, ref_m1, common_keys),
            "max_diff_iter4_vs_m2": _max_macro_diff(ref_iter4, ref_m2, common_keys),
            "max_diff_iter4_vs_m3": _max_macro_diff(ref_iter4, ref_m3, common_keys),
            "max_diff_m1_vs_m3": _max_macro_diff(ref_m1, ref_m3, common_keys),
        }

    # 2. Stage 3 Integrated System = ①_orig(R) + ②_orig(R) + ③_DeformGrad
    # (Only Module ③ passed Stage 2 independent verification.)
    stage3_vs_b0 = {}
    r_vs_b0 = {}
    stage3_vs_r_loo = {}
    gate_net_checks = {}

    for k in ("1", "3", "5"):
        b0_k = iter4["methods"]["B0_32_Ref"][k]
        r_k = m3["methods"]["M3_Off_R"][k]
        s3_k = m3["methods"]["M3_Deformable_Grad"][k]

        cmp_s3_b0 = compare_methods(s3_k, b0_k)
        cmp_r_b0 = compare_methods(r_k, b0_k)
        cmp_s3_r = m3["comparisons"]["DeformGrad_vs_OffR"][k]

        stage3_vs_b0[k] = cmp_s3_b0
        r_vs_b0[k] = cmp_r_b0
        stage3_vs_r_loo[k] = cmp_s3_r
        gate_net_checks[k] = {
            "Gate_Net_per_k": bool(cmp_s3_b0["gates"]["Gate_Net"]),
            "dice_ci_low_gt_0": bool(cmp_s3_b0["dice"]["ci95"][0] > 0.0),
            "dice_delta": round(cmp_s3_b0["dice"]["mean_delta"], 4),
            "dice_ci95": [round(x, 4) for x in cmp_s3_b0["dice"]["ci95"]],
            "dice_folds": int(cmp_s3_b0["dice"]["positive_folds"]),
            "ap512_delta": round(cmp_s3_b0["ap_512"]["mean_delta"], 4),
            "ap512_ci95": [round(x, 4) for x in cmp_s3_b0["ap_512"]["ci95"]],
            "ap512_folds": int(cmp_s3_b0["ap_512"]["positive_folds"]),
            "neg_fp_delta": round(cmp_s3_b0["neg_fp"]["mean_delta"], 4),
            "neg_fp_ci95": [round(x, 4) for x in cmp_s3_b0["neg_fp"]["ci95"]],
            "neg_fp_improved_folds": int(cmp_s3_b0["neg_fp"]["improved_folds"]),
            "bf1_2px_delta": round(cmp_s3_b0["boundary_f1"]["mean_delta"], 4),
            "bf1_2px_ci95": [round(x, 4) for x in cmp_s3_b0["boundary_f1"]["ci95"]],
            "small_lt5pct_recall_delta": round(cmp_s3_b0["small_lt5pct_recall_delta"], 4),
        }

    overall_gate_net_pass = (
        all(gate_net_checks[k]["Gate_Net_per_k"] for k in ("1", "3", "5"))
        and gate_net_checks["1"]["dice_ci_low_gt_0"]
        and gate_net_checks["3"]["dice_ci_low_gt_0"]
    )

    # 3. Failure / Fallback / Case Diagnostics
    diagnostics = {
        "module_1_hardneg": {},
        "module_2_recovery": {},
        "module_3_deformgrad": {},
    }
    for k in ("1", "3", "5"):
        # M1 worst negatives across Base / RandomBG / HardNegH
        m1_base_imgs = m1["methods"]["M1_Base_R"][k]["images"]
        m1_rand_imgs = m1["methods"]["M1_Ctrl_EqualBudgetRandomBG"][k]["images"]
        m1_hard_imgs = m1["methods"]["M1_Cand_HardNegH"][k]["images"]
        worst_negs = _worst_negatives(m1_base_imgs, top_n=5)
        for row in worst_negs:
            qid = row["id"]
            row["rand_bg_neg_fp"] = round(float(m1_rand_imgs[qid]["neg_fp"]), 4)
            row["hard_neg_fp"] = round(float(m1_hard_imgs[qid]["neg_fp"]), 4)
            row["hard_vs_base_delta"] = round(float(m1_hard_imgs[qid]["neg_fp"]) - float(m1_base_imgs[qid]["neg_fp"]), 4)
        diagnostics["module_1_hardneg"][k] = {
            "worst_negatives": worst_negs,
            "neg_deltas_hard_vs_base": _top_image_deltas(m1_hard_imgs, m1_base_imgs, "neg_fp", positive_only=False, top_n=5),
        }

        # M2 worst missed positives & FullToken spillover on negatives
        m2_r_imgs = m2["methods"]["M2_SelfRescue_R"][k]["images"]
        m2_no_imgs = m2["methods"]["M2_NoRecovery"][k]["images"]
        m2_ft_imgs = m2["methods"]["M2_Ctrl_FullTokenReverify"][k]["images"]
        missed = _worst_missed_positives(m2_r_imgs, top_n=8)
        for row in missed:
            qid = row["id"]
            row["no_rescue_recall"] = round(float(m2_no_imgs[qid]["recall"]), 4)
            row["full_token_recall"] = round(float(m2_ft_imgs[qid]["recall"]), 4)
        ft_neg_spill = _top_image_deltas(m2_ft_imgs, m2_r_imgs, "neg_fp", positive_only=False, top_n=5)
        diagnostics["module_2_recovery"][k] = {
            "worst_missed_positives_in_R": missed,
            "full_token_neg_fp_spillover": ft_neg_spill["top_gains"],
            "stage0_coverage": m2["stage0_coverage_by_k"][k],
            "seedless_subset": m2["seedless_subset_by_k"][k],
        }

        # M3 top gains & worst drops (DeformGrad vs Off_R) and strata
        m3_off_imgs = m3["methods"]["M3_Off_R"][k]["images"]
        m3_def_imgs = m3["methods"]["M3_Deformable_Grad"][k]["images"]
        m3_rig_imgs = m3["methods"]["M3_Rigid_Grad"][k]["images"]
        diagnostics["module_3_deformgrad"][k] = {
            "bf1_2px_deform_vs_off": _top_image_deltas(m3_def_imgs, m3_off_imgs, "boundary_f1", positive_only=True, top_n=5),
            "bf1_2px_deform_vs_rigid": _top_image_deltas(m3_def_imgs, m3_rig_imgs, "boundary_f1", positive_only=True, top_n=5),
            "strata_off_r": m3["methods"]["M3_Off_R"][k]["strata"],
            "strata_rigid_grad": m3["methods"]["M3_Rigid_Grad"][k]["strata"],
            "strata_deform_grad": m3["methods"]["M3_Deformable_Grad"][k]["strata"],
            "movement_diagnostics": m3["diagnostics_by_k"][k],
        }

    # 4. Summary object
    out = {
        "schema": "three_module_stage3_integration_v1",
        "integrated_configuration": {
            "module_1_semantic": "Original R (M3_Pruned_Ref) -- Hard-Negative H candidate REJECTED in Stage 2",
            "module_2_recovery": "Original R (Self-Rescue) -- Seedless Crop-Zoom stopped at Stage 0 (<25% coverage)",
            "module_3_boundary": "Enhanced M3_Deformable_Grad (Narrow-Band Multi-Control-Point DP + Scharr Gradient) -- PASSED all Stage 2 gates",
            "ap_512_note": "M3_Deformable_Grad refines the binary 512x512 mask (M_final); AP_512 and AP_32 are evaluated on upstream continuous P_coarse / scores_32 and are therefore identical to R.",
        },
        "baseline_consistency_max_abs_diff": baseline_consistency,
        "runtime_sec": {
            "equivalence_120ep": round(float(eq["elapsed_sec"]), 1),
            "m1_hardneg_120ep": round(float(m1["elapsed_sec"]), 1),
            "m2_stage0_and_fulltoken_120ep": round(float(m2["elapsed_sec"]), 1),
            "m3_deformgrad_120ep": round(float(m3["elapsed_sec"]), 1),
            "total_stage2_cpu_kernel": round(float(master["total_elapsed_sec"]), 1),
        },
        "stage3_overall_gate_net_pass": bool(overall_gate_net_pass),
        "stage3_gate_net_by_k": gate_net_checks,
        "stage3_vs_B0_32_Ref": stage3_vs_b0,
        "R_vs_B0_32_Ref": r_vs_b0,
        "stage3_LOO_replace_M3_with_orig_R": stage3_vs_r_loo,
        "R_internal_LOO_remove_M2_SelfRescue": {k: m2["comparisons"]["SelfRescueR_vs_NoRecovery"][k] for k in ("1", "3", "5")},
        "reference_macros": {
            "B0_32_Ref": {k: iter4["methods"]["B0_32_Ref"][k]["macro"] for k in ("1", "3", "5")},
            "R_M3_Pruned_Ref": {k: m3["methods"]["M3_Off_R"][k]["macro"] for k in ("1", "3", "5")},
            "Stage3_R_plus_M3_DeformGrad": {k: m3["methods"]["M3_Deformable_Grad"][k]["macro"] for k in ("1", "3", "5")},
        },
        "diagnostics": diagnostics,
    }

    out_path = stage2_dir / "stage3_integration_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    # Print concise terminal report
    print("\n=== CROSS-RUN BASELINE CONSISTENCY (max abs diff across 12 macro metrics) ===")
    print(json.dumps(baseline_consistency, indent=2))

    print(f"\n=== STAGE 3 INTEGRATED (R + M3_DeformGrad) vs B0_32_Ref | OVERALL GATE-NET PASS: {overall_gate_net_pass} ===")
    for k in ("1", "3", "5"):
        b0m = out["reference_macros"]["B0_32_Ref"][k]
        rm = out["reference_macros"]["R_M3_Pruned_Ref"][k]
        s3m = out["reference_macros"]["Stage3_R_plus_M3_DeformGrad"][k]
        g = gate_net_checks[k]
        rb = r_vs_b0[k]
        print(f"\n--- K={k} ---")
        print(f"  B0_32_Ref    : Dice={b0m['dice']:.4f} IoU={b0m['iou']:.4f} Rec={b0m['recall']:.4f} Prec={b0m['precision']:.4f} BF1@2={b0m['boundary_f1']:.4f} NegFP={b0m['neg_fp']:.4f} AP512={b0m['ap_512']:.4f}")
        print(f"  R (M3_Pruned): Dice={rm['dice']:.4f} IoU={rm['iou']:.4f} Rec={rm['recall']:.4f} Prec={rm['precision']:.4f} BF1@2={rm['boundary_f1']:.4f} BF1@1={rm['bf1_1px']:.4f} NegFP={rm['neg_fp']:.4f} AP512={rm['ap_512']:.4f}")
        print(f"  Stage3 (R+③) : Dice={s3m['dice']:.4f} IoU={s3m['iou']:.4f} Rec={s3m['recall']:.4f} Prec={s3m['precision']:.4f} BF1@2={s3m['boundary_f1']:.4f} BF1@1={s3m['bf1_1px']:.4f} NegFP={s3m['neg_fp']:.4f} AP512={s3m['ap_512']:.4f}")
        print(
            f"  Stage3 vs B0 : dDice={g['dice_delta']:+.4f} CI95={g['dice_ci95']} ({g['dice_folds']}/4) | "
            f"dAP512={g['ap512_delta']:+.4f} ({g['ap512_folds']}/4) | dNegFP={g['neg_fp_delta']:+.4f} ({g['neg_fp_improved_folds']}/4) | "
            f"dBF1@2={g['bf1_2px_delta']:+.4f} CI95={g['bf1_2px_ci95']} | dSm<5R={g['small_lt5pct_recall_delta']:+.4f} | Gate_Net={g['Gate_Net_per_k']}"
        )
        print(
            f"  R vs B0      : dDice={rb['dice']['mean_delta']:+.4f} CI95={[round(x,4) for x in rb['dice']['ci95']]} | "
            f"dBF1@2={rb['boundary_f1']['mean_delta']:+.4f} | dNegFP={rb['neg_fp']['mean_delta']:+.4f} | Gate_Net={rb['gates']['Gate_Net']}"
        )

    print("\n=== FAILURE / FALLBACK / TOP-DROP HIGHLIGHTS (K=5) ===")
    print("M1 Worst Negatives (K=5):")
    for row in diagnostics["module_1_hardneg"]["5"]["worst_negatives"]:
        print(" ", row)
    print("M1 Worst Negatives (K=1):")
    for row in diagnostics["module_1_hardneg"]["1"]["worst_negatives"]:
        print(" ", row)
    print("M2 Worst Missed Positives in R (K=5):")
    for row in diagnostics["module_2_recovery"]["5"]["worst_missed_positives_in_R"][:5]:
        print(" ", row)
    print("M2 FullToken NegFP Spillover (K=5):")
    for row in diagnostics["module_2_recovery"]["5"]["full_token_neg_fp_spillover"][:3]:
        print(" ", row)
    print("M3 DeformGrad vs Off_R Top 5 Gains (K=5, BF1@2px):")
    for row in diagnostics["module_3_deformgrad"]["5"]["bf1_2px_deform_vs_off"]["top_gains"]:
        print(" ", row)
    print("M3 DeformGrad vs Off_R Worst 5 Drops (K=5, BF1@2px):")
    for row in diagnostics["module_3_deformgrad"]["5"]["bf1_2px_deform_vs_off"]["worst_drops"]:
        print(" ", row)
    print("M3 Strata Comparison (K=5):")
    for name in ("strata_off_r", "strata_rigid_grad", "strata_deform_grad"):
        print(f"  {name}:", json.dumps(diagnostics["module_3_deformgrad"]["5"][name]))


if __name__ == "__main__":
    main()
