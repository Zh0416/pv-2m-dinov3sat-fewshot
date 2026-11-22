from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("results/stage2_independent")
    master = json.loads((root / "stage2_master_summary.json").read_text(encoding="utf-8"))
    eq = json.loads((root / "equivalence_summary.json").read_text(encoding="utf-8"))
    m1 = json.loads((root / "m1_hardneg_summary.json").read_text(encoding="utf-8"))
    m2 = json.loads((root / "m2_cropzoom_stage0_summary.json").read_text(encoding="utf-8"))
    m3 = json.loads((root / "m3_deformgrad_summary.json").read_text(encoding="utf-8"))

    print("=== MASTER VERDICTS ===")
    print(json.dumps(master, indent=2))

    print("\n=== 0. 120-EPISODE L vs R EQUIVALENCE ===")
    print(json.dumps(eq, indent=2))

    print("\n=== 1. MODULE 1: HARD-NEGATIVE REVERIFICATION ===")
    for k in ("1", "3", "5"):
        print(f"\n--- M1 K={k} ---")
        for vn in ("M1_Base_R", "M1_Ctrl_EqualBudgetRandomBG", "M1_Cand_HardNegH"):
            mc = m1["methods"][vn][k]["macro"]
            dg = m1["diagnostics_by_k"][k][vn]
            eqr = m1["equal_recall_by_k"][k]["variants"][vn]["at_base_recall"]
            print(
                f"  {vn:28s}: Dice={mc['dice']:.4f} Prec={mc['precision']:.4f} Rec={mc['recall']:.4f} "
                f"NegFP={mc['neg_fp']:.4f} NegClean={mc['neg_clean']:.4f} NegMax={mc['neg_max_fp']:.4f} "
                f"PosBgFP={mc['pos_bg_fp']:.4f} AP512={mc['ap_512']:.4f} | "
                f"EqRec(NegFP={eqr['neg_site_fp_at_recall']:.4f}, BgFP={eqr['pos_bg_fp_at_recall']:.4f}, thr={eqr['matched_threshold']:.3f}) | "
                f"Act={dg['active_episodes']}/40 Bank={dg['mean_bank_tokens']:.1f}"
            )
        for cname in ("HardNegH_vs_BaseR", "RandomBG_vs_BaseR", "HardNegH_vs_RandomBG"):
            c = m1["comparisons"][cname][k]
            print(
                f"  {cname:22s}: dDice={c['dice']['mean_delta']:+.4f} CI95={[round(x,4) for x in c['dice']['ci95']]} | "
                f"dNegFP={c['neg_fp']['mean_delta']:+.4f} CI95={[round(x,4) for x in c['neg_fp']['ci95']]} folds={c['neg_fp']['improved_folds']}/4 | "
                f"dPrec={c['precision']['mean_delta']:+.4f} folds={c['precision']['positive_folds']}/4 | "
                f"dRec={c['recall']['mean_delta']:+.4f} | Gate_FP={c['gates']['Gate_FP']}"
            )

    print("\n=== 2. MODULE 2: STAGE 0 COVERAGE AUDIT & FULL-TOKEN CONTROL ===")
    for k in ("1", "3", "5"):
        print(f"\n--- M2 K={k} ---")
        print("  Stage0:", json.dumps(m2["stage0_coverage_by_k"][k]))
        print("  Seedless:", json.dumps(m2["seedless_subset_by_k"][k]))
        for vn in ("M2_NoRecovery", "M2_SelfRescue_R", "M2_Ctrl_FullTokenReverify"):
            mc = m2["methods"][vn][k]["macro"]
            sm = m2["methods"][vn][k]["strata"]["small_lt5pct"]["recall"]
            ty = m2["methods"][vn][k]["strata"]["tiny_lt1pct"]["recall"]
            print(
                f"  {vn:26s}: Dice={mc['dice']:.4f} Rec={mc['recall']:.4f} Sm<5R={sm:.4f} Tiny<1R={ty:.4f} "
                f"Det={mc['detected']:.4f} NegFP={mc['neg_fp']:.4f} PosBgFP={mc['pos_bg_fp']:.4f} AP512={mc['ap_512']:.4f}"
            )
        for cname in ("SelfRescueR_vs_NoRecovery", "FullTokenCtrl_vs_SelfRescueR"):
            c = m2["comparisons"][cname][k]
            print(
                f"  {cname:28s}: dDice={c['dice']['mean_delta']:+.4f} CI95={[round(x,4) for x in c['dice']['ci95']]} | "
                f"dRec={c['recall']['mean_delta']:+.4f} folds={c['recall']['positive_folds']}/4 | "
                f"dSm<5R={c['small_lt5pct_recall_delta']:+.4f} | dNegFP={c['neg_fp']['mean_delta']:+.4f} | Gate_FN={c['gates']['Gate_FN']}"
            )

    print("\n=== 3. MODULE 3: DEFORMABLE CONTOUR DP + GRADIENT ===")
    for k in ("1", "3", "5"):
        print(f"\n--- M3 K={k} ---")
        print("  MoveDiag:", json.dumps(m3["diagnostics_by_k"][k]))
        for vn in ("M3_Off_R", "M3_Rigid_Grad", "M3_Deformable_Grad", "Offline_Rigid_GT_Ref", "Offline_Deform_GT_Ref", "Offline_Loose_PM8px"):
            if k != "5" and vn.startswith("Offline_"):
                continue
            mc = m3["methods"][vn][k]["macro"]
            sm = m3["methods"][vn][k]["strata"]["small_lt5pct"]["recall"]
            print(
                f"  {vn:22s}: BF1@1={mc['bf1_1px']:.4f} BF1@2={mc['boundary_f1']:.4f} Dice={mc['dice']:.4f} "
                f"IoU={mc['iou']:.4f} Rec={mc['recall']:.4f} Sm<5R={sm:.4f} NegFP={mc['neg_fp']:.4f}"
            )
        for cname in ("DeformGrad_vs_OffR", "RigidGrad_vs_OffR", "DeformGrad_vs_RigidGrad"):
            c = m3["comparisons"][cname][k]
            print(
                f"  {cname:24s}: dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI95={[round(x,4) for x in c['boundary_f1']['ci95']]} folds={c['boundary_f1']['positive_folds']}/4 | "
                f"dBF1@1={c['bf1_1px']['mean_delta']:+.4f} CI95={[round(x,4) for x in c['bf1_1px']['ci95']]} | "
                f"dDice={c['dice']['mean_delta']:+.4f} | dSm<5R={c['small_lt5pct_recall_delta']:+.4f} | dNegFP={c['neg_fp']['mean_delta']:+.4f}"
            )
    off_c = m3["comparisons"]["Offline_DeformGTRef_vs_RigidGTRef_5shot"]
    print(
        f"\n  Offline_DeformGTRef_vs_RigidGTRef_5shot: dBF1@2={off_c['boundary_f1']['mean_delta']:+.4f} CI95={[round(x,4) for x in off_c['boundary_f1']['ci95']]} | "
        f"dBF1@1={off_c['bf1_1px']['mean_delta']:+.4f} CI95={[round(x,4) for x in off_c['bf1_1px']['ci95']]} | dDice={off_c['dice']['mean_delta']:+.4f}"
    )


if __name__ == "__main__":
    main()
