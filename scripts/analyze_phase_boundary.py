from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    summary_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/phase_boundary_dev/phase_boundary_summary.json")
    d = json.loads(summary_path.read_text(encoding="utf-8"))

    variants_order = [
        "Oracle_PM8px",
        "A_M3_Pruned",
        "B_GradAlign",
        "C_TexVar",
        "D_BilateralTerm",
        "E_FreqAmp_NoPhase",
        "F_FullPhase",
        "G_pi_2",
        "G_pi",
        "G_3pi_2",
        "G_Counterfactual_Mean",
        "G_FixedTauF_pi_2",
        "G_FixedTauF_pi",
        "G_FixedTauF_3pi_2",
        "G_FixedTauF_Mean",
        "H_NoSemGuard",
    ]

    for k in ("1", "3", "5"):
        print(f"\n================ K={k} MAIN & ABLATION TABLE ================")
        print(f"{'Variant':22s} {'BF1@1':>7s} {'BF1@2':>7s} {'Dice':>7s} {'IoU':>7s} {'Rec':>7s} {'Sm<5R':>7s} {'NegFP':>7s} {'Det':>7s} {'AP512*':>7s}")
        for v in variants_order:
            m = d["methods"][v][k]["macro"]
            sm = d["methods"][v][k]["strata"]["small_lt5pct"]["recall"]
            print(
                f"{v:22s} {m['bf1_1px']:7.4f} {m['boundary_f1']:7.4f} {m['dice']:7.4f} "
                f"{m['iou']:7.4f} {m['recall']:7.4f} {sm:7.4f} {m['neg_fp']:7.4f} "
                f"{m['detected']:7.4f} {m['ap_512']:7.4f}"
            )

    print("\n================ 5-SHOT ALL VARIANTS VS A_M3_Pruned ================")
    for v in variants_order:
        if v == "A_M3_Pruned":
            continue
        cdata = d["comparisons_vs_A"][v]["5"]
        b2 = cdata["boundary_f1"]
        b1 = cdata["bf1_1px"]
        dc = cdata["dice"]
        nf = cdata["neg_fp"]
        print(
            f"{v:22s} vs A: dBF1@2={b2['mean_delta']:+.4f} CI95=[{b2['ci95'][0]:+.4f},{b2['ci95'][1]:+.4f}] "
            f"folds={b2['positive_folds']}/4 ({[round(x,4) for x in b2['fold_deltas']]}) | "
            f"dBF1@1={b1['mean_delta']:+.4f} | dDice={dc['mean_delta']:+.4f} | dNegFP={nf['mean_delta']:+.4f}"
        )

    print("\n================ 5-SHOT CAUSAL PHASE COMPARISONS ================")
    for cname, cdata in d["phase_causal_comparisons_5shot"].items():
        b2 = cdata["boundary_f1"]
        b1 = cdata["bf1_1px"]
        dc = cdata["dice"]
        nf = cdata["neg_fp"]
        print(
            f"{cname:30s}: dBF1@2={b2['mean_delta']:+.4f} CI95=[{b2['ci95'][0]:+.4f},{b2['ci95'][1]:+.4f}] "
            f"CI97.5=[{b2['ci975_bonferroni'][0]:+.4f},{b2['ci975_bonferroni'][1]:+.4f}] folds={b2['positive_folds']}/4 | "
            f"dBF1@1={b1['mean_delta']:+.4f} | dDice={dc['mean_delta']:+.4f} | dNegFP={nf['mean_delta']:+.4f}"
        )

    print("\n================ 5-SHOT EQUAL-BUDGET COMPARISONS ================")
    for pkey, pdata in d["equal_budget_by_k"]["5"].items():
        pc = pdata["paired_comparison"]["boundary_f1"]
        pc1 = pdata["paired_comparison"]["bf1_1px"]
        print(
            f"{pkey:36s}: common_arc={pdata['common_evaluable_arc_fraction']:.3f} excl={pdata['condition_exclusion_rate']:.3f} "
            f"natA={pdata['natural_moved_arc_a_fraction']:.4f} natB={pdata['natural_moved_arc_b_fraction']:.4f} "
            f"matA={pdata['matched_moved_arc_a_fraction']:.4f} matB={pdata['matched_moved_arc_b_fraction']:.4f} "
            f"mean|d|A={pdata['mean_abs_d_a']:.2f} mean|d|B={pdata['mean_abs_d_b']:.2f} suff={pdata['sufficient_common_coverage']}"
        )
        print(
            f"    dBF1@2={pc['mean_delta']:+.4f} CI95=[{pc['ci95'][0]:+.4f},{pc['ci95'][1]:+.4f}] "
            f"CI97.5=[{pc['ci975_bonferroni'][0]:+.4f},{pc['ci975_bonferroni'][1]:+.4f}] | dBF1@1={pc1['mean_delta']:+.4f}"
        )

    print("\n================ DIAGNOSTICS BY K ================")
    for k in ("1", "3", "5"):
        dk = d["diagnostics_by_k"][k]
        print(f"\n--- K={k} ---")
        print(
            f"TotalArc={dk['total_contour_arc_len_px']} ValidCarrierArc={dk['carrier_valid_arc_len_px']} "
            f"({dk['carrier_valid_arc_fraction']:.2%}) SmallValidCov={dk['small_target_carrier_valid_arc_fraction']:.2%}"
        )
        print("p_floor_mean:", dk["p_floor_mean"], "p_floor_fallbacks:", dk["p_floor_fallback_episodes"])
        print("Component fallbacks:", dk["component_fallback_counts"])
        print("tau_mean:", {kk: round(vv, 4) for kk, vv in dk["tau_mean_by_variant"].items()})
        print("tau_fallbacks:", dk["tau_fallback_episodes_by_variant"])
        print("mean_null_blocks:", {kk: round(vv, 1) for kk, vv in dk["mean_null_blocks_by_variant"].items()})
        print("Variant movement:")
        for vn, mv in dk["variant_movement"].items():
            if vn == "A_M3_Pruned":
                continue
            print(
                f"  {vn:22s}: moved_arc={mv['moved_arc_fraction']:.2%} (of valid:{mv['moved_arc_fraction_of_valid_carrier']:.2%}) "
                f"in={mv['inward_fraction_of_moved']:.1%} out={mv['outward_fraction_of_moved']:.1%} "
                f"mean|d|={mv['mean_abs_d_moved_px']:.2f}px net_area={mv['mean_net_area_delta_px_per_ep_img']:+.1f}px"
            )

    print("\n================ 5-SHOT REPRESENTATIVE SITE CASES (F vs A) ================")
    f_sites = d["methods"]["F_FullPhase"]["5"]["sites"]
    a_sites = d["methods"]["A_M3_Pruned"]["5"]["sites"]
    pos_diffs = []
    for s, fv in f_sites.items():
        if fv["positive_images"] > 0:
            av = a_sites[s]
            pos_diffs.append((fv["boundary_f1"] - av["boundary_f1"], s, av["boundary_f1"], fv["boundary_f1"], fv["dice"] - av["dice"]))
    pos_diffs.sort(reverse=True)
    print("Top 5 positive-site gains (BF1@2px):")
    for row in pos_diffs[:5]:
        print(f"  {row[1]:18s}: A={row[2]:.4f} -> F={row[3]:.4f} (dBF1@2={row[0]:+.4f}, dDice={row[4]:+.4f})")
    print("Worst 5 positive-site drops (BF1@2px):")
    for row in pos_diffs[-5:]:
        print(f"  {row[1]:18s}: A={row[2]:.4f} -> F={row[3]:.4f} (dBF1@2={row[0]:+.4f}, dDice={row[4]:+.4f})")

    f_imgs = d["methods"]["F_FullPhase"]["5"]["images"]
    a_imgs = d["methods"]["A_M3_Pruned"]["5"]["images"]
    print("\nSelected panel images at K=5 (A vs F vs H):")
    h_imgs = d["methods"]["H_NoSemGuard"]["5"]["images"]
    for qid in ("正样本/PV-73.tif", "正样本/PV-49.tif", "正样本/PV-55.tif", "正样本/PV-80.tif", "负样本/PV-93.tif"):
        ai, fi, hi = a_imgs[qid], f_imgs[qid], h_imgs[qid]
        if ai["positive"]:
            print(
                f"  {qid:18s}: A(BF1@2={ai['boundary_f1']:.4f}, BF1@1={ai['bf1_1px']:.4f}, Dice={ai['dice']:.4f}) -> "
                f"F(BF1@2={fi['boundary_f1']:.4f}, BF1@1={fi['bf1_1px']:.4f}, Dice={fi['dice']:.4f}) | "
                f"H(BF1@2={hi['boundary_f1']:.4f}, Dice={hi['dice']:.4f})"
            )
        else:
            print(
                f"  {qid:18s}: A(NegFP={ai['neg_fp']:.4f}) -> F(NegFP={fi['neg_fp']:.4f}) | H(NegFP={hi['neg_fp']:.4f})"
            )


if __name__ == "__main__":
    main()
