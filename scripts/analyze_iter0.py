from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "results" / "iter0" / "iter0_summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    print("=== ITERATION 0 SUMMARY ===")
    print("Device:", data["device"], "Backbone:", data["backbone"])
    for m_name, by_k in data["methods"].items():
        print(f"\n--- {m_name} ---")
        for k_str, agg in sorted(by_k.items(), key=lambda x: int(x[0])):
            m = agg["macro"]
            cal = agg["calibration_summary"]
            orc = data["oracle_raw_threshold_diagnostic"][m_name][k_str]
            print(
                f"K={k_str}: Dice={m['dice']:.4f}, IoU={m['iou']:.4f}, AP512={m['ap_512']:.4f}, "
                f"AP32={m['ap_32']:.4f}, Recall={m['recall']:.4f}, Prec={m['precision']:.4f}, "
                f"PosBgFP={m['pos_bg_fp']:.4f}, BF1={m['boundary_f1']:.4f}, DetRec={m['detected']:.4f} | "
                f"NegFP={m['neg_fp']:.4f}, NegClean={m['neg_clean']:.4f}, NegMaxFP={m['neg_max_fp']:.4f} ({m['neg_worst_id']}) | "
                f"Calib(b={cal['bias_mean']:.3f}+-{cal['bias_std']:.3f}, T={cal['temp_mean']:.3f}) | "
                f"OracleRaw(thr={orc['best_thr']:.3f}, imgDice={orc['best_img_dice']:.4f}, imgRec={orc['best_img_recall']:.4f}, imgPrec={orc['best_img_precision']:.4f}, negFP={orc['best_img_neg_fp']:.4f})"
            )
            for s_name, s_val in agg["strata"].items():
                print(
                    f"   [{s_name} n={s_val['count']}] Dice={s_val['dice']:.4f} Recall={s_val['recall']:.4f} Prec={s_val['precision']:.4f} Det={s_val['detected']:.4f}"
                )

    print("\n=== B0_Contrast vs B0_PosOnly ===")
    for k_str, cmp in sorted(data["comparisons_contrast_vs_posonly"].items(), key=lambda x: int(x[0])):
        print(
            f"K={k_str}: dDice={cmp['dice']['mean_delta']:+.4f} {cmp['dice']['ci95']} (folds {cmp['dice']['positive_folds']}/4), "
            f"dRec={cmp['recall']['mean_delta']:+.4f}, dPrec={cmp['precision']['mean_delta']:+.4f}, "
            f"dAP512={cmp['ap_512']['mean_delta']:+.4f}, dNegFP={cmp['neg_fp']['mean_delta']:+.4f} "
            f"(improved folds {cmp['neg_fp']['improved_folds']}/4) | Gates={cmp['gates']}"
        )

    for k_str in ("1", "3", "5"):
        tax = data["error_taxonomy"][k_str]
        print(f"\n=== B0_Contrast K={k_str} Worst FN (Misses) ===")
        for x in tax["worst_fn_positive_images"][:8]:
            print(f"  {x['id']} (area={x['target_fraction']:.2%}): rec={x['recall']:.3f}, dice={x['dice']:.3f}, prec={x['precision']:.3f}, det={x['detected']:.2f}")
        print(f"=== B0_Contrast K={k_str} Worst Neg FP (False Alarms) ===")
        for x in tax["worst_neg_fp_images"][:8]:
            print(f"  {x['id']}: neg_fp={x['neg_fp']:.4f}, clean={x['neg_clean']:.2f}, maxP={x['max_prob_mean']:.3f}")
        print(f"=== B0_Contrast K={k_str} Worst Pos-Bg FP ===")
        for x in tax["worst_pos_bg_fp_images"][:8]:
            print(f"  {x['id']} (area={x['target_fraction']:.2%}): bg_fp={x['pos_bg_fp']:.4f}, prec={x['precision']:.3f}, dice={x['dice']:.3f}")


if __name__ == "__main__":
    main()
