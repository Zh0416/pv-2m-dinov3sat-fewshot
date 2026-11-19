from __future__ import annotations

import json
from pathlib import Path


def _print_suite(title: str, methods_dict: dict) -> None:
    print(f"\n=== {title} ===")
    for m_name, by_k in methods_dict.items():
        print(f"\n--- {m_name} ---")
        for k in ("1", "3", "5"):
            agg = by_k[k]
            m = agg["macro"]
            sd = agg["strata"].get("small_lt5pct", {}).get("dice", 0.0)
            sr = agg["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            print(
                f"  K={k}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} AP512={m['ap_512']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} BF1={m['boundary_f1']:.4f} "
                f"PosBg={m['pos_bg_fp']:.4f} NegFP={m['neg_fp']:.4f} NegClean={m['neg_clean']:.4f} "
                f"SmallD={sd:.4f} SmallR={sr:.4f}"
            )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "results" / "v2_final" / "v2_sealed_and_ext185_summary.json").read_text(encoding="utf-8"))
    print("Ext185 dataset counts:", data["ext185_stats"])

    _print_suite("V2 SEALED TEST (21 IMAGES / 19 SITES)", data["sealed21"]["methods"])
    print("\n=== SEALED TEST PAIRED COMPARISONS ===")
    for cname, by_k in data["sealed21"]["comparisons"].items():
        print(f"\n[{cname}]")
        for k in ("1", "3", "5"):
            c = by_k[k]
            print(
                f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x, 4) for x in c['dice']['ci95']]} | "
                f"dAP512={c['ap_512']['mean_delta']:+.4f} dRec={c['recall']['mean_delta']:+.4f} dPrec={c['precision']['mean_delta']:+.4f} | "
                f"dSmallR={c['small_lt5pct_recall_delta']:+.4f} dBF1={c['boundary_f1']['mean_delta']:+.4f} | "
                f"dNegFP={c['neg_fp']['mean_delta']:+.4f} (rel={c['neg_fp']['relative_reduction']:+.1%})"
            )

    _print_suite("EXT185 PROTOCOL A: CROSS-DATASET (pv-100scence Support -> 185 Query)", data["ext185"]["cross_dataset"])
    _print_suite("EXT185 PROTOCOL B: IN-DOMAIN FEW-SHOT (K from 185 -> Remaining Query)", data["ext185"]["in_domain"])


if __name__ == "__main__":
    main()
