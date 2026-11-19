from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "results" / "v2_dev" / "v2_dev_summary.json").read_text(encoding="utf-8"))

    print("=== V2 DEV ABLATIONS SUMMARY (79 IMAGES / 4 FOLDS x 10 EPISODES) ===")
    for m_name, by_k in data["methods"].items():
        print(f"\n--- {m_name} ---")
        for k in ("1", "3", "5"):
            agg = by_k[k]
            m = agg["macro"]
            sd = agg["strata"].get("small_lt5pct", {}).get("dice", 0.0)
            sr = agg["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            imgs = agg["images"]
            diag = (
                f"PV93={imgs['负样本/PV-93.tif']['neg_fp']:.3f} "
                f"PV100={imgs['负样本/PV-100.tif']['neg_fp']:.3f} "
                f"PV96={imgs['负样本/PV-96.tif']['neg_fp']:.3f} | "
                f"PV55r={imgs['正样本/PV-55.tif']['recall']:.3f} "
                f"PV49r={imgs['正样本/PV-49.tif']['recall']:.3f} "
                f"PV73r={imgs['正样本/PV-73.tif']['recall']:.3f}"
            )
            print(
                f"  K={k}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} AP512={m['ap_512']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} BF1={m['boundary_f1']:.4f} "
                f"PosBg={m['pos_bg_fp']:.4f} NegFP={m['neg_fp']:.4f} NegClean={m['neg_clean']:.4f} "
                f"SmallD={sd:.4f} SmallR={sr:.4f} || {diag}"
            )

    print("\n=== V2 PAIRED COMPARISONS & ACCEPTANCE GATES ===")
    for cname, by_k in data["comparisons"].items():
        print(f"\n[{cname}]")
        for k in ("1", "3", "5"):
            c = by_k[k]
            g = c["gates"]
            print(
                f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x, 4) for x in c['dice']['ci95']]} "
                f"(f={c['dice']['positive_folds']}/4) | "
                f"dAP512={c['ap_512']['mean_delta']:+.4f} (f={c['ap_512']['positive_folds']}/4) | "
                f"dRec={c['recall']['mean_delta']:+.4f} dPrec={c['precision']['mean_delta']:+.4f} | "
                f"dSmallR={c['small_lt5pct_recall_delta']:+.4f} dBF1={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x, 4) for x in c['boundary_f1']['ci95']]} | "
                f"dNegFP={c['neg_fp']['mean_delta']:+.4f} (f={c['neg_fp']['improved_folds']}/4, rel={c['neg_fp']['relative_reduction']:+.1%}) | "
                f"Gates[FP={g['Gate_FP']} FN={g['Gate_FN']} Net={g['Gate_Net']}]"
            )


if __name__ == "__main__":
    main()
