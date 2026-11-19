from __future__ import annotations

import json
from pathlib import Path

from pvfs.metrics import compare_methods


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "results" / "iter1" / "iter1_summary.json").read_text(encoding="utf-8"))

    print("=== ITERATION 1 METHODS SUMMARY ===")
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
                f"PV80bg={imgs['正样本/PV-80.tif']['pos_bg_fp']:.3f} "
                f"PV21bg={imgs['正样本/PV-21.tif']['pos_bg_fp']:.3f} "
                f"PV12bg={imgs['正样本/PV-12.tif']['pos_bg_fp']:.3f} | "
                f"PV73r={imgs['正样本/PV-73.tif']['recall']:.3f} "
                f"PV55r={imgs['正样本/PV-55.tif']['recall']:.3f} "
                f"PV49r={imgs['正样本/PV-49.tif']['recall']:.3f}"
            )
            print(
                f"  K={k}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} AP512={m['ap_512']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} BF1={m['boundary_f1']:.4f} "
                f"PosBg={m['pos_bg_fp']:.4f} NegFP={m['neg_fp']:.4f} NegClean={m['neg_clean']:.4f} "
                f"SmallD={sd:.4f} SmallR={sr:.4f} || {diag}"
            )

    print("\n=== PAIRED COMPARISONS & ACCEPTANCE GATES ===")
    extra_pairs = [
        ("M1_Scale_Core", "B0_32_Ref"),
        ("M1_Full", "M1_Scale"),
        ("M1_Scale_Core_Calib", "M1_Scale"),
    ]
    comps = dict(data["comparisons"])
    for cand, ref in extra_pairs:
        cname = f"{cand}_vs_{ref}"
        if cname not in comps:
            comps[cname] = {
                k: compare_methods(data["methods"][cand][k], data["methods"][ref][k])
                for k in ("1", "3", "5")
            }

    for cname, by_k in comps.items():
        print(f"\n[{cname}]")
        for k in ("1", "3", "5"):
            c = by_k[k]
            g = c["gates"]
            print(
                f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x, 4) for x in c['dice']['ci95']]} "
                f"(f={c['dice']['positive_folds']}/4) | "
                f"dAP512={c['ap_512']['mean_delta']:+.4f} CI{[round(x, 4) for x in c['ap_512']['ci95']]} "
                f"(f={c['ap_512']['positive_folds']}/4) | "
                f"dRec={c['recall']['mean_delta']:+.4f} dPrec={c['precision']['mean_delta']:+.4f} "
                f"dSmallR={c['small_lt5pct_recall_delta']:+.4f} dBF1={c['boundary_f1']['mean_delta']:+.4f} | "
                f"dNegFP={c['neg_fp']['mean_delta']:+.4f} (f={c['neg_fp']['improved_folds']}/4, rel={c['neg_fp']['relative_reduction']:+.1%}) | "
                f"Gates[FP={g['Gate_FP']} FN={g['Gate_FN']} Net={g['Gate_Net']}]"
            )

    for variant in ("M1_Scale", "M1_Full"):
        for k in ("1", "3", "5"):
            imgs = list(data["methods"][variant][k]["images"].values())
            pos = [v for v in imgs if v["positive"]]
            neg = [v for v in imgs if not v["positive"]]
            print(f"\n=== {variant} K={k} WORST RESIDUAL ERRORS ===")
            print("  Worst FN:", [(x["id"], round(x["recall"], 3), round(x["dice"], 3), round(x["target_fraction"], 4)) for x in sorted(pos, key=lambda v: v["recall"])[:6]])
            print("  Worst NegFP:", [(x["id"], round(x["neg_fp"], 4), round(x["neg_clean"], 2)) for x in sorted(neg, key=lambda v: -v["neg_fp"])[:5]])
            print("  Worst PosBgFP:", [(x["id"], round(x["pos_bg_fp"], 4), round(x["precision"], 3)) for x in sorted(pos, key=lambda v: -v["pos_bg_fp"])[:5]])


if __name__ == "__main__":
    main()
