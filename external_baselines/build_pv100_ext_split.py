from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def _stratum(tf: float) -> str:
    if tf < 0.01:
        return "tiny_lt1pct"
    if tf < 0.05:
        return "small_1_5pct"
    if tf < 0.15:
        return "mid_5_15pct"
    return "large_ge15pct"


def build_pv100_ext_split(repo_root: Path) -> Path:
    split_dir = repo_root / "configs" / "locked_split"
    mf = json.loads((split_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    sp = json.loads((split_dir / "split_lock.json").read_text(encoding="utf-8"))
    assign = sp["assignment"]

    pos_by_site: dict[str, list[dict]] = defaultdict(list)
    neg_records: list[dict] = []

    for row in mf["images"]:
        rid = row["id"]
        info = assign[rid]
        rec = {
            "id": rid,
            "site": info["site"],
            "orig_partition": info["partition"],
            "positive": bool(row["positive"]),
            "target_fraction": float(row["target_fraction"]),
            "image_sha256": row["image_sha256"],
            "label_sha256": row["label_sha256"],
            "meters_per_pixel": list(row["meters_per_pixel"]),
        }
        if rec["positive"]:
            pos_by_site[rec["site"]].append(rec)
        else:
            neg_records.append(rec)

    # Stratify the 49 independent positive sites by mean target_fraction and select 10 (~20%) for support_pool
    site_meta = []
    for s, items in sorted(pos_by_site.items()):
        mean_tf = float(np.mean([x["target_fraction"] for x in items]))
        items_sorted = sorted(items, key=lambda x: (x["target_fraction"], x["id"]))
        rep = items_sorted[len(items_sorted) // 2]
        site_meta.append({"site": s, "mean_tf": mean_tf, "stratum": _stratum(mean_tf), "items": items_sorted, "rep": rep})

    by_st: dict[str, list[dict]] = defaultdict(list)
    for sm in site_meta:
        by_st[sm["stratum"]].append(sm)

    # Target 10 support sites proportionally across strata
    want_counts = {"tiny_lt1pct": 1, "small_1_5pct": 3, "mid_5_15pct": 3, "large_ge15pct": 3}
    sup_sites: list[dict] = []
    chosen_site_names: set[str] = set()
    for st in ("tiny_lt1pct", "small_1_5pct", "mid_5_15pct", "large_ge15pct"):
        arr = sorted(by_st[st], key=lambda x: (x["mean_tf"], x["site"]))
        want = want_counts[st]
        if len(arr) <= want:
            picks = arr
        else:
            idxs = np.linspace(0, len(arr) - 1, want).round().astype(int)
            seen: set[int] = set()
            picks = []
            for idx in idxs:
                cur = int(idx)
                while cur in seen and cur + 1 < len(arr):
                    cur += 1
                seen.add(cur)
                picks.append(arr[cur])
        for p in picks:
            sup_sites.append(p)
            chosen_site_names.add(p["site"])

    assert len(chosen_site_names) == 10, f"Expected 10 support sites, got {len(chosen_site_names)}"

    support_records: list[dict] = []
    query_records: list[dict] = []
    for sm in site_meta:
        if sm["site"] in chosen_site_names:
            for r in sm["items"]:
                support_records.append({**r, "ext_partition": "support_pool"})
        else:
            for r in sm["items"]:
                query_records.append({**r, "ext_partition": "query_set"})

    for r in sorted(neg_records, key=lambda x: x["id"]):
        query_records.append({**r, "ext_partition": "query_set"})

    # Verify zero site overlap & exact 100 total images
    sup_site_set = {r["site"] for r in support_records}
    qry_site_set = {r["site"] for r in query_records}
    assert len(sup_site_set & qry_site_set) == 0
    assert len(support_records) + len(query_records) == 100

    # Lock 5 deterministic episodes per K (K=1, 3, 5), each drawing K distinct support sites (1 rep image/site)
    sup_reps = sorted([sm["rep"] for sm in sup_sites], key=lambda x: x["id"])
    rng = np.random.default_rng(20261127)
    episodes = []
    for k in (1, 3, 5):
        for ep_i in range(5):
            perm = sorted(int(i) for i in rng.choice(len(sup_reps), size=k, replace=False))
            picked = [sup_reps[i] for i in perm]
            episodes.append(
                {
                    "k": k,
                    "episode": ep_i,
                    "support_ids": [x["id"] for x in picked],
                    "support_sites": [x["site"] for x in picked],
                }
            )

    payload = {
        "schema": "pv100-ext-site-split-lock-v1",
        "seed": 20261127,
        "episodes_per_k": 5,
        "summary": {
            "total_images": 100,
            "support_pool_sites": len(sup_site_set),
            "support_pool_images": len(support_records),
            "support_rep_images": len(sup_reps),
            "query_pos_sites": len({r["site"] for r in query_records if r["positive"]}),
            "query_pos_images": sum(1 for r in query_records if r["positive"]),
            "query_neg_sites": len({r["site"] for r in query_records if not r["positive"]}),
            "query_neg_images": sum(1 for r in query_records if not r["positive"]),
            "query_total_sites": len(qry_site_set),
            "query_total_images": len(query_records),
        },
        "support_rep_ids": [x["id"] for x in sup_reps],
        "support_records": support_records,
        "query_records": query_records,
        "episodes": episodes,
    }
    out_path = repo_root / "external_baselines" / "configs" / "pv100_ext_split_lock.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved locked split:", out_path, payload["summary"])
    return out_path


if __name__ == "__main__":
    build_pv100_ext_split(Path(__file__).resolve().parents[1])
