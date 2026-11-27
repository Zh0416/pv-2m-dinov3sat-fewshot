from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def build_pv100_ext_split(repo_root: Path) -> Path:
    """Build a locked 4-fold site-disjoint cross-evaluation split across ALL 100 PV-100scence images.

    - 49 independent positive sites stratified by mean target_fraction into 4 folds (12, 12, 12, 13 sites);
    - 20 independent negative sites evenly distributed into 4 folds (5, 5, 5, 5 sites);
    - For each fold f, support_pool = all positive sites from the other 3 folds (~37 sites, zero leakage);
      query_set = all positive + negative images in fold f;
    - 3 deterministic episodes per fold per K (12 episodes/K, 36 total), each drawing K distinct support sites.
    """
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

    # Sort positive sites by mean target_fraction and snake-assign to 4 folds for balanced area distribution
    pos_site_list = []
    for s, items in sorted(pos_by_site.items()):
        mean_tf = float(np.mean([x["target_fraction"] for x in items]))
        items_sorted = sorted(items, key=lambda x: (x["target_fraction"], x["id"]))
        rep = items_sorted[len(items_sorted) // 2]
        pos_site_list.append({"site": s, "mean_tf": mean_tf, "items": items_sorted, "rep": rep})
    pos_site_list.sort(key=lambda x: (x["mean_tf"], x["site"]))

    all_records: list[dict] = []
    pos_reps_by_fold: dict[int, list[dict]] = defaultdict(list)
    query_ids_by_fold: dict[int, list[str]] = defaultdict(list)

    for idx, sm in enumerate(pos_site_list):
        block = idx // 4
        fold = (idx % 4) if (block % 2 == 0) else (3 - (idx % 4))
        pos_reps_by_fold[fold].append(sm["rep"])
        for r in sm["items"]:
            all_records.append({**r, "ext_fold": fold})
            query_ids_by_fold[fold].append(r["id"])

    for idx, r in enumerate(sorted(neg_records, key=lambda x: x["id"])):
        fold = idx % 4
        all_records.append({**r, "ext_fold": fold})
        query_ids_by_fold[fold].append(r["id"])

    assert len(all_records) == 100

    # Lock 3 deterministic episodes per fold per K (12 episodes/K), drawing K distinct positive sites from other 3 folds
    rng = np.random.default_rng(20261127)
    episodes_per_fold = 3
    episodes = []
    for k in (1, 3, 5):
        for f in range(4):
            sup_pool = [r for ff in range(4) if ff != f for r in pos_reps_by_fold[ff]]
            sup_pool.sort(key=lambda x: x["id"])
            q_ids = sorted(query_ids_by_fold[f])
            q_sites = {r["site"] for r in all_records if r["ext_fold"] == f}
            for ep_i in range(episodes_per_fold):
                perm = sorted(int(i) for i in rng.choice(len(sup_pool), size=k, replace=False))
                picked = [sup_pool[i] for i in perm]
                assert len({x["site"] for x in picked} & q_sites) == 0
                episodes.append(
                    {
                        "k": k,
                        "fold": f,
                        "episode": ep_i,
                        "support_ids": [x["id"] for x in picked],
                        "support_sites": [x["site"] for x in picked],
                        "query_ids": q_ids,
                    }
                )

    payload = {
        "schema": "pv100-ext-4fold-site-cv-lock-v2",
        "seed": 20261127,
        "num_folds": 4,
        "episodes_per_fold": episodes_per_fold,
        "summary": {
            "total_images": 100,
            "pos_sites": len(pos_site_list),
            "pos_images": sum(1 for r in all_records if r["positive"]),
            "neg_sites": len(neg_records),
            "neg_images": len(neg_records),
            "total_sites": len(pos_site_list) + len(neg_records),
            "episodes_per_k": 4 * episodes_per_fold,
        },
        "records": sorted(all_records, key=lambda x: x["id"]),
        "episodes": episodes,
    }
    out_path = repo_root / "external_baselines" / "configs" / "pv100_ext_split_lock.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved 4-fold site-CV lock:", out_path, payload["summary"])
    return out_path


if __name__ == "__main__":
    build_pv100_ext_split(Path(__file__).resolve().parents[1])
