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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sp = json.loads((root / "configs" / "pv663_split_lock.json").read_text(encoding="utf-8"))
    recs = sp["records"]

    sup_by_site: dict[str, list[dict]] = defaultdict(list)
    qry_by_site: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        (sup_by_site if r["partition"] == "support_pool" else qry_by_site)[r["site"]].append(r)

    def pick_rep(by_site: dict[str, list[dict]]) -> list[dict]:
        out = []
        for s in sorted(by_site):
            arr = sorted(by_site[s], key=lambda x: (x["target_fraction"], x["id"]))
            out.append(arr[len(arr) // 2])
        return out

    def strat_pick(items: list[dict], counts: dict[str, int]) -> list[dict]:
        by_st: dict[str, list[dict]] = defaultdict(list)
        for r in items:
            by_st[_stratum(r["target_fraction"])].append(r)
        chosen: list[dict] = []
        for st in ("tiny_lt1pct", "small_1_5pct", "mid_5_15pct", "large_ge15pct"):
            arr = sorted(by_st[st], key=lambda x: (x["target_fraction"], x["id"]))
            want = counts[st]
            if len(arr) <= want:
                chosen.extend(arr)
            else:
                idxs = np.linspace(0, len(arr) - 1, want).round().astype(int)
                seen: set[int] = set()
                for idx in idxs:
                    cur = int(idx)
                    while cur in seen and cur + 1 < len(arr):
                        cur += 1
                    seen.add(cur)
                    chosen.append(arr[cur])
        return chosen

    sup_chosen = strat_pick(pick_rep(sup_by_site), {"tiny_lt1pct": 3, "small_1_5pct": 4, "mid_5_15pct": 4, "large_ge15pct": 4})
    qry_chosen = strat_pick(pick_rep(qry_by_site), {"tiny_lt1pct": 12, "small_1_5pct": 12, "mid_5_15pct": 12, "large_ge15pct": 12})

    # Verify zero site overlap
    assert len({x["site"] for x in sup_chosen} & {x["site"] for x in qry_chosen}) == 0

    rng = np.random.default_rng(20261125)
    episodes = []
    for k in (1, 3, 5):
        for ep_i in range(6):
            perm = sorted(int(i) for i in rng.choice(len(sup_chosen), size=k, replace=False))
            s_items = [sup_chosen[i] for i in perm]
            episodes.append(
                {
                    "k": k,
                    "episode": ep_i,
                    "support_ids": [x["id"] for x in s_items],
                    "support_sites": [x["site"] for x in s_items],
                }
            )

    payload = {
        "schema": "pv663-stratified-subset-v1",
        "min_support_query_distance_km": sp["min_support_query_distance_km"],
        "episodes_per_k": 6,
        "support_records": sup_chosen,
        "query_records": qry_chosen,
        "unique_support_ids": [x["id"] for x in sup_chosen],
        "query_ids": [x["id"] for x in qry_chosen],
        "episodes": episodes,
    }
    out_p = root / "configs" / "pv663_subset_lock.json"
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {out_p}: {len(sup_chosen)} sup sites, {len(qry_chosen)} qry sites, {len(episodes)} episodes")


if __name__ == "__main__":
    main()
