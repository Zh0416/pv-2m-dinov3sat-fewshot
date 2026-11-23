from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.io import write_json

COORD_RE = re.compile(r"^PV\d+_(\d+)_(\d+)$")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def cluster_sites_5km(records: list[dict[str, Any]], radius_km: float = 5.0) -> list[dict[str, Any]]:
    """Assign deterministic 5 km geographic connected-component site IDs."""
    n = len(records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        lat_i, lon_i = records[i]["lat"], records[i]["lon"]
        for j in range(i + 1, n):
            if haversine_km(lat_i, lon_i, records[j]["lat"], records[j]["lon"]) <= radius_km:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Sort site clusters deterministically by smallest image ID inside each cluster
    ordered_clusters = sorted(groups.values(), key=lambda idxs: min(records[k]["id"] for k in idxs))
    for s_idx, idxs in enumerate(ordered_clusters):
        site_id = f"pv663_site_{s_idx:03d}"
        for k in idxs:
            records[k]["site"] = site_id
    return records


def build_pv663_split_and_episodes(
    dataset_root: Path,
    *,
    split_out: Path,
    episodes_out: Path,
    radius_km: float = 5.0,
    support_site_stride: int = 5,
    ks: tuple[int, ...] = (1, 3, 5),
    num_episodes: int = 10,
    seed: int = 20261124,
) -> tuple[dict[str, Any], dict[str, Any]]:
    img_dir = dataset_root / "images"
    mask_dir = dataset_root / "mask"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and mask/ under {dataset_root}")

    records: list[dict[str, Any]] = []
    for img_path in sorted(img_dir.glob("*.bmp")):
        mask_name = f"{img_path.stem}_label.bmp"
        mask_path = mask_dir / mask_name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask {mask_path} for {img_path.name}")
        m = COORD_RE.match(img_path.stem)
        if not m:
            raise ValueError(f"Unexpected PV663 filename format: {img_path.name}")
        lat = int(m.group(1)) / 10000.0
        lon = int(m.group(2)) / 10000.0

        arr = np.asarray(Image.open(mask_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape != (1024, 1024):
            raise ValueError(f"Unexpected mask shape {arr.shape} for {mask_name}")
        mask_1024 = arr > 0
        frac_1024 = float(mask_1024.mean())
        mask_512 = mask_1024.astype(np.float32).reshape(512, 2, 512, 2).mean(axis=(1, 3)) >= 0.5
        if mask_1024.any() and not mask_512.any():
            mask_512 = mask_1024.reshape(512, 2, 512, 2).any(axis=(1, 3))
        frac_512 = float(mask_512.mean())

        records.append(
            {
                "id": img_path.name,
                "image_rel": f"images/{img_path.name}",
                "mask_rel": f"mask/{mask_name}",
                "lat": lat,
                "lon": lon,
                "positive": bool(mask_512.any()),
                "target_fraction_1024": frac_1024,
                "target_fraction": frac_512,
            }
        )

    if len(records) != 663:
        raise ValueError(f"Expected 663 images in pv663, found {len(records)}")

    cluster_sites_5km(records, radius_km=radius_km)

    # Stratified deterministic site split: sort sites by median target_fraction, pick 1 of every `stride` sites for support_pool
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_site[r["site"]].append(r)

    site_order = sorted(
        by_site.keys(),
        key=lambda s: (float(np.median([x["target_fraction"] for x in by_site[s]])), len(by_site[s]), s),
    )

    support_sites: set[str] = set()
    query_sites: set[str] = set()
    for idx, s in enumerate(site_order):
        if idx % support_site_stride == 2:
            support_sites.add(s)
        else:
            query_sites.add(s)

    for r in records:
        r["partition"] = "support_pool" if r["site"] in support_sites else "query_set"
        r["fold"] = None

    sup_recs = [r for r in records if r["partition"] == "support_pool"]
    qry_recs = [r for r in records if r["partition"] == "query_set"]

    # Hard geographic isolation verification (> 5 km between ANY support image and ANY query image)
    min_dist_km = float("inf")
    for sr in sup_recs:
        for qr in qry_recs:
            d = haversine_km(sr["lat"], sr["lon"], qr["lat"], qr["lon"])
            if d < min_dist_km:
                min_dist_km = d
    if min_dist_km <= radius_km:
        raise AssertionError(f"Spatial leakage detected: min support-query distance = {min_dist_km:.3f} km <= {radius_km} km")

    def _strata_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "total": len(rows),
            "positive": sum(1 for r in rows if r["positive"]),
            "negative": sum(1 for r in rows if not r["positive"]),
            "tiny_lt1pct": sum(1 for r in rows if r["positive"] and r["target_fraction"] < 0.01),
            "small_1_5pct": sum(1 for r in rows if r["positive"] and 0.01 <= r["target_fraction"] < 0.05),
            "mid_5_15pct": sum(1 for r in rows if r["positive"] and 0.05 <= r["target_fraction"] < 0.15),
            "large_ge15pct": sum(1 for r in rows if r["positive"] and r["target_fraction"] >= 0.15),
        }

    split_payload = {
        "schema": "pv663-site-split-lock-v1",
        "dataset": "zh0416/pv-dinov3sat-663-v1",
        "site_radius_km": radius_km,
        "seed": seed,
        "min_support_query_distance_km": round(min_dist_km, 4),
        "summary": {
            "total_images": len(records),
            "total_sites": len(by_site),
            "support_pool_sites": len(support_sites),
            "query_set_sites": len(query_sites),
            "support_pool": _strata_counts(sup_recs),
            "query_set": _strata_counts(qry_recs),
        },
        "records": records,
    }

    # Build deterministic K-shot episodes sampling distinct support sites -> querying ALL query_set images
    sup_by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sup_recs:
        if r["positive"]:
            sup_by_site[r["site"]].append(r)
    sup_site_list = sorted(sup_by_site.keys())
    query_ids = [r["id"] for r in qry_recs]

    episodes: list[dict[str, Any]] = []
    unique_support_ids: set[str] = set()
    for k in ks:
        if k > len(sup_site_list):
            raise ValueError(f"Requested K={k} exceeds available support sites ({len(sup_site_list)})")
        for ep in range(num_episodes):
            rng = random.Random(seed + 1000 * int(k) + ep)
            chosen_sites = rng.sample(sup_site_list, int(k))
            chosen_ids = [rng.choice(sup_by_site[s])["id"] for s in chosen_sites]
            unique_support_ids.update(chosen_ids)
            episodes.append(
                {
                    "k": int(k),
                    "episode": int(ep),
                    "support_sites": chosen_sites,
                    "support_ids": chosen_ids,
                    "query_count": len(query_ids),
                }
            )

    episodes_payload = {
        "schema": "pv663-episodes-lock-v1",
        "dataset": "zh0416/pv-dinov3sat-663-v1",
        "seed": seed,
        "ks": [int(k) for k in ks],
        "episodes_per_k": int(num_episodes),
        "unique_support_ids": sorted(unique_support_ids),
        "query_ids": query_ids,
        "episodes": episodes,
    }

    write_json(split_out, split_payload)
    write_json(episodes_out, episodes_payload)
    return split_payload, episodes_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build locked 5km site-disjoint support/query split and 1/3/5-shot episodes for pv663.")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"E:\PV-DINOV3SAT\datasets"))
    parser.add_argument("--split-out", type=Path, default=REPO_ROOT / "configs" / "pv663_split_lock.json")
    parser.add_argument("--episodes-out", type=Path, default=REPO_ROOT / "configs" / "pv663_episodes_lock.json")
    parser.add_argument("--episodes-per-k", type=int, default=10)
    args = parser.parse_args()

    split_p, ep_p = build_pv663_split_and_episodes(
        args.dataset_root,
        split_out=args.split_out,
        episodes_out=args.episodes_out,
        num_episodes=args.episodes_per_k,
    )
    print("PV663 Split Summary:", split_p["summary"])
    print(f"Min support-query Haversine distance: {split_p['min_support_query_distance_km']:.3f} km")
    print(f"Unique support IDs across 30 episodes: {len(ep_p['unique_support_ids'])} | Query IDs: {len(ep_p['query_ids'])}")


if __name__ == "__main__":
    main()
