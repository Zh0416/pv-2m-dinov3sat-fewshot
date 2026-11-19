from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import read_json, sha256_file, sha256_json, write_json

SHOT_VALUES = (1, 3, 5)
FOLDS = (0, 1, 2, 3)
EPISODES_PER_FOLD = 10
BASE_SEED = 20261119


def generate_episodes(dev_records: list[dict], *, ks: tuple[int, ...] = SHOT_VALUES,
                      episodes_per_fold: int = EPISODES_PER_FOLD,
                      base_seed: int = BASE_SEED) -> dict:
    """Generate deterministic, site-disjoint K-shot episodes across the 4 dev folds."""
    by_id = {r["id"]: r for r in dev_records}
    if any(r["partition"] != "dev" for r in dev_records):
        raise PermissionError("Episodes may only be built from dev records")

    episodes = []
    for k in ks:
        for fold in FOLDS:
            query_rows = sorted((r for r in dev_records if r["fold"] == fold), key=lambda r: r["id"])
            query_ids = [r["id"] for r in query_rows]
            query_sites = {r["site"] for r in query_rows}

            pos_by_site: dict[str, list[str]] = defaultdict(list)
            for r in sorted(dev_records, key=lambda x: x["id"]):
                if r["fold"] != fold and r["positive"]:
                    pos_by_site[r["site"]].append(r["id"])

            candidate_sites = sorted(pos_by_site)
            if len(candidate_sites) < k:
                raise ValueError(f"Fold {fold} only has {len(candidate_sites)} positive training sites (< {k})")

            for ep in range(episodes_per_fold):
                seed = base_seed + 1000 * k + 100 * fold + ep
                rng = np.random.default_rng(seed)
                chosen_indices = rng.choice(len(candidate_sites), size=k, replace=False)
                chosen_sites = [candidate_sites[int(idx)] for idx in chosen_indices]
                support_ids = []
                for site in chosen_sites:
                    pool = pos_by_site[site]
                    pick = int(rng.integers(0, len(pool)))
                    support_ids.append(pool[pick])

                support_sites = {by_id[sid]["site"] for sid in support_ids}
                if not support_sites.isdisjoint(query_sites):
                    raise AssertionError("Support/query site leakage detected")
                if len(support_sites) != k or len(set(support_ids)) != k:
                    raise AssertionError("Support sites must be distinct")

                episodes.append(
                    {
                        "k": int(k),
                        "fold": int(fold),
                        "episode": int(ep),
                        "seed": int(seed),
                        "support_ids": support_ids,
                        "support_sites": chosen_sites,
                        "query_ids": query_ids,
                    }
                )

    return {
        "schema": "pvfs-episodes-v1",
        "ks": list(ks),
        "folds": list(FOLDS),
        "episodes_per_fold": episodes_per_fold,
        "base_seed": base_seed,
        "total_episodes": len(episodes),
        "episodes": episodes,
    }


def save_episodes_lock(split_dir: Path, dev_records: list[dict], out_path: Path) -> dict:
    payload = generate_episodes(dev_records)
    payload["manifest_sha256"] = sha256_file(split_dir / "dataset_manifest.json")
    payload["split_sha256"] = sha256_file(split_dir / "split_lock.json")
    payload["episodes_sha256"] = sha256_json(payload["episodes"])
    write_json(out_path, payload)
    return payload


def load_and_verify_episodes(split_dir: Path, lock_path: Path, dev_records: list[dict]) -> dict:
    payload = read_json(lock_path)
    if payload.get("manifest_sha256") != sha256_file(split_dir / "dataset_manifest.json"):
        raise ValueError("Manifest hash mismatch vs episodes_lock.json")
    if payload.get("split_sha256") != sha256_file(split_dir / "split_lock.json"):
        raise ValueError("Split hash mismatch vs episodes_lock.json")
    if payload.get("episodes_sha256") != sha256_json(payload["episodes"]):
        raise ValueError("Episodes integrity hash mismatch")
    by_id = {r["id"]: r for r in dev_records}
    for ep in payload["episodes"]:
        sup_sites = {by_id[sid]["site"] for sid in ep["support_ids"]}
        qry_sites = {by_id[qid]["site"] for qid in ep["query_ids"]}
        if not sup_sites.isdisjoint(qry_sites):
            raise ValueError("Locked episode violates support/query site isolation")
        if any(not by_id[sid]["positive"] for sid in ep["support_ids"]):
            raise ValueError("Support set contains non-positive image")
    return payload
