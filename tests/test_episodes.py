from __future__ import annotations

import unittest
from pathlib import Path

from pvfs.data import load_dev_records
from pvfs.episodes import generate_episodes, load_and_verify_episodes


class EpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.split_dir = Path(__file__).resolve().parents[1] / "configs" / "locked_split"

    def test_dev_only_and_site_disjoint_episodes(self) -> None:
        _, _, dev_records = load_dev_records(self.split_dir)
        self.assertEqual(len(dev_records), 79)
        by_id = {r["id"]: r for r in dev_records}

        payload = generate_episodes(dev_records)
        self.assertEqual(payload["total_episodes"], 120)
        for ep in payload["episodes"]:
            sup_sites = {by_id[s]["site"] for s in ep["support_ids"]}
            qry_sites = {by_id[q]["site"] for q in ep["query_ids"]}
            self.assertEqual(len(ep["support_ids"]), ep["k"])
            self.assertEqual(len(sup_sites), ep["k"])
            self.assertTrue(sup_sites.isdisjoint(qry_sites))
            self.assertTrue(all(by_id[s]["positive"] for s in ep["support_ids"]))

    def test_locked_episodes_file_matches(self) -> None:
        lock = self.split_dir / "episodes_lock.json"
        if lock.is_file():
            _, _, dev_records = load_dev_records(self.split_dir)
            payload = load_and_verify_episodes(self.split_dir, lock, dev_records)
            self.assertEqual(payload["total_episodes"], 120)


if __name__ == "__main__":
    unittest.main()
