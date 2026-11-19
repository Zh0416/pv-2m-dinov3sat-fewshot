from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from pvfs.data import area_target_32, load_dev_records
from pvfs.final_sealed_eval import (
    SEALED_VARIANTS,
    aggregate_sealed_episodes,
    build_sealed_test_episodes,
    compare_sealed_methods,
    load_sealed_test_records,
)
from pvfs.metrics import evaluate_episode


class FinalSealedEvalTests(unittest.TestCase):
    def test_sealed_split_episodes_and_aggregation(self) -> None:
        split_dir = Path(__file__).resolve().parents[1] / "configs" / "locked_split"
        _, _, dev_records = load_dev_records(split_dir)
        _, _, test_records = load_sealed_test_records(split_dir)

        self.assertEqual(len(test_records), 21)
        self.assertEqual(sum(r["positive"] for r in test_records), 17)
        self.assertEqual(sum(not r["positive"] for r in test_records), 4)
        self.assertEqual(len({r["site"] for r in test_records}), 19)

        eps = build_sealed_test_episodes(dev_records, test_records, ks=(1, 3, 5), num_episodes=4)
        dev_pos_ids = {r["id"] for r in dev_records if r["positive"]}
        test_ids = [r["id"] for r in test_records]
        for k_str, k_eps in eps.items():
            self.assertEqual(len(k_eps), 4)
            for ep in k_eps:
                self.assertEqual(len(ep["support_ids"]), int(k_str))
                self.assertTrue(set(ep["support_ids"]).issubset(dev_pos_ids))
                self.assertEqual(ep["query_ids"], test_ids)

        # Synthetic sealed aggregation sanity check (handles fold=None cleanly)
        query_items = []
        p_good, p_ref, s_map = {}, {}, {}
        for idx, r in enumerate(test_records):
            m = np.zeros((512, 512), dtype=np.uint8)
            pg = np.zeros((512, 512), dtype=np.float32)
            pr = np.zeros((512, 512), dtype=np.float32)
            sc = np.zeros((32, 32), dtype=np.float32)
            if r["positive"]:
                m[64:192, 64:192] = 1
                pg[64:192, 64:192] = 0.90
                pr[64:160, 64:160] = 0.80
                sc[4:12, 4:12] = 0.80
            else:
                pr[0:16, 0:16] = 0.75 if idx % 2 == 0 else 0.1
            query_items.append({**r, "mask": m, "y32": area_target_32(m)})
            p_good[r["id"]] = pg
            p_ref[r["id"]] = pr
            s_map[r["id"]] = sc

        good_agg = aggregate_sealed_episodes([evaluate_episode(query_items, p_good, s_map) for _ in range(3)], expected_episodes=3)
        ref_agg = aggregate_sealed_episodes([evaluate_episode(query_items, p_ref, s_map) for _ in range(3)], expected_episodes=3)
        self.assertEqual(good_agg["macro"]["n_pos_sites"], 15)
        self.assertEqual(good_agg["macro"]["n_neg_sites"], 4)
        self.assertAlmostEqual(good_agg["macro"]["dice"], 1.0, places=5)
        cmp = compare_sealed_methods(good_agg, ref_agg)
        self.assertGreater(cmp["dice"]["mean_delta"], 0.10)
        self.assertEqual(len(SEALED_VARIANTS), 5)


if __name__ == "__main__":
    unittest.main()
