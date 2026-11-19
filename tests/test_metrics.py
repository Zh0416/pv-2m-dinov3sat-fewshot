from __future__ import annotations

import unittest

import numpy as np

from pvfs.metrics import aggregate_episodes, average_precision, boundary_f1, compare_methods, evaluate_episode


class MetricTests(unittest.TestCase):
    def test_ap_and_boundary_f1_exact(self) -> None:
        y = np.array([1, 0, 1, 0], dtype=bool)
        s = np.array([0.9, 0.4, 0.8, 0.1], dtype=np.float32)
        self.assertAlmostEqual(average_precision(y, s), 1.0, places=6)

        m = np.zeros((32, 32), dtype=bool)
        m[8:24, 8:24] = True
        self.assertAlmostEqual(boundary_f1(m, m), 1.0, places=6)
        shifted = np.zeros((32, 32), dtype=bool)
        shifted[9:25, 8:24] = True
        self.assertAlmostEqual(boundary_f1(m, shifted), 1.0, places=6)

    def test_site_aggregation_and_gates(self) -> None:
        def make_ep(site_scores: dict[str, tuple[bool, float, float]]) -> dict:
            qrecs, probs, s32 = [], {}, {}
            for qid, (pos, p_fg, p_bg) in site_scores.items():
                mask = np.zeros((512, 512), dtype=np.uint8)
                y32 = np.zeros((32, 32), dtype=np.float32)
                prob = np.full((512, 512), p_bg, dtype=np.float32)
                c32 = np.full((32, 32), p_bg, dtype=np.float32)
                if pos:
                    mask[128:256, 128:256] = 1
                    y32[8:16, 8:16] = 1.0
                    prob[128:256, 128:256] = p_fg
                    c32[8:16, 8:16] = p_fg
                    if p_fg < 0.5:
                        prob[0:128, 0:128] = 0.55
                        c32[0:8, 0:8] = 0.55
                qrecs.append({
                    "id": qid,
                    "site": qid,
                    "fold": int(qid[-1]) % 4,
                    "positive": pos,
                    "target_fraction": float(mask.mean()),
                    "mask": mask,
                    "y32": y32,
                })
                probs[qid] = prob
                s32[qid] = c32
            return evaluate_episode(qrecs, probs, s32)

        ref_eps = [
            make_ep({
                f"pos-{f}": (True, 0.45, 0.10) for f in range(4)
            } | {
                f"neg-{f}": (False, 0.0, 0.60) for f in range(4)
            })
            for _ in range(2)
        ]
        cand_eps = [
            make_ep({
                f"pos-{f}": (True, 0.85, 0.10) for f in range(4)
            } | {
                f"neg-{f}": (False, 0.0, 0.10) for f in range(4)
            })
            for _ in range(2)
        ]
        ref_agg = aggregate_episodes(ref_eps, episodes_per_fold=2)
        cand_agg = aggregate_episodes(cand_eps, episodes_per_fold=2)
        cmp = compare_methods(cand_agg, ref_agg)
        self.assertTrue(cmp["gates"]["Gate_FP"])
        self.assertTrue(cmp["gates"]["Gate_FN"])
        self.assertTrue(cmp["gates"]["Gate_Net"])


if __name__ == "__main__":
    unittest.main()
