from __future__ import annotations

import unittest

import numpy as np
import torch

from pvfs.iter1_methods import ITER1_VARIANTS, _to_tokens, run_iter1_episode


class Iter1Tests(unittest.TestCase):
    def _make_scene(self, seed: int, kind: str) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        rng = np.random.default_rng(seed)
        mask = np.zeros((512, 512), dtype=np.uint8)
        d32 = rng.normal(0.0, 0.08, size=(1024, 32, 32)).astype(np.float32)
        d64 = rng.normal(0.0, 0.08, size=(1024, 64, 64)).astype(np.float32)
        m64 = rng.normal(0.0, 0.08, size=(1024, 64, 64)).astype(np.float32)
        d32[0] -= 1.0
        d64[0] -= 1.0
        m64[0] -= 1.0

        if kind == "pv":
            d32[:, 8:16, 8:16] = 0.0
            d32[1, 8:16, 8:16] = 2.0
            d64[:, 16:32, 16:32] = 0.0
            d64[1, 16:32, 16:32] = 2.0
            m64[:, 16:32, 16:32] = 0.0
            m64[2, 16:32, 16:32] = 2.0
            mask[128:256, 128:256] = 1
        elif kind == "greenhouse_confounder":
            # Deep layers look PV-like, mid layer stays background-like!
            d32[:, 8:16, 8:16] = 0.0
            d32[1, 8:16, 8:16] = 1.8
            d64[:, 16:32, 16:32] = 0.0
            d64[1, 16:32, 16:32] = 1.8
            m64[:, 16:32, 16:32] = 0.0
            m64[0, 16:32, 16:32] = -1.8
        return (
            {
                "deep_32": _to_tokens(torch.from_numpy(d32), "cpu"),
                "deep_64": _to_tokens(torch.from_numpy(d64), "cpu"),
                "mid_64": _to_tokens(torch.from_numpy(m64), "cpu"),
            },
            mask,
        )

    def test_iter1_variants_and_mid_veto_suppresses_confounder(self) -> None:
        feats_tokens = {"deep_32": {}, "deep_64": {}, "mid_64": {}}
        masks = {}
        for i in range(3):
            t, m = self._make_scene(10 + i, "pv")
            for k in feats_tokens:
                feats_tokens[k][f"sup-{i}"] = t[k]
            masks[f"sup-{i}"] = m
        for qid, kind in (("q-pv", "pv"), ("q-gh", "greenhouse_confounder")):
            t, m = self._make_scene(50 if kind == "pv" else 51, kind)
            for k in feats_tokens:
                feats_tokens[k][qid] = t[k]
            masks[qid] = m

        res = {}
        for cfg in ITER1_VARIANTS:
            p512, s32 = run_iter1_episode(feats_tokens, masks, ["sup-0", "sup-1", "sup-2"], ["q-pv", "q-gh"], cfg)
            self.assertEqual(p512["q-pv"].shape, (512, 512))
            self.assertEqual(s32["q-pv"].shape, (32, 32))
            res[cfg.name] = p512

        # Mid-layer veto in M1_Full must drop greenhouse confounder probability below M1_Scale_Core_Calib
        gh_no_mid = float(res["M1_Scale_Core_Calib"]["q-gh"][140:240, 140:240].mean())
        gh_with_mid = float(res["M1_Full"]["q-gh"][140:240, 140:240].mean())
        self.assertLess(gh_with_mid, gh_no_mid)
        self.assertGreater(float(res["M1_Full"]["q-pv"][140:240, 140:240].mean()), 0.7)


if __name__ == "__main__":
    unittest.main()
