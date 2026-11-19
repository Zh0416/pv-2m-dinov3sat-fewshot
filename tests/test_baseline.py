from __future__ import annotations

import unittest

import numpy as np
import torch

from pvfs.baseline import normalize_dino_tokens, run_b0_episode


class BaselineTests(unittest.TestCase):
    def _synthetic_scene(self, seed: int, positive: bool) -> tuple[torch.Tensor, np.ndarray]:
        rng = np.random.default_rng(seed)
        feat = rng.normal(0.0, 0.1, size=(1024, 32, 32)).astype(np.float32)
        feat[0] -= 1.0  # background direction
        mask = np.zeros((512, 512), dtype=np.uint8)
        if positive:
            feat[:, 8:16, 8:16] = 0.0
            feat[1, 8:16, 8:16] = 2.0  # orthogonal PV direction
            mask[128:256, 128:256] = 1
        return normalize_dino_tokens(feat, device="cpu"), mask

    def test_b0_k1_and_k3_separate_pv_from_background(self) -> None:
        toks, masks = {}, {}
        for i in range(4):
            toks[f"sup-{i}"], masks[f"sup-{i}"] = self._synthetic_scene(10 + i, True)
        toks["q-pos"], masks["q-pos"] = self._synthetic_scene(20, True)
        toks["q-neg"], masks["q-neg"] = self._synthetic_scene(21, False)

        for sup in (["sup-0"], ["sup-0", "sup-1", "sup-2"]):
            probs, scores, info = run_b0_episode(toks, masks, sup, ["q-pos", "q-neg"])
            self.assertEqual(probs["q-pos"].shape, (512, 512))
            self.assertEqual(scores["q-pos"].shape, (32, 32))
            self.assertGreater(info["n_fg_patches"], 0)
            self.assertGreater(probs["q-pos"][160:220, 160:220].mean(), 0.7)
            self.assertLess((probs["q-neg"] >= 0.5).mean(), 0.05)


if __name__ == "__main__":
    unittest.main()
