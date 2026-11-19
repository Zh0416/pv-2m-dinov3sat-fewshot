from __future__ import annotations

import unittest

import numpy as np
import torch

from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.iter1_methods import _to_tokens
from pvfs.iter3_methods import compute_dino_affinity_weights
from pvfs.iter5_methods import (
    ITER5_VARIANTS,
    Iter5Config,
    _apply_mid_verified_rescue,
    run_iter5_episode,
)


class Iter5Tests(unittest.TestCase):
    def test_mid_verified_rescue_blocks_low_mid_seeds(self) -> None:
        torch.manual_seed(505)
        n = 64 * 64
        c = 64
        pv_vec = torch.randn(c)
        pv_vec = pv_vec / pv_vec.norm()
        bg_vec = torch.randn(c)
        bg_vec = bg_vec - (bg_vec @ pv_vec) * pv_vec
        bg_vec = bg_vec / bg_vec.norm()

        q_tok = bg_vec[None, :].repeat(n, 1)
        seed_slice = torch.zeros(n, dtype=torch.bool)
        seed_slice[650:686] = True
        wing_slice = torch.zeros(n, dtype=torch.bool)
        wing_slice[714:742] = True
        q_tok[seed_slice] = pv_vec
        q_tok[wing_slice] = pv_vec
        q_tok = torch.nn.functional.normalize(q_tok, dim=-1)

        base_comb = torch.full((n,), -0.15, dtype=torch.float32)
        base_comb[seed_slice] = 0.25
        base_comb[wing_slice] = 0.02
        p_base = torch.full((n,), 0.10, dtype=torch.float32)
        p_base[seed_slice] = 0.80
        p_base[wing_slice] = 0.40

        cfg = Iter5Config(name="TestTight", seed_prob_thr=0.72, min_seeds=6, seed_mid_margin=0.02, qsc_weight=0.45)

        # Case 1: seeds fail mid-layer check (like dark greenhouse) -> no rescue boost
        sm_low = torch.zeros(n, dtype=torch.float32)
        sm_low[seed_slice] = 0.15  # < 0.20 + 0.02
        out_blocked = _apply_mid_verified_rescue(base_comb, p_base, sm_low, 0.20, q_tok, cfg)
        self.assertTrue(torch.allclose(out_blocked, base_comb))

        # Case 2: seeds pass mid-layer check -> coherent PV wing boosted above base
        sm_high = torch.zeros(n, dtype=torch.float32)
        sm_high[seed_slice] = 0.32  # > 0.20 + 0.02
        out_rescued = _apply_mid_verified_rescue(base_comb, p_base, sm_high, 0.20, q_tok, cfg)
        self.assertGreater(float(out_rescued[wing_slice].mean()), float(base_comb[wing_slice].mean()) + 0.05)

    def test_iter5_variants_end_to_end(self) -> None:
        torch.manual_seed(506)
        c = 64
        pv_main = torch.randn(c)
        bg_vec = torch.randn(c)

        masks, toks, affs = {}, {}, {}
        for idx, sid in enumerate(["s0", "s1", "s2"]):
            m512 = np.zeros((512, 512), dtype=np.uint8)
            r0 = 64 + idx * 16
            m512[r0 : r0 + 128, 64:192] = 1
            masks[sid] = m512
            m64 = m512.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
            f64 = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.10
            f64[:, torch.from_numpy(m64)] = pv_main[:, None] + torch.randn(c, int(m64.sum())) * 0.08
            f32 = torch.nn.functional.avg_pool2d(f64[None], 2, stride=2)[0]
            t64 = _to_tokens(f64, "cpu")
            toks.setdefault("deep_64", {})[sid] = t64
            toks.setdefault("mid_64", {})[sid] = t64
            toks.setdefault("deep_32", {})[sid] = _to_tokens(f32, "cpu")
            affs[sid] = compute_dino_affinity_weights(t64, tau=0.12)

        q_m = np.zeros((512, 512), dtype=np.uint8)
        q_m[96:224, 96:224] = 1
        masks["q_pos"] = q_m
        m64_q = q_m.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
        f64_q = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.10
        f64_q[:, torch.from_numpy(m64_q)] = pv_main[:, None] + torch.randn(c, int(m64_q.sum())) * 0.08
        t64_q = _to_tokens(f64_q, "cpu")
        toks["deep_64"]["q_pos"] = t64_q
        toks["mid_64"]["q_pos"] = t64_q
        toks["deep_32"]["q_pos"] = _to_tokens(torch.nn.functional.avg_pool2d(f64_q[None], 2, stride=2)[0], "cpu")
        affs["q_pos"] = compute_dino_affinity_weights(t64_q, tau=0.12)

        masks["q_neg"] = np.zeros((512, 512), dtype=np.uint8)
        f64_n = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.12
        t64_n = _to_tokens(f64_n, "cpu")
        toks["deep_64"]["q_neg"] = t64_n
        toks["mid_64"]["q_neg"] = t64_n
        toks["deep_32"]["q_neg"] = _to_tokens(torch.nn.functional.avg_pool2d(f64_n[None], 2, stride=2)[0], "cpu")
        affs["q_neg"] = compute_dino_affinity_weights(t64_n, tau=0.12)

        all_ids = ["s0", "s1", "s2", "q_pos", "q_neg"]
        pos_ids = ["s0", "s1", "s2"]
        caches = {
            key: build_pairwise_topk_cache(toks[key], masks, all_ids, pos_ids, grid_hw=(64 if "64" in key else 32))
            for key in ("deep_64", "mid_64", "deep_32")
        }

        for cfg in ITER5_VARIANTS:
            for sids in (["s0"], ["s0", "s1", "s2"]):
                p512, s32 = run_iter5_episode(caches, toks, masks, affs, sids, ["q_pos", "q_neg"], cfg)
                self.assertEqual(p512["q_pos"].shape, (512, 512))
                self.assertEqual(s32["q_pos"].shape, (32, 32))
                self.assertTrue(np.isfinite(p512["q_pos"]).all())
                self.assertGreater(float(p512["q_pos"][110:210, 110:210].mean()), 0.55)
                self.assertEqual(float((p512["q_neg"] >= 0.5).mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
