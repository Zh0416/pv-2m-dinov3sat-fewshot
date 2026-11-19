from __future__ import annotations

import unittest

import numpy as np
import torch

from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.iter1_methods import _to_tokens
from pvfs.iter3_methods import (
    ITER3_VARIANTS,
    Iter3Config,
    compute_dino_affinity_weights,
    fast_guided_filter_512,
    remove_weak_tiny_speckles,
    run_iter3_episode,
)


class Iter3Tests(unittest.TestCase):
    def test_guided_filter_speckle_removal_and_variants(self) -> None:
        torch.manual_seed(303)
        c = 64
        pv_main = torch.randn(c)
        bg_vec = torch.randn(c)

        masks, toks, lumas, affs = {}, {}, {}, {}
        for idx, sid in enumerate(["s0", "s1", "s2"]):
            m512 = np.zeros((512, 512), dtype=np.uint8)
            r0 = 64 + idx * 16
            m512[r0 : r0 + 128, 64:192] = 1
            masks[sid] = m512
            m64 = m512.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
            f64 = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.10
            f64[:, torch.from_numpy(m64)] = pv_main[:, None] + torch.randn(c, int(m64.sum())) * 0.08
            f32 = torch.nn.functional.avg_pool2d(f64[None], 2, stride=2)[0]
            toks_64 = _to_tokens(f64, "cpu")
            toks.setdefault("deep_64", {})[sid] = toks_64
            toks.setdefault("mid_64", {})[sid] = toks_64
            toks.setdefault("deep_32", {})[sid] = _to_tokens(f32, "cpu")
            luma = np.full((512, 512), 0.65, dtype=np.float32)
            luma[m512 > 0] = 0.25
            lumas[sid] = torch.from_numpy(luma)
            affs[sid] = compute_dino_affinity_weights(toks_64, tau=0.12)

        # Query with crisp 2m edge in RGB luma + small hole inside 64x64 PV block
        q_m = np.zeros((512, 512), dtype=np.uint8)
        q_m[96:224, 96:224] = 1
        masks["q_pos"] = q_m
        m64_q = q_m.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
        f64_q = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.10
        f64_q[:, torch.from_numpy(m64_q)] = pv_main[:, None] + torch.randn(c, int(m64_q.sum())) * 0.08
        # Inject one weakened hole token inside PV block so Affinity propagation fills it
        f64_q[:, 18, 18] = bg_vec * 0.35 + pv_main * 0.65
        t64_q = _to_tokens(f64_q, "cpu")
        toks["deep_64"]["q_pos"] = t64_q
        toks["mid_64"]["q_pos"] = t64_q
        toks["deep_32"]["q_pos"] = _to_tokens(torch.nn.functional.avg_pool2d(f64_q[None], 2, stride=2)[0], "cpu")
        luma_q = np.full((512, 512), 0.70, dtype=np.float32)
        luma_q[96:224, 96:224] = 0.20
        lumas["q_pos"] = torch.from_numpy(luma_q)
        affs["q_pos"] = compute_dino_affinity_weights(t64_q, tau=0.12)

        masks["q_neg"] = np.zeros((512, 512), dtype=np.uint8)
        f64_n = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.12
        t64_n = _to_tokens(f64_n, "cpu")
        toks["deep_64"]["q_neg"] = t64_n
        toks["mid_64"]["q_neg"] = t64_n
        toks["deep_32"]["q_neg"] = _to_tokens(torch.nn.functional.avg_pool2d(f64_n[None], 2, stride=2)[0], "cpu")
        lumas["q_neg"] = torch.full((512, 512), 0.65, dtype=torch.float32)
        affs["q_neg"] = compute_dino_affinity_weights(t64_n, tau=0.12)

        all_ids = ["s0", "s1", "s2", "q_pos", "q_neg"]
        pos_ids = ["s0", "s1", "s2"]
        caches = {
            key: build_pairwise_topk_cache(toks[key], masks, all_ids, pos_ids, grid_hw=(64 if "64" in key else 32))
            for key in ("deep_64", "mid_64", "deep_32")
        }

        for cfg in ITER3_VARIANTS:
            for sids in (["s0"], ["s0", "s1", "s2"]):
                p512, s32 = run_iter3_episode(caches, toks, masks, lumas, affs, sids, ["q_pos", "q_neg"], cfg)
                self.assertEqual(p512["q_pos"].shape, (512, 512))
                self.assertEqual(s32["q_pos"].shape, (32, 32))
                self.assertTrue(np.isfinite(p512["q_pos"]).all())
                self.assertGreater(float(p512["q_pos"][110:210, 110:210].mean()), 0.55)
                self.assertEqual(float((p512["q_neg"] >= 0.5).mean()), 0.0)

        # Unit check for speckle guard: removes weak <24px noise while preserving strong small PV
        p_sp = np.zeros((512, 512), dtype=np.float32)
        p_sp[10:13, 10:14] = 0.58   # 12 px, peak 0.58 < 0.70 -> suppressed
        p_sp[100:103, 100:104] = 0.85  # 12 px, peak 0.85 >= 0.70 -> preserved
        p_sp[200:210, 200:210] = 0.55  # 100 px >= 24 -> preserved
        cleaned = remove_weak_tiny_speckles(p_sp, max_area=24, max_peak=0.70)
        self.assertLess(float(cleaned[10:13, 10:14].max()), 0.50)
        self.assertGreater(float(cleaned[100:103, 100:104].min()), 0.80)
        self.assertGreater(float(cleaned[200:210, 200:210].min()), 0.50)

        # Unit check for guided filter snapping blurry step to RGB boundary
        guide = torch.zeros((512, 512), dtype=torch.float32)
        guide[:, :256] = 0.8
        guide[:, 256:] = 0.2
        blurry = torch.zeros((512, 512), dtype=torch.float32)
        blurry[:, :250] = 0.1
        blurry[:, 250:262] = torch.linspace(0.1, 0.9, 12)[None, :]
        blurry[:, 262:] = 0.9
        aligned = fast_guided_filter_512(guide, blurry, radius=5, eps=0.001, blend=0.8)
        self.assertGreater(float(aligned[:, 257].mean() - blurry[:, 257].mean()), 0.05)


if __name__ == "__main__":
    unittest.main()
