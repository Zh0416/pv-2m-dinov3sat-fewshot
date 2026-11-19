from __future__ import annotations

import unittest

import numpy as np
import torch

from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.iter1_methods import _to_tokens
from pvfs.iter2_methods import ITER2_VARIANTS, Iter2Config, run_iter2_episode


class Iter2Tests(unittest.TestCase):
    def test_iter2_variants_self_rescue_and_scale_suppression(self) -> None:
        torch.manual_seed(2026)
        c = 64
        pv_main = torch.randn(c)
        pv_shifted = torch.randn(c) * 0.25 + pv_main * 0.75  # related solar farm style inside same query
        bg_vec = torch.randn(c)
        conf_vec = torch.randn(c)

        def make_img_tokens(
            fg_mask_64: np.ndarray,
            pv_vec: torch.Tensor,
            extra_mask_64: np.ndarray | None = None,
            extra_vec: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # Build deep_64, mid_64, deep_32, early_32 normalized token tensors
            noise64 = torch.randn(c, 64, 64) * 0.10
            m64_t = torch.from_numpy(fg_mask_64)
            f64 = bg_vec[:, None, None] + noise64
            f64[:, m64_t] = pv_vec[:, None] + noise64[:, m64_t]
            if extra_mask_64 is not None and extra_vec is not None:
                em = torch.from_numpy(extra_mask_64)
                f64[:, em] = extra_vec[:, None] + noise64[:, em]

            # Mid_64 differentiates confounder cleanly
            mid64 = f64.clone()
            if extra_mask_64 is not None:
                mid64[:, torch.from_numpy(extra_mask_64)] = bg_vec[:, None] + torch.randn(c, int(extra_mask_64.sum())) * 0.08

            # Coarse 32x32 features (confounder looks strong at 32m!)
            f32 = torch.nn.functional.avg_pool2d(f64[None], 2, stride=2)[0]
            if extra_mask_64 is not None:
                em32 = torch.nn.functional.avg_pool2d(torch.from_numpy(extra_mask_64).float()[None, None], 2, stride=2)[0, 0] > 0.2
                f32[:, em32] = pv_main[:, None] + torch.randn(c, int(em32.sum())) * 0.05

            e32 = f32.clone()
            if extra_mask_64 is not None:
                em32 = torch.nn.functional.avg_pool2d(torch.from_numpy(extra_mask_64).float()[None, None], 2, stride=2)[0, 0] > 0.2
                e32[:, em32] = bg_vec[:, None] + torch.randn(c, int(em32.sum())) * 0.05

            return _to_tokens(f64, "cpu"), _to_tokens(mid64, "cpu"), _to_tokens(f32, "cpu"), _to_tokens(e32, "cpu")

        masks = {}
        toks = {"deep_64": {}, "mid_64": {}, "deep_32": {}, "early_32": {}}
        for idx, sid in enumerate(["s0", "s1", "s2"]):
            m512 = np.zeros((512, 512), dtype=np.uint8)
            r0 = 64 + idx * 16
            m512[r0 : r0 + 128, 64:192] = 1
            masks[sid] = m512
            m64 = m512.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
            d64, md64, d32, e32 = make_img_tokens(m64, pv_main)
            toks["deep_64"][sid] = d64
            toks["mid_64"][sid] = md64
            toks["deep_32"][sid] = d32
            toks["early_32"][sid] = e32

        # Query image: core PV matching pv_main + second PV wing in pv_shifted + blurry confounder
        q_mask = np.zeros((512, 512), dtype=np.uint8)
        q_mask[48:112, 48:112] = 1     # core seed wing
        q_mask[160:240, 160:240] = 1   # shifted wing (same internal appearance)
        masks["q_pos"] = q_mask
        q_core64 = np.zeros((64, 64), dtype=bool)
        q_core64[6:14, 6:14] = True
        q_shift64 = np.zeros((64, 64), dtype=bool)
        q_shift64[20:30, 20:30] = True
        q_conf64 = np.zeros((64, 64), dtype=bool)
        q_conf64[44:54, 44:54] = True

        d64, md64, d32, e32 = make_img_tokens(q_core64, pv_main, q_conf64, conf_vec)
        # Inject shifted PV wing + slightly distinct sub-feature so Q+ rescue must link core->shifted wing
        shifted_base = (pv_main * 0.45 + pv_shifted * 0.55)
        shifted_raw = shifted_base[:, None] + torch.randn(c, int(q_shift64.sum())) * 0.05
        shift_norm = torch.nn.functional.normalize(shifted_raw, dim=0).T
        q_shift_flat = torch.from_numpy(q_shift64.ravel())
        d64[q_shift_flat] = shift_norm
        md64[q_shift_flat] = shift_norm
        toks["deep_64"]["q_pos"] = d64
        toks["mid_64"]["q_pos"] = md64
        toks["deep_32"]["q_pos"] = d32
        toks["early_32"]["q_pos"] = e32

        # Clean negative with blurry coarse confounder
        masks["q_neg"] = np.zeros((512, 512), dtype=np.uint8)
        d64_n, md64_n, d32_n, e32_n = make_img_tokens(np.zeros((64, 64), dtype=bool), pv_main, q_conf64, conf_vec)
        toks["deep_64"]["q_neg"] = d64_n
        toks["mid_64"]["q_neg"] = md64_n
        toks["deep_32"]["q_neg"] = d32_n
        toks["early_32"]["q_neg"] = e32_n

        all_ids = ["s0", "s1", "s2", "q_pos", "q_neg"]
        pos_ids = ["s0", "s1", "s2"]
        caches = {
            key: build_pairwise_topk_cache(toks[key], masks, all_ids, pos_ids, grid_hw=(64 if "64" in key else 32), core_purity=0.85, top_m=3)
            for key in ("deep_64", "mid_64", "deep_32", "early_32")
        }

        by_name: dict[str, Iter2Config] = {cfg.name: cfg for cfg in ITER2_VARIANTS}
        for cfg in ITER2_VARIANTS:
            for sids in (["s0"], ["s0", "s1", "s2"]):
                p512, s32 = run_iter2_episode(caches, toks, masks, sids, ["q_pos", "q_neg"], cfg)
                self.assertEqual(p512["q_pos"].shape, (512, 512))
                self.assertEqual(s32["q_pos"].shape, (32, 32))
                self.assertTrue(np.isfinite(p512["q_pos"]).all())

        p_base, _ = run_iter2_episode(caches, toks, masks, ["s0", "s1", "s2"], ["q_pos", "q_neg"], by_name["M2_LayerComp"])
        p_full, _ = run_iter2_episode(caches, toks, masks, ["s0", "s1", "s2"], ["q_pos", "q_neg"], by_name["M2_Full"])
        p_shot, _ = run_iter2_episode(caches, toks, masks, ["s0", "s1", "s2"], ["q_pos", "q_neg"], by_name["M2_ShotCal"])

        # Confounder probability drops below 0.50 (0 FP) from M2_ShotCal -> M2_LayerComp / M2_Full
        self.assertGreater(float(p_shot["q_neg"][352:432, 352:432].mean()), 0.50)
        self.assertLess(float(p_full["q_neg"][352:432, 352:432].mean()), 0.50)
        self.assertEqual(float((p_full["q_neg"] >= 0.5).mean()), 0.0)
        # Core wing stays high and QSC rescues without boosting orthogonal confounder
        self.assertGreater(float(p_full["q_pos"][48:112, 48:112].mean()), 0.50)
        self.assertLessEqual(float(p_full["q_pos"][352:432, 352:432].mean()), float(p_base["q_pos"][352:432, 352:432].mean()) + 1e-3)
        self.assertGreaterEqual(float(p_full["q_pos"][48:112, 48:112].mean()), float(p_base["q_pos"][48:112, 48:112].mean()) - 0.05)


if __name__ == "__main__":
    unittest.main()
