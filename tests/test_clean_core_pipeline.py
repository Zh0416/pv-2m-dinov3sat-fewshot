from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from pvfs.fast_bank import build_pairwise_topk_cache, compute_dino_affinity_weights, score_from_pairwise, to_tokens_norm
from pvfs.modules import DeformableBoundaryConfig, ThreeModulePipeline, get_b0_32_config, get_clean_core_config


class CleanCorePipelineTests(unittest.TestCase):
    def test_clean_core_modules_and_deformgrad_end_to_end(self) -> None:
        torch.manual_seed(20261125)
        ids = ["s0", "s1", "s2", "q_pos", "q_neg"]
        pos_ids = ["s0", "s1", "s2"]
        mask_by_id: dict[str, np.ndarray] = {}
        rgb_by_id: dict[str, np.ndarray] = {}
        tokens_by_key: dict[str, dict[str, torch.Tensor]] = {"deep_32": {}, "deep_64": {}, "mid_64": {}}

        fg_d = F.normalize(torch.randn(16), dim=0)
        bg_d = F.normalize(torch.randn(16), dim=0)
        fg_m = F.normalize(torch.randn(16), dim=0)
        bg_m = F.normalize(torch.randn(16), dim=0)

        for idx, rid in enumerate(ids):
            mask = np.zeros((512, 512), dtype=np.uint8)
            rgb = np.full((512, 512, 3), 185, dtype=np.uint8)
            if rid != "q_neg":
                y0, y1 = 128 + idx * 8, 288 + idx * 8
                x0, x1 = 128 + idx * 8, 288 + idx * 8
                mask[y0:y1, x0:x1] = 1
                rgb[y0:y1, x0:x1] = [45, 65, 100]
            mask_by_id[rid] = mask
            rgb_by_id[rid] = rgb

            for key, hw, fv, bv in (("deep_32", 32, fg_d, bg_d), ("deep_64", 64, fg_d, bg_d), ("mid_64", 64, fg_m, bg_m)):
                p = 512 // hw
                mg = mask.reshape(hw, p, hw, p).mean(axis=(1, 3)) >= 0.5
                fmap = torch.randn(16, hw, hw) * 0.10
                fmap[:, mg] += fv[:, None]
                fmap[:, ~mg] += bv[:, None]
                tokens_by_key[key][rid] = to_tokens_norm(F.normalize(fmap, dim=0), "cpu")

        caches = {
            key: build_pairwise_topk_cache(tokens_by_key[key], mask_by_id, ids, pos_ids, grid_hw=(64 if "64" in key else 32))
            for key in ("deep_32", "deep_64", "mid_64")
        }
        aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=0.12) for rid in ids}

        pipe_b0 = ThreeModulePipeline(get_b0_32_config())
        pipe_clean = ThreeModulePipeline(get_clean_core_config())
        pipe_deform = ThreeModulePipeline(
            get_clean_core_config(),
            refiner_cfg=DeformableBoundaryConfig(enabled=True, mode="deformable_grad"),
        )

        for sup in (["s0"], ["s0", "s1", "s2"]):
            out_b0 = pipe_b0.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup, ["q_pos", "q_neg"])
            out_clean = pipe_clean.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup, ["q_pos", "q_neg"])
            out_def = pipe_deform.run_episode(
                caches, tokens_by_key, mask_by_id, aff_by_id, sup, ["q_pos", "q_neg"], rgb_by_id=rgb_by_id
            )

            # Verify Module ① uses pure linear mid-layer fusion (S_sem = 0.75*S_64 + 0.25*Up(S_32) + 0.35*S_mid, zero Softplus veto)
            use_core = False if len(sup) == 1 else True
            s32, _, _ = score_from_pairwise(caches["deep_32"], "q_pos", sup, use_core=use_core, shot_mode="top2_soft")
            s64, _, _ = score_from_pairwise(caches["deep_64"], "q_pos", sup, use_core=use_core, shot_mode="top2_soft")
            sm, _, _ = score_from_pairwise(caches["mid_64"], "q_pos", sup, use_core=use_core, shot_mode="top2_soft")
            s32_up = F.interpolate(s32.reshape(1, 1, 32, 32), size=(64, 64), mode="bilinear", align_corners=False).reshape(-1)
            expected_s_sem = 0.75 * s64 + 0.25 * s32_up + 0.35 * sm
            torch.testing.assert_close(out_clean.semantic_by_id["q_pos"].s_sem, expected_s_sem, atol=1e-6, rtol=1e-6)

            # Verify m_final == m_coarse when DeformGrad is off, and strong separation on q_pos vs q_neg
            np.testing.assert_array_equal(
                out_clean.boundary_by_id["q_pos"].m_final, out_clean.boundary_by_id["q_pos"].m_coarse
            )
            self.assertGreater(float(out_clean.boundary_by_id["q_pos"].m_coarse[mask_by_id["q_pos"] > 0].mean()), 0.85)
            self.assertLess(float(out_clean.boundary_by_id["q_neg"].m_coarse.mean()), 0.05)
            self.assertGreater(float(out_def.boundary_by_id["q_pos"].m_final[mask_by_id["q_pos"] > 0].mean()), 0.85)
            self.assertTrue(np.isfinite(out_b0.boundary_by_id["q_pos"].p_coarse).all())


if __name__ == "__main__":
    unittest.main()
