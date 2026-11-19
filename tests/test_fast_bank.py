from __future__ import annotations

import unittest

import numpy as np
import torch

from pvfs.fast_bank import build_pairwise_topk_cache, score_from_pairwise
from pvfs.iter1_methods import _build_bank_for_scale, _score_bank, _to_tokens


class FastBankTests(unittest.TestCase):
    def test_pairwise_cache_matches_direct_score_bank(self) -> None:
        torch.manual_seed(7)
        grid_hw = 16  # fast synthetic 256-token test (upscales cleanly from 512x512 mask patches)
        # Note: select_fg_bg_indices assumes 512 // grid_hw, so use 32x32 grid with 512x512 masks.
        grid_hw = 32
        feats = {
            f"img_{i}": _to_tokens(torch.randn(16, grid_hw, grid_hw), "cpu")
            for i in range(4)
        }
        masks = {}
        for i in range(3):
            m = np.zeros((512, 512), dtype=np.uint8)
            m[64 + i * 32 : 192 + i * 32, 64:224] = 1
            masks[f"img_{i}"] = m
        masks["img_3"] = np.zeros((512, 512), dtype=np.uint8)

        all_ids = ["img_0", "img_1", "img_2", "img_3"]
        pos_ids = ["img_0", "img_1", "img_2"]
        support_ids = ["img_0", "img_1", "img_2"]

        cache = build_pairwise_topk_cache(
            feats, masks, all_ids, pos_ids, grid_hw=grid_hw, core_purity=0.85, top_m=3
        )
        direct_bank = _build_bank_for_scale(feats, masks, support_ids, grid_hw, core_purity=0.85)

        for qid in all_ids:
            for balanced, shot_mode in ((False, "pooled"), (True, "balanced_70_30")):
                direct = _score_bank(
                    feats[qid],
                    direct_bank["fg_per_img"],
                    direct_bank["bg"],
                    top_m=3,
                    alpha_bg=0.5,
                    balanced=balanced,
                )
                fast, _, _ = score_from_pairwise(
                    cache, qid, support_ids, use_core=True, shot_mode=shot_mode, alpha_bg=0.5
                )
                max_diff = float((direct - fast).abs().max().item())
                self.assertLess(max_diff, 2e-3, f"Mismatch for {qid} ({shot_mode}): {max_diff}")

        # Verify LOO exclusion matches subset bank
        loo_fast, _, _ = score_from_pairwise(
            cache,
            "img_1",
            support_ids,
            use_core=True,
            shot_mode="balanced_70_30",
            alpha_bg=0.5,
            exclude_pos_idx=1,
        )
        fgs_loo = [direct_bank["fg_per_img"][0], direct_bank["fg_per_img"][2]]
        bgs_loo = direct_bank["bg"][direct_bank["bg_src"] != 1]
        loo_direct = _score_bank(feats["img_1"], fgs_loo, bgs_loo, top_m=3, alpha_bg=0.5, balanced=True)
        self.assertLess(float((loo_direct - loo_fast).abs().max().item()), 2e-3)


if __name__ == "__main__":
    unittest.main()
