from __future__ import annotations

import unittest

import numpy as np
import torch

from pvfs.fast_bank import build_pairwise_topk_cache, score_from_pairwise, select_fg_bg_indices, to_tokens_norm


def _direct_score_bank(
    q_tok: torch.Tensor,
    fg_per_img: list[torch.Tensor],
    bg_all: torch.Tensor,
    *,
    top_m: int = 3,
    alpha_bg: float = 0.50,
    shot_mode: str = "pooled",
) -> torch.Tensor:
    if len(fg_per_img) == 1 or shot_mode == "pooled":
        fg_cat = torch.cat(fg_per_img, dim=0)
        s_pos = (q_tok @ fg_cat.T).clamp(-1.0, 1.0).topk(min(top_m, fg_cat.shape[0]), dim=1).values.mean(dim=1)
    else:
        per_shot = torch.stack(
            [(q_tok @ fg.T).clamp(-1.0, 1.0).topk(min(top_m, fg.shape[0]), dim=1).values.mean(dim=1) for fg in fg_per_img],
            dim=0,
        )
        if shot_mode == "balanced_70_30":
            s_pos = 0.70 * per_shot.max(dim=0).values + 0.30 * per_shot.mean(dim=0)
        elif shot_mode == "top2_soft":
            ps = per_shot.sort(dim=0, descending=True).values
            s_pos = 0.75 * ps[0] + 0.25 * ps[1] if ps.shape[0] == 2 else 0.65 * ps[0] + 0.25 * ps[1] + 0.10 * ps[2:].mean(dim=0)
        else:
            raise ValueError(shot_mode)
    s_neg = (q_tok @ bg_all.T).clamp(-1.0, 1.0).topk(min(top_m, bg_all.shape[0]), dim=1).values.mean(dim=1)
    return s_pos - alpha_bg * s_neg


class FastBankTests(unittest.TestCase):
    def test_pairwise_cache_matches_direct_score_bank(self) -> None:
        torch.manual_seed(7)
        grid_hw = 32
        feats = {f"img_{i}": to_tokens_norm(torch.randn(16, grid_hw, grid_hw), "cpu") for i in range(4)}
        masks = {}
        for i in range(3):
            m = np.zeros((512, 512), dtype=np.uint8)
            m[64 + i * 32 : 192 + i * 32, 64:224] = 1
            masks[f"img_{i}"] = m
        masks["img_3"] = np.zeros((512, 512), dtype=np.uint8)

        all_ids = ["img_0", "img_1", "img_2", "img_3"]
        pos_ids = ["img_0", "img_1", "img_2"]
        support_ids = ["img_0", "img_1", "img_2"]

        cache = build_pairwise_topk_cache(feats, masks, all_ids, pos_ids, grid_hw=grid_hw, core_purity=0.85, top_m=3)

        fg_per_img, bg_list, bg_src = [], [], []
        for idx, sid in enumerate(support_ids):
            fg_m, bg_m, _ = select_fg_bg_indices(masks[sid], grid_hw, 0.85)
            fg_per_img.append(feats[sid][torch.from_numpy(np.flatnonzero(fg_m))])
            bg_t = feats[sid][torch.from_numpy(np.flatnonzero(bg_m))]
            bg_list.append(bg_t)
            bg_src.append(torch.full((bg_t.shape[0],), idx, dtype=torch.long))
        bg_all = torch.cat(bg_list, dim=0)
        bg_src_t = torch.cat(bg_src, dim=0)

        for qid in all_ids:
            for shot_mode in ("pooled", "balanced_70_30", "top2_soft"):
                direct = _direct_score_bank(feats[qid], fg_per_img, bg_all, shot_mode=shot_mode)
                fast, _, _ = score_from_pairwise(cache, qid, support_ids, use_core=True, shot_mode=shot_mode, alpha_bg=0.5)
                self.assertLess(float((direct - fast).abs().max().item()), 2e-3)

        loo_fast, _, _ = score_from_pairwise(
            cache, "img_1", support_ids, use_core=True, shot_mode="top2_soft", alpha_bg=0.5, exclude_pos_idx=1
        )
        loo_direct = _direct_score_bank(
            feats["img_1"], [fg_per_img[0], fg_per_img[2]], bg_all[bg_src_t != 1], shot_mode="top2_soft"
        )
        self.assertLess(float((loo_direct - loo_fast).abs().max().item()), 2e-3)


if __name__ == "__main__":
    unittest.main()
