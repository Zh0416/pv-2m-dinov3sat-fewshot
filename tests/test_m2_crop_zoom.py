from __future__ import annotations

import unittest

import numpy as np
import torch

from pvfs.modules.crop_zoom import (
    CropZoomConfig,
    audit_offline_candidate_coverage,
    extract_reflect_padded_crop,
    project_crop_scores_max_increment,
    propose_seedless_candidate_windows,
)


class M2CropZoomTests(unittest.TestCase):
    def test_seedless_proposal_and_border_adjacent_reflect_crop(self) -> None:
        cfg = CropZoomConfig(enabled=True, mode="crop_zoom_only")
        s_sem = np.full((64, 64), -0.10, dtype=np.float32)
        p_sem = np.full((64, 64), 0.10, dtype=np.float32)
        # Inject two sub-threshold peaks: one near top-left image border (tok 2, 2), one interior (tok 32, 32)
        s_sem[1:4, 1:4] = 0.08  # local prominence = 0.18 >= 0.08, p_sem = 0.35 <= 0.65
        p_sem[1:4, 1:4] = 0.35
        s_sem[31:34, 31:34] = 0.06
        p_sem[31:34, 31:34] = 0.30

        wins = propose_seedless_candidate_windows(s_sem, p_sem, cfg)
        self.assertEqual(len(wins), 2)
        # Border window has y0 < 0 and x0 < 0, must extract cleanly with reflect padding
        border_w = min(wins, key=lambda w: w.center_tok_y)
        self.assertLess(border_w.y0, 0)
        rgb = np.arange(512 * 512 * 3, dtype=np.uint32).reshape(512, 512, 3) % 256
        crop, valid_box = extract_reflect_padded_crop(rgb.astype(np.uint8), border_w)
        self.assertEqual(crop.shape, (128, 128, 3))
        self.assertGreater(valid_box[0], 0)
        self.assertEqual(valid_box[1], 128)

    def test_overlapping_windows_take_max_increment_not_additive_sum(self) -> None:
        cfg = CropZoomConfig(enabled=True, mode="crop_zoom_only", nms_min_center_dist_tokens=4)
        s_sem = np.full((64, 64), 0.0, dtype=np.float32)
        p_sem = np.full((64, 64), 0.20, dtype=np.float32)
        s_sem[20, 20] = 0.15
        s_sem[20, 26] = 0.14  # 6 tokens apart -> their 16x16 windows overlap in core!

        wins = propose_seedless_candidate_windows(s_sem, p_sem, cfg)
        self.assertEqual(len(wins), 2)

        s_base = torch.zeros((4096,), dtype=torch.float32)
        # Both windows predict +0.20 in their overlap -> max-increment must be 0.20, NOT 0.40!
        win_scores = [
            (wins[0], torch.full((16, 16), 0.20, dtype=torch.float32)),
            (wins[1], torch.full((16, 16), 0.20, dtype=torch.float32)),
        ]
        s_out, diag = project_crop_scores_max_increment(s_base, win_scores, max_writeback_delta=0.30)
        self.assertAlmostEqual(float(s_out.max().item()), 0.20, places=6)
        self.assertEqual(diag["crop_windows_applied"], 2)

    def test_offline_gt_coverage_audit(self) -> None:
        gt = np.zeros((512, 512), dtype=bool)
        gt[248:272, 248:272] = True  # small missed PV component around token (32, 32)
        m0 = np.zeros((512, 512), dtype=bool)  # completely missed
        s_sem = np.full((64, 64), -0.10, dtype=np.float32)
        p_sem = np.full((64, 64), 0.10, dtype=np.float32)
        s_sem[31:34, 31:34] = 0.08
        p_sem[31:34, 31:34] = 0.35

        wins = propose_seedless_candidate_windows(s_sem, p_sem)
        audit = audit_offline_candidate_coverage(gt, m0, wins)
        self.assertEqual(audit["missed_components"], 1)
        self.assertEqual(audit["covered_missed_components"], 1)


if __name__ == "__main__":
    unittest.main()
