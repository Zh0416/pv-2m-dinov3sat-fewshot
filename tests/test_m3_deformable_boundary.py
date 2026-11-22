from __future__ import annotations

import math
import unittest

import numpy as np

from pvfs.modules.deformable_boundary import (
    DeformableBoundaryConfig,
    build_deformable_loops,
    refine_mask_deformable_grad,
)
from pvfs.phase_boundary import boundary_f1_at_radius, check_topology_preserved


class M3DeformableBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = DeformableBoundaryConfig(enabled=True, mode="deformable_grad")

    def test_zero_shift_and_disabled_pixel_identical(self) -> None:
        h, w = 128, 128
        m0 = np.zeros((h, w), dtype=bool)
        m0[32:96, 32:96] = True
        rgb_flat = np.full((h, w, 3), 128, dtype=np.uint8)
        prob = np.where(m0, 0.85, 0.15).astype(np.float32)

        # Disabled switch -> exact copy
        res_off = refine_mask_deformable_grad(
            m0, rgb_flat, prob, cfg=DeformableBoundaryConfig(enabled=False, mode="off")
        )
        np.testing.assert_array_equal(res_off["mask"], m0)

        # Flat zero-gradient image -> zero shift -> exact copy without redrawing
        res_zero = refine_mask_deformable_grad(m0, rgb_flat, prob, cfg=self.cfg)
        np.testing.assert_array_equal(res_zero["mask"], m0)
        self.assertEqual(res_zero["moved_loops"], 0)

    def test_disjoint_voronoi_arc_corridors_no_double_counting(self) -> None:
        h, w = 128, 128
        m0 = np.zeros((h, w), dtype=bool)
        m0[28:100, 28:100] = True
        _, _, loops = build_deformable_loops(m0, self.cfg)
        self.assertEqual(len(loops), 1)
        seen: set[tuple[int, int]] = set()
        for arc in loops[0].arcs:
            pts = set(zip(arc.corridor_y.tolist(), arc.corridor_x.tolist()))
            self.assertEqual(len(seen.intersection(pts)), 0, "Corridor pixels must never be double-counted across arcs")
            seen.update(pts)

    def test_tilted_edge_deformation_beats_coarse_on_gradient_and_offline_gt_ref(self) -> None:
        """Tilted 5-degree edge: coarse mask is axis-aligned, true edge slopes by 5 deg across 72px (~6.3px shift)."""
        h, w = 140, 140
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        # True mask has a tilted right edge: x <= 96 + (y - 70) * tan(5 deg)
        tilt_x = 96.0 + (yy - 70.0) * math.tan(math.radians(5.0))
        true_mask = (yy >= 32) & (yy < 108) & (xx >= 32) & (xx <= tilt_x)
        # Axis-aligned coarse mask
        coarse_mask = (yy >= 32) & (yy < 108) & (xx >= 32) & (xx <= 96)

        rgb = np.where(true_mask[..., None], 70, 190).astype(np.uint8)
        rgb = np.repeat(rgb, 3, axis=-1)
        prob = np.where(coarse_mask, 0.85, 0.40).astype(np.float32)

        bf1_coarse = boundary_f1_at_radius(true_mask, coarse_mask, radius=1)

        # 1. Gradient-driven deformable DP
        res_grad = refine_mask_deformable_grad(
            coarse_mask, rgb, prob, allow_outward=True, p_floor=0.25, tau_grad_per_px=0.01, cfg=self.cfg
        )
        bf1_grad = boundary_f1_at_radius(true_mask, res_grad["mask"], radius=1)
        self.assertTrue(check_topology_preserved(coarse_mask, res_grad["mask"]))
        self.assertGreater(bf1_grad, 0.95)
        self.assertGreater(bf1_grad, bf1_coarse + 0.10)

        # 2. Offline GT reachable reference
        res_gt = refine_mask_deformable_grad(
            coarse_mask, None, prob, allow_outward=True, p_floor=None, gt_mask_offline=true_mask, cfg=self.cfg
        )
        bf1_gt_ref = boundary_f1_at_radius(true_mask, res_gt["mask"], radius=1)
        self.assertGreater(bf1_gt_ref, 0.95)
        self.assertGreater(bf1_gt_ref, bf1_coarse + 0.10)

    def test_slender_array_and_adjacent_components_preserve_topology(self) -> None:
        h, w = 128, 128
        m0 = np.zeros((h, w), dtype=bool)
        # Two slender adjacent arrays separated by 3px
        m0[24:104, 30:42] = True
        m0[24:104, 45:57] = True
        # Strong distracting RGB edge right between them
        rgb = np.full((h, w, 3), 180, dtype=np.uint8)
        rgb[24:104, 28:59] = 60
        prob = np.where(m0, 0.85, 0.45).astype(np.float32)

        res = refine_mask_deformable_grad(
            m0, rgb, prob, allow_outward=True, p_floor=0.25, tau_grad_per_px=0.01, cfg=self.cfg
        )
        self.assertTrue(check_topology_preserved(m0, res["mask"]))


if __name__ == "__main__":
    unittest.main()
