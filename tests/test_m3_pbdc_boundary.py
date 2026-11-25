from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

from pvfs.metrics import boundary_f1_at_radius
from pvfs.modules.deformable_boundary import (
    DeformableBoundaryConfig,
    check_topology_preserved,
    refine_mask_deformable_grad,
)
from pvfs.modules.pbdc_boundary import PBDCConfig, refine_mask_pbdc


class M3PBDCBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pbdc_cfg = PBDCConfig(enabled=True)
        self.old_cfg = DeformableBoundaryConfig(enabled=True, mode="deformable_grad")

    def test_1_hd_08m_striped_pv_internal_seams_and_bloat_trim(self) -> None:
        """0.8m HD scenario: dark-blue PV with strong 1px bright white mounting seams every 5px.

        Coarse mask is bloated outward by +2px into soil (typical PV663 case where Recall > Prec).
        Old Scharr Deformable_Grad gets sucked onto internal bright seams (-3..-5px inside),
        whereas PBDC ignores internal 1px seams and snaps cleanly to the true outer PV edge!
        """
        h, w = 160, 160
        true_mask = np.zeros((h, w), dtype=bool)
        true_mask[40:120, 40:120] = True

        # Soil exterior: warm brown [145, 130, 110]; PV interior: dark blue [40, 55, 85] + bright [220,220,225] seams!
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:] = [145, 130, 110]
        rgb[true_mask] = [40, 55, 85]
        for r_seam in range(44, 118, 5):
            rgb[r_seam, 41:119] = [225, 225, 230]  # much stronger Scharr gradient than outer PV-soil edge!
        for c_seam in range(44, 118, 5):
            rgb[41:119, c_seam] = [225, 225, 230]

        # Coarse mask bloated by +2px on all sides, steep HD P_coarse
        coarse_mask = np.zeros((h, w), dtype=bool)
        coarse_mask[38:122, 38:122] = True
        prob = gaussian_filter(coarse_mask.astype(np.float32), sigma=2.2)

        bf1_coarse = boundary_f1_at_radius(true_mask, coarse_mask, radius=1)
        out_old = refine_mask_deformable_grad(
            coarse_mask, rgb, prob, allow_outward=True, p_floor=0.28, tau_grad_per_px=0.05, cfg=self.old_cfg
        )
        out_pbdc = refine_mask_pbdc(coarse_mask, rgb, prob, k_shots=5, p_floor_loo=0.28, cfg=self.pbdc_cfg)

        bf1_old = boundary_f1_at_radius(true_mask, out_old["mask"], radius=1)
        bf1_pbdc = boundary_f1_at_radius(true_mask, out_pbdc["mask"], radius=1)

        self.assertTrue(check_topology_preserved(coarse_mask, out_pbdc["mask"]))
        self.assertGreater(bf1_pbdc, bf1_coarse + 0.25)
        self.assertGreater(bf1_pbdc, bf1_old + 0.10)
        self.assertGreater(bf1_pbdc, 0.90)

    def test_2_already_aligned_sharp_mask_stays_locked_at_zero_drift(self) -> None:
        """When M_coarse == true_mask on a sharp 0.8m PV array (even with internal seams), PBDC causes zero degradation."""
        h, w = 160, 160
        true_mask = np.zeros((h, w), dtype=bool)
        true_mask[40:120, 40:120] = True
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:] = [150, 138, 120]
        rgb[true_mask] = [42, 58, 88]
        for r_seam in range(44, 118, 5):
            rgb[r_seam, 41:119] = [220, 220, 225]

        prob = gaussian_filter(true_mask.astype(np.float32), sigma=1.6)
        coarse_mask = prob >= 0.50  # already exact!
        out_pbdc = refine_mask_pbdc(coarse_mask, rgb, prob, k_shots=5, p_floor_loo=0.28, cfg=self.pbdc_cfg)
        bf1_pbdc = boundary_f1_at_radius(true_mask, out_pbdc["mask"], radius=1)
        self.assertGreaterEqual(bf1_pbdc, 0.98)

    def test_3_adjacent_bright_road_trap_rejected(self) -> None:
        """Adjacent bright concrete road 3px outside PV array must NOT pull contour across the road."""
        h, w = 160, 160
        true_mask = np.zeros((h, w), dtype=bool)
        true_mask[40:120, 40:100] = True
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:] = [95, 110, 85]  # medium grass/soil
        rgb[40:120, 40:100] = [48, 62, 88]  # PV array
        rgb[36:124, 101:105] = [240, 240, 240]  # super-bright concrete road right outside right edge!

        # Coarse mask under-segmented by 1px on right edge (x <= 98), with blurred P_coarse bleeding onto road
        coarse_mask = np.zeros((h, w), dtype=bool)
        coarse_mask[40:120, 40:99] = True
        prob = gaussian_filter(true_mask.astype(np.float32), sigma=3.5)

        out_old = refine_mask_deformable_grad(
            coarse_mask, rgb, prob, allow_outward=True, p_floor=0.25, tau_grad_per_px=0.05, cfg=self.old_cfg
        )
        out_pbdc = refine_mask_pbdc(coarse_mask, rgb, prob, k_shots=5, p_floor_loo=0.25, cfg=self.pbdc_cfg)

        # Road pixels (x in 101..104) swallowed:
        road_swallowed_old = int(out_old["mask"][45:115, 101:105].sum())
        road_swallowed_pbdc = int(out_pbdc["mask"][45:115, 101:105].sum())
        bf1_pbdc = boundary_f1_at_radius(true_mask, out_pbdc["mask"], radius=1)

        self.assertLessEqual(road_swallowed_pbdc, road_swallowed_old)
        self.assertEqual(road_swallowed_pbdc, 0)
        self.assertGreater(bf1_pbdc, 0.94)

    def test_4_blurry_2m_tilted_edge_snaps_accurately(self) -> None:
        """2m blurry tilted PV array: axis-aligned coarse mask snaps cleanly to 5-deg tilted blurred boundary."""
        h, w = 150, 150
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        tilt_x = 96.0 + (yy - 75.0) * math.tan(math.radians(5.0))
        true_mask = (yy >= 35) & (yy < 115) & (xx >= 35) & (xx <= tilt_x)
        coarse_mask = (yy >= 35) & (yy < 115) & (xx >= 35) & (xx <= 95)

        rgb_f = np.where(true_mask[..., None], np.array([50.0, 68.0, 98.0]), np.array([155.0, 148.0, 132.0]))
        rgb = np.clip(gaussian_filter(rgb_f, sigma=(1.2, 1.2, 0.0)), 0, 255).astype(np.uint8)
        prob = np.clip(gaussian_filter(coarse_mask.astype(np.float32), sigma=4.0), 0.0, 1.0)
        # Ensure coarse_mask corresponds to prob >= 0.50
        coarse_m = prob >= 0.50

        bf1_coarse = boundary_f1_at_radius(true_mask, coarse_m, radius=1)
        out_pbdc = refine_mask_pbdc(coarse_m, rgb, prob, k_shots=3, p_floor_loo=0.25, cfg=self.pbdc_cfg)
        bf1_pbdc = boundary_f1_at_radius(true_mask, out_pbdc["mask"], radius=1)

        self.assertTrue(check_topology_preserved(coarse_m, out_pbdc["mask"]))
        self.assertGreater(bf1_pbdc, bf1_coarse + 0.04)
        self.assertGreater(bf1_pbdc, 0.84)

    def test_5_pure_isoluminant_chromatic_boundary_snaps(self) -> None:
        """Isoluminant blue PV vs green vegetation (identical grayscale L -> zero grayscale Scharr gradient!)."""
        h, w = 140, 140
        true_mask = np.zeros((h, w), dtype=bool)
        true_mask[36:104, 36:100] = True
        coarse_mask = np.zeros((h, w), dtype=bool)
        coarse_mask[36:104, 36:97] = True  # missed 3px on right edge

        # PV: R=50, G=75, B=140 -> L = 0.299*50 + 0.587*75 + 0.114*140 = 14.95 + 44.025 + 15.96 = 74.935
        # Veg: R=60, G=90, B=36 -> L = 0.299*60 + 0.587*90 + 0.114*36 = 17.94 + 52.83 + 4.104 = 74.874 (virtually identical L!)
        rng = np.random.default_rng(42)
        rgb_f = np.zeros((h, w, 3), dtype=np.float32)
        rgb_f[:] = [60.0, 90.0, 36.0]
        rgb_f[true_mask] = [50.0, 75.0, 140.0]
        iso_noise = rng.normal(0.0, 3.0, size=(h, w, 1)).astype(np.float32)
        rgb = np.clip(rgb_f + iso_noise, 0, 255).astype(np.uint8)
        prob = np.where(coarse_mask, 0.75, 0.18).astype(np.float32)
        prob[36:104, 97:102] = 0.42  # uncertain transition zone

        bf1_coarse = boundary_f1_at_radius(true_mask, coarse_mask, radius=1)
        out_old = refine_mask_deformable_grad(
            coarse_mask, rgb, prob, allow_outward=True, p_floor=0.30, tau_grad_per_px=0.05, cfg=self.old_cfg
        )
        out_pbdc = refine_mask_pbdc(coarse_mask, rgb, prob, k_shots=5, p_floor_loo=0.30, cfg=self.pbdc_cfg)

        bf1_old = boundary_f1_at_radius(true_mask, out_old["mask"], radius=1)
        bf1_pbdc = boundary_f1_at_radius(true_mask, out_pbdc["mask"], radius=1)
        self.assertGreater(bf1_pbdc, bf1_coarse + 0.10)
        self.assertGreater(bf1_pbdc, bf1_old + 0.08)


if __name__ == "__main__":
    unittest.main()
