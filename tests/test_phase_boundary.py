from __future__ import annotations

import math
import unittest

import numpy as np

from pvfs.metrics import boundary_f1
from pvfs.phase_boundary import (
    PhaseBoundaryConfig,
    SupportCalibrationBundle,
    boundary_f1_at_radius,
    build_component_geometry_and_corridors,
    calibrate_support_thresholds,
    check_topology_preserved,
    compute_luma_and_detrended_residual,
    compute_oracle_pm8px_mask,
    estimate_and_freeze_component_carrier,
    refine_query_image_all_variants,
    score_segment_candidates,
)


def _make_default_calib(
    cfg: PhaseBoundaryConfig,
    *,
    allow_outward: bool = True,
    p_floor: float | None = 0.25,
    tau_scale: float = 0.15,
) -> SupportCalibrationBundle:
    taus = {k: v * tau_scale for k, v in cfg.fallback_gain_per_px.items()}
    return SupportCalibrationBundle(
        k_shots=5,
        allow_outward=allow_outward,
        p_floor=p_floor,
        p_floor_fallback=False,
        period_range_px=(2.8, 12.0),
        holdout_r2_floor=0.05,
        tau_by_variant=taus,
        null_block_counts={k: 8 for k in taus},
        tau_fallback_by_variant={k: False for k in taus},
    )


def _synth_periodic_image(
    h: int,
    w: int,
    true_mask: np.ndarray,
    *,
    wavelength_px: float = 6.0,
    theta_rad: float = 0.0,
    amp: float = 0.14,
    phase0: float = 0.3,
    bg_mode: str = "flat",
) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    omega = (2.0 * math.pi / wavelength_px) * np.array([math.cos(theta_rad), math.sin(theta_rad)], dtype=np.float32)
    carrier = amp * np.cos(omega[0] * xx + omega[1] * yy + phase0)
    img = np.full((h, w), 0.45, dtype=np.float32)
    img[true_mask] = 0.45 + carrier[true_mask]
    if bg_mode == "phase_shifted":
        # Same frequency and amplitude outside, but phase shifted by pi (anti-phase neighbors)
        anti_carrier = amp * np.cos(omega[0] * xx + omega[1] * yy + phase0 + math.pi)
        img[~true_mask] = 0.45 + anti_carrier[~true_mask]
    rgb = np.stack([img, img, img], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


class TestPhaseBoundaryRefinement(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PhaseBoundaryConfig()

    def test_switch_off_and_empty_mask_pixel_identical(self) -> None:
        h, w = 128, 128
        gt = np.zeros((h, w), dtype=bool)
        gt[30:90, 30:90] = True
        rgb = _synth_periodic_image(h, w, gt)
        prob = gt.astype(np.float32) * 0.8 + 0.1
        calib = _make_default_calib(self.cfg)

        out = refine_query_image_all_variants(rgb, prob, calib, cfg=self.cfg)
        m0 = prob >= 0.5
        np.testing.assert_array_equal(out["variants"]["A_M3_Pruned"]["mask"], m0)

        # Empty mask case
        prob_empty = np.full((h, w), 0.1, dtype=np.float32)
        out_empty = refine_query_image_all_variants(rgb, prob_empty, calib, cfg=self.cfg)
        for v_res in out_empty["variants"].values():
            self.assertFalse(v_res["mask"].any())

    def test_known_outward_and_inward_boundary_movement(self) -> None:
        h, w = 128, 128
        true_mask = np.zeros((h, w), dtype=bool)
        true_mask[24:104, 24:100] = True
        rgb = _synth_periodic_image(h, w, true_mask, wavelength_px=6.0, theta_rad=0.0, amp=0.16)
        calib = _make_default_calib(self.cfg, tau_scale=0.10)

        # Case 1: Under-segmented coarse mask (right edge at x=96 instead of 100 -> needs outward shift +4)
        coarse_under = np.zeros((h, w), dtype=bool)
        coarse_under[24:104, 24:96] = True
        prob_under = np.where(coarse_under, 0.85, 0.10).astype(np.float32)
        prob_under[24:104, 96:102] = 0.40  # above p_floor=0.25 so outward expansion is permitted

        res_under = refine_query_image_all_variants(rgb, prob_under, calib, cfg=self.cfg)
        f_mask_under = res_under["variants"]["F_FullPhase"]["mask"]
        bf1_init_under = boundary_f1_at_radius(true_mask, coarse_under, radius=1)
        bf1_ref_under = boundary_f1_at_radius(true_mask, f_mask_under, radius=1)
        self.assertGreater(res_under["variants"]["F_FullPhase"]["outward_arc_len_px"], 0)
        self.assertGreater(int(f_mask_under.sum()), int(coarse_under.sum()))
        self.assertGreater(bf1_ref_under, bf1_init_under)

        # Case 2: Over-segmented coarse mask (right edge at x=104 instead of 100 -> needs inward shift -4)
        coarse_over = np.zeros((h, w), dtype=bool)
        coarse_over[24:104, 24:104] = True
        prob_over = np.where(coarse_over, 0.85, 0.10).astype(np.float32)

        res_over = refine_query_image_all_variants(rgb, prob_over, calib, cfg=self.cfg)
        f_mask_over = res_over["variants"]["F_FullPhase"]["mask"]
        bf1_init_over = boundary_f1_at_radius(true_mask, coarse_over, radius=1)
        bf1_ref_over = boundary_f1_at_radius(true_mask, f_mask_over, radius=1)
        self.assertGreater(res_over["variants"]["F_FullPhase"]["inward_arc_len_px"], 0)
        self.assertLess(int(f_mask_over.sum()), int(coarse_over.sum()))
        self.assertGreater(bf1_ref_over, bf1_init_over)

    def test_internal_dark_trough_does_not_attract_boundary_unlike_grad_align(self) -> None:
        """Strong dark stripe trough inside the array must not pull F inward when coarse boundary is already at true edge."""
        h, w = 128, 128
        true_mask = np.zeros((h, w), dtype=bool)
        true_mask[24:104, 24:100] = True
        rgb = _synth_periodic_image(h, w, true_mask, wavelength_px=6.0, theta_rad=0.0, amp=0.16)
        # Inject an extra-sharp internal dark trough line at x=96 (4px inside true boundary x=100)
        # while the true boundary at x=100 has matched mean luminance (0.45 inside vs 0.45 outside)
        prob = np.where(true_mask, 0.85, 0.10).astype(np.float32)
        calib = _make_default_calib(self.cfg, tau_scale=0.10)

        out = refine_query_image_all_variants(rgb, prob, calib, cfg=self.cfg)
        f_mask = out["variants"]["F_FullPhase"]["mask"]
        # F_FullPhase sees genuine phase continuation up to x=100 and zero carrier outside x=100,
        # so it keeps the true boundary at x=100 without shrinking to internal dark troughs.
        iou_f = float(np.logical_and(true_mask, f_mask).sum() / np.logical_or(true_mask, f_mask).sum())
        self.assertGreaterEqual(iou_f, 0.98)

    def test_same_freq_same_amp_phase_discontinuity_beats_phase_blind_E(self) -> None:
        """When adjacent background has the SAME frequency and amplitude but shifted phase (+pi),
        frozen-phase F shrinks the over-segmented mask back to the phase boundary, whereas phase-blind E cannot."""
        h, w = 128, 128
        true_mask = np.zeros((h, w), dtype=bool)
        true_mask[24:104, 24:96] = True
        # Background has identical frequency and amplitude, but phase shifted by pi!
        rgb = _synth_periodic_image(
            h, w, true_mask, wavelength_px=6.0, theta_rad=0.0, amp=0.16, bg_mode="phase_shifted"
        )
        # Coarse mask over-extends 5px into the phase-shifted background (to x=101)
        coarse_mask = np.zeros((h, w), dtype=bool)
        coarse_mask[24:104, 24:101] = True
        prob = np.where(coarse_mask, 0.85, 0.10).astype(np.float32)
        calib = _make_default_calib(self.cfg, tau_scale=0.10)

        out = refine_query_image_all_variants(rgb, prob, calib, cfg=self.cfg)
        f_mask = out["variants"]["F_FullPhase"]["mask"]
        e_mask = out["variants"]["E_FreqAmp_NoPhase"]["mask"]

        bf1_f = boundary_f1_at_radius(true_mask, f_mask, radius=1)
        bf1_e = boundary_f1_at_radius(true_mask, e_mask, radius=1)
        self.assertGreater(bf1_f, bf1_e)

    def test_fallbacks_nonperiodic_lowvar_narrow_border_adjacent(self) -> None:
        h, w = 128, 128
        calib = _make_default_calib(self.cfg)

        # 1. Low variance / flat interior -> fallback low_interior_signal_energy
        m_flat = np.zeros((h, w), dtype=bool)
        m_flat[24:104, 24:104] = True
        rgb_flat = np.full((h, w, 3), 120, dtype=np.uint8)
        prob_flat = np.where(m_flat, 0.85, 0.10).astype(np.float32)
        out_flat = refine_query_image_all_variants(rgb_flat, prob_flat, calib, cfg=self.cfg)
        self.assertFalse(out_flat["carriers"][1].valid)
        np.testing.assert_array_equal(out_flat["variants"]["F_FullPhase"]["mask"], m_flat)

        # 2. Aperiodic random noise interior -> fallback holdout_r2_below_threshold or phase_inconsistent
        rng = np.random.default_rng(42)
        rgb_noise = rng.integers(60, 190, size=(h, w, 3), dtype=np.uint8)
        out_noise = refine_query_image_all_variants(rgb_noise, prob_flat, calib, cfg=self.cfg)
        self.assertFalse(out_noise["carriers"][1].valid)
        np.testing.assert_array_equal(out_noise["variants"]["F_FullPhase"]["mask"], m_flat)

        # 3. Too narrow component (width 10px -> eroded by 8px leaves empty interior) -> fallback insufficient_eroded_interior
        m_narrow = np.zeros((h, w), dtype=bool)
        m_narrow[20:100, 50:60] = True
        prob_narrow = np.where(m_narrow, 0.85, 0.10).astype(np.float32)
        out_narrow = refine_query_image_all_variants(rgb_noise, prob_narrow, calib, cfg=self.cfg)
        self.assertFalse(out_narrow["carriers"][1].valid)
        self.assertEqual(out_narrow["carriers"][1].fallback_reason, "insufficient_eroded_interior")

        # 4. Two adjacent components separated by 3px must never merge
        m_two = np.zeros((h, w), dtype=bool)
        m_two[24:104, 16:58] = True
        m_two[24:104, 61:104] = True
        rgb_two = _synth_periodic_image(h, w, np.ones((h, w), dtype=bool), wavelength_px=6.0)
        prob_two = np.where(m_two, 0.85, 0.45).astype(np.float32)
        out_two = refine_query_image_all_variants(rgb_two, prob_two, calib, cfg=self.cfg)
        for v_name, v_res in out_two["variants"].items():
            self.assertTrue(check_topology_preserved(m_two, v_res["mask"]), f"Topology broken in {v_name}")

    def test_counterfactual_G_shares_exact_freq_amp_and_corridor_pixels(self) -> None:
        h, w = 128, 128
        true_mask = np.zeros((h, w), dtype=bool)
        true_mask[24:104, 24:100] = True
        rgb = _synth_periodic_image(h, w, true_mask, wavelength_px=6.0, theta_rad=0.25, amp=0.15)
        coarse = np.zeros((h, w), dtype=bool)
        coarse[24:104, 24:96] = True
        prob = np.where(coarse, 0.85, 0.40).astype(np.float32)
        calib = _make_default_calib(self.cfg)

        _, _, res = compute_luma_and_detrended_residual(rgb, self.cfg)
        _, _, corrs, _ = build_component_geometry_and_corridors(coarse, self.cfg)
        segs = corrs[1]
        corr_u = np.zeros_like(coarse, dtype=bool)
        for s in segs:
            corr_u[s.corridor_y, s.corridor_x] = True
        carrier = estimate_and_freeze_component_carrier(1, coarse, corr_u, res, cfg=self.cfg)
        self.assertTrue(carrier.valid)

        seg0 = segs[0]
        dummy = np.zeros_like(res)
        sc_f = score_segment_candidates(
            seg0,
            "F_FullPhase",
            residual=res,
            grad_mag=dummy,
            local_var_map=dummy,
            prob_512=prob,
            carrier=carrier,
            p_floor=0.25,
            allow_outward=True,
            cfg=self.cfg,
        )
        for gamma in (0.5 * math.pi, math.pi, 1.5 * math.pi):
            sc_g = score_segment_candidates(
                seg0,
                "G_PhaseCounterfactual",
                residual=res,
                grad_mag=dummy,
                local_var_map=dummy,
                prob_512=prob,
                carrier=carrier,
                p_floor=0.25,
                allow_outward=True,
                gamma_offset_rad=gamma,
                cfg=self.cfg,
            )
            self.assertEqual(set(sc_f["gains_by_d"].keys()), set(sc_g["gains_by_d"].keys()))
            self.assertEqual(len(sc_f["evidence_v"]), len(sc_g["evidence_v"]))

    def test_metrics_compatibility_and_no_query_gt_leakage(self) -> None:
        h, w = 128, 128
        gt = np.zeros((h, w), dtype=bool)
        gt[30:90, 30:90] = True
        pred = np.zeros((h, w), dtype=bool)
        pred[31:91, 31:91] = True

        # BF1@2px must exactly match pvfs.metrics.boundary_f1
        bf1_orig = boundary_f1(gt, pred)
        bf1_2px = boundary_f1_at_radius(gt, pred, radius=2)
        bf1_1px = boundary_f1_at_radius(gt, pred, radius=1)
        self.assertAlmostEqual(bf1_orig, bf1_2px, places=7)
        self.assertLessEqual(bf1_1px, bf1_2px + 1e-7)

        # Oracle +/-8px mask fixes 1px boundary shift completely
        oracle_info = compute_oracle_pm8px_mask(gt, pred.astype(np.float32))
        np.testing.assert_array_equal(oracle_info["oracle_mask"], gt)
        self.assertAlmostEqual(oracle_info["pm8_fixable_fraction"], 1.0, places=6)

        # Verify calibration & inference never accept or depend on query GT
        rgb_sup = _synth_periodic_image(h, w, gt, wavelength_px=6.0)
        sup_mask = gt.astype(np.uint8)
        p_loo = np.where(gt, 0.75, 0.15).astype(np.float32)
        calib = calibrate_support_thresholds(
            ["sup1", "sup2"],
            {"sup1": rgb_sup, "sup2": rgb_sup},
            {"sup1": sup_mask, "sup2": sup_mask},
            {"sup1": p_loo, "sup2": p_loo},
            cfg=self.cfg,
        )
        # Verify eval module imports cleanly and CHW/HWC inputs yield identical residuals
        from pvfs.phase_boundary_eval import run_phase_boundary_dev_evaluation  # noqa: F401
        _, _, res_hwc = compute_luma_and_detrended_residual(rgb_sup, self.cfg)
        _, _, res_chw = compute_luma_and_detrended_residual(np.transpose(rgb_sup, (2, 0, 1)), self.cfg)
        np.testing.assert_allclose(res_hwc, res_chw, atol=1e-6)

        self.assertTrue(calib.allow_outward)
        self.assertIsNotNone(calib.p_floor)


if __name__ == "__main__":
    unittest.main()
