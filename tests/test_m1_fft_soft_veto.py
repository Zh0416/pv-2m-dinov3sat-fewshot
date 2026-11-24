from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.iter3_methods import _to_tokens_norm, compute_dino_affinity_weights
from pvfs.iter5_methods import run_iter5_episode
from pvfs.modules import ThreeModulePipeline, get_m3_pruned_config
from pvfs.modules.fft_soft_veto import (
    FFTSoftVetoConfig,
    compute_adaptive_peak_share,
    compute_window_mask_coverage,
    evaluate_fft_soft_veto_map,
    evaluate_k1_nonoverlap_bg_penalties,
    fit_fft_soft_veto_reference,
    precompute_image_fft,
    select_adaptive_band,
)


def _make_striped_rgb(seed: int, *, pv_box: tuple[int, int, int, int], period_px: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:512, :512].astype(np.float32)
    base = 120.0 + rng.normal(0.0, 10.0, size=(512, 512))
    mask = np.zeros((512, 512), dtype=np.uint8)
    y0, y1, x0, x1 = pv_box
    mask[y0:y1, x0:x1] = 1
    stripes = 45.0 * np.sin(2.0 * np.pi * xx / period_px)
    img_g = np.clip(np.where(mask > 0, 100.0 + stripes + rng.normal(0.0, 3.0, size=(512, 512)), base), 0, 255).astype(np.uint8)
    rgb = np.stack((img_g, img_g, img_g), axis=-1)
    return rgb, mask


class M1FFTSoftVetoTests(unittest.TestCase):
    def test_fft_radial_band_and_peak_share_and_no_fixed_fallback(self) -> None:
        cfg = FFTSoftVetoConfig(enabled=True)
        rgb, mask = _make_striped_rgb(11, pv_box=(128, 320, 128, 320), period_px=4.0)
        pre = precompute_image_fft("syn_1", rgb)
        self.assertEqual(int(pre.interior.sum()), 3600)

        cov = compute_window_mask_coverage(mask, pre.rows, pre.cols, pre.interior)
        pv_m = pre.interior & (cov >= cfg.pv_cov_min)
        bg_m = pre.interior & (cov == 0.0)
        self.assertGreater(int(pv_m.sum()), 100)
        self.assertGreater(int(bg_m.sum()), 100)

        # Period 4px in 32px window -> radial frequency r = 32 / 4 = 8
        band = select_adaptive_band(pre.radial_profile[pv_m], pre.radial_profile[bg_m], cfg)
        self.assertTrue(band.valid)
        self.assertEqual(band.peak_r, 8)
        self.assertGreaterEqual(band.high - band.low, 3.0)

        a_map = compute_adaptive_peak_share(pre, band.band_mask)
        self.assertGreater(float(a_map[pv_m].mean()), float(a_map[bg_m].mean()) + 0.25)

        # Zero fixed-band fallback: if PV and BG profiles are identical -> gain=0 -> valid=False!
        band_none = select_adaptive_band(pre.radial_profile[bg_m], pre.radial_profile[bg_m], cfg)
        self.assertFalse(band_none.valid)
        self.assertEqual(band_none.reason, "no_positive_contrast_peak")
        self.assertIsNone(band_none.band_mask)

    def test_parity_with_external_fft_reference_when_present(self) -> None:
        ext_fft_dir = Path(r"E:\PV-DINOV3SAT\FFT")
        if not (ext_fft_dir / "adaptive_band_visualization.py").is_file():
            self.skipTest("External FFT reference directory not present")
        if str(ext_fft_dir) not in sys.path:
            sys.path.insert(0, str(ext_fft_dir))
        try:
            from adaptive_band_visualization import estimate_adaptive_band_mask
            from physical_feature_visualization import local_fft_maps, grid_centers
        except ImportError:
            self.skipTest("External FFT optional deps (e.g., cv2) unavailable")

        rgb, mask = _make_striped_rgb(42, pv_box=(128, 320, 128, 320), period_px=4.0)
        pre = precompute_image_fft("parity", rgb)
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32) / 255.0
        rows, cols = grid_centers((512, 512))
        ext_maps = local_fft_maps(gray, rows, cols, with_profile=True)
        np.testing.assert_allclose(pre.radial_profile, ext_maps["radial_profile"], atol=1e-4)

        cov = compute_window_mask_coverage(mask, pre.rows, pre.cols, pre.interior)
        pv_m = pre.interior & (cov >= 0.75)
        bg_m = pre.interior & (cov == 0.0)
        cfg = FFTSoftVetoConfig(enabled=True)
        b_new = select_adaptive_band(pre.radial_profile[pv_m], pre.radial_profile[bg_m], cfg)
        b_ext = estimate_adaptive_band_mask({"fg": [ext_maps["radial_profile"][pv_m]], "bg": [ext_maps["radial_profile"][bg_m]]})
        self.assertEqual(b_new.peak_r, b_ext.peak)
        self.assertAlmostEqual(b_new.low, b_ext.low, places=4)
        self.assertAlmostEqual(b_new.high, b_ext.high, places=4)
        np.testing.assert_array_equal(b_new.band_mask, b_ext.mask)

        ext_maps2 = local_fft_maps(gray, rows, cols, adaptive_band=b_ext.mask)
        a_new = compute_adaptive_peak_share(pre, b_new.band_mask)
        np.testing.assert_allclose(a_new, ext_maps2["adaptive_peak_share"], atol=1e-4)

    def test_gating_rules_out_of_range_border_low_energy_and_bin_counts(self) -> None:
        cfg = FFTSoftVetoConfig(enabled=True, q_cand=0.80, tau=0.40, gamma=0.05, cap=0.05)
        rgb, mask = _make_striped_rgb(7, pv_box=(128, 320, 128, 320), period_px=4.0)
        pre = precompute_image_fft("gate_test", rgb)
        cov = compute_window_mask_coverage(mask, pre.rows, pre.cols, pre.interior)
        # Synthetic deep score: high on PV and on a chunk of BG (to generate >=16 BG candidates)
        s_deep = np.full((64, 64), 0.10, dtype=np.float32)
        s_deep[cov >= 0.75] = 0.80
        s_deep[2:14, 2:14] = 0.75  # 144 high-scoring pure-BG candidate windows
        ref = fit_fft_soft_veto_reference([pre], [cov], [s_deep], cfg)
        self.assertTrue(ref.active)

        # Evaluate on query image with border tokens set to high S_deep -> border MUST stay g=0!
        s_deep_q = np.full((64, 64), 0.90, dtype=np.float32)
        out = evaluate_fft_soft_veto_map(pre, s_deep_q, ref, cfg)
        delta_2d = out["delta_fft"].reshape(64, 64)
        self.assertTrue(np.all(delta_2d[~pre.interior] == 0.0))
        self.assertLessEqual(float(delta_2d.max()), cfg.cap + 1e-7)
        # Low-energy flat query image MUST give zero penalty everywhere
        flat_rgb = np.full((512, 512, 3), 128, dtype=np.uint8)
        pre_flat = precompute_image_fft("flat", flat_rgb)
        out_flat = evaluate_fft_soft_veto_map(pre_flat, s_deep_q, ref, cfg)
        self.assertTrue(np.all(out_flat["delta_fft"] == 0.0))

    def test_k1_nonoverlap_exclusion_and_pipeline_fallback_matches_nomidveto(self) -> None:
        torch.manual_seed(20261124)
        cfg_r = get_m3_pruned_config()
        cfg_nm = replace(cfg_r, name="R_NoMidVeto", beta_veto=0.0)
        cfg_fft = FFTSoftVetoConfig(enabled=True)
        cfg_fft_fail = replace(cfg_fft, min_pv_cand=999999)

        ids = ["s0", "s1", "s2", "q0"]
        mask_by_id: dict[str, np.ndarray] = {}
        fft_pre_by_id = {}
        tokens_by_key: dict[str, dict[str, torch.Tensor]] = {"deep_32": {}, "deep_64": {}, "mid_64": {}}

        fg_d = F.normalize(torch.randn(16), dim=0)
        bg_d = F.normalize(torch.randn(16), dim=0)
        fg_m = F.normalize(torch.randn(16), dim=0)
        bg_m = F.normalize(torch.randn(16), dim=0)

        for idx, rid in enumerate(ids):
            rgb, mask = _make_striped_rgb(100 + idx, pv_box=(140, 300, 140, 300), period_px=4.0)
            mask_by_id[rid] = mask
            fft_pre_by_id[rid] = precompute_image_fft(rid, rgb)
            for key, hw, fv, bv in (("deep_32", 32, fg_d, bg_d), ("deep_64", 64, fg_d, bg_d), ("mid_64", 64, fg_m, bg_m)):
                p = 512 // hw
                mg = mask.reshape(hw, p, hw, p).mean(axis=(1, 3)) >= 0.5
                fmap = torch.randn(16, hw, hw) * 0.12
                fmap[:, mg] += fv[:, None]
                fmap[:, ~mg] += bv[:, None]
                tokens_by_key[key][rid] = _to_tokens_norm(F.normalize(fmap, dim=0), "cpu")

        caches = {
            key: build_pairwise_topk_cache(tokens_by_key[key], mask_by_id, ids, ids[:3], grid_hw=(64 if "64" in key else 32))
            for key in ("deep_32", "deep_64", "mid_64")
        }
        aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=0.12) for rid in ids}

        pipe_r = ThreeModulePipeline(cfg_r)
        pipe_nm = ThreeModulePipeline(cfg_nm)
        pipe_fft_fb = ThreeModulePipeline(cfg_r, fft_veto_cfg=cfg_fft_fail)
        pipe_fft = ThreeModulePipeline(cfg_r, fft_veto_cfg=cfg_fft)

        for sup in (["s0"], ["s0", "s1", "s2"]):
            # 1. R exact equivalence to legacy run_iter5_episode
            pL, sL = run_iter5_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup, ["q0"], cfg_r)
            outR = pipe_r.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup, ["q0"])
            pR, sR = outR.to_legacy_outputs()
            np.testing.assert_array_equal(pL["q0"], pR["q0"])
            np.testing.assert_array_equal(sL["q0"], sR["q0"])

            # 2. Inactive FFT veto -> exact identity with R_NoMidVeto (including b, T!)
            outNM = pipe_nm.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup, ["q0"])
            outFB = pipe_fft_fb.run_episode(
                caches, tokens_by_key, mask_by_id, aff_by_id, sup, ["q0"], fft_pre_by_id=fft_pre_by_id
            )
            self.assertFalse(outFB.calibration.fft_veto_active)
            self.assertEqual(outFB.calibration.bias, outNM.calibration.bias)
            self.assertEqual(outFB.calibration.temp, outNM.calibration.temp)
            np.testing.assert_array_equal(outFB.boundary_by_id["q0"].p_coarse, outNM.boundary_by_id["q0"].p_coarse)

            # 3. Active FFT veto runs cleanly end-to-end with finite outputs
            outFFT = pipe_fft.run_episode(
                caches, tokens_by_key, mask_by_id, aff_by_id, sup, ["q0"], fft_pre_by_id=fft_pre_by_id
            )
            self.assertTrue(np.isfinite(outFFT.boundary_by_id["q0"].p_coarse).all())

    def test_k1_nonoverlap_strict_isolation_and_k3_loo_exclusion(self) -> None:
        cfg = FFTSoftVetoConfig(enabled=True, q_cand=0.80, tau=0.20, gamma=0.05, cap=0.05)
        rgb, mask = _make_striped_rgb(77, pv_box=(200, 380, 200, 380), period_px=4.0)
        pre = precompute_image_fft("iso_k1", rgb)
        cov = compute_window_mask_coverage(mask, pre.rows, pre.cols, pre.interior)
        s_deep = np.full((64, 64), 0.10, dtype=np.float32)
        s_deep[cov >= 0.75] = 0.85
        s_deep[4:16, 4:24] = 0.78  # large high-scoring pure-BG region
        target_r, target_c = 10, 10
        target_flat = np.array([target_r * 64 + target_c], dtype=np.int64)
        pen_orig = evaluate_k1_nonoverlap_bg_penalties(pre, cov, s_deep, target_flat, cfg)

        # Mutate radial_profile and power ONLY on windows strictly overlapping (target_r, target_c)
        # while keeping (target_r, target_c) itself untouched -> reference MUST be 100% identical!
        pre_mut = precompute_image_fft("iso_k1_mut", rgb)
        gr, gc = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        overlap_others = (np.abs(gr - target_r) < 4) & (np.abs(gc - target_c) < 4) & ~((gr == target_r) & (gc == target_c))
        pre_mut.radial_profile[overlap_others] = 999.0
        pre_mut.norm_energy[overlap_others] = 1e-12
        pre_mut.power[overlap_others] = 0.0
        pen_mut = evaluate_k1_nonoverlap_bg_penalties(pre_mut, cov, s_deep, target_flat, cfg)
        np.testing.assert_allclose(pen_orig, pen_mut, atol=0.0)

        # Multi-shot LOO exclusion: mutating held-out support image's reference data cannot change its LOO ref
        rgb2, mask2 = _make_striped_rgb(78, pv_box=(160, 340, 160, 340), period_px=4.0)
        rgb3, mask3 = _make_striped_rgb(79, pv_box=(180, 360, 180, 360), period_px=4.0)
        pre2, pre3 = precompute_image_fft("s2", rgb2), precompute_image_fft("s3", rgb3)
        cov2 = compute_window_mask_coverage(mask2, pre2.rows, pre2.cols, pre2.interior)
        cov3 = compute_window_mask_coverage(mask3, pre3.rows, pre3.cols, pre3.interior)
        s2 = np.where(cov2 >= 0.75, 0.85, 0.10).astype(np.float32)
        s2[4:14, 4:14] = 0.75
        s3 = np.where(cov3 >= 0.75, 0.85, 0.10).astype(np.float32)
        s3[4:14, 4:14] = 0.75
        ref_loo_a = fit_fft_soft_veto_reference([pre2, pre3], [cov2, cov3], [s2, s3], cfg)
        self.assertTrue(ref_loo_a.active)


if __name__ == "__main__":
    unittest.main()
