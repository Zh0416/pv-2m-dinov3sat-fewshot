from __future__ import annotations

import unittest

import numpy as np
import torch

from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.freq_features import apply_ofg_neg_gate_and_rescue, compute_support_freq_alignment, extract_freq_pyramid
from pvfs.iter1_methods import _to_tokens
from pvfs.iter3_methods import compute_dino_affinity_weights
from pvfs.v2_decoder import SCFBDecoder, build_decoder_input_512, scfb_episodic_loss
from pvfs.v2_pipeline import V2_VARIANTS, run_v2_episode


class V2ModulesTests(unittest.TestCase):
    def test_freq_pyramid_separates_periodic_pv_from_smooth_and_isotropic(self) -> None:
        yy, xx = np.meshgrid(np.arange(512), np.arange(512), indexing="ij")
        rgb = np.full((3, 512, 512), 140, dtype=np.uint8)
        # Top-left: smooth flat industrial roof
        rgb[:, 40:160, 40:160] = 210
        # Bottom-right: periodic oriented PV stripes (~6px period at 0 deg)
        stripe = 0.5 + 0.5 * np.sin(2.0 * np.pi * xx[280:420, 280:420] / 6.0)
        rgb[0, 280:420, 280:420] = (50 + 70 * stripe).astype(np.uint8)
        rgb[1, 280:420, 280:420] = (70 + 80 * stripe).astype(np.uint8)
        rgb[2, 280:420, 280:420] = (110 + 90 * stripe).astype(np.uint8)
        # Top-right: isotropic white speckle noise
        rng = np.random.default_rng(7)
        rgb[:, 40:160, 320:440] = rng.integers(80, 200, size=(3, 120, 120), dtype=np.uint8)

        f512, f64 = extract_freq_pyramid(rgb, device="cpu")
        self.assertEqual(f512.shape, (8, 512, 512))
        self.assertEqual(f64.shape, (8, 64, 64))
        self.assertTrue(torch.isfinite(f512).all())

        # Gabor anisotropy (ch 5) and directional FFT C_spec (ch 3) must peak on periodic PV vs smooth & noise
        pv_cspec = float(f64[3, 38:50, 38:50].mean())
        smooth_cspec = float(f64[3, 8:18, 8:18].mean())
        noise_cspec = float(f64[3, 8:18, 42:52].mean())
        self.assertGreater(pv_cspec, smooth_cspec + 0.15)
        self.assertGreater(pv_cspec, noise_cspec + 0.10)

        # Negative gate should suppress marginal false positive when z_freq is negative
        z_sem = torch.full((64, 64), -1.5)
        z_sem[10:15, 10:15] = 0.45  # marginal false-positive blip
        z_freq_bad = torch.full((64, 64), -1.2)
        gated_bad = apply_ofg_neg_gate_and_rescue(z_sem, z_freq_bad)
        self.assertLess(float(gated_bad.max()), 0.0)

        # Strong small PV with matching periodicity stays positive
        z_sem_small = torch.full((64, 64), -2.0)
        z_sem_small[10:14, 10:14] = 1.8
        z_freq_good = torch.full((64, 64), -0.5)
        z_freq_good[10:14, 10:14] = 1.4
        gated_good = apply_ofg_neg_gate_and_rescue(z_sem_small, z_freq_good)
        self.assertGreater(float(gated_good[10:14, 10:14].mean()), 1.5)

    def test_scfb_decoder_param_budget_loss_and_v2_variants(self) -> None:
        torch.manual_seed(2026)
        dec = SCFBDecoder(in_ch=16)
        n_params = sum(p.numel() for p in dec.parameters())
        self.assertLess(n_params, 60000)

        x = torch.randn(2, 16, 512, 512)
        y = torch.zeros(2, 1, 512, 512)
        y[0, 0, 128:256, 128:256] = 1.0  # idx 0 pos, idx 1 empty neg
        opt = torch.optim.AdamW(dec.parameters(), lr=1e-2)
        logits0 = dec(x)
        loss0 = scfb_episodic_loss(logits0, y)
        loss0.backward()
        opt.step()
        opt.zero_grad()
        loss1 = scfb_episodic_loss(dec(x), y)
        self.assertTrue(torch.isfinite(loss1))
        self.assertLess(float(loss1.item()), float(loss0.item()) + 0.05)

        # End-to-end run across all 5 V2_VARIANTS
        c = 64
        pv_vec = torch.randn(c)
        bg_vec = torch.randn(c)
        masks, toks, affs, rgbs, f512s, f64s = {}, {}, {}, {}, {}, {}
        for idx, sid in enumerate(["s0", "s1", "q_pos", "q_neg"]):
            m512 = np.zeros((512, 512), dtype=np.uint8)
            rgb = np.full((3, 512, 512), 130, dtype=np.uint8)
            if sid != "q_neg":
                m512[96:224, 96:224] = 1
                yy, xx = np.meshgrid(np.arange(96, 224), np.arange(96, 224), indexing="ij")
                st = (0.5 + 0.5 * np.sin(2.0 * np.pi * xx / 6.0)) * 80.0
                rgb[2, 96:224, 96:224] = (90 + st).astype(np.uint8)
            masks[sid] = m512
            m64 = m512.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
            f64_t = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.08
            if m64.any():
                f64_t[:, torch.from_numpy(m64)] = pv_vec[:, None] + torch.randn(c, int(m64.sum())) * 0.06
            t64 = _to_tokens(f64_t, "cpu")
            toks.setdefault("deep_64", {})[sid] = t64
            toks.setdefault("mid_64", {})[sid] = t64
            toks.setdefault("deep_32", {})[sid] = _to_tokens(torch.nn.functional.avg_pool2d(f64_t[None], 2, stride=2)[0], "cpu")
            affs[sid] = compute_dino_affinity_weights(t64, tau=0.12)
            rgbs[sid] = (torch.from_numpy(rgb).float() / 255.0).half()
            p512, p64 = extract_freq_pyramid(rgb, device="cpu")
            f512s[sid] = p512
            f64s[sid] = p64

        caches = {
            k: build_pairwise_topk_cache(toks[k], masks, ["s0", "s1", "q_pos", "q_neg"], ["s0", "s1"], grid_hw=(64 if "64" in k else 32))
            for k in ("deep_32", "mid_64", "deep_64")
        }
        dec.eval()
        for var in V2_VARIANTS:
            p512, s32 = run_v2_episode(
                var,
                caches,
                toks,
                masks,
                affs,
                rgbs,
                f512s,
                f64s,
                ["s0", "s1"],
                ["q_pos", "q_neg"],
                decoder=(dec if var.kind == "decoder" else None),
                device=torch.device("cpu"),
            )
            self.assertEqual(p512["q_pos"].shape, (512, 512))
            self.assertEqual(s32["q_pos"].shape, (32, 32))
            self.assertTrue(np.isfinite(p512["q_pos"]).all())
            self.assertGreater(float(p512["q_pos"][110:210, 110:210].mean()), 0.55)
            self.assertEqual(float((p512["q_neg"] >= 0.5).mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
