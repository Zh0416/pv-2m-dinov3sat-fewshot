from __future__ import annotations

import unittest

import numpy as np
import torch

from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.iter1_methods import _to_tokens
from pvfs.iter3_methods import compute_dino_affinity_weights
from pvfs.iter4_methods import ITER4_VARIANTS, Iter4Config, run_iter4_episode


class Iter4Tests(unittest.TestCase):
    def test_whitening_and_ridge_suppress_shared_clutter_confounder(self) -> None:
        torch.manual_seed(404)
        c = 64
        # Shared high-variance clutter axis present in roofs/greenhouses AND slightly in PV,
        # plus true PV material axis (especially clear in mid_64) vs greenhouse distractor axis.
        shared_grid_axis = torch.randn(c)
        shared_grid_axis = shared_grid_axis / shared_grid_axis.norm()
        pv_material = torch.randn(c)
        pv_material = pv_material - (pv_material @ shared_grid_axis) * shared_grid_axis
        pv_material = pv_material / pv_material.norm()
        bg_base = torch.randn(c)
        bg_base = bg_base / bg_base.norm()

        # Construct tokens where support background has strong variations along shared_grid_axis
        # (so Whitening + Ridge downweight shared_grid_axis!) and PV is distinct along pv_material.
        def make_img(fg_mask_64: np.ndarray, extra_conf_64: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            noise = torch.randn(c, 64, 64) * 0.08
            # Spatial variation along shared_grid_axis inside background so cov captures its variance!
            grid_amp = (torch.rand(1, 64, 64) * 1.4 - 0.3)
            f64 = bg_base[:, None, None] + grid_amp * shared_grid_axis[:, None, None] + noise
            m_t = torch.from_numpy(fg_mask_64)
            f64[:, m_t] = 0.65 * shared_grid_axis[:, None] + 0.85 * pv_material[:, None] + noise[:, m_t]
            if extra_conf_64 is not None:
                cm = torch.from_numpy(extra_conf_64)
                # Greenhouse confounder: high shared_grid_axis, ZERO pv_material -> fools raw cosine!
                f64[:, cm] = 1.40 * shared_grid_axis[:, None] + noise[:, cm]

            mid64 = f64.clone()
            mid64[:, m_t] = 0.30 * shared_grid_axis[:, None] + 1.10 * pv_material[:, None] + noise[:, m_t]
            if extra_conf_64 is not None:
                mid64[:, torch.from_numpy(extra_conf_64)] = 0.80 * shared_grid_axis[:, None] + bg_base[:, None] * 0.5 + noise[:, torch.from_numpy(extra_conf_64)]

            f32 = torch.nn.functional.avg_pool2d(f64[None], 2, stride=2)[0]
            return _to_tokens(f64, "cpu"), _to_tokens(mid64, "cpu"), _to_tokens(f32, "cpu")

        masks, toks, affs = {}, {"deep_64": {}, "mid_64": {}, "deep_32": {}}, {}
        for idx, sid in enumerate(["s0", "s1", "s2"]):
            m512 = np.zeros((512, 512), dtype=np.uint8)
            r0 = 64 + idx * 16
            m512[r0 : r0 + 128, 64:192] = 1
            masks[sid] = m512
            m64 = m512.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
            d64, md64, d32 = make_img(m64)
            toks["deep_64"][sid] = d64
            toks["mid_64"][sid] = md64
            toks["deep_32"][sid] = d32
            affs[sid] = compute_dino_affinity_weights(d64, tau=0.12)

        q_m = np.zeros((512, 512), dtype=np.uint8)
        q_m[96:224, 96:224] = 1
        masks["q_pos"] = q_m
        m64_q = q_m.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
        conf64 = np.zeros((64, 64), dtype=bool)
        conf64[42:54, 42:54] = True
        d64_q, md64_q, d32_q = make_img(m64_q, conf64)
        toks["deep_64"]["q_pos"], toks["mid_64"]["q_pos"], toks["deep_32"]["q_pos"] = d64_q, md64_q, d32_q
        affs["q_pos"] = compute_dino_affinity_weights(d64_q, tau=0.12)

        masks["q_neg"] = np.zeros((512, 512), dtype=np.uint8)
        d64_n, md64_n, d32_n = make_img(np.zeros((64, 64), dtype=bool), conf64)
        toks["deep_64"]["q_neg"], toks["mid_64"]["q_neg"], toks["deep_32"]["q_neg"] = d64_n, md64_n, d32_n
        affs["q_neg"] = compute_dino_affinity_weights(d64_n, tau=0.12)

        all_ids = ["s0", "s1", "s2", "q_pos", "q_neg"]
        pos_ids = ["s0", "s1", "s2"]
        caches = {
            key: build_pairwise_topk_cache(toks[key], masks, all_ids, pos_ids, grid_hw=(64 if "64" in key else 32))
            for key in ("deep_64", "mid_64", "deep_32")
        }

        by_name: dict[str, Iter4Config] = {cfg.name: cfg for cfg in ITER4_VARIANTS}
        for cfg in ITER4_VARIANTS:
            for sids in (["s0"], ["s0", "s1", "s2"]):
                p512, s32 = run_iter4_episode(caches, toks, masks, affs, sids, ["q_pos", "q_neg"], cfg)
                self.assertEqual(p512["q_pos"].shape, (512, 512))
                self.assertEqual(s32["q_pos"].shape, (32, 32))
                self.assertTrue(np.isfinite(p512["q_pos"]).all())
                self.assertGreater(float(p512["q_pos"][110:210, 110:210].mean()), 0.55)

        p_base, _ = run_iter4_episode(caches, toks, masks, affs, ["s0", "s1", "s2"], ["q_pos", "q_neg"], by_name["M3_Pruned_Ref"])
        p_full, _ = run_iter4_episode(caches, toks, masks, affs, ["s0", "s1", "s2"], ["q_pos", "q_neg"], by_name["M4_Full"])
        # Confounder probability in q_neg[336:432, 336:432] must be lower in M4_Full than M3_Pruned_Ref
        self.assertLess(
            float(p_full["q_neg"][336:432, 336:432].mean()),
            float(p_base["q_neg"][336:432, 336:432].mean()),
        )
        self.assertEqual(float((p_full["q_neg"] >= 0.5).mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
