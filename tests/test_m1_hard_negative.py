from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.iter1_methods import _to_tokens
from pvfs.iter3_methods import compute_dino_affinity_weights
from pvfs.modules import ThreeModulePipeline, get_m3_pruned_config
from pvfs.modules.hard_negative import HardNegConfig, build_support_bg_subbank


class M1HardNegativeTests(unittest.TestCase):
    def _build_confounder_scene(self):
        torch.manual_seed(777)
        c = 64
        pv_vec = F.normalize(torch.randn(c), dim=0)
        bg_vec = F.normalize(torch.randn(c), dim=0)
        # Greenhouse-like confounder: high cosine similarity (~0.70) to PV!
        orth = F.normalize(torch.randn(c), dim=0)
        conf_vec = F.normalize(0.72 * pv_vec + 0.69 * orth, dim=0)

        masks, toks, affs = {}, {}, {}
        for idx, sid in enumerate(["s0", "s1", "s2"]):
            m512 = np.zeros((512, 512), dtype=np.uint8)
            r0 = 32 + idx * 16
            m512[r0 : r0 + 128, 32:160] = 1
            masks[sid] = m512
            m64 = m512.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
            f64 = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.08
            f64[:, torch.from_numpy(m64)] = pv_vec[:, None] + torch.randn(c, int(m64.sum())) * 0.05
            # Put a realistic rare greenhouse confounder patch (~0.6% of background) in each support image
            f64[:, 48:53, 48:53] = conf_vec[:, None, None] + torch.randn(c, 5, 5) * 0.04
            t64 = _to_tokens(f64, "cpu")
            toks.setdefault("deep_64", {})[sid] = t64
            toks.setdefault("mid_64", {})[sid] = t64
            toks.setdefault("deep_32", {})[sid] = _to_tokens(F.avg_pool2d(f64[None], 2, stride=2)[0], "cpu")
            affs[sid] = compute_dino_affinity_weights(t64, tau=0.12)

        # Positive query containing both true PV and greenhouse confounder
        q_pos = np.zeros((512, 512), dtype=np.uint8)
        q_pos[64:192, 64:192] = 1
        masks["q_pos"] = q_pos
        m64_q = q_pos.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
        f64_q = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.08
        f64_q[:, torch.from_numpy(m64_q)] = pv_vec[:, None] + torch.randn(c, int(m64_q.sum())) * 0.05
        f64_q[:, 36:52, 36:52] = conf_vec[:, None, None] + torch.randn(c, 16, 16) * 0.04
        t64_q = _to_tokens(f64_q, "cpu")
        toks["deep_64"]["q_pos"] = t64_q
        toks["mid_64"]["q_pos"] = t64_q
        toks["deep_32"]["q_pos"] = _to_tokens(F.avg_pool2d(f64_q[None], 2, stride=2)[0], "cpu")
        affs["q_pos"] = compute_dino_affinity_weights(t64_q, tau=0.12)

        # Pure negative query with large greenhouse confounder
        masks["q_neg"] = np.zeros((512, 512), dtype=np.uint8)
        f64_n = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.08
        f64_n[:, 24:48, 24:48] = conf_vec[:, None, None] + torch.randn(c, 24, 24) * 0.04
        t64_n = _to_tokens(f64_n, "cpu")
        toks["deep_64"]["q_neg"] = t64_n
        toks["mid_64"]["q_neg"] = t64_n
        toks["deep_32"]["q_neg"] = _to_tokens(F.avg_pool2d(f64_n[None], 2, stride=2)[0], "cpu")
        affs["q_neg"] = compute_dino_affinity_weights(t64_n, tau=0.12)

        all_ids = ["s0", "s1", "s2", "q_pos", "q_neg"]
        pos_ids = ["s0", "s1", "s2"]
        caches = {
            key: build_pairwise_topk_cache(toks[key], masks, all_ids, pos_ids, grid_hw=(64 if "64" in key else 32))
            for key in ("deep_64", "mid_64", "deep_32")
        }
        return caches, toks, masks, affs

    def test_loo_exclusion_and_permutation_invariance(self) -> None:
        caches, toks, masks, affs = self._build_confounder_scene()
        cfg = get_m3_pruned_config()
        hcfg = HardNegConfig(enabled=True, mode="hard_neg_H")

        # Bank built on [s0, s1] must never contain s2
        bank_01 = build_support_bg_subbank(toks["deep_64"], masks, ["s0", "s1"], hcfg)
        self.assertTrue(bank_01["valid"])
        self.assertNotIn("s2", bank_01["token_indices_by_sid"])

        # Permutation invariance: [s0, s1, s2] vs [s2, s0, s1]
        pipe_a = ThreeModulePipeline(cfg, hard_neg_cfg=hcfg)
        pipe_b = ThreeModulePipeline(cfg, hard_neg_cfg=hcfg)
        out_a = pipe_a.run_episode(caches, toks, masks, affs, ["s0", "s1", "s2"], ["q_pos", "q_neg"])
        out_b = pipe_b.run_episode(caches, toks, masks, affs, ["s2", "s0", "s1"], ["q_pos", "q_neg"])

        self.assertAlmostEqual(out_a.calibration.bias, out_b.calibration.bias, places=6)
        self.assertAlmostEqual(out_a.calibration.hard_neg_t_h, out_b.calibration.hard_neg_t_h, places=6)
        np.testing.assert_allclose(
            out_a.boundary_by_id["q_neg"].p_coarse,
            out_b.boundary_by_id["q_neg"].p_coarse,
            atol=1e-6,
        )

    def test_abstention_when_no_reliable_H_returns_exact_R(self) -> None:
        caches, toks, masks, affs = self._build_confounder_scene()
        cfg = get_m3_pruned_config()
        # Impossibly high confusion threshold -> empty H -> must fallback to exact R
        hcfg_impossible = HardNegConfig(enabled=True, mode="hard_neg_H", per_support_min_confusion_sim=0.999)

        pipe_r = ThreeModulePipeline(cfg)
        pipe_h = ThreeModulePipeline(cfg, hard_neg_cfg=hcfg_impossible)
        out_r = pipe_r.run_episode(caches, toks, masks, affs, ["s0", "s1", "s2"], ["q_pos", "q_neg"])
        out_h = pipe_h.run_episode(caches, toks, masks, affs, ["s0", "s1", "s2"], ["q_pos", "q_neg"])

        self.assertFalse(out_h.calibration.hard_neg_active)
        self.assertEqual(out_h.calibration.bias, out_r.calibration.bias)
        self.assertEqual(out_h.calibration.temp, out_r.calibration.temp)
        for qid in ("q_pos", "q_neg"):
            self.assertTrue(torch.equal(out_h.semantic_by_id[qid].s_sem, out_r.semantic_by_id[qid].s_sem))
            np.testing.assert_array_equal(out_h.boundary_by_id[qid].p_coarse, out_r.boundary_by_id[qid].p_coarse)

    def test_hard_neg_H_suppresses_confounder_stronger_than_equal_budget_random_bg(self) -> None:
        caches, toks, masks, affs = self._build_confounder_scene()
        cfg = get_m3_pruned_config()
        hcfg_rand = HardNegConfig(enabled=True, mode="random_bg")
        hcfg_hard = HardNegConfig(enabled=True, mode="hard_neg_H")

        out_r = ThreeModulePipeline(cfg).run_episode(caches, toks, masks, affs, ["s0", "s1", "s2"], ["q_pos", "q_neg"])
        out_rand = ThreeModulePipeline(cfg, hard_neg_cfg=hcfg_rand).run_episode(caches, toks, masks, affs, ["s0", "s1", "s2"], ["q_pos", "q_neg"])
        out_hard = ThreeModulePipeline(cfg, hard_neg_cfg=hcfg_hard).run_episode(caches, toks, masks, affs, ["s0", "s1", "s2"], ["q_pos", "q_neg"])

        # Equal budget check
        self.assertEqual(
            out_rand.calibration.diagnostics["hard_neg_bank_tokens"],
            out_hard.calibration.diagnostics["hard_neg_bank_tokens"],
        )
        # Confounder region on q_neg (24:48, 24:48 in 64x64 grid -> 192:384 in 512x512)
        s_conf_r = float(out_r.semantic_by_id["q_neg"].s_sem.reshape(64, 64)[24:48, 24:48].mean())
        s_conf_rand = float(out_rand.semantic_by_id["q_neg"].s_sem.reshape(64, 64)[24:48, 24:48].mean())
        s_conf_hard = float(out_hard.semantic_by_id["q_neg"].s_sem.reshape(64, 64)[24:48, 24:48].mean())
        s_pv_r = float(out_r.semantic_by_id["q_pos"].s_sem.reshape(64, 64)[8:24, 8:24].mean())
        s_pv_rand = float(out_rand.semantic_by_id["q_pos"].s_sem.reshape(64, 64)[8:24, 8:24].mean())
        s_pv_hard = float(out_hard.semantic_by_id["q_pos"].s_sem.reshape(64, 64)[8:24, 8:24].mean())
        # Hard-negative bank H increases PV-vs-confounder separation strictly beyond both R and equal-budget random_bg
        self.assertGreater(s_pv_hard - s_conf_hard, s_pv_rand - s_conf_rand)
        self.assertGreater(s_pv_hard - s_conf_hard, s_pv_r - s_conf_r)
        p_neg_r = float(out_r.boundary_by_id["q_neg"].p_coarse[192:384, 192:384].mean())
        p_neg_rand = float(out_rand.boundary_by_id["q_neg"].p_coarse[192:384, 192:384].mean())
        p_neg_hard = float(out_hard.boundary_by_id["q_neg"].p_coarse[192:384, 192:384].mean())
        self.assertLess(p_neg_hard, p_neg_rand)
        self.assertLess(p_neg_hard, p_neg_r)
        # True PV region on q_pos remains high probability
        p_pv_hard = float(out_hard.boundary_by_id["q_pos"].p_coarse[80:176, 80:176].mean())
        self.assertGreater(p_pv_hard, 0.70)


if __name__ == "__main__":
    unittest.main()
