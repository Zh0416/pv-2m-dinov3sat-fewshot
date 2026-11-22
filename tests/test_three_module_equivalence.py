from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from pvfs.fast_bank import build_pairwise_topk_cache, score_from_pairwise, select_fg_bg_indices
from pvfs.iter1_methods import _to_tokens
from pvfs.iter3_methods import _ex_neigh_bg_contrast, compute_dino_affinity_weights, propagate_affinity
from pvfs.iter5_methods import ITER5_VARIANTS, Iter5Config, _apply_mid_verified_rescue, run_iter5_episode
from pvfs.modules import ThreeModulePipeline, get_b0_32_config, get_m3_pruned_config, run_three_module_episode


@torch.inference_mode()
def _trace_legacy_iter5_intermediates(
    caches: dict[str, dict],
    tokens_by_key: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    aff_by_id: dict[str, torch.Tensor] | None,
    support_ids: list[str],
    query_ids: list[str],
    cfg: Iter5Config,
) -> dict:
    """Exact mirror of pvfs.iter5_methods.run_iter5_episode exposing internal intermediates for L vs R audit."""
    k = len(support_ids)
    use_core = cfg.k1_use_core if k == 1 else cfg.kge2_use_core
    core_purity = 0.85 if use_core else 0.50
    use_core_calib = use_core and not cfg.calib_v2

    mid_veto_ref = None
    if cfg.use_64 and cfg.use_mid_veto:
        if k >= 2:
            mid_vals = []
            for idx, sid in enumerate(support_ids):
                sm, _, _ = score_from_pairwise(
                    caches["mid_64"], sid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=idx
                )
                bm = caches["mid_64"]["bg_masks"][caches["mid_64"]["id_to_pos"][sid]]
                if bm.any():
                    mid_vals.append(sm[torch.from_numpy(np.flatnonzero(bm)).to(sm.device)])
            if mid_vals:
                mid_veto_ref = float(torch.quantile(torch.cat(mid_vals), 0.85).item())
        else:
            c_bg_m = _ex_neigh_bg_contrast(tokens_by_key["mid_64"][support_ids[0]], mask_by_id[support_ids[0]], 64, core_purity=core_purity)
            if c_bg_m is not None:
                mid_veto_ref = float(torch.quantile(c_bg_m, 0.85).item())
        if mid_veto_ref is None:
            mid_veto_ref = 0.20

    def raw_score(qid: str, ex_idx: int | None):
        s32, _, _ = score_from_pairwise(
            caches["deep_32"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx
        )
        if not cfg.use_64:
            return s32, None
        s64, _, _ = score_from_pairwise(
            caches["deep_64"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx
        )
        s32_up = F.interpolate(s32.reshape(1, 1, 32, 32), size=(64, 64), mode="bilinear", align_corners=False).reshape(-1)
        comb = cfg.w_s64 * s64 + (1.0 - cfg.w_s64) * s32_up
        sm = None
        if cfg.use_mid_veto:
            sm, _, _ = score_from_pairwise(
                caches["mid_64"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx
            )
            comb = comb + cfg.lambda_mid * sm - cfg.beta_veto * F.softplus((mid_veto_ref - sm) * 10.0) / 10.0
        return comb, sm

    if k >= 2:
        ref_key = "deep_64" if cfg.use_64 else "deep_32"
        fg_items, bg_list = [], []
        for idx, sid in enumerate(support_ids):
            comb_s, _ = raw_score(sid, idx)
            p_i = caches[ref_key]["id_to_pos"][sid]
            fg_m = caches[ref_key]["fg_core_masks"][p_i] if use_core_calib else caches[ref_key]["fg_all_masks"][p_i]
            bg_m = caches[ref_key]["bg_masks"][p_i]
            fg_idx = torch.from_numpy(np.flatnonzero(fg_m)).to(comb_s.device)
            if cfg.calib_v2 or cfg.name == "M1_Full_Ref":
                fg_items.append(float(torch.median(comb_s[fg_idx]).item()))
            else:
                fg_items.append(comb_s[fg_idx])
            if bg_m.any():
                bg_list.append(comb_s[torch.from_numpy(np.flatnonzero(bg_m)).to(comb_s.device)])
        if bg_list:
            mu_fg = float(np.mean(fg_items)) if isinstance(fg_items[0], float) else float(torch.median(torch.cat(fg_items)).item())
            mu_bg = float(torch.quantile(torch.cat(bg_list), cfg.bg_q_calib).item())
            if mu_fg > mu_bg + 0.02:
                bias = cfg.w_fg_calib * mu_fg + (1.0 - cfg.w_fg_calib) * mu_bg
                temp = max((mu_fg - mu_bg) / (2.0 * math.log(4.0)), 0.04)
            else:
                bias, temp = mu_bg + 0.05, 0.06
        else:
            bias, temp = 0.30, 0.06
    else:
        sid0 = support_ids[0]
        c32_b = _ex_neigh_bg_contrast(tokens_by_key["deep_32"][sid0], mask_by_id[sid0], 32, core_purity=core_purity)
        if not cfg.use_64:
            bias = (float(torch.quantile(c32_b, cfg.bg_q_calib).item()) + cfg.k1_margin) if c32_b is not None else 0.30
        else:
            c64_b = _ex_neigh_bg_contrast(tokens_by_key["deep_64"][sid0], mask_by_id[sid0], 64, core_purity=core_purity)
            cm_b = _ex_neigh_bg_contrast(tokens_by_key["mid_64"][sid0], mask_by_id[sid0], 64, core_purity=core_purity) if cfg.use_mid_veto else None
            if c64_b is not None and c32_b is not None:
                _, bg64_m, _ = select_fg_bg_indices(mask_by_id[sid0], 64, core_purity)
                _, bg32_m, _ = select_fg_bg_indices(mask_by_id[sid0], 32, core_purity)
                bg32_flat = np.flatnonzero(bg32_m)
                c32_full = torch.zeros((32, 32), device=c64_b.device)
                c32_full[torch.from_numpy(bg32_flat // 32).to(c64_b.device), torch.from_numpy(bg32_flat % 32).to(c64_b.device)] = c32_b
                bg64_flat = np.flatnonzero(bg64_m)
                bg64_t = torch.from_numpy(bg64_flat).to(c64_b.device)
                c32_at_64 = c32_full[(bg64_t // 64) // 2, (bg64_t % 64) // 2]
                bg_s = cfg.w_s64 * c64_b + (1.0 - cfg.w_s64) * c32_at_64
                if cfg.use_mid_veto and cm_b is not None and mid_veto_ref is not None:
                    bg_s = bg_s + cfg.lambda_mid * cm_b - cfg.beta_veto * F.softplus((mid_veto_ref - cm_b) * 10.0) / 10.0
                bias = float(torch.quantile(bg_s, cfg.bg_q_calib).item()) + cfg.k1_margin
            else:
                bias = 0.30
        temp = 0.06

    out_hw = 64 if cfg.use_64 else 32
    per_q = {}
    for qid in query_ids:
        comb, sm = raw_score(qid, None)
        p_base = torch.sigmoid((comb - bias) / temp)
        if cfg.use_64 and cfg.use_self_rescue:
            comb_fin = _apply_mid_verified_rescue(comb, p_base, sm, mid_veto_ref, tokens_by_key["deep_64"][qid], cfg)
            p_grid = torch.sigmoid((comb_fin - bias) / temp)
            s32_out = F.avg_pool2d(comb_fin.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
        else:
            comb_fin = comb
            p_grid = p_base
            s32_out = comb.reshape(32, 32) if out_hw == 32 else F.avg_pool2d(comb.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]

        if cfg.use_64 and cfg.use_affinity:
            w_aff = aff_by_id[qid] if aff_by_id is not None else compute_dino_affinity_weights(tokens_by_key["deep_64"][qid], tau=cfg.aff_tau)
            p_after_aff = propagate_affinity(p_grid.reshape(64, 64), w_aff, steps=cfg.aff_steps, eta=cfg.aff_eta)
        else:
            p_after_aff = p_grid

        p512 = F.interpolate(p_after_aff.reshape(1, 1, out_hw, out_hw), size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
        per_q[qid] = {
            "s_sem": comb,
            "p_sem": p_base,
            "s_mid": sm,
            "s_res": comb_fin,
            "p_res": p_grid,
            "probs_512": p512.cpu().numpy().astype(np.float32),
            "scores_32": s32_out.cpu().numpy().astype(np.float32),
        }
    return {"bias": float(bias), "temp": float(temp), "mid_veto_ref": mid_veto_ref, "per_q": per_q}


class ThreeModuleEquivalenceTests(unittest.TestCase):
    def _build_synthetic_bundle(self):
        torch.manual_seed(20261122)
        c = 64
        pv_main = torch.randn(c)
        pv_wing = pv_main * 0.8 + torch.randn(c) * 0.25
        bg_vec = torch.randn(c)
        conf_vec = pv_main * 0.45 + bg_vec * 0.55

        masks, toks, affs = {}, {}, {}
        for idx, sid in enumerate(["s0", "s1", "s2", "s3", "s4"]):
            m512 = np.zeros((512, 512), dtype=np.uint8)
            r0 = 48 + idx * 24
            m512[r0 : r0 + 112, 64:192] = 1
            masks[sid] = m512
            m64 = m512.reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
            f64 = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.12
            f64[:, torch.from_numpy(m64)] = pv_main[:, None] + torch.randn(c, int(m64.sum())) * 0.08
            # Inject a confounder patch in support background
            f64[:, 44:54, 44:54] = conf_vec[:, None, None] + torch.randn(c, 10, 10) * 0.06
            f32 = F.avg_pool2d(f64[None], 2, stride=2)[0]
            t64 = _to_tokens(f64, "cpu")
            toks.setdefault("deep_64", {})[sid] = t64
            toks.setdefault("mid_64", {})[sid] = _to_tokens(f64 * 0.9 + torch.randn_like(f64) * 0.05, "cpu")
            toks.setdefault("deep_32", {})[sid] = _to_tokens(f32, "cpu")
            affs[sid] = compute_dino_affinity_weights(t64, tau=0.12)

        # Positive query with a main PV block + a weaker shifted PV wing (triggers self-rescue)
        q_pos = np.zeros((512, 512), dtype=np.uint8)
        q_pos[96:224, 96:224] = 1
        q_pos[256:320, 256:320] = 1
        masks["q_pos"] = q_pos
        m64_main = np.zeros((64, 64), dtype=bool)
        m64_main[12:28, 12:28] = True
        m64_wing = np.zeros((64, 64), dtype=bool)
        m64_wing[32:40, 32:40] = True
        f64_q = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.12
        f64_q[:, torch.from_numpy(m64_main)] = pv_main[:, None] + torch.randn(c, int(m64_main.sum())) * 0.08
        f64_q[:, torch.from_numpy(m64_wing)] = pv_wing[:, None] + torch.randn(c, int(m64_wing.sum())) * 0.08
        t64_q = _to_tokens(f64_q, "cpu")
        toks["deep_64"]["q_pos"] = t64_q
        toks["mid_64"]["q_pos"] = _to_tokens(f64_q * 0.9 + torch.randn_like(f64_q) * 0.05, "cpu")
        toks["deep_32"]["q_pos"] = _to_tokens(F.avg_pool2d(f64_q[None], 2, stride=2)[0], "cpu")
        affs["q_pos"] = compute_dino_affinity_weights(t64_q, tau=0.12)

        # Pure negative query with a confounder patch
        masks["q_neg"] = np.zeros((512, 512), dtype=np.uint8)
        f64_n = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.12
        f64_n[:, 20:34, 20:34] = conf_vec[:, None, None] + torch.randn(c, 14, 14) * 0.06
        t64_n = _to_tokens(f64_n, "cpu")
        toks["deep_64"]["q_neg"] = t64_n
        toks["mid_64"]["q_neg"] = _to_tokens(f64_n * 0.9 + torch.randn_like(f64_n) * 0.05, "cpu")
        toks["deep_32"]["q_neg"] = _to_tokens(F.avg_pool2d(f64_n[None], 2, stride=2)[0], "cpu")
        affs["q_neg"] = compute_dino_affinity_weights(t64_n, tau=0.12)

        all_ids = ["s0", "s1", "s2", "s3", "s4", "q_pos", "q_neg"]
        pos_ids = ["s0", "s1", "s2", "s3", "s4"]
        caches = {
            key: build_pairwise_topk_cache(toks[key], masks, all_ids, pos_ids, grid_hw=(64 if "64" in key else 32))
            for key in ("deep_64", "mid_64", "deep_32")
        }
        return caches, toks, masks, affs

    def test_exact_equivalence_all_variants_and_shots(self) -> None:
        caches, toks, masks, affs = self._build_synthetic_bundle()
        query_ids = ["q_pos", "q_neg"]

        for cfg in ITER5_VARIANTS:
            pipe = ThreeModulePipeline(cfg)
            for sids in (["s0"], ["s0", "s1", "s2"], ["s0", "s1", "s2", "s3", "s4"]):
                # Legacy L direct call
                p512_L, s32_L = run_iter5_episode(caches, toks, masks, affs, sids, query_ids, cfg)
                trace_L = _trace_legacy_iter5_intermediates(caches, toks, masks, affs, sids, query_ids, cfg)

                # Refactored R structured call + drop-in legacy wrapper
                out_R = pipe.run_episode(caches, toks, masks, affs, sids, query_ids)
                p512_R, s32_R = out_R.to_legacy_outputs()
                p512_wrap, s32_wrap = run_three_module_episode(caches, toks, masks, affs, sids, query_ids, cfg)

                # 1. Episode calibration exact match
                self.assertEqual(out_R.calibration.bias, trace_L["bias"])
                self.assertEqual(out_R.calibration.temp, trace_L["temp"])
                self.assertEqual(out_R.calibration.mid_veto_ref, trace_L["mid_veto_ref"])

                # 2. Per-query exact stage-by-stage equality
                for qid in query_ids:
                    qL = trace_L["per_q"][qid]
                    sem_R = out_R.semantic_by_id[qid]
                    rec_R = out_R.recovery_by_id[qid]
                    bnd_R = out_R.boundary_by_id[qid]

                    self.assertTrue(torch.equal(sem_R.s_sem, qL["s_sem"]), f"s_sem mismatch {cfg.name} k={len(sids)} {qid}")
                    self.assertTrue(torch.equal(sem_R.p_sem, qL["p_sem"]), f"p_sem mismatch {cfg.name} k={len(sids)} {qid}")
                    if qL["s_mid"] is None:
                        self.assertIsNone(sem_R.s_mid)
                    else:
                        self.assertTrue(torch.equal(sem_R.s_mid, qL["s_mid"]))

                    self.assertTrue(torch.equal(rec_R.s_res, qL["s_res"]), f"s_res mismatch {cfg.name} k={len(sids)} {qid}")
                    self.assertTrue(torch.equal(rec_R.p_res, qL["p_res"]), f"p_res mismatch {cfg.name} k={len(sids)} {qid}")

                    np.testing.assert_array_equal(p512_R[qid], p512_L[qid])
                    np.testing.assert_array_equal(s32_R[qid], s32_L[qid])
                    np.testing.assert_array_equal(p512_wrap[qid], p512_L[qid])
                    np.testing.assert_array_equal(s32_wrap[qid], s32_L[qid])

                    # 0.50 masks strictly identical, and M_final == M_coarse when pixel refinement is off
                    m_legacy = p512_L[qid] >= 0.50
                    np.testing.assert_array_equal(bnd_R.m_coarse, m_legacy)
                    np.testing.assert_array_equal(bnd_R.m_final, m_legacy)

    def test_m3_pruned_and_b0_config_helpers(self) -> None:
        m3 = get_m3_pruned_config()
        b0 = get_b0_32_config()
        self.assertEqual(m3.name, "M3_Pruned_Ref")
        self.assertEqual(m3.shot_mode, "top2_soft")
        self.assertEqual(m3.w_fg_calib, 0.38)
        self.assertEqual(m3.seed_prob_thr, 0.66)
        self.assertEqual(m3.min_seeds, 4)
        self.assertEqual(m3.seed_mid_margin, -1.0)
        self.assertEqual(m3.qsc_weight, 0.40)
        self.assertEqual(b0.name, "B0_32_Ref")
        self.assertFalse(b0.use_64)


if __name__ == "__main__":
    unittest.main()
