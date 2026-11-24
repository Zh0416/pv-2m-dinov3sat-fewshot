from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..fast_bank import score_from_pairwise, select_fg_bg_indices
from ..iter3_methods import _ex_neigh_bg_contrast
from ..iter5_methods import Iter5Config
from .contracts import SemanticCalibration, SemanticQueryOutput
from .fft_soft_veto import (
    FFTSoftVetoConfig,
    FFTSoftVetoReference,
    ImageFFTPrecompute,
    compute_window_mask_coverage,
    evaluate_fft_soft_veto_map,
    evaluate_k1_nonoverlap_bg_penalties,
    fit_fft_soft_veto_reference,
)
from .hard_negative import (
    HardNegConfig,
    apply_hard_neg_penalty,
    build_support_bg_subbank,
    compute_margin_against_bank,
)


class SemanticLocalizationModule:
    """Module ①: Photovoltaic Semantic Localization & False-Positive Suppression.

    Implements the exact behavior-equivalent logic of M3_Pruned_Ref / Iter5Config:
      1. Support-bank Top-3 foreground/background matching across deep_64, deep_32, mid_64;
      2. Dual-scale fusion + unilateral mid-layer Softplus suppression (S_sem);
      3. Support-set leave-one-out (K>=2) or neighbor-excluded self-match (K=1) calibration
         producing episode (bias, temp) and continuous P_sem = sigmoid((S_sem - bias) / temp).

    No probability thresholding is performed inside Module ①.
    """

    def __init__(
        self,
        cfg: Iter5Config,
        *,
        hard_neg_cfg: HardNegConfig | dict[str, Any] | None = None,
        fft_veto_cfg: FFTSoftVetoConfig | dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        if isinstance(hard_neg_cfg, HardNegConfig):
            self.hard_neg_cfg: HardNegConfig | None = hard_neg_cfg if hard_neg_cfg.enabled else None
        elif isinstance(hard_neg_cfg, dict) and hard_neg_cfg.get("enabled", False):
            self.hard_neg_cfg = HardNegConfig(**{k: v for k, v in hard_neg_cfg.items() if k in HardNegConfig.__dataclass_fields__})
        else:
            self.hard_neg_cfg = None
        if isinstance(fft_veto_cfg, FFTSoftVetoConfig):
            self.fft_veto_cfg: FFTSoftVetoConfig | None = fft_veto_cfg if fft_veto_cfg.enabled else None
        elif isinstance(fft_veto_cfg, dict) and fft_veto_cfg.get("enabled", False):
            self.fft_veto_cfg = FFTSoftVetoConfig(**{k: v for k, v in fft_veto_cfg.items() if k in FFTSoftVetoConfig.__dataclass_fields__})
        else:
            self.fft_veto_cfg = None
        self._hard_neg_cache: dict[tuple, dict[str, Any]] = {}
        self._support_cov_cache: dict[str, np.ndarray] = {}

    @torch.inference_mode()
    def _estimate_mid_veto_ref(
        self,
        caches: dict[str, dict],
        tokens_by_key: dict[str, dict[str, torch.Tensor]],
        mask_by_id: dict[str, np.ndarray],
        support_ids: list[str],
        *,
        use_core: bool,
        core_purity: float,
    ) -> tuple[float | None, str]:
        cfg = self.cfg
        if not (cfg.use_64 and cfg.use_mid_veto):
            return None, "disabled"

        k = len(support_ids)
        mid_veto_ref: float | None = None
        source = "fallback_default"
        if k >= 2:
            mid_vals = []
            for idx, sid in enumerate(support_ids):
                sm, _, _ = score_from_pairwise(
                    caches["mid_64"],
                    sid,
                    support_ids,
                    use_core=use_core,
                    shot_mode=cfg.shot_mode,
                    exclude_pos_idx=idx,
                )
                bm = caches["mid_64"]["bg_masks"][caches["mid_64"]["id_to_pos"][sid]]
                if bm.any():
                    mid_vals.append(sm[torch.from_numpy(np.flatnonzero(bm)).to(sm.device)])
            if mid_vals:
                mid_veto_ref = float(torch.quantile(torch.cat(mid_vals), 0.85).item())
                source = "support_loo_q85"
        else:
            c_bg_m = _ex_neigh_bg_contrast(
                tokens_by_key["mid_64"][support_ids[0]],
                mask_by_id[support_ids[0]],
                64,
                core_purity=core_purity,
            )
            if c_bg_m is not None:
                mid_veto_ref = float(torch.quantile(c_bg_m, 0.85).item())
                source = "k1_ex_neigh_q85"

        if mid_veto_ref is None:
            mid_veto_ref = 0.20
            source = "fallback_default_0.20"
        return mid_veto_ref, source

    @torch.inference_mode()
    def compute_raw_semantic_score(
        self,
        caches: dict[str, dict],
        qid: str,
        support_ids: list[str],
        *,
        use_core: bool,
        mid_veto_ref: float | None,
        exclude_pos_idx: int | None = None,
        return_f64: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute fused S_sem (64x64 or 32x32 flattened) and S_mid in exact legacy order."""
        cfg = self.cfg
        s32, _, _ = score_from_pairwise(
            caches["deep_32"],
            qid,
            support_ids,
            use_core=use_core,
            shot_mode=cfg.shot_mode,
            exclude_pos_idx=exclude_pos_idx,
        )
        if not cfg.use_64:
            return (s32, None, None) if return_f64 else (s32, None)

        s64, f64, _ = score_from_pairwise(
            caches["deep_64"],
            qid,
            support_ids,
            use_core=use_core,
            shot_mode=cfg.shot_mode,
            exclude_pos_idx=exclude_pos_idx,
        )
        s32_up = F.interpolate(
            s32.reshape(1, 1, 32, 32),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        ).reshape(-1)
        comb = cfg.w_s64 * s64 + (1.0 - cfg.w_s64) * s32_up
        sm = None
        if cfg.use_mid_veto:
            sm, _, _ = score_from_pairwise(
                caches["mid_64"],
                qid,
                support_ids,
                use_core=use_core,
                shot_mode=cfg.shot_mode,
                exclude_pos_idx=exclude_pos_idx,
            )
            if self.fft_veto_cfg is not None:
                # Replace mid-layer Softplus veto only; retain deep localization + 0.35 * S_mid
                comb = comb + cfg.lambda_mid * sm
            else:
                comb = comb + cfg.lambda_mid * sm - cfg.beta_veto * F.softplus((mid_veto_ref - sm) * 10.0) / 10.0
        if return_f64:
            return comb, sm, f64
        return comb, sm

    @torch.inference_mode()
    def _compute_deep_score_only(
        self,
        caches: dict[str, dict],
        qid: str,
        support_ids: list[str],
        *,
        use_core: bool,
        exclude_pos_idx: int | None = None,
    ) -> torch.Tensor:
        """Compute S_deep = w_s64 * S_64 + (1 - w_s64) * Up(S_32) flattened [4096]."""
        cfg = self.cfg
        s32, _, _ = score_from_pairwise(
            caches["deep_32"],
            qid,
            support_ids,
            use_core=use_core,
            shot_mode=cfg.shot_mode,
            exclude_pos_idx=exclude_pos_idx,
        )
        s64, _, _ = score_from_pairwise(
            caches["deep_64"],
            qid,
            support_ids,
            use_core=use_core,
            shot_mode=cfg.shot_mode,
            exclude_pos_idx=exclude_pos_idx,
        )
        s32_up = F.interpolate(
            s32.reshape(1, 1, 32, 32),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        ).reshape(-1)
        return cfg.w_s64 * s64 + (1.0 - cfg.w_s64) * s32_up

    @torch.inference_mode()
    def _k1_ex_neigh_deep_map(
        self,
        tokens_by_key: dict[str, dict[str, torch.Tensor]],
        mask_512: np.ndarray,
        sid0: str,
        *,
        core_purity: float,
    ) -> np.ndarray:
        """Compute 1-shot spatial-neighbor-excluded S_deep [64, 64] on sid0 for both PV and BG windows."""

        def _grid_ex_contrast(tok: torch.Tensor, grid_hw: int) -> np.ndarray:
            radius = 2 if grid_hw == 64 else 1
            fg_c, bg_m, _ = select_fg_bg_indices(mask_512, grid_hw, core_purity)
            fg_flat = np.flatnonzero(fg_c)
            bg_flat = np.flatnonzero(bg_m)
            out = np.full(grid_hw * grid_hw, np.nan, dtype=np.float32)
            if len(fg_flat) == 0 or len(bg_flat) == 0:
                return out.reshape(grid_hw, grid_hw)

            dev = tok.device
            all_idx = torch.arange(grid_hw * grid_hw, device=dev)
            all_yx = torch.stack((all_idx // grid_hw, all_idx % grid_hw), dim=1)
            fg_t = torch.from_numpy(fg_flat).to(dev)
            bg_t = torch.from_numpy(bg_flat).to(dev)
            fg_yx = torch.stack((fg_t // grid_hw, fg_t % grid_hw), dim=1)
            bg_yx = torch.stack((bg_t // grid_hw, bg_t % grid_hw), dim=1)

            fg_vecs = tok[fg_t]
            bg_vecs = tok[bg_t]

            near_f = ((all_yx[:, 0:1] - fg_yx[:, 0:1].T).abs() <= radius) & (
                (all_yx[:, 1:2] - fg_yx[:, 1:2].T).abs() <= radius
            )
            near_b = ((all_yx[:, 0:1] - bg_yx[:, 0:1].T).abs() <= radius) & (
                (all_yx[:, 1:2] - bg_yx[:, 1:2].T).abs() <= radius
            )
            cnt_f = (~near_f).sum(dim=1)
            cnt_b = (~near_b).sum(dim=1)
            valid = (cnt_f >= 1) & (cnt_b >= 1)
            if not valid.any():
                return out.reshape(grid_hw, grid_hw)

            v_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            q_vecs = tok[v_idx]
            nf_v = near_f[v_idx]
            nb_v = near_b[v_idx]

            sim_f = (q_vecs @ fg_vecs.T).clamp(-1.0, 1.0).masked_fill(nf_v, -2.0)
            m_f = min(3, fg_vecs.shape[0])
            top_f = sim_f.topk(m_f, dim=1).values
            valid_f = top_f > -1.5
            s_pos = (top_f * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1).float()

            sim_b = (q_vecs @ bg_vecs.T).clamp(-1.0, 1.0).masked_fill(nb_v, -2.0)
            m_b = min(3, bg_vecs.shape[0])
            top_b = sim_b.topk(m_b, dim=1).values
            valid_b = top_b > -1.5
            s_neg = (top_b * valid_b).sum(dim=1) / valid_b.sum(dim=1).clamp_min(1).float()

            vals = (s_pos - 0.50 * s_neg).detach().cpu().numpy().astype(np.float32)
            out[v_idx.detach().cpu().numpy()] = vals
            return out.reshape(grid_hw, grid_hw)

        c64 = _grid_ex_contrast(tokens_by_key["deep_64"][sid0], 64)
        c32 = _grid_ex_contrast(tokens_by_key["deep_32"][sid0], 32)
        c32_fin = np.isfinite(c32)
        c32_safe = np.where(c32_fin, c32, 0.0).astype(np.float32)
        c32_up = (
            F.interpolate(torch.from_numpy(c32_safe)[None, None], size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
            .numpy()
            .astype(np.float32)
        )
        # Propagate any invalid 32x32 cells cleanly to their 2x2 children at 64x64
        c32_valid_64 = np.repeat(np.repeat(c32_fin, 2, axis=0), 2, axis=1)
        s_deep = self.cfg.w_s64 * c64 + (1.0 - self.cfg.w_s64) * c32_up
        s_deep[~c32_valid_64] = np.nan
        return s_deep.astype(np.float32)

    @torch.inference_mode()
    def _prepare_fft_veto_calibration(
        self,
        caches: dict[str, dict],
        tokens_by_key: dict[str, dict[str, torch.Tensor]],
        mask_by_id: dict[str, np.ndarray],
        support_ids: list[str],
        fft_pre_by_id: dict[str, ImageFFTPrecompute] | None,
        *,
        use_core: bool,
        core_purity: float,
    ) -> dict[str, Any]:
        """Build full-support adaptive FFT soft-veto reference + strict LOO/non-overlap calibration penalties."""
        fcfg = self.fft_veto_cfg
        if fcfg is None or not self.cfg.use_64:
            return {"active": False, "ref_full": FFTSoftVetoReference(active=False, reason="disabled")}
        if fft_pre_by_id is None:
            raise ValueError("fft_pre_by_id is required when fft_veto_cfg is enabled")

        def _get_cov(sid: str) -> np.ndarray:
            if sid not in self._support_cov_cache:
                pre = fft_pre_by_id[sid]
                self._support_cov_cache[sid] = compute_window_mask_coverage(
                    mask_by_id[sid], pre.rows, pre.cols, pre.interior, window_size=fcfg.window_size
                )
            return self._support_cov_cache[sid]

        k = len(support_ids)
        pre_list = [fft_pre_by_id[sid] for sid in support_ids]
        cov_list = [_get_cov(sid) for sid in support_ids]

        if k >= 2:
            s_deep_full_list: list[np.ndarray] = []
            for idx, sid in enumerate(support_ids):
                sd = self._compute_deep_score_only(caches, sid, support_ids, use_core=use_core, exclude_pos_idx=idx)
                s_deep_full_list.append(sd.detach().cpu().numpy().reshape(64, 64).astype(np.float32))
            ref_full = fit_fft_soft_veto_reference(pre_list, cov_list, s_deep_full_list, fcfg)
            if not ref_full.active:
                return {"active": False, "ref_full": ref_full}

            # Strict LOO calibration: when scoring support_ids[idx] for (b, T), exclude support_ids[idx] from band & histogram!
            loo_deltas: dict[str, np.ndarray] = {}
            loo_active_count = 0
            for idx, sid in enumerate(support_ids):
                other_ids = [s for j, s in enumerate(support_ids) if j != idx]
                other_pres = [fft_pre_by_id[s] for s in other_ids]
                other_covs = [_get_cov(s) for s in other_ids]
                other_s_deeps: list[np.ndarray] = []
                if len(other_ids) >= 2:
                    for j, oid in enumerate(other_ids):
                        sd_o = self._compute_deep_score_only(
                            caches, oid, other_ids, use_core=use_core, exclude_pos_idx=j
                        )
                        other_s_deeps.append(sd_o.detach().cpu().numpy().reshape(64, 64).astype(np.float32))
                else:
                    other_s_deeps.append(
                        self._k1_ex_neigh_deep_map(tokens_by_key, mask_by_id[other_ids[0]], other_ids[0], core_purity=core_purity)
                    )
                ref_loo = fit_fft_soft_veto_reference(other_pres, other_covs, other_s_deeps, fcfg)
                if ref_loo.active:
                    loo_active_count += 1
                    loo_deltas[sid] = evaluate_fft_soft_veto_map(pre_list[idx], s_deep_full_list[idx], ref_loo, fcfg)["delta_fft"]
                else:
                    loo_deltas[sid] = np.zeros(64 * 64, dtype=np.float32)

            return {
                "active": True,
                "ref_full": ref_full,
                "loo_deltas": loo_deltas,
                "loo_active_count": loo_active_count,
            }

        # K == 1
        sid0 = support_ids[0]
        s_deep_ex0 = self._k1_ex_neigh_deep_map(tokens_by_key, mask_by_id[sid0], sid0, core_purity=core_purity)
        ref_full = fit_fft_soft_veto_reference([pre_list[0]], [cov_list[0]], [s_deep_ex0], fcfg)
        if not ref_full.active:
            return {"active": False, "ref_full": ref_full}
        return {
            "active": True,
            "ref_full": ref_full,
            "k1_pre": pre_list[0],
            "k1_cov": cov_list[0],
            "k1_s_deep_ex": s_deep_ex0,
        }

    @torch.inference_mode()
    def _prepare_hard_neg_calibration(
        self,
        caches: dict[str, dict],
        tokens_by_key: dict[str, dict[str, torch.Tensor]],
        mask_by_id: dict[str, np.ndarray],
        support_ids: list[str],
        *,
        use_core: bool,
        core_purity: float,
        use_core_calib: bool,
    ) -> dict[str, Any]:
        """Build episode bank H (or equal-budget random_bg) and support-only t_H reference."""
        hcfg = self.hard_neg_cfg
        if hcfg is None or not self.cfg.use_64:
            return {"active": False, "reason": "disabled"}

        toks_64 = tokens_by_key["deep_64"]
        bank_full = build_support_bg_subbank(
            toks_64, mask_by_id, support_ids, hcfg, cache_store=self._hard_neg_cache
        )
        if not bank_full["valid"]:
            return {
                "active": False,
                "reason": bank_full["reason"],
                "bank_total_tokens": bank_full["total_tokens"],
            }

        k = len(support_ids)
        if k >= 2:
            loo_margins: dict[str, torch.Tensor] = {}
            loo_banks: dict[str, dict[str, Any]] = {}
            fg_margin_pool: list[torch.Tensor] = []
            for idx, sid in enumerate(support_ids):
                other_sids = [s for s in support_ids if s != sid]
                b_loo = build_support_bg_subbank(
                    toks_64, mask_by_id, other_sids, hcfg, cache_store=self._hard_neg_cache
                )
                loo_banks[sid] = b_loo
                if not b_loo["valid"]:
                    continue
                _, f64_s, _ = score_from_pairwise(
                    caches["deep_64"],
                    sid,
                    support_ids,
                    use_core=use_core,
                    shot_mode=self.cfg.shot_mode,
                    exclude_pos_idx=idx,
                )
                m_s = compute_margin_against_bank(
                    toks_64[sid], f64_s, b_loo["tokens"], top_m=hcfg.top_m_bank
                )
                loo_margins[sid] = m_s
                p_i = caches["deep_64"]["id_to_pos"][sid]
                fg_m = (
                    caches["deep_64"]["fg_core_masks"][p_i]
                    if use_core_calib
                    else caches["deep_64"]["fg_all_masks"][p_i]
                )
                if fg_m.any():
                    fg_margin_pool.append(m_s[torch.from_numpy(np.flatnonzero(fg_m)).to(m_s.device)])

            if not fg_margin_pool:
                return {
                    "active": False,
                    "reason": "no_valid_loo_bank_for_t_h",
                    "bank_total_tokens": bank_full["total_tokens"],
                }
            t_h = float(torch.quantile(torch.cat(fg_margin_pool), hcfg.t_h_fg_quantile).item())
            return {
                "active": True,
                "reason": "ok",
                "t_h": t_h,
                "bank_full": bank_full,
                "loo_margins": loo_margins,
                "loo_banks": loo_banks,
                "bank_total_tokens": bank_full["total_tokens"],
            }

        # K == 1: ex-neighbor foreground & background margin estimation on support_ids[0]
        sid0 = support_ids[0]
        tok0 = toks_64[sid0]
        dev = tok0.device
        fg_m, bg_m, _ = select_fg_bg_indices(mask_by_id[sid0], 64, core_purity)
        fg_idx = np.flatnonzero(fg_m)
        bg_idx = np.flatnonzero(bg_m)
        h_idx_list = bank_full["token_indices_by_sid"].get(sid0, [])
        if len(fg_idx) < hcfg.k1_min_independent_fg_tokens or len(h_idx_list) < hcfg.min_total_tokens:
            return {"active": False, "reason": "k1_insufficient_independent_tokens", "bank_total_tokens": len(h_idx_list)}

        fg_t = torch.from_numpy(fg_idx).to(dev)
        h_t = torch.tensor(h_idx_list, dtype=torch.long, device=dev)
        fg_vecs = tok0[fg_t]
        h_vecs = bank_full["tokens"]

        fg_yx = torch.stack((fg_t // 64, fg_t % 64), dim=1)
        h_yx = torch.stack((h_t // 64, h_t % 64), dim=1)

        near_ff = ((fg_yx[:, 0:1] - fg_yx[:, 0:1].T).abs() <= hcfg.k1_ex_neigh_radius) & (
            (fg_yx[:, 1:2] - fg_yx[:, 1:2].T).abs() <= hcfg.k1_ex_neigh_radius
        )
        near_fh = ((fg_yx[:, 0:1] - h_yx[:, 0:1].T).abs() <= hcfg.k1_ex_neigh_radius) & (
            (fg_yx[:, 1:2] - h_yx[:, 1:2].T).abs() <= hcfg.k1_ex_neigh_radius
        )
        valid_fg = ((~near_ff).sum(dim=1) >= 3) & ((~near_fh).sum(dim=1) >= 3)
        if int(valid_fg.sum().item()) < hcfg.k1_min_independent_fg_tokens:
            return {"active": False, "reason": "k1_insufficient_ex_neigh_fg", "bank_total_tokens": len(h_idx_list)}

        m_f = min(3, int(fg_vecs.shape[0]))
        f_self_fg = (fg_vecs @ fg_vecs.T).clamp(-1.0, 1.0).masked_fill(near_ff, -2.0).topk(m_f, dim=1).values.mean(dim=1)
        m_fg = compute_margin_against_bank(fg_vecs, f_self_fg, h_vecs, top_m=hcfg.top_m_bank, exclude_mask=near_fh)
        t_h = float(torch.quantile(m_fg[valid_fg], hcfg.t_h_fg_quantile).item())

        # Also compute ex-neighbor background margins for K=1 bias recalibration
        k1_bg_margin: torch.Tensor | None = None
        if len(bg_idx) > 0:
            bg_t = torch.from_numpy(bg_idx).to(dev)
            bg_vecs = tok0[bg_t]
            bg_yx = torch.stack((bg_t // 64, bg_t % 64), dim=1)
            near_bf = ((bg_yx[:, 0:1] - fg_yx[:, 0:1].T).abs() <= hcfg.k1_ex_neigh_radius) & (
                (bg_yx[:, 1:2] - fg_yx[:, 1:2].T).abs() <= hcfg.k1_ex_neigh_radius
            )
            near_bh = ((bg_yx[:, 0:1] - h_yx[:, 0:1].T).abs() <= hcfg.k1_ex_neigh_radius) & (
                (bg_yx[:, 1:2] - h_yx[:, 1:2].T).abs() <= hcfg.k1_ex_neigh_radius
            )
            if int((~near_bf).sum(dim=1).min().item()) >= 1 and int((~near_bh).sum(dim=1).min().item()) >= 1:
                f_bg = (bg_vecs @ fg_vecs.T).clamp(-1.0, 1.0).masked_fill(near_bf, -2.0).topk(m_f, dim=1).values.mean(dim=1)
                k1_bg_margin = compute_margin_against_bank(bg_vecs, f_bg, h_vecs, top_m=hcfg.top_m_bank, exclude_mask=near_bh)

        return {
            "active": True,
            "reason": "ok",
            "t_h": t_h,
            "bank_full": bank_full,
            "k1_bg_margin": k1_bg_margin,
            "bank_total_tokens": bank_full["total_tokens"],
        }

    @torch.inference_mode()
    def calibrate_episode(
        self,
        caches: dict[str, dict],
        tokens_by_key: dict[str, dict[str, torch.Tensor]],
        mask_by_id: dict[str, np.ndarray],
        support_ids: list[str],
        *,
        fft_pre_by_id: dict[str, ImageFFTPrecompute] | None = None,
    ) -> SemanticCalibration:
        """Fit support-only episode calibration parameters (t_mid, bias, temp) + optional hard-neg / FFT soft-veto."""
        cfg = self.cfg
        k = len(support_ids)
        use_core = cfg.k1_use_core if k == 1 else cfg.kge2_use_core
        core_purity = 0.85 if use_core else 0.50
        use_core_calib = use_core and not cfg.calib_v2

        mid_veto_ref, mid_src = self._estimate_mid_veto_ref(
            caches,
            tokens_by_key,
            mask_by_id,
            support_ids,
            use_core=use_core,
            core_purity=core_purity,
        )

        hn_prep = self._prepare_hard_neg_calibration(
            caches,
            tokens_by_key,
            mask_by_id,
            support_ids,
            use_core=use_core,
            core_purity=core_purity,
            use_core_calib=use_core_calib,
        )
        hn_active = bool(hn_prep.get("active", False))
        t_h = hn_prep.get("t_h") if hn_active else None
        bank_full = hn_prep.get("bank_full") if hn_active else None

        fft_prep = self._prepare_fft_veto_calibration(
            caches,
            tokens_by_key,
            mask_by_id,
            support_ids,
            fft_pre_by_id,
            use_core=use_core,
            core_purity=core_purity,
        )
        fft_active = bool(fft_prep.get("active", False))
        fft_ref_full: FFTSoftVetoReference = fft_prep["ref_full"]

        calib_source = "fallback"
        if k >= 2:
            ref_key = "deep_64" if cfg.use_64 else "deep_32"
            fg_items, bg_list = [], []
            loo_margins = hn_prep.get("loo_margins", {}) if hn_active else {}
            fft_loo_deltas = fft_prep.get("loo_deltas", {}) if fft_active else {}
            for idx, sid in enumerate(support_ids):
                comb_s, _ = self.compute_raw_semantic_score(
                    caches,
                    sid,
                    support_ids,
                    use_core=use_core,
                    mid_veto_ref=mid_veto_ref,
                    exclude_pos_idx=idx,
                )
                if hn_active and sid in loo_margins and self.hard_neg_cfg is not None and t_h is not None:
                    comb_s = apply_hard_neg_penalty(comb_s, loo_margins[sid], t_h, self.hard_neg_cfg)
                if fft_active and sid in fft_loo_deltas:
                    comb_s = comb_s - torch.from_numpy(fft_loo_deltas[sid]).to(comb_s.device)
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
                    calib_source = "support_loo_normal"
                else:
                    bias, temp = mu_bg + 0.05, 0.06
                    calib_source = "support_loo_low_sep_fallback"
            else:
                bias, temp = 0.30, 0.06
                calib_source = "support_loo_no_bg_fallback"
        else:
            sid0 = support_ids[0]
            c32_b = _ex_neigh_bg_contrast(
                tokens_by_key["deep_32"][sid0],
                mask_by_id[sid0],
                32,
                core_purity=core_purity,
            )
            if not cfg.use_64:
                if c32_b is not None:
                    bias = float(torch.quantile(c32_b, cfg.bg_q_calib).item()) + cfg.k1_margin
                    calib_source = "k1_ex_neigh_32"
                else:
                    bias = 0.30
                    calib_source = "k1_fallback_32"
            else:
                c64_b = _ex_neigh_bg_contrast(
                    tokens_by_key["deep_64"][sid0],
                    mask_by_id[sid0],
                    64,
                    core_purity=core_purity,
                )
                cm_b = (
                    _ex_neigh_bg_contrast(
                        tokens_by_key["mid_64"][sid0],
                        mask_by_id[sid0],
                        64,
                        core_purity=core_purity,
                    )
                    if cfg.use_mid_veto
                    else None
                )
                if c64_b is not None and c32_b is not None:
                    _, bg64_m, _ = select_fg_bg_indices(mask_by_id[sid0], 64, core_purity)
                    _, bg32_m, _ = select_fg_bg_indices(mask_by_id[sid0], 32, core_purity)
                    bg32_flat = np.flatnonzero(bg32_m)
                    c32_full = torch.zeros((32, 32), device=c64_b.device)
                    c32_full[
                        torch.from_numpy(bg32_flat // 32).to(c64_b.device),
                        torch.from_numpy(bg32_flat % 32).to(c64_b.device),
                    ] = c32_b
                    bg64_flat = np.flatnonzero(bg64_m)
                    bg64_t = torch.from_numpy(bg64_flat).to(c64_b.device)
                    c32_at_64 = c32_full[(bg64_t // 64) // 2, (bg64_t % 64) // 2]
                    bg_s = cfg.w_s64 * c64_b + (1.0 - cfg.w_s64) * c32_at_64
                    if cfg.use_mid_veto and cm_b is not None and mid_veto_ref is not None:
                        if self.fft_veto_cfg is not None:
                            bg_s = bg_s + cfg.lambda_mid * cm_b
                        else:
                            bg_s = bg_s + cfg.lambda_mid * cm_b - cfg.beta_veto * F.softplus((mid_veto_ref - cm_b) * 10.0) / 10.0
                    k1_bg_m = hn_prep.get("k1_bg_margin") if hn_active else None
                    if hn_active and k1_bg_m is not None and self.hard_neg_cfg is not None and t_h is not None:
                        bg_s = apply_hard_neg_penalty(bg_s, k1_bg_m, t_h, self.hard_neg_cfg)
                    if fft_active and self.fft_veto_cfg is not None:
                        k1_pen = evaluate_k1_nonoverlap_bg_penalties(
                            fft_prep["k1_pre"],
                            fft_prep["k1_cov"],
                            fft_prep["k1_s_deep_ex"],
                            bg64_flat,
                            self.fft_veto_cfg,
                        )
                        bg_s = bg_s - torch.from_numpy(k1_pen).to(bg_s.device)
                    bias = float(torch.quantile(bg_s, cfg.bg_q_calib).item()) + cfg.k1_margin
                    calib_source = "k1_ex_neigh_64"
                else:
                    bias = 0.30
                    calib_source = "k1_fallback_64"
            temp = 0.06

        return SemanticCalibration(
            k_shots=k,
            use_core=use_core,
            core_purity=core_purity,
            use_core_calib=use_core_calib,
            mid_veto_ref=mid_veto_ref,
            bias=float(bias),
            temp=float(temp),
            hard_neg_active=hn_active,
            hard_neg_t_h=t_h,
            hard_neg_bank=bank_full,
            fft_veto_active=fft_active,
            fft_veto_ref=fft_ref_full,
            diagnostics={
                "mid_veto_source": mid_src,
                "calib_source": calib_source,
                "hard_neg_enabled": self.hard_neg_cfg is not None,
                "hard_neg_mode": self.hard_neg_cfg.mode if self.hard_neg_cfg is not None else "off",
                "hard_neg_active": hn_active,
                "hard_neg_reason": hn_prep.get("reason", "disabled"),
                "hard_neg_bank_tokens": int(hn_prep.get("bank_total_tokens", 0)),
                "hard_neg_t_h": t_h,
                "fft_veto_enabled": self.fft_veto_cfg is not None,
                "fft_veto_active": fft_active,
                "fft_veto_reason": fft_ref_full.reason,
                "fft_veto_band_selected": fft_ref_full.band_selected,
                "fft_veto_band_low": fft_ref_full.band_low,
                "fft_veto_band_high": fft_ref_full.band_high,
                "fft_veto_band_peak_r": fft_ref_full.band_peak_r,
                "fft_veto_energy_thr": fft_ref_full.energy_thr,
                "fft_veto_t_cand": fft_ref_full.t_cand,
                "fft_veto_n_pv_band": fft_ref_full.n_pv_band_windows,
                "fft_veto_n_bg_band": fft_ref_full.n_bg_band_windows,
                "fft_veto_n_pv_cand": fft_ref_full.n_pv_cand_windows,
                "fft_veto_n_bg_cand": fft_ref_full.n_bg_cand_windows,
                "fft_veto_num_merged_bins": fft_ref_full.num_merged_bins,
                "fft_veto_num_eligible_bins": fft_ref_full.num_eligible_bins,
            },
        )

    @torch.inference_mode()
    def infer_query(
        self,
        caches: dict[str, dict],
        qid: str,
        support_ids: list[str],
        calib: SemanticCalibration,
        *,
        q_tok_64: torch.Tensor | None = None,
        fft_pre_q: ImageFFTPrecompute | None = None,
    ) -> SemanticQueryOutput:
        """Compute Module ① outputs (S_sem, P_sem, S_mid, t_mid) for one query image."""
        fft_diag: dict[str, Any] = {
            "fft_veto_applied": False,
            "cand_tokens": 0,
            "g_active_tokens": 0,
            "penalized_tokens": 0,
            "penalty_rate_all": 0.0,
            "penalty_rate_cand": 0.0,
            "mean_penalty_all": 0.0,
            "mean_penalty_penalized": 0.0,
            "max_penalty": 0.0,
            "mean_e_bg_g_active": 0.0,
        }
        if calib.hard_neg_active and self.hard_neg_cfg is not None and calib.hard_neg_t_h is not None:
            if q_tok_64 is None:
                raise ValueError("q_tok_64 is required when hard_neg_active is True")
            s_sem_raw, s_mid, f64_q = self.compute_raw_semantic_score(
                caches,
                qid,
                support_ids,
                use_core=calib.use_core,
                mid_veto_ref=calib.mid_veto_ref,
                exclude_pos_idx=None,
                return_f64=True,
            )
            assert f64_q is not None
            m_q = compute_margin_against_bank(
                q_tok_64, f64_q, calib.hard_neg_bank["tokens"], top_m=self.hard_neg_cfg.top_m_bank
            )
            s_sem = apply_hard_neg_penalty(s_sem_raw, m_q, calib.hard_neg_t_h, self.hard_neg_cfg)
            hn_applied = True
            mean_pen = float((s_sem_raw - s_sem).mean().item())
            max_pen = float((s_sem_raw - s_sem).max().item())
        elif calib.fft_veto_active and self.fft_veto_cfg is not None and calib.fft_veto_ref is not None:
            if fft_pre_q is None:
                raise ValueError("fft_pre_q is required when fft_veto_active is True")
            s_deep_q = self._compute_deep_score_only(
                caches,
                qid,
                support_ids,
                use_core=calib.use_core,
                exclude_pos_idx=None,
            )
            s_mid, _, _ = score_from_pairwise(
                caches["mid_64"],
                qid,
                support_ids,
                use_core=calib.use_core,
                shot_mode=self.cfg.shot_mode,
                exclude_pos_idx=None,
            )
            fft_out = evaluate_fft_soft_veto_map(
                fft_pre_q,
                s_deep_q.detach().cpu().numpy(),
                calib.fft_veto_ref,
                self.fft_veto_cfg,
            )
            delta_t = torch.from_numpy(fft_out["delta_fft"]).to(s_deep_q.device)
            s_sem = s_deep_q + self.cfg.lambda_mid * s_mid - delta_t
            hn_applied = False
            mean_pen = 0.0
            max_pen = 0.0
            fft_diag = {"fft_veto_applied": True, **fft_out["diagnostics"]}
        else:
            s_sem, s_mid = self.compute_raw_semantic_score(
                caches,
                qid,
                support_ids,
                use_core=calib.use_core,
                mid_veto_ref=calib.mid_veto_ref,
                exclude_pos_idx=None,
                return_f64=False,
            )
            hn_applied = False
            mean_pen = 0.0
            max_pen = 0.0

        p_sem = torch.sigmoid((s_sem - calib.bias) / calib.temp)
        grid_hw = 64 if self.cfg.use_64 else 32
        return SemanticQueryOutput(
            qid=qid,
            grid_hw=grid_hw,
            s_sem=s_sem,
            p_sem=p_sem,
            s_mid=s_mid,
            t_mid=calib.mid_veto_ref,
            diagnostics={
                "hard_neg_applied": hn_applied,
                "hard_neg_mean_penalty": mean_pen,
                "hard_neg_max_penalty": max_pen,
                **fft_diag,
            },
        )
