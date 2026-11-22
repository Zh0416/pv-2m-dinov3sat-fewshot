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
    ) -> None:
        self.cfg = cfg
        if isinstance(hard_neg_cfg, HardNegConfig):
            self.hard_neg_cfg: HardNegConfig | None = hard_neg_cfg if hard_neg_cfg.enabled else None
        elif isinstance(hard_neg_cfg, dict) and hard_neg_cfg.get("enabled", False):
            self.hard_neg_cfg = HardNegConfig(**{k: v for k, v in hard_neg_cfg.items() if k in HardNegConfig.__dataclass_fields__})
        else:
            self.hard_neg_cfg = None
        self._hard_neg_cache: dict[tuple, dict[str, Any]] = {}

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
            comb = comb + cfg.lambda_mid * sm - cfg.beta_veto * F.softplus((mid_veto_ref - sm) * 10.0) / 10.0
        if return_f64:
            return comb, sm, f64
        return comb, sm

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
    ) -> SemanticCalibration:
        """Fit support-only episode calibration parameters (t_mid, bias, temp) + optional hard-neg bank."""
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

        calib_source = "fallback"
        if k >= 2:
            ref_key = "deep_64" if cfg.use_64 else "deep_32"
            fg_items, bg_list = [], []
            loo_margins = hn_prep.get("loo_margins", {}) if hn_active else {}
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
                        bg_s = bg_s + cfg.lambda_mid * cm_b - cfg.beta_veto * F.softplus((mid_veto_ref - cm_b) * 10.0) / 10.0
                    k1_bg_m = hn_prep.get("k1_bg_margin") if hn_active else None
                    if hn_active and k1_bg_m is not None and self.hard_neg_cfg is not None and t_h is not None:
                        bg_s = apply_hard_neg_penalty(bg_s, k1_bg_m, t_h, self.hard_neg_cfg)
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
            diagnostics={
                "mid_veto_source": mid_src,
                "calib_source": calib_source,
                "hard_neg_enabled": self.hard_neg_cfg is not None,
                "hard_neg_mode": self.hard_neg_cfg.mode if self.hard_neg_cfg is not None else "off",
                "hard_neg_active": hn_active,
                "hard_neg_reason": hn_prep.get("reason", "disabled"),
                "hard_neg_bank_tokens": int(hn_prep.get("bank_total_tokens", 0)),
                "hard_neg_t_h": t_h,
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
    ) -> SemanticQueryOutput:
        """Compute Module ① outputs (S_sem, P_sem, S_mid, t_mid) for one query image."""
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
            },
        )
