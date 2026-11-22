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


class SemanticLocalizationModule:
    """Module ①: Photovoltaic Semantic Localization & False-Positive Suppression.

    Implements the exact behavior-equivalent logic of M3_Pruned_Ref / Iter5Config:
      1. Support-bank Top-3 foreground/background matching across deep_64, deep_32, mid_64;
      2. Dual-scale fusion + unilateral mid-layer Softplus suppression (S_sem);
      3. Support-set leave-one-out (K>=2) or neighbor-excluded self-match (K=1) calibration
         producing episode (bias, temp) and continuous P_sem = sigmoid((S_sem - bias) / temp).

    No probability thresholding is performed inside Module ①.
    """

    def __init__(self, cfg: Iter5Config, *, hard_neg_cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg
        self.hard_neg_cfg = hard_neg_cfg

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
            return s32, None

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
        return comb, sm

    @torch.inference_mode()
    def calibrate_episode(
        self,
        caches: dict[str, dict],
        tokens_by_key: dict[str, dict[str, torch.Tensor]],
        mask_by_id: dict[str, np.ndarray],
        support_ids: list[str],
    ) -> SemanticCalibration:
        """Fit support-only episode calibration parameters (t_mid, bias, temp)."""
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

        calib_source = "fallback"
        if k >= 2:
            ref_key = "deep_64" if cfg.use_64 else "deep_32"
            fg_items, bg_list = [], []
            for idx, sid in enumerate(support_ids):
                comb_s, _ = self.compute_raw_semantic_score(
                    caches,
                    sid,
                    support_ids,
                    use_core=use_core,
                    mid_veto_ref=mid_veto_ref,
                    exclude_pos_idx=idx,
                )
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
            diagnostics={
                "mid_veto_source": mid_src,
                "calib_source": calib_source,
                "hard_neg_enabled": bool(self.hard_neg_cfg and self.hard_neg_cfg.get("enabled", False)),
            },
        )

    @torch.inference_mode()
    def infer_query(
        self,
        caches: dict[str, dict],
        qid: str,
        support_ids: list[str],
        calib: SemanticCalibration,
    ) -> SemanticQueryOutput:
        """Compute Module ① outputs (S_sem, P_sem, S_mid, t_mid) for one query image."""
        s_sem, s_mid = self.compute_raw_semantic_score(
            caches,
            qid,
            support_ids,
            use_core=calib.use_core,
            mid_veto_ref=calib.mid_veto_ref,
            exclude_pos_idx=None,
        )
        p_sem = torch.sigmoid((s_sem - calib.bias) / calib.temp)
        grid_hw = 64 if self.cfg.use_64 else 32
        return SemanticQueryOutput(
            qid=qid,
            grid_hw=grid_hw,
            s_sem=s_sem,
            p_sem=p_sem,
            s_mid=s_mid,
            t_mid=calib.mid_veto_ref,
            diagnostics={"hard_neg_applied": False},
        )
