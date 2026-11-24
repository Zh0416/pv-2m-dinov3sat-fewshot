from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from ..fast_bank import ex_neigh_bg_contrast, score_from_pairwise, select_fg_bg_indices
from .contracts import ModelConfig, SemanticCalibration, SemanticQueryOutput


class SemanticLocalizationModule:
    """Module ①: Multi-Scale & Mid-Layer Photovoltaic Semantic Localization.

    Converged clean implementation:
      1. Support-bank Top-3 foreground-vs-background matching across deep_64, deep_32, and mid_64;
      2. Linear multi-scale & mid-layer fusion: S_sem = 0.75 * S_64 + 0.25 * Up(S_32) + 0.35 * S_mid
         (disproven mid-layer Softplus penalty removed);
      3. Zero-leakage support leave-one-out (K>=2) or spatial-neighbor-excluded (K=1) calibration
         producing episode (bias, temp) and continuous P_sem = sigmoid((S_sem - bias) / temp).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    @torch.inference_mode()
    def compute_raw_semantic_score(
        self,
        caches: dict[str, dict],
        qid: str,
        support_ids: list[str],
        *,
        use_core: bool,
        exclude_pos_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute fused S_sem (64x64 or 32x32 flattened) and S_mid."""
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
        if cfg.use_mid_fusion:
            sm, _, _ = score_from_pairwise(
                caches["mid_64"],
                qid,
                support_ids,
                use_core=use_core,
                shot_mode=cfg.shot_mode,
                exclude_pos_idx=exclude_pos_idx,
            )
            comb = comb + cfg.lambda_mid * sm
        return comb, sm

    @torch.inference_mode()
    def calibrate_episode(
        self,
        caches: dict[str, dict],
        tokens_by_key: dict[str, dict[str, torch.Tensor]],
        mask_by_id: dict[str, np.ndarray],
        support_ids: list[str],
    ) -> SemanticCalibration:
        """Fit support-only episode calibration parameters (bias, temp) with zero query leakage."""
        cfg = self.cfg
        k = len(support_ids)
        use_core = cfg.k1_use_core if k == 1 else cfg.kge2_use_core
        core_purity = 0.85 if use_core else 0.50
        use_core_calib = use_core and not cfg.calib_v2

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
                    exclude_pos_idx=idx,
                )
                p_i = caches[ref_key]["id_to_pos"][sid]
                fg_m = caches[ref_key]["fg_core_masks"][p_i] if use_core_calib else caches[ref_key]["fg_all_masks"][p_i]
                bg_m = caches[ref_key]["bg_masks"][p_i]
                fg_idx = torch.from_numpy(np.flatnonzero(fg_m)).to(comb_s.device)
                if cfg.calib_v2:
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
            c32_b = ex_neigh_bg_contrast(
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
                c64_b = ex_neigh_bg_contrast(
                    tokens_by_key["deep_64"][sid0],
                    mask_by_id[sid0],
                    64,
                    core_purity=core_purity,
                )
                cm_b = (
                    ex_neigh_bg_contrast(
                        tokens_by_key["mid_64"][sid0],
                        mask_by_id[sid0],
                        64,
                        core_purity=core_purity,
                    )
                    if cfg.use_mid_fusion
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
                    bg64_t = torch.from_numpy(np.flatnonzero(bg64_m)).to(c64_b.device)
                    c32_at_64 = c32_full[(bg64_t // 64) // 2, (bg64_t % 64) // 2]
                    bg_s = cfg.w_s64 * c64_b + (1.0 - cfg.w_s64) * c32_at_64
                    if cfg.use_mid_fusion and cm_b is not None:
                        bg_s = bg_s + cfg.lambda_mid * cm_b
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
            bias=float(bias),
            temp=float(temp),
            diagnostics={"calib_source": calib_source},
        )

    @torch.inference_mode()
    def infer_query(
        self,
        caches: dict[str, dict],
        qid: str,
        support_ids: list[str],
        calib: SemanticCalibration,
    ) -> SemanticQueryOutput:
        """Compute Module ① outputs (S_sem, P_sem, S_mid) for one query image."""
        s_sem, s_mid = self.compute_raw_semantic_score(
            caches,
            qid,
            support_ids,
            use_core=calib.use_core,
            exclude_pos_idx=None,
        )
        grid_hw = 64 if self.cfg.use_64 else 32
        p_sem = torch.sigmoid((s_sem - calib.bias) / calib.temp)
        return SemanticQueryOutput(
            qid=qid,
            grid_hw=grid_hw,
            s_sem=s_sem,
            p_sem=p_sem,
            s_mid=s_mid,
            diagnostics={
                "max_p_sem": float(p_sem.max().item()),
                "mean_p_sem": float(p_sem.mean().item()),
            },
        )
