from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .contracts import ModelConfig, RecoveryQueryOutput, SemanticCalibration, SemanticQueryOutput


class SmallTargetRecoveryModule:
    """Module ②: Gated Intra-Image Self-Rescue (Small / Sparse / Domain-Shifted PV Arrays)."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    @torch.inference_mode()
    def _apply_self_rescue(
        self,
        s_sem: torch.Tensor,
        p_sem: torch.Tensor,
        q_tok_64: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        cfg = self.cfg
        prominence = float((torch.quantile(s_sem, 0.99) - torch.quantile(s_sem, 0.50)).item())
        diag: dict[str, Any] = {
            "self_rescue_enabled": True,
            "self_rescue_triggered": False,
            "prominence": prominence,
            "n_fg_seeds": 0,
            "n_bg_seeds": 0,
            "fallback_reason": "none",
        }
        if prominence < cfg.min_prominence:
            diag["fallback_reason"] = "low_prominence"
            return s_sem, diag

        seed_mask = p_sem >= cfg.seed_prob_thr
        n_fg = int(seed_mask.sum().item())
        diag["n_fg_seeds"] = n_fg
        if n_fg < cfg.min_seeds:
            diag["fallback_reason"] = "insufficient_fg_seeds"
            return s_sem, diag

        bg_mask = p_sem <= 0.30
        n_bg = int(bg_mask.sum().item())
        diag["n_bg_seeds"] = n_bg
        if n_bg < 32:
            diag["fallback_reason"] = "insufficient_bg_seeds"
            return s_sem, diag

        seed_idx = torch.nonzero(seed_mask, as_tuple=False).squeeze(1)
        if seed_idx.numel() > 256:
            seed_idx = seed_idx[s_sem[seed_idx].topk(256).indices]

        bg_idx = torch.nonzero(bg_mask, as_tuple=False).squeeze(1)
        if bg_idx.numel() > 1024:
            bg_idx = bg_idx[:: max(1, bg_idx.numel() // 1024)]

        q_fg = q_tok_64[seed_idx]
        q_bg = q_tok_64[bg_idx]
        m_fg = min(3, q_fg.shape[0])
        m_bg = min(3, q_bg.shape[0])

        top_fg = (q_tok_64 @ q_fg.T).clamp(-1.0, 1.0).topk(m_fg, dim=1).values.mean(dim=1)
        top_bg = (q_tok_64 @ q_bg.T).clamp(-1.0, 1.0).topk(m_bg, dim=1).values.mean(dim=1)
        s_self = top_fg - 0.85 * top_bg

        seed_yx = torch.stack((seed_idx // 64, seed_idx % 64), dim=1)
        near_ss = ((seed_yx[:, 0:1] - seed_yx[:, 0:1].T).abs() <= 2) & ((seed_yx[:, 1:2] - seed_yx[:, 1:2].T).abs() <= 2)
        vc_ss = int((~near_ss).sum(dim=1).min().item())
        if vc_ss >= 1:
            fg_loo = (q_fg @ q_fg.T).clamp(-1.0, 1.0).masked_fill(near_ss, -2.0).topk(min(m_fg, vc_ss), dim=1).values.mean(dim=1)
            fg_bg = (q_fg @ q_bg.T).clamp(-1.0, 1.0).topk(m_bg, dim=1).values.mean(dim=1)
            fg_anchor = float(torch.quantile(fg_loo - 0.85 * fg_bg, 0.25).item())
            fg_sim_floor = max(0.55, 0.78 * float(torch.quantile(fg_loo, 0.20).item()))
        else:
            fg_anchor = float(torch.quantile(s_self[seed_idx], 0.25).item())
            fg_sim_floor = max(0.55, 0.78 * float(torch.quantile(top_fg[seed_idx], 0.20).item()))
        bg_anchor = float(torch.quantile(s_self[bg_idx], 0.95).item())
        mid_anchor = 0.45 * fg_anchor + 0.55 * bg_anchor

        delta = (s_self - mid_anchor).clamp(-0.15, 0.35)
        fg_gate = torch.sigmoid((top_fg - fg_sim_floor) / 0.04) * torch.sigmoid(((top_fg - top_bg) - 0.05) / 0.04)
        boost = torch.where(delta > 0.0, cfg.qsc_weight * delta * fg_gate, 0.20 * cfg.qsc_weight * delta)
        s_res = s_sem + boost
        diag["self_rescue_triggered"] = True
        diag["max_pos_boost"] = float(boost.max().item())
        diag["min_neg_boost"] = float(boost.min().item())
        return s_res, diag

    @torch.inference_mode()
    def infer_query(
        self,
        sem_out: SemanticQueryOutput,
        calib: SemanticCalibration,
        *,
        q_tok_64: torch.Tensor | None = None,
    ) -> RecoveryQueryOutput:
        """Compute Module ② outputs (S_res, P_res, pre-propagation scores_32)."""
        cfg = self.cfg
        if cfg.use_64 and cfg.use_self_rescue:
            if q_tok_64 is None:
                raise ValueError("q_tok_64 is required when use_64 and use_self_rescue are True")
            s_res, diag = self._apply_self_rescue(sem_out.s_sem, sem_out.p_sem, q_tok_64)
            p_res = torch.sigmoid((s_res - calib.bias) / calib.temp)
            scores_32 = F.avg_pool2d(s_res.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
        else:
            s_res = sem_out.s_sem
            p_res = sem_out.p_sem
            scores_32 = s_res.reshape(32, 32) if sem_out.grid_hw == 32 else F.avg_pool2d(s_res.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
            diag = {
                "self_rescue_enabled": False,
                "self_rescue_triggered": False,
                "fallback_reason": "self_rescue_disabled",
            }

        return RecoveryQueryOutput(
            qid=sem_out.qid,
            grid_hw=sem_out.grid_hw,
            s_res=s_res,
            p_res=p_res,
            scores_32=scores_32,
            diagnostics=diag,
        )
