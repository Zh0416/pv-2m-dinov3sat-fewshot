from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..fast_bank import compute_dino_affinity_weights, propagate_affinity
from .contracts import BoundaryQueryOutput, ModelConfig, RecoveryQueryOutput
from .deformable_boundary import DeformableBoundaryConfig, refine_mask_deformable_grad
from .pbdc_boundary import PBDCConfig, refine_mask_pbdc


class BoundaryRefinementModule:
    """Module ③: Feature-Affinity Propagation & Pixel-Level Boundary Refinement (PBDC or Legacy DeformGrad)."""

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        threshold: float = 0.50,
        refiner_cfg: PBDCConfig | DeformableBoundaryConfig | dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.threshold = float(threshold)
        if isinstance(refiner_cfg, (PBDCConfig, DeformableBoundaryConfig)):
            self.refiner_cfg: PBDCConfig | DeformableBoundaryConfig | None = refiner_cfg if refiner_cfg.enabled else None
        elif isinstance(refiner_cfg, dict) and refiner_cfg.get("enabled", False):
            mode = refiner_cfg.get("mode", "pbdc")
            if mode == "pbdc":
                self.refiner_cfg = PBDCConfig(
                    **{k: v for k, v in refiner_cfg.items() if k in PBDCConfig.__dataclass_fields__}
                )
            else:
                self.refiner_cfg = DeformableBoundaryConfig(
                    **{k: v for k, v in refiner_cfg.items() if k in DeformableBoundaryConfig.__dataclass_fields__}
                )
        else:
            self.refiner_cfg = None

    @torch.inference_mode()
    def infer_query(
        self,
        rec_out: RecoveryQueryOutput,
        *,
        aff_weights: torch.Tensor | None = None,
        q_tok_64: torch.Tensor | None = None,
        rgb_512: np.ndarray | None = None,
        k_shots: int = 5,
        allow_outward: bool = True,
        p_floor: float | None = 0.28,
        tau_grad_per_px: float | None = None,
    ) -> BoundaryQueryOutput:
        """Compute Module ③ outputs (P_coarse, M_coarse, M_final)."""
        cfg = self.cfg
        p_grid = rec_out.p_res
        out_hw = rec_out.grid_hw

        if cfg.use_64 and cfg.use_affinity:
            if aff_weights is not None:
                w_aff = aff_weights
            elif q_tok_64 is not None:
                w_aff = compute_dino_affinity_weights(q_tok_64, tau=cfg.aff_tau)
            else:
                raise ValueError("Either aff_weights or q_tok_64 must be supplied when use_affinity is True")
            p_grid = propagate_affinity(
                p_grid.reshape(64, 64),
                w_aff,
                steps=cfg.aff_steps,
                eta=cfg.aff_eta,
            )

        p512 = F.interpolate(
            p_grid.reshape(1, 1, out_hw, out_hw),
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        p_coarse = p512.detach().cpu().numpy().astype(np.float32)
        m_coarse = (p_coarse >= self.threshold).astype(bool)

        if isinstance(self.refiner_cfg, PBDCConfig) and self.refiner_cfg.enabled:
            if rgb_512 is None:
                raise ValueError("rgb_512 is required when PBDC boundary refinement is enabled")
            ref = refine_mask_pbdc(
                m_coarse,
                rgb_512,
                p_coarse,
                k_shots=k_shots,
                p_floor_loo=p_floor,
                cfg=self.refiner_cfg,
            )
            m_final = ref["mask"]
            diag: dict[str, Any] = {
                "affinity_applied": bool(cfg.use_64 and cfg.use_affinity),
                "pixel_refine_mode": "pbdc",
                "pixel_refine_enabled": True,
                "pixel_refine_applied": ref["moved_loops"] > 0,
                **{k: v for k, v in ref.items() if k != "mask"},
            }
        elif isinstance(self.refiner_cfg, DeformableBoundaryConfig) and self.refiner_cfg.enabled:
            tau = float(tau_grad_per_px) if tau_grad_per_px is not None else self.refiner_cfg.fallback_tau_grad_per_px
            ref = refine_mask_deformable_grad(
                m_coarse,
                rgb_512,
                p_coarse,
                allow_outward=allow_outward,
                p_floor=p_floor,
                tau_grad_per_px=tau,
                cfg=self.refiner_cfg,
            )
            m_final = ref["mask"]
            diag = {
                "affinity_applied": bool(cfg.use_64 and cfg.use_affinity),
                "pixel_refine_mode": "deformable_grad",
                "pixel_refine_enabled": True,
                "pixel_refine_applied": ref["moved_loops"] > 0,
                **{k: v for k, v in ref.items() if k != "mask"},
            }
        else:
            m_final = m_coarse.copy()
            diag = {
                "affinity_applied": bool(cfg.use_64 and cfg.use_affinity),
                "pixel_refine_mode": "off",
                "pixel_refine_enabled": False,
                "pixel_refine_applied": False,
            }
        return BoundaryQueryOutput(
            qid=rec_out.qid,
            p_coarse=p_coarse,
            m_coarse=m_coarse,
            m_final=m_final,
            diagnostics=diag,
        )
