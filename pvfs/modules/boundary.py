from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..iter3_methods import compute_dino_affinity_weights, propagate_affinity
from ..iter5_methods import Iter5Config
from .contracts import BoundaryQueryOutput, RecoveryQueryOutput


class BoundaryRefinementModule:
    """Module ③: Training-Free Contour Output & Optional Pixel-Level Boundary Refinement.

    Baseline behavior (exact equivalent to M3_Pruned_Ref / Iter5Config):
      1. Optional 5x5 DINO feature-affinity random walk propagation on the 64x64 grid (eta=0.45, steps=2);
      2. Bilinear upsampling (align_corners=False) to 512x512 -> continuous P_coarse;
      3. Thresholding at 0.50 -> binary M_coarse;
      4. When pixel-level refinement is disabled, M_final = M_coarse exactly (pixel-identical).
    """

    def __init__(
        self,
        cfg: Iter5Config,
        *,
        threshold: float = 0.50,
        refiner_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.threshold = float(threshold)
        self.refiner_cfg = refiner_cfg

    @torch.inference_mode()
    def infer_query(
        self,
        rec_out: RecoveryQueryOutput,
        *,
        aff_weights: torch.Tensor | None = None,
        q_tok_64: torch.Tensor | None = None,
        rgb_512: np.ndarray | None = None,
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

        refine_enabled = bool(self.refiner_cfg and self.refiner_cfg.get("enabled", False))
        # When pixel refinement is off (R baseline), M_final is strictly identical to M_coarse
        m_final = m_coarse.copy()
        diag: dict[str, Any] = {
            "affinity_applied": bool(cfg.use_64 and cfg.use_affinity),
            "pixel_refine_enabled": refine_enabled,
            "pixel_refine_applied": False,
        }
        return BoundaryQueryOutput(
            qid=rec_out.qid,
            p_coarse=p_coarse,
            m_coarse=m_coarse,
            m_final=m_final,
            diagnostics=diag,
        )
