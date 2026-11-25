from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion
import torch

from ..metrics import DISK_8
from .boundary import BoundaryRefinementModule
from .contracts import ModelConfig, ThreeModuleEpisodeOutput
from .deformable_boundary import DeformableBoundaryConfig
from .pbdc_boundary import PBDCConfig
from .recovery import SmallTargetRecoveryModule
from .semantic import SemanticLocalizationModule


def get_clean_core_config() -> ModelConfig:
    """Return the converged clean 3-module configuration (dual-scale + mid-fusion + Self-Rescue + Affinity)."""
    return ModelConfig(name="CleanCore")


def get_b0_32_config() -> ModelConfig:
    """Return the single-scale 32x32 DINOv3-SAT baseline configuration (B0_32_Ref)."""
    return ModelConfig(
        name="B0_32_Ref",
        use_64=False,
        shot_mode="pooled",
        k1_use_core=False,
        kge2_use_core=False,
        calib_v2=False,
        w_fg_calib=0.50,
        bg_q_calib=0.90,
        k1_margin=0.080,
        use_mid_fusion=False,
        use_self_rescue=False,
        use_affinity=False,
    )


def calibrate_deform_p_floor_loo(
    support_ids: list[str],
    mask_by_id: dict[str, np.ndarray],
    loo_probs: dict[str, np.ndarray] | None,
) -> tuple[bool, float | None]:
    """Support-only outward semantic floor calibration for Module ③: K=1 forbid outward; K>=2 LOO missed-band q25."""
    if len(support_ids) == 1 or loo_probs is None:
        return False, None
    missed_vals, edge_vals = [], []
    for sid in support_ids:
        gt = np.asarray(mask_by_id[sid] > 0, dtype=bool)
        if not gt.any() or sid not in loo_probs:
            continue
        p_loo = loo_probs[sid]
        inner = gt & (~binary_erosion(gt, structure=DISK_8, border_value=0))
        if not inner.any():
            inner = gt
        missed = inner & (p_loo < 0.50)
        if int(missed.sum()) >= 16:
            missed_vals.append(p_loo[missed])
        edge_vals.append(p_loo[inner])
    pool = missed_vals if missed_vals else edge_vals
    if not pool:
        return True, 0.35
    q = float(np.quantile(np.concatenate(pool), 0.25))
    return True, float(np.clip(q, 0.20, 0.48))


class ThreeModulePipeline:
    """Converged 3-module training-free pipeline for few-shot PV segmentation.

    Pipeline flow:
      Shared Frozen DINOv3-SAT Features
        -> Module ① SemanticLocalizationModule  (S_sem, P_sem, bias/temp)
        -> Module ② SmallTargetRecoveryModule   (S_res, P_res, scores_32)
        -> Module ③ BoundaryRefinementModule    (P_coarse, M_coarse, M_final)
    """

    def __init__(
        self,
        cfg: ModelConfig | None = None,
        *,
        refiner_cfg: PBDCConfig | DeformableBoundaryConfig | dict[str, Any] | None = None,
        threshold: float = 0.50,
    ) -> None:
        self.cfg = cfg if cfg is not None else get_clean_core_config()
        self.semantic = SemanticLocalizationModule(self.cfg)
        self.recovery = SmallTargetRecoveryModule(self.cfg)
        self.boundary = BoundaryRefinementModule(self.cfg, threshold=threshold, refiner_cfg=refiner_cfg)

    @torch.inference_mode()
    def run_episode(
        self,
        caches: dict[str, dict],
        tokens_by_key: dict[str, dict[str, torch.Tensor]],
        mask_by_id: dict[str, np.ndarray],
        aff_by_id: dict[str, torch.Tensor] | None,
        support_ids: list[str],
        query_ids: list[str],
        *,
        rgb_by_id: dict[str, np.ndarray] | None = None,
        tau_grad_per_px: float | None = 0.20,
    ) -> ThreeModuleEpisodeOutput:
        calib = self.semantic.calibrate_episode(caches, tokens_by_key, mask_by_id, support_ids)

        allow_outward = True
        p_floor: float | None = 0.28
        if self.boundary.refiner_cfg is not None and self.boundary.refiner_cfg.enabled:
            if len(support_ids) == 1:
                allow_outward, p_floor = False, None
            else:
                loo_probs: dict[str, np.ndarray] = {}
                for sid in support_ids:
                    other_ids = [x for x in support_ids if x != sid]
                    c_loo = self.semantic.calibrate_episode(caches, tokens_by_key, mask_by_id, other_ids)
                    q_tok_s = tokens_by_key["deep_64"][sid] if (self.cfg.use_64 and "deep_64" in tokens_by_key) else None
                    sem_s = self.semantic.infer_query(caches, sid, other_ids, c_loo)
                    rec_s = self.recovery.infer_query(sem_s, c_loo, q_tok_64=q_tok_s)
                    aff_s = aff_by_id[sid] if (aff_by_id is not None and sid in aff_by_id) else None
                    bnd_s = BoundaryRefinementModule(self.cfg, threshold=self.boundary.threshold).infer_query(
                        rec_s, aff_weights=aff_s, q_tok_64=q_tok_s
                    )
                    loo_probs[sid] = bnd_s.p_coarse
                allow_outward, p_floor = calibrate_deform_p_floor_loo(support_ids, mask_by_id, loo_probs)

        semantic_by_id = {}
        recovery_by_id = {}
        boundary_by_id = {}

        for qid in query_ids:
            q_tok_64 = tokens_by_key["deep_64"][qid] if (self.cfg.use_64 and "deep_64" in tokens_by_key) else None
            sem_out = self.semantic.infer_query(caches, qid, support_ids, calib)
            rec_out = self.recovery.infer_query(sem_out, calib, q_tok_64=q_tok_64)
            aff_w = aff_by_id[qid] if (aff_by_id is not None and qid in aff_by_id) else None
            rgb_q = rgb_by_id[qid] if (rgb_by_id is not None and qid in rgb_by_id) else None
            bnd_out = self.boundary.infer_query(
                rec_out,
                aff_weights=aff_w,
                q_tok_64=q_tok_64,
                rgb_512=rgb_q,
                k_shots=len(support_ids),
                allow_outward=allow_outward,
                p_floor=p_floor,
                tau_grad_per_px=tau_grad_per_px,
            )

            semantic_by_id[qid] = sem_out
            recovery_by_id[qid] = rec_out
            boundary_by_id[qid] = bnd_out

        return ThreeModuleEpisodeOutput(
            cfg_name=self.cfg.name,
            support_ids=list(support_ids),
            query_ids=list(query_ids),
            calibration=calib,
            semantic_by_id=semantic_by_id,
            recovery_by_id=recovery_by_id,
            boundary_by_id=boundary_by_id,
        )
