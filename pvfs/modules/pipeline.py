from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..iter5_methods import ITER5_VARIANTS, Iter5Config
from .boundary import BoundaryRefinementModule
from .contracts import ThreeModuleEpisodeOutput
from .recovery import SmallTargetRecoveryModule
from .semantic import SemanticLocalizationModule


def get_iter5_variant_config(name: str) -> Iter5Config:
    for cfg in ITER5_VARIANTS:
        if cfg.name == name:
            return cfg
    raise KeyError(f"Unknown Iter5 variant: {name}")


def get_m3_pruned_config() -> Iter5Config:
    """Return the exact locked M3_Pruned_Ref configuration from pvfs.iter5_methods."""
    return get_iter5_variant_config("M3_Pruned_Ref")


def get_b0_32_config() -> Iter5Config:
    """Return the historical repository baseline B0_32_Ref configuration."""
    return get_iter5_variant_config("B0_32_Ref")


class ThreeModulePipeline:
    """Behavior-equivalent 3-module decomposition (R) of M3_Pruned_Ref / Iter5Config.

    Pipeline flow:
      Shared Frozen DINOv3-SAT Features
        -> Module ① SemanticLocalizationModule  (S_sem, P_sem, S_mid, t_mid, bias/temp)
        -> Module ② SmallTargetRecoveryModule   (S_res, P_res, pre-propagation scores_32)
        -> Module ③ BoundaryRefinementModule    (P_coarse, M_coarse, M_final)
    """

    def __init__(
        self,
        cfg: Iter5Config,
        *,
        hard_neg_cfg: dict[str, Any] | None = None,
        crop_zoom_cfg: dict[str, Any] | None = None,
        refiner_cfg: dict[str, Any] | None = None,
        threshold: float = 0.50,
    ) -> None:
        self.cfg = cfg
        self.semantic = SemanticLocalizationModule(cfg, hard_neg_cfg=hard_neg_cfg)
        self.recovery = SmallTargetRecoveryModule(cfg, crop_zoom_cfg=crop_zoom_cfg)
        self.boundary = BoundaryRefinementModule(cfg, threshold=threshold, refiner_cfg=refiner_cfg)

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
    ) -> ThreeModuleEpisodeOutput:
        calib = self.semantic.calibrate_episode(caches, tokens_by_key, mask_by_id, support_ids)

        semantic_by_id = {}
        recovery_by_id = {}
        boundary_by_id = {}

        for qid in query_ids:
            sem_out = self.semantic.infer_query(caches, qid, support_ids, calib)
            q_tok_64 = tokens_by_key["deep_64"][qid] if (self.cfg.use_64 and "deep_64" in tokens_by_key) else None
            rec_out = self.recovery.infer_query(sem_out, calib, q_tok_64=q_tok_64)
            aff_w = aff_by_id[qid] if (aff_by_id is not None and qid in aff_by_id) else None
            rgb_q = rgb_by_id[qid] if (rgb_by_id is not None and qid in rgb_by_id) else None
            bnd_out = self.boundary.infer_query(rec_out, aff_weights=aff_w, q_tok_64=q_tok_64, rgb_512=rgb_q)

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


@torch.inference_mode()
def run_three_module_episode(
    caches: dict[str, dict],
    tokens_by_key: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    aff_by_id: dict[str, torch.Tensor] | None,
    support_ids: list[str],
    query_ids: list[str],
    cfg: Iter5Config,
    *,
    hard_neg_cfg: dict[str, Any] | None = None,
    crop_zoom_cfg: dict[str, Any] | None = None,
    refiner_cfg: dict[str, Any] | None = None,
    rgb_by_id: dict[str, np.ndarray] | None = None,
    return_structured: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | ThreeModuleEpisodeOutput:
    """Drop-in behavior-equivalent replacement for run_iter5_episode when all candidate switches are off."""
    pipe = ThreeModulePipeline(
        cfg,
        hard_neg_cfg=hard_neg_cfg,
        crop_zoom_cfg=crop_zoom_cfg,
        refiner_cfg=refiner_cfg,
    )
    out = pipe.run_episode(
        caches,
        tokens_by_key,
        mask_by_id,
        aff_by_id,
        support_ids,
        query_ids,
        rgb_by_id=rgb_by_id,
    )
    if return_structured:
        return out
    return out.to_legacy_outputs()
