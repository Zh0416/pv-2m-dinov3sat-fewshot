from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ModelConfig:
    """Converged training-free 3-module configuration (zero trainable parameters, zero disproven branches)."""

    name: str = "CleanCore"
    use_64: bool = True
    shot_mode: str = "top2_soft"
    k1_use_core: bool = False
    kge2_use_core: bool = True
    calib_v2: bool = True
    w_fg_calib: float = 0.38
    bg_q_calib: float = 0.95
    k1_margin: float = 0.075
    w_s64: float = 0.75
    use_mid_fusion: bool = True
    lambda_mid: float = 0.35
    use_self_rescue: bool = True
    seed_prob_thr: float = 0.66
    min_seeds: int = 4
    min_prominence: float = 0.14
    qsc_weight: float = 0.40
    use_affinity: bool = True
    aff_steps: int = 2
    aff_eta: float = 0.45
    aff_tau: float = 0.12


@dataclass(frozen=True)
class SemanticCalibration:
    """Episode-level support calibration state produced by Module ① (Semantic Localization)."""

    k_shots: int
    use_core: bool
    core_purity: float
    use_core_calib: bool
    bias: float
    temp: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticQueryOutput:
    """Per-query continuous semantic scores and probabilities produced by Module ①."""

    qid: str
    grid_hw: int
    s_sem: torch.Tensor  # flattened [grid_hw * grid_hw] raw semantic score
    p_sem: torch.Tensor  # flattened [grid_hw * grid_hw] continuous probability (unthresholded)
    s_mid: torch.Tensor | None  # flattened [4096] mid_64 contrast score when use_64 & use_mid_fusion
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryQueryOutput:
    """Per-query recovered score/probability and pre-propagation scores_32 produced by Module ②."""

    qid: str
    grid_hw: int
    s_res: torch.Tensor  # flattened [grid_hw * grid_hw] score after Self-Rescue
    p_res: torch.Tensor  # flattened [grid_hw * grid_hw] probability under Module ①'s (bias, temp)
    scores_32: torch.Tensor  # [32, 32] pre-propagation raw score map
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryQueryOutput:
    """Per-query 512x512 probability and binary masks produced by Module ③ (Boundary)."""

    qid: str
    p_coarse: np.ndarray  # [512, 512] float32 continuous probability after affinity + bilinear upsample
    m_coarse: np.ndarray  # [512, 512] bool binary mask (p_coarse >= 0.50)
    m_final: np.ndarray  # [512, 512] bool final mask (equals m_coarse when pixel refinement is off)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreeModuleEpisodeOutput:
    """Full structured output of one episode through Modules ① -> ② -> ③."""

    cfg_name: str
    support_ids: list[str]
    query_ids: list[str]
    calibration: SemanticCalibration
    semantic_by_id: dict[str, SemanticQueryOutput]
    recovery_by_id: dict[str, RecoveryQueryOutput]
    boundary_by_id: dict[str, BoundaryQueryOutput]

    def to_legacy_outputs(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return (probs_512, scores_32) for continuous-probability evaluation."""
        probs_512 = {qid: self.boundary_by_id[qid].p_coarse for qid in self.query_ids}
        scores_32 = {
            qid: self.recovery_by_id[qid].scores_32.detach().cpu().numpy().astype(np.float32)
            for qid in self.query_ids
        }
        return probs_512, scores_32

    def coarse_masks_512(self) -> dict[str, np.ndarray]:
        return {qid: self.boundary_by_id[qid].m_coarse for qid in self.query_ids}

    def final_masks_512(self) -> dict[str, np.ndarray]:
        return {qid: self.boundary_by_id[qid].m_final for qid in self.query_ids}
