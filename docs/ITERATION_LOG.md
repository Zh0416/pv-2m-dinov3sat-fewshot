# Iteration Log

Chronological scientific record of all hypotheses, implementations, Kaggle GPU runs, strict ablations, and acceptance verdicts.

---

## Iteration 0 — Protocol Lock, Baseline `B0` & Deep Error Taxonomy (Pending Kaggle Run)
- **Objective**: Establish the strictly zero-leakage $K \in \{1, 3, 5\}$-shot pure frozen `DINOv3-SAT` baseline (`B0-PosOnly` & `B0-Contrast`), verify infrastructure on Kaggle GPU, and diagnose exact root causes of **漏检 (FN)** and **误检 (FP)** on 2m imagery.
- **Method**:
  - Frozen `dinov3_vitl16` (SAT-493M, commit `6876159a11b4df116f30f667f8c9888617df0751`), `512×512` input, arithmetic mean of last 4 normalized blocks (`20..23`) $\rightarrow 1024 \times 32 \times 32$, L2-normalized per token.
  - Support patch selection: foreground purity $\ge 0.5$ (with highest-fraction fallback for tiny targets), pure background `y32 == 0`.
  - Query matching: Top-$m=3$ foreground cosine $S^+(u)$ (`B0-PosOnly`) and $S^+(u) - 0.5 S^-(u)$ (`B0-Contrast`).
  - Zero-leakage support calibration:
    - $K \ge 2$: leave-one-support-image-out (`fg_loo` median & `bg_loo` 90th percentile).
    - $K = 1$: support `bg` excluding $3\times 3$ neighborhood 90th percentile $+ 0.08$, $T = 0.06$.
  - Upsampling: bilinear interpolation `32×32 -> 512×512`, evaluated at fixed `0.50`.
- **Status**: Implementing code & pushing Kaggle GPU kernel.
