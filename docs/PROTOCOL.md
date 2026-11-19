# Experiment Protocol: 2m Few-Shot PV Segmentation (`K = 1, 3, 5`)

## 1. Dataset & Sealed-Test Isolation
- **Source**: `zh0416/pv-100scence` (`/kaggle/input/datasets/zh0416/pv-100scence/PV-多场景`), 100 paired `512×512` uint8 RGB GeoTIFFs & binary `0/1` PV masks (`~1.54–2.49 m/pixel`).
- **Locked Split** (`configs/locked_split/split_lock.json`, status `locked_after_manual_qc`):
  - **Development (79 images, 47 5 km sites)**: 63 positive (34 positive sites, including 3 mixed sites) + 16 negative (16 negative sites). Partitioned into 4 site-disjoint validation folds (`fold-0`: 20, `fold-1`: 19, `fold-2`: 20, `fold-3`: 20; 4 negatives per fold).
  - **Test (21 images, 19 5 km sites)**: 17 positive + 4 negative. **Strictly sealed**: never opened, cached, or evaluated during Iterations 0–N.

## 2. Deterministic $K$-Shot Episode Protocol (`K ∈ {1, 3, 5}`)
- For each $K \in \{1, 3, 5\}$, each outer fold $f \in \{0, 1, 2, 3\}$, and each episode $e \in \{0, \dots, 9\}$ (**40 episodes per $K$, 120 episodes total**):
  - **Support Set $\mathcal{S}_{K,f,e}$**: Sample $K$ **distinct positive 5 km sites** without replacement from fold $f$'s training pool (`partition == "dev"` and `fold != f`), using deterministic seed `20261119 + 1000 * K + 100 * f + e`, then sample 1 positive image from each chosen site.
  - **Annotation Budget**: Strictly **$K$ positive labeled images** (and their internal `mask == 0` background pixels). No external negative-only images are added in the primary $K$-shot protocol.
  - **Query Set $\mathcal{Q}_f$**: All 19–20 validation images of fold $f$. Every site in $\mathcal{Q}_f$ is strictly disjoint from every site in $\mathcal{S}_{K,f,e}$.
  - **Zero Leakage**: No trainable weights, thresholds, or calibrations may read query labels or non-support training images. All hyperparameters are locked across folds/episodes.

## 3. Backbone Contract (Pure Frozen DINOv3-SAT)
- Architecture: `dinov3_vitl16` from `facebookresearch/dinov3` pinned commit `6876159a11b4df116f30f667f8c9888617df0751`.
- Weights: `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` (SHA-256 prefix `eadcf0ff`).
- Preprocessing: RGB uint8 `/ 255.0`, normalized with SAT-493M mean `(0.430, 0.411, 0.296)` and std `(0.213, 0.156, 0.143)`, `eval()`, `requires_grad_(False)`, `float32`.

## 4. Statistical Aggregation Unit
- To avoid pseudo-replication across the $E=10$ episodes per fold, for every site $g$ we first average its metrics across the 10 episodes of fold $f$ to obtain a single independent site-level score $\bar{m}_g$.
- This yields **34 independent positive site observations** and **16 independent negative site observations** across the development set, paired identically between candidate and control methods.
- Significance intervals use 10,000-draw paired site bootstrap (`seed = 20261120`), alongside per-fold directional consistency (`positive_folds / 4`).
