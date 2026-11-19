# 2m-Resolution Few-Shot PV Segmentation with Pure Frozen DINOv3-SAT

This repository investigates how a **strictly frozen `DINOv3-SAT` (ViT-L/16, SAT-493M)** backbone can accurately recognize and segment **photovoltaic (PV) arrays** on **2m-resolution satellite imagery** under **$K \in \{1, 3, 5\}$-shot** supervision, with an explicit focus on solving **false negatives (漏检)** and **false positives (误检)** (especially agricultural/aquaculture greenhouses, bright industrial metal roofs, regular fishponds, and boundary bleed).

## Research Principles
1. **Pure Frozen DINOv3-SAT Backbone**: No auxiliary multimodal encoders (e.g., GeoRSCLIP) and no multi-hundred-thousand-parameter query-set-trained decoders. Every method operates strictly on the $K \in \{1, 3, 5\}$ labeled support images per episode.
2. **Zero Data Leakage**:
   - The 100-image `zh0416/pv-100scence` dataset (`512×512` RGB GeoTIFF, ~1.54–2.49 m/pixel) uses the locked, manually QC-verified 5 km site-grouped split: **79 development images** (63 positive, 16 negative, 47 sites across 4 folds) and **21 sealed test images** (17 positive, 4 negative, 19 sites).
   - **Test images remain strictly sealed** across all development iterations and ablations.
   - All $K$-shot support episodes are deterministic, shared identically across all methods/ablations, and strictly site-disjoint from the query fold (`episodes_lock.json`).
3. **Iterative, Hypothesis-Driven Engineering**:
   - Every iteration documents its hypothesis, exact mathematical formulation, Kaggle GPU execution record, quantitative ablation results, and explicit Pass/Fail verdict against [ACCEPTANCE_CRITERIA.md](docs/ACCEPTANCE_CRITERIA.md) in [ITERATION_LOG.md](docs/ITERATION_LOG.md).

## Repository Layout
- `configs/locked_split/`: Locked `dataset_manifest.json`, `split_lock.json`, `QC.confirmed-20261117.json`, and `episodes_lock.json`.
- `docs/`: [PROTOCOL.md](docs/PROTOCOL.md), [ACCEPTANCE_CRITERIA.md](docs/ACCEPTANCE_CRITERIA.md), [ITERATION_LOG.md](docs/ITERATION_LOG.md), and per-iteration reports.
- `pvfs/`: Core library (`data.py`, `episodes.py`, `metrics.py`, `dinov3_backbone.py`, `baseline.py`, `runner.py`, plus iterative modules).
- `scripts/`: Local verification and Kaggle execution entrypoints.
- `kaggle/`: Self-contained Kaggle GPU kernel configurations and runners.
- `results/`: Compact JSON summaries, ablation tables, and diagnostic error reports pulled from Kaggle (no raw GeoTIFFs or model weights committed).
- `tests/`: Unit tests verifying zero-leakage episode construction, metric correctness, and module contracts.
