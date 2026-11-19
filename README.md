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
- `pvfs/`: Core library (`data.py`, `episodes.py`, `metrics.py`, `dinov3_backbone.py`, `baseline.py`, `fast_bank.py`, `iter1_methods.py`, `iter2_methods.py`, `iter3_methods.py`, `runner.py`).
- `scripts/`: Local verification and ablation analysis entrypoints.
- `kaggle/`: Self-contained Kaggle GPU kernel configurations and runners.
- `results/`: Compact JSON summaries, ablation tables, and diagnostic error panels pulled from Kaggle (no raw GeoTIFFs or model weights committed).
- `tests/`: Unit tests verifying zero-leakage episode construction, metric correctness, pairwise cache parity, and Iter 1–3 contracts.

## Current Verified Progress (79 Dev Images, 4 Folds × 10 Episodes; 21 Test Images Strictly Sealed)
| Stage / Method | $K=1$ Dice / AP512 / NegFP | $K=3$ Dice / AP512 / NegFP | $K=5$ Dice / AP512 / NegFP | Boundary F1@2px ($K=1/3/5$) |
| :--- | :---: | :---: | :---: | :---: |
| `Iter 0/1: B0_Contrast` (`deep_32`, Baseline) | 0.6252 / 0.7933 / 0.0589 | 0.6663 / 0.8473 / 0.0308 | 0.7267 / 0.8647 / 0.0184 | 0.2172 / 0.2326 / 0.2751 |
| `Iter 1: M1_Scale` (`deep_64+32`) | 0.6710 / 0.8613 / 0.0560 | 0.7403 / 0.9006 / 0.0367 | 0.7773 / 0.9121 / 0.0210 | 0.2981 / 0.3566 / 0.4100 |
| `Iter 1: M1_Full` (`+Core+Mid64 Veto`) | 0.6505 / 0.8594 / 0.0781 | 0.7235 / 0.9206 / **0.0269** | 0.7674 / 0.9375 / **0.0176** | 0.2862 / 0.3331 / 0.3618 |
| `Iter 2: M2_NoScale_LOO` (`+Cal28+SelfRescue`) | **0.6823** / 0.8581 / 0.0477 | **0.7571** / 0.9245 / 0.0475 | **0.8244** / 0.9389 / 0.0330 | 0.3303 / 0.4057 / **0.4945** |
| `Iter 3: M3_Plus_Affinity` (Pruned Best) | **0.6817 / 0.8729 / 0.0340** | **0.7528 / 0.9295** / 0.0370 | **0.8182 / 0.9439** / 0.0280 | **0.3439 / 0.3990 / 0.4905** |
| **Net Gain (`M3_Plus_Affinity` vs `B0`)** | **+0.0565 / +0.0797 / -42.3%** | **+0.0865 / +0.0822** / +0.0061 | **+0.0915 / +0.0792** / +0.0096 | **+0.1267 / +0.1664 / +0.2154** |

See [ITERATION_LOG.md](docs/ITERATION_LOG.md) for full LOO ablations, pruned modules, and visual diagnostic notes.
