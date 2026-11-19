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
- `pvfs/`: Core library (`data.py`, `episodes.py`, `metrics.py`, `dinov3_backbone.py`, `baseline.py`, `fast_bank.py`, `iter1_methods.py`–`iter5_methods.py`, `final_sealed_eval.py`, `runner.py`).
- `scripts/`: Local verification, diagnostic, and Kaggle result analysis entrypoints.
- `kaggle/`: Self-contained Kaggle GPU kernel configurations and runners.
- `results/`: Compact JSON summaries, ablation tables, and diagnostic error panels pulled from Kaggle (no raw GeoTIFFs or model weights committed).
- `tests/`: Unit tests verifying zero-leakage episode construction, metric correctness, pairwise cache parity, Iter 1–5 contracts, and sealed test aggregation.

## Final Verified Benchmark (Pure Frozen DINOv3-SAT, No Extra Backbone or Learned Decoder)

### 1. Development Split (`79` Images / `47` Sites, 4 Folds × 10 Episodes)
| Stage / Method | $K=1$ Dice / AP512 / NegFP | $K=3$ Dice / AP512 / NegFP | $K=5$ Dice / AP512 / NegFP | Boundary F1@2px ($K=1/3/5$) |
| :--- | :---: | :---: | :---: | :---: |
| `B0_32_Ref` (`deep_32` Baseline) | 0.6252 / 0.7933 / 0.0589 | 0.6663 / 0.8473 / 0.0308 | 0.7267 / 0.8647 / 0.0184 | 0.2172 / 0.2326 / 0.2751 |
| `M1_Full_Ref` (`+RoPE-1024+Core+MidVeto`) | 0.6490 / 0.8590 / 0.0813 | 0.7236 / 0.9206 / **0.0269** | 0.7674 / 0.9375 / **0.0176** | 0.2844 / 0.3331 / 0.3618 |
| `M5_BalCal` (**Gate-Locked Winner**) | 0.6534 / 0.8659 / **0.0293** | 0.7263 / 0.9245 / 0.0310 | 0.7848 / 0.9416 / 0.0198 | 0.3015 / 0.3578 / 0.4126 |
| `M5_TightRescue` (`+Mid-Verified Rescue`) | 0.6771 / 0.8714 / 0.0318 | 0.7552 / 0.9277 / 0.0347 | 0.8097 / **0.9455** / 0.0219 | 0.3396 / 0.4058 / 0.4745 |
| `M3_Pruned_Ref` (**Peak-Dice Mode**) | **0.6817 / 0.8729** / 0.0340 | **0.7681 / 0.9321** / 0.0393 | **0.8230** / 0.9450 / 0.0273 | **0.3439 / 0.4252 / 0.5051** |

### 2. One-Shot Sealed Test Split (`21` Unseen Images / `19` Sites × `40` Episodes)
| Variant | $K=1$ Dice / AP512 / NegFP | $K=3$ Dice / AP512 / NegFP | $K=5$ Dice / AP512 / NegFP | Boundary F1@2px ($K=1/3/5$) |
| :--- | :---: | :---: | :---: | :---: |
| `B0_32_Ref` (Baseline) | 0.5705 / 0.7431 / 0.0668 | 0.6262 / 0.8063 / 0.0125 | 0.6390 / 0.8245 / 0.0150 | 0.1967 / 0.2373 / 0.2479 |
| `M1_Full_Ref` | 0.6101 / 0.7968 / 0.0817 | 0.6700 / 0.8777 / **0.0037** | 0.6980 / 0.8989 / **0.0045** | 0.2800 / 0.2973 / 0.3178 |
| `M5_BalCal` (`-63%` NegFP) | 0.5946 / 0.8096 / **0.0247** | 0.6799 / 0.8830 / 0.0046 | 0.7136 / 0.9039 / 0.0055 | 0.2897 / 0.3401 / 0.3675 |
| `M5_TightRescue` | 0.6177 / 0.8085 / 0.0271 | 0.7117 / 0.8836 / 0.0057 | 0.7456 / **0.9054** / 0.0062 | 0.3307 / 0.4164 / 0.4461 |
| `M3_Pruned_Ref` (`+0.1438` Dice @ $K=5$) | **0.6297 / 0.8101** / 0.0283 | **0.7388 / 0.8859** / 0.0084 | **0.7828** / 0.9008 / 0.0102 | **0.3410 / 0.4444 / 0.4860** |
| **Best Gain vs `B0_32_Ref`** | **+0.0592 / +0.0670 / -63.0%** | **+0.1126 / +0.0797 / -63.2%** | **+0.1438 / +0.0809 / -63.1%** | **+0.1443 / +0.2071 / +0.2381** |

See [ITERATION_LOG.md](docs/ITERATION_LOG.md) for full incremental + Leave-One-Out (LOO) ablations (including pruned `Calib_v1`, `M2_LayerComp`, `M2_ScaleSharp`, `M3_ShotDehub`, `M3_Guided`, `M4_Whiten`, `M4_Ridge`) and visual diagnostic panels.
