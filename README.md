# Few-Shot Photovoltaic Segmentation — Converged Clean Core (`DINOv3-SAT`)

Training-free few-shot ($K \in \{1, 3, 5\}$) photovoltaic (PV) segmentation built on frozen `DINOv3-SAT ViT-L/16`.
All disproven or zero-gain experimental branches (`FFTSoftVeto`, `HardNegH`, `CropZoom`, `PhaseBoundary`, mid-layer `Softplus` veto, and historical `iter1–4` ablations) have been pruned.

## Converged Three-Module Architecture (`pvfs/modules/`)

1. **Module ① — Multi-Scale & Mid-Layer Semantic Localization (`semantic.py`)**
   - Frozen `DINOv3-SAT ViT-L/16` features: `deep_32` (`512×512`, blocks 20–23), `deep_64` (`1024×1024`, blocks 20–23), and `mid_64` (`1024×1024`, blocks 12–15).
   - High-purity support core bank (`purity >= 0.85` for $K \ge 2$) with Top-3 foreground-vs-background matching (`top2_soft` multi-shot pooling):
     $$S_{\rm sem} = 0.75\,S_{64} + 0.25\,\operatorname{Up}(S_{32}) + 0.35\,S_{\rm mid}.$$
   - Zero-leakage support leave-one-out ($K \ge 2$) or spatial-neighbor-excluded ($K = 1$) `(bias, temp)` calibration $\rightarrow P_{\rm sem}$.

2. **Module ② — Gated Intra-Image Self-Rescue (`recovery.py`)**
   - Uses high-confidence query seeds ($P_{\rm sem} \ge 0.66$) and background anchors ($P_{\rm sem} \le 0.30$) within the query image to recover sparse, small, or intra-scene domain-shifted PV arrays $\rightarrow S_{\rm res}, P_{\rm res}$.

3. **Module ③ — Feature-Affinity Propagation & Optional Deformable Contour Refinement (`boundary.py`, `deformable_boundary.py`)**
   - `64×64` local $5\times 5$ DINO cosine-affinity random walk (2 steps, $\eta = 0.45$) + bilinear upsampling to `512×512` $\rightarrow P_{\rm coarse}, M_{\rm coarse}$.
   - Optional narrow-band multi-control-point deformable contour DP refinement (`M3_Deformable_Grad`) $\rightarrow M_{\rm final}$.

## Repository Structure

```text
pvfs/
  data.py, episodes.py, io.py        # Verified dataset & site-disjoint episode loaders
  dinov3_backbone.py, cache_builder.py # Frozen DINOv3-SAT feature extraction
  fast_bank.py                       # Pairwise Top-3 bank matching & DINO affinity ops
  metrics.py                         # Site-macro Dice, IoU, AP512, BF1@1px/2px, NegFP & bootstrap
  pv663_eval.py                      # Cross-dataset PV663 (116 query sites) streaming evaluator
  modules/
    contracts.py                     # ModelConfig & module I/O dataclasses
    semantic.py                      # Module ①
    recovery.py                      # Module ②
    boundary.py                      # Module ③
    deformable_boundary.py           # Deformable contour DP refinement
    pipeline.py                      # ThreeModulePipeline (CleanCore & B0_32_Ref)
configs/
  locked_split/                      # Locked PV-100scence split & 120 paired episodes
  pv663_split_lock.json              # Locked 5km site-disjoint PV663 split & episodes
  three_module_exp_m3_deformgrad.json# Deformable contour geometry configuration
scripts/
  run_dev_eval.py                    # 120-episode PV-100scence Dev evaluation
  run_pv663_eval.py                  # PV663 1/3/5-shot evaluation
  build_episodes.py, build_pv663_split.py
tests/                               # Unit test suite
```

## Quick Test & Usage

```bash
# Run unit test suite
python -m unittest discover -s tests -v

# Run 120-episode PV-100scence Dev evaluation from cached features
python scripts/run_dev_eval.py --cache-pt /path/to/dev_features_v1.pt --device cuda
```
