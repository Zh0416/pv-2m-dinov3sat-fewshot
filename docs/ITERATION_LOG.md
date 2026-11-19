# Iteration Log

Chronological scientific record of all hypotheses, implementations, Kaggle GPU runs, strict ablations, and acceptance verdicts.

---

## Iteration 0 — Protocol Lock, Baseline `B0` & Deep Error Taxonomy (`COMPLETED` on Kaggle T4)
- **Kaggle Kernel**: `zh0416/pv-2m-dinov3-sat-iter0-baseline` (`KernelWorkerStatus.COMPLETE`, CUDA Tesla T4, 256s runtime).
- **Results artifact**: [`results/iter0/iter0_summary.json`](../results/iter0/iter0_summary.json), [`results/iter0/pv-2m-dinov3-sat-iter0-baseline.log`](../results/iter0/pv-2m-dinov3-sat-iter0-baseline.log).
- **Method**:
  - Frozen `dinov3_vitl16` (SAT-493M, commit `6876159a`), `512×512` input, arithmetic mean of normalized blocks `20..23` $\rightarrow 1024 \times 32 \times 32$, L2-normalized per token.
  - Support patch selection: foreground purity `y32 >= 0.5` (with highest-fraction fallback), pure background `y32 == 0`.
  - Query matching: Top-$m=3$ foreground cosine $S^+(u)$ (`B0_PosOnly`) and $S^+(u) - 0.5 S^-(u)$ (`B0_Contrast`).
  - Zero-leakage support calibration (`K >= 2`: LOO fg median / bg q90; `K = 1`: support `bg` ex-$3\times 3$ q90 $+ 0.08$, $T=0.06$), bilinear upsampling `32×32 -> 512×512`.

### Quantitative Baseline Table (34 Pos Sites, 16 Neg Sites, 4 Folds × 10 Episodes)
| Method | Shot $K$ | Dice@0.5 | IoU@0.5 | AP (512) | AP (32) | Recall (FN↓) | Prec | PosBgFP | BF1@2px | Small `<5%` Dice / Rec | NegFP (FP↓) | NegClean | NegMaxFP (Worst) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_PosOnly` | 1 | 0.3580 | 0.2610 | 0.7850 | 0.7545 | 0.2800 | 0.7770 | 0.0073 | 0.0822 | 0.3477 / 0.2841 | 0.0029 | 0.9375 | 0.0340 (`PV-93`) |
| `B0_Contrast` | 1 | **0.6252** | **0.4977** | **0.7933** | **0.7763** | **0.7133** | 0.6682 | 0.0893 | **0.2172** | **0.4584 / 0.6775** | 0.0589 | 0.5188 | 0.2085 (`PV-93`) |
| `B0_PosOnly` | 3 | 0.6273 | 0.4962 | 0.8034 | 0.7882 | 0.6862 | 0.7111 | 0.0784 | 0.2007 | 0.5073 / 0.6271 | 0.0376 | 0.6562 | 0.1475 (`PV-93`) |
| `B0_Contrast` | 3 | **0.6663** | **0.5438** | **0.8473** | **0.8309** | 0.6715 | **0.7973** | **0.0476** | **0.2326** | **0.5438 / 0.6099** | **0.0308** | **0.7563** | 0.1774 (`PV-100`) |
| `B0_PosOnly` | 5 | 0.6781 | 0.5452 | 0.8239 | 0.8067 | 0.7114 | 0.7356 | 0.0632 | 0.2284 | 0.5765 / 0.6418 | 0.0231 | 0.7063 | 0.1561 (`PV-93`) |
| `B0_Contrast` | 5 | **0.7267** | **0.6052** | **0.8647** | **0.8502** | **0.7165** | **0.8161** | **0.0402** | **0.2751** | **0.6265 / 0.6447** | **0.0184** | **0.8125** | 0.1660 (`PV-93`) |

- **Paired Comparison (`B0_Contrast` vs `B0_PosOnly`)**:
  - $K=1$: $\Delta\text{Dice}=+0.2672$ (4/4 folds; `B0_PosOnly` suffered severe calibration threshold shift `b=0.658` vs oracle `0.525`), $\Delta\text{AP}_{512}=+0.0083$, $\Delta\text{NegFP}=+0.0559$ (higher FP due to low $K=1$ bias `b=0.266`).
  - $K=3$: $\Delta\text{Dice}=+0.0390$ `[+0.0113, +0.0670]` (3/4 folds), $\Delta\text{AP}_{512}=+0.0440$, $\Delta\text{NegFP}=-0.0067$ (`Gate-FP=True`, `Gate-Net=True`).
  - $K=5$: $\Delta\text{Dice}=+0.0486$ `[+0.0239, +0.0748]` (4/4 folds), $\Delta\text{AP}_{512}=+0.0408$, $\Delta\text{NegFP}=-0.0048$ (`Gate-FP=True`, `Gate-Net=True`).

### Root-Cause Error Taxonomy (Directly Observed in `B0_Contrast` Panels & Per-Image Metrics)
1. **误检 (False Positives)**:
   - **FP-1 (Greenhouse Confounders, persists even at $K=5$)**: `负样本/PV-93.tif` reaches `NegFP=0.2085` ($K=1$) / `0.1660` ($K=5$, `maxP=0.905`); `负样本/PV-100.tif` reaches `0.1508` ($K=1$) / `0.1774` ($K=3$); inside positive images, adjacent aquaculture/plastic greenhouses cause huge Pos-Bg-FP (`PV-80.tif`: `bg_fp=0.2607`, `prec=0.477`; `PV-21.tif`: `bg_fp=0.1542–0.2277`, `prec=0.464–0.578`; `PV-13.tif`: `bg_fp=0.1038–0.2736`). **Cause**: Final blocks `20..23` become invariant to material reflectance/fine ribs and encode generic "periodic rectangular array" geometry, while few-shot support backgrounds rarely include greenhouses, AND `purity >= 0.5` lets mixed edge tokens leak into `fg_bank`.
   - **FP-2 (Urban/Port Regular Grids)**: `PV-96.tif` (port piers/containers: `0.1508` at $K=1$, `0.0508` at $K=5$), `PV-92.tif`, `PV-99.tif`.
   - **FP-3 (Coarse `32×32` Bleed & Boundary Blur)**: `Boundary F1@2px` is only `0.217–0.275`; tiny target `PV-12.tif` (`1.84%`) has `bg_fp=0.104–0.232` (`prec=0.08–0.10`) because `16px` (`32m`) patches bleed over fishpond dikes.
2. **漏检 (False Negatives)**:
   - **FN-1 (Sparse/Floating/Domain-Shifted PV Farms)**: `正样本/PV-73.tif` (`area=18.87%`) has **`Recall=0.090` ($K=1$), `0.021` ($K=3$), `0.071` ($K=5$)**! `正样本/PV-55.tif` (`12.83%`) has `Recall=0.282–0.344`. Both are sparse/bright-edged pond PV whose local sub-patches partially match (`max` hotspots exist) while the bulk falls below global support similarity.
   - **FN-2 (Small/Narrow/Hilly PV Diluted at `32m` Stride)**: `tiny_lt1pct` (`n=2`) gets only `Dice=0.170–0.324`; `small_lt5pct` (`n=14`) gets `Dice=0.458–0.627` (`PV-49`: `rec=0.24–0.36`, `PV-03`: `rec=0.24–0.46`, `PV-26`: `rec=0.35–0.38`). Narrow rows (`4–10 px`) are diluted inside `16×16` patches.
   - **FN-3 (Support Calibration Bias)**: At $K=3$, `OracleRaw` reaches `imgDice=0.7172` (`imgRec=0.8158`) vs deployed `B0_Contrast` `Dice=0.6663` (`Recall=0.6715`) because median `fg_loo` is dominated by large interior PV blocks.
