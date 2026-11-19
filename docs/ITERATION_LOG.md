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

---

## Dev Feature Cache & Multi-Layer/RoPE-1024 Probe (`COMPLETED` on Kaggle T4)
- **Kaggle Kernel**: `zh0416/pv-2m-dinov3-sat-dev-cache` (`KernelWorkerStatus.COMPLETE`, CUDA Tesla T4, 578s runtime, `79` dev images ONLY; `21` sealed test images untouched).
- **Artifact**: `/kaggle/working/dev_features_v1.pt` (`1.852 GiB`, normalized fp16 `early_32 [8..11]`, `mid_32 [12..15]`, `semi_32 [16..19]`, `deep_32 [20..23]`, `mid_64 [12..15 @1024]`, `deep_64 [20..23 @1024]`), [`results/cache_probe/layer_scale_probe.json`](../results/cache_probe/layer_scale_probe.json), [`results/cache_probe/pv-2m-dinov3-sat-dev-cache.log`](../results/cache_probe/pv-2m-dinov3-sat-dev-cache.log).
- **Key Probe Findings**:
  - `deep_64` (`1024×1024` native RoPE, `16 m/token`) strongly outperforms `deep_32` (`512×512`): $K=1$ Dice `0.6251 → 0.6588`, AP512 `0.7933 → 0.8553`; $K=3$ Dice `0.6663 → 0.7426 (+0.0763)`, AP512 `0.8473 → 0.8944`, BF1 `0.2327 → 0.3717`; $K=5$ Dice `0.7267 → 0.7692`, AP512 `0.8647 → 0.9034`, BF1 `0.2751 → 0.4077`.
  - `mid_64` (`blocks 12..15 @1024`) alone has lower recall (`0.315–0.661`), **but cuts `PV-80.tif` adjacent aquaculture-greenhouse `PosBgFP` by 66% at $K=5$ (`0.230 → 0.078`) and `PV-21.tif` by 61% (`0.076 → 0.030`)**, confirming intermediate blocks preserve rib/material contrast lost in `20..23`.
  - However, `PV-93.tif` still produces `NegFP=0.142` in `deep_64` and `0.190` in `mid_64` ($K=5$), and `PV-73.tif` pond PV still sits at `Rec=0.038–0.177`.

---

## Iteration 1 — Multi-Scale RoPE Fusion + Interior Core Balanced Bank + Mid-Layer Material Veto (`COMPLETED` on Kaggle T4)
- **Kaggle Kernel**: `zh0416/pv-2m-dinov3-sat-iter1` (`KernelWorkerStatus.COMPLETE`, CUDA Tesla T4, 1015s runtime).
- **Results artifact**: [`results/iter1/iter1_summary.json`](../results/iter1/iter1_summary.json), [`results/iter1/pv-2m-dinov3-sat-iter1.log`](../results/iter1/pv-2m-dinov3-sat-iter1.log), [`scripts/analyze_iter1.py`](../scripts/analyze_iter1.py), [`pvfs/iter1_methods.py`](../pvfs/iter1_methods.py).
- **Ablation Variants**:
  - `B0_32_Ref`: `512×512` `deep_32`, purity `≥0.50`, pooled bank.
  - `M1_Scale`: `0.70*deep_64 + 0.30*deep_32` dual-RoPE score fusion.
  - `M1_Scale_Core`: `+ core_purity≥0.85` interior PV tokens + balanced per-shot pooling (`0.7*max + 0.3*mean` for $K \ge 2$).
  - `M1_Scale_Core_Calib`: `+ Calib_v1` (mean per-shot fg median + bg $q_{95}$ for $K\ge2$; bg $q_{95}+0.075$ for $K=1$).
  - `M1_Full`: `+ mid_64` support-background asymmetric veto ($\lambda_{\text{mid}}=0.35$, $\beta_{\text{veto}}=0.40$).
  - `M1_NoCore_LOO`: `M1_Full` without interior-core & balanced pooling (strict leave-one-module-out check).

### Quantitative Iteration 1 Ablation Table (Dev 79 images, 4 Folds × 10 Episodes)
| Variant | $K$ | Dice@0.5 | IoU@0.5 | AP512 | Recall | Prec | PosBgFP | BF1@2px | Small `<5%` D / R | NegFP | NegClean | `PV-93` / `PV-80bg` / `PV-73rec` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_32_Ref` | 1 | 0.6251 | 0.4977 | 0.7933 | 0.7132 | 0.6682 | 0.0893 | 0.2172 | 0.4583 / 0.6774 | 0.0589 | 0.5188 | 0.208 / 0.286 / 0.090 |
| `M1_Scale` | 1 | **0.6710** | **0.5572** | **0.8613** | 0.7529 | 0.7188 | 0.0790 | **0.2981** | **0.5223 / 0.7169** | 0.0560 | 0.5813 | 0.183 / 0.297 / 0.170 |
| `M1_Scale_Core` | 1 | 0.6480 | 0.5276 | 0.8591 | **0.8400** | 0.6237 | 0.1486 | 0.2790 | 0.4516 / 0.8269 | 0.1224 | 0.3625 | 0.242 / 0.411 / 0.451 |
| `M1_Scale_Core_Calib` | 1 | 0.6518 | 0.5351 | 0.8628 | 0.7742 | 0.6922 | 0.1027 | 0.2905 | 0.4836 / 0.7571 | 0.0807 | 0.4875 | 0.206 / 0.322 / 0.376 |
| `M1_Full` | 1 | 0.6505 | 0.5337 | 0.8594 | 0.7743 | 0.6876 | 0.1021 | 0.2862 | 0.4852 / 0.7473 | 0.0781 | 0.4625 | 0.217 / 0.308 / 0.421 |
| `M1_NoCore_LOO` | 1 | 0.6492 | 0.5410 | 0.8568 | 0.6706 | **0.7636** | **0.0467** | 0.2875 | 0.5222 / 0.6072 | **0.0306** | **0.6937** | **0.160 / 0.186** / 0.117 |
| `B0_32_Ref` | 3 | 0.6663 | 0.5438 | 0.8473 | 0.6715 | 0.7973 | 0.0476 | 0.2327 | 0.5438 / 0.6099 | 0.0308 | 0.7563 | 0.128 / 0.205 / 0.021 |
| `M1_Scale` | 3 | 0.7403 | 0.6331 | 0.9006 | **0.7546** | 0.8187 | 0.0430 | 0.3566 | 0.6052 / 0.6687 | 0.0367 | 0.7688 | 0.152 / 0.234 / 0.032 |
| `M1_Scale_Core` | 3 | **0.7561** | **0.6510** | 0.9156 | 0.7318 | 0.8651 | 0.0260 | **0.3653** | **0.6206 / 0.6526** | 0.0335 | 0.7937 | 0.159 / 0.229 / 0.017 |
| `M1_Scale_Core_Calib` | 3 | 0.7117 | 0.6101 | 0.9170 | 0.6731 | 0.8607 | 0.0203 | 0.3300 | 0.5924 / 0.5949 | **0.0267** | 0.8187 | **0.119** / 0.205 / 0.002 |
| `M1_Full` | 3 | 0.7235 | 0.6202 | **0.9206** | 0.6852 | **0.8789** | **0.0202** | 0.3331 | 0.6041 / 0.6020 | 0.0269 | **0.8250** | 0.143 / **0.182** / 0.007 |
| `M1_NoCore_LOO` | 3 | 0.7106 | 0.6047 | 0.9028 | 0.7129 | 0.8341 | 0.0361 | 0.3278 | 0.5909 / 0.6261 | 0.0314 | 0.8125 | 0.136 / 0.211 / 0.003 |
| `B0_32_Ref` | 5 | 0.7267 | 0.6052 | 0.8647 | 0.7165 | 0.8161 | 0.0402 | 0.2751 | 0.6265 / 0.6447 | 0.0184 | 0.8125 | 0.166 / 0.261 / 0.071 |
| `M1_Scale` | 5 | **0.7773** | **0.6770** | 0.9121 | **0.7737** | 0.8437 | 0.0317 | **0.4100** | **0.6643 / 0.6846** | 0.0210 | 0.8125 | 0.166 / 0.254 / 0.098 |
| `M1_Scale_Core` | 5 | 0.7739 | 0.6729 | 0.9286 | 0.7289 | 0.8992 | 0.0187 | 0.3843 | 0.6484 / 0.6363 | 0.0197 | 0.8313 | 0.173 / 0.240 / 0.089 |
| `M1_Scale_Core_Calib` | 5 | 0.7606 | 0.6562 | 0.9300 | 0.7045 | 0.8972 | 0.0165 | 0.3516 | 0.6369 / 0.6070 | 0.0183 | 0.8438 | 0.171 / 0.233 / 0.089 |
| `M1_Full` | 5 | 0.7674 | 0.6648 | **0.9375** | 0.7102 | **0.9137** | **0.0141** | 0.3618 | 0.6470 / 0.6153 | 0.0176 | **0.8500** | 0.187 / **0.180** / 0.094 |
| `M1_NoCore_LOO` | 5 | 0.7699 | 0.6685 | 0.9168 | 0.7451 | 0.8697 | 0.0240 | 0.3902 | 0.6549 / 0.6464 | **0.0170** | 0.8312 | 0.181 / 0.187 / 0.100 |

### Paired Acceptance Gate Verdicts & Critical Diagnostic Takeaways
1. **`M1_Scale` vs `B0_32_Ref` PASSES robustly**: `K=1` passes `Gate_FP=True, Gate_FN=True, Gate_Net=True` ($\Delta\text{Dice}=+0.0459$ `[0.0310, 0.0637]`, 4/4 folds; $\Delta\text{AP}_{512}=+0.0680$, 4/4 folds); `K=3` gains $\Delta\text{Dice}=+0.0740$ `[0.0522, 0.0971]`, $\Delta\text{AP}_{512}=+0.0533$ (4/4 folds); `K=5` gains $\Delta\text{Dice}=+0.0507$ `[0.0239, 0.0725]`, $\Delta\text{AP}_{512}=+0.0474$ (4/4 folds).
2. **`Core` + `mid_64` Veto achieve massive ranking & precision gains**: Adding `Core` + `mid_64` pushes **`AP512` to `0.9206` ($K=3$) and `0.9375` ($K=5$)** (both **+0.073 over `B0_32_Ref`, 4/4 folds**!), raises `Precision` to **`0.8789` / `0.9137`**, and cuts `PosBgFP` by **>55%** (`0.0476→0.0202`, `0.0402→0.0141`). `M1_Full` beats `M1_NoCore_LOO` by `+0.0178` / `+0.0207` `AP512` across 4/4 folds (`K=3,5`) and beats `M1_Scale_Core_Calib` on both `Dice` and `AP512` (`K=3,5`).
3. **Why `Dice@0.5` dipped in `M1_Scale_Core_Calib` and `M1_Full` (PRUNED CALIBRATION BUG)**:
   - On $K \ge 2$, `Calib_v1` (`bias = 0.45*mu_fg + 0.55*mu_bg`) anchored 45% to **interior core (`purity>=0.85`) LOO medians**, which sit far above real boundary/sparse PV scores (`PV-45` had `Prec=1.000, Rec=0.405`; `PV-49` `Prec=0.994, Rec=0.385`; `PV-70` `Prec=0.898, Rec=0.402`), throwing away `+10..+20%` Recall even while `AP512` kept climbing (`0.9375`).
   - On $K=1$, `core_purity=0.85` lowered support `ex-bg` quantiles (`s_pos` became tighter to pure cores) without scaling `margin_1` or anchoring on `ex-fg`, causing `NegFP=0.0781` while `M1_NoCore_LOO` (`purity=0.50`) achieved `NegFP=0.0306` (`Prec=0.7636`).
4. **Remaining Genuine Failure Modes Targeted for Iteration 2 (confirmed visually via `results/iter0/panels/*.png` + `Iter 1` error taxonomy)**:
   - **Genuine FN**: `PV-73.tif` (pond PV, heat map already highlights target structure faintly above surrounding water, `Rec=0.007–0.094`), `PV-55.tif` (adjacent identical PV ponds missed around bright-red detected PV cores!), `PV-10.tif`, `PV-30.tif`, `PV-03.tif`, `PV-26.tif`.
   - **Genuine FP**: `PV-93.tif` (`NegFP=0.143–0.217`, `NegClean=0.0–0.1`, pale translucent greenhouse complex), `PV-100.tif` (dark framed greenhouse), `PV-96.tif` (port complex), `PV-80.tif` (`PosBgFP=0.180`, adjacent narrow aquaculture greenhouses), `PV-12.tif` (`PosBgFP` pond sheen).

---

## Probe 2 — Fast Pairwise Cache Engine & Multi-Layer Confounder Physics (`COMPLETED` on Kaggle T4)
- **Kaggle Kernel**: `zh0416/pv-2m-dinov3-sat-probe2` (`KernelWorkerStatus.COMPLETE`, CUDA Tesla T4, 310s runtime).
- **Artifact**: [`pvfs/fast_bank.py`](../pvfs/fast_bank.py) (verified `<2e-3` equivalent vs direct scoring), [`pvfs/probe2.py`](../pvfs/probe2.py), [`results/probe2/probe2_summary.json`](../results/probe2/probe2_summary.json), [`results/probe2/pv-2m-dinov3-sat-probe2.log`](../results/probe2/pv-2m-dinov3-sat-probe2.log).
- **Critical Diagnostic Discoveries**:
  1. Anchoring `K>=2` calibration on `fg_all (purity>=0.50)` + `top2_soft` pooling (`CalFix_Top2_25_75`) pushed $K=5$ Dice to `0.8124` (`BF1=0.4743`, `Rec=0.8417`) and $K=3$ to `0.7412` (`AP512=0.9226`), **but** lowering the global threshold globally also pushed `NegFP` from `0.0176 → 0.0360` ($K=5$) and `0.0269 → 0.0518` ($K=3$).
  2. Multi-layer `K=3` physics showed why naive `s_neg` scaling fails: pond PV `PV-73` has `s_neg > s_pos` (`0.681 > 0.556` in `deep_64`) due to pond-water matching support `bg_bank`, whereas 5/6 missed PV farms (`PV-55, PV-49, PV-03, PV-10, PV-30`) already contain strong local high-res seeds (`q99 = 0.383–0.431`, prominence `q99 - q50 = +0.204..+0.333`) whose remaining modules are missed simply by intra-farm style variation.
  3. Across `deep_32 → deep_64`, genuine PV increased in raw $S^+$ (`+0.020..+0.083`), whereas pale greenhouse `PV-93` decreased slightly (`0.701 → 0.691`); `PV-100` and `PV-96` had near-zero `mid_64` margin (`sp-sn = +0.008` and `+0.001`) and negative `early_32` margin (`-0.024`, `-0.019`).

---

## Iteration 2 — Shot/Calibration Fix + Scale-Sharpening + Layer Competition + Gated Self-Rescue (`COMPLETED` on Kaggle T4)
- **Kaggle Kernel**: `zh0416/pv-2m-dinov3-sat-iter2` (`KernelWorkerStatus.COMPLETE`, CUDA Tesla T4, 439s runtime for 8 variants × 120 episodes).
- **Results artifact**: [`results/iter2/iter2_summary.json`](../results/iter2/iter2_summary.json), [`results/iter2/pv-2m-dinov3-sat-iter2.log`](../results/iter2/pv-2m-dinov3-sat-iter2.log), [`scripts/analyze_iter2.py`](../scripts/analyze_iter2.py), [`pvfs/iter2_methods.py`](../pvfs/iter2_methods.py).
- **Note on `B0_32_Ref` in Iter 2**: Line 411 of `iter2_methods.py` calibrated `B0_32_Ref` (`calib_v2=False, kge2_use_core=False`) on `fg_core_masks` (`0.85`) instead of `fg_all_masks` (`0.50`), making `Iter 2 B0_32_Ref` stricter (`0.6551 / 0.6965`) at $K=3,5$; `M1_Full_Ref` matched Iter 1 `M1_Full` to `0.0001` (`0.7236 / 0.7674`), and all `M2_*` variants (`calib_v2=True`) used `fg_all_masks` as designed. Below we compare against both `M1_Full_Ref` and exact `Iter 1 B0_32_Ref` (`0.6251 / 0.6663 / 0.7267`).

### Quantitative Iteration 2 Ablation Table (Dev 79 images, 4 Folds × 10 Episodes)
| Variant | $K$ | Dice@0.5 | IoU@0.5 | AP512 | Recall | Prec | PosBgFP | BF1@2px | Small `<5%` D / R | NegFP | NegClean | `PV-93` / `PV-100` / `PV-96` / `PV-55rec` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M1_Full_Ref` | 1 | 0.6490 | 0.5319 | 0.8590 | **0.7784** | 0.6821 | 0.1057 | 0.2844 | 0.4813 / **0.7521** | 0.0813 | 0.4500 | 0.220 / 0.157 / 0.227 / **0.881** |
| `M2_ShotCal` | 1 | 0.6498 | 0.5414 | 0.8566 | 0.6752 | 0.7621 | 0.0485 | 0.2879 | 0.5211 / 0.6127 | 0.0318 | **0.6937** | 0.162 / 0.084 / 0.111 / 0.740 |
| `M2_ScaleSharp` | 1 | 0.6400 | 0.5302 | 0.8520 | 0.6446 | 0.7714 | **0.0407** | 0.2854 | 0.5231 / 0.5847 | **0.0286** | **0.6937** | **0.140** / **0.080** / **0.103** / 0.709 |
| `M2_LayerComp` | 1 | 0.6562 | 0.5452 | 0.8472 | 0.6939 | 0.7459 | 0.0550 | 0.2987 | 0.5135 / 0.6320 | 0.0402 | 0.6062 | 0.169 / 0.102 / 0.139 / 0.768 |
| `M2_Full` | 1 | 0.6774 | 0.5701 | 0.8527 | 0.7202 | 0.7507 | 0.0513 | **0.3336** | 0.5357 / 0.6796 | 0.0438 | 0.5938 | 0.179 / 0.110 / 0.142 / 0.820 |
| `M2_NoLayer_LOO` | 1 | 0.6655 | 0.5590 | **0.8589** | 0.6750 | **0.7756** | **0.0381** | 0.3233 | **0.5498** / 0.6323 | 0.0318 | 0.6875 | 0.152 / 0.089 / 0.108 / 0.776 |
| `M2_NoScale_LOO` | 1 | **0.6823** | **0.5757** | 0.8581 | 0.7470 | 0.7428 | 0.0597 | 0.3303 | 0.5356 / 0.7149 | 0.0477 | 0.5812 | 0.198 / 0.118 / 0.152 / 0.836 |
| `M1_Full_Ref` | 3 | 0.7236 | 0.6202 | 0.9206 | 0.6852 | **0.8789** | **0.0202** | 0.3331 | 0.6042 / 0.6020 | **0.0269** | **0.8250** | **0.143 / 0.149 / 0.108** / 0.463 |
| `M2_ShotCal` | 3 | 0.7353 | 0.6322 | 0.9234 | 0.7593 | 0.8251 | 0.0400 | 0.3715 | 0.5867 / 0.6847 | 0.0451 | 0.7375 | 0.177 / 0.255 / 0.208 / 0.577 |
| `M2_ScaleSharp` | 3 | 0.7366 | 0.6326 | 0.9188 | 0.7503 | 0.8333 | 0.0363 | 0.3807 | 0.5917 / 0.6750 | 0.0429 | 0.7438 | 0.165 / 0.240 / 0.202 / 0.583 |
| `M2_LayerComp` | 3 | 0.7348 | 0.6318 | 0.9172 | 0.7476 | 0.8347 | 0.0366 | 0.3759 | 0.5893 / 0.6715 | 0.0430 | 0.7438 | 0.179 / 0.236 / 0.190 / 0.564 |
| `M2_Full` | 3 | 0.7573 | 0.6580 | 0.9205 | 0.7720 | 0.8405 | 0.0342 | 0.4130 | 0.6162 / 0.7118 | 0.0456 | 0.7500 | 0.191 / 0.251 / 0.194 / 0.639 |
| `M2_NoLayer_LOO` | 3 | **0.7602** | **0.6604** | 0.9221 | 0.7756 | 0.8396 | 0.0336 | **0.4180** | **0.6177** / 0.7144 | 0.0458 | 0.7438 | 0.179 / 0.255 / 0.206 / 0.659 |
| `M2_NoScale_LOO` | 3 | 0.7571 | 0.6583 | **0.9245** | **0.7798** | 0.8348 | 0.0370 | 0.4057 | 0.6132 / **0.7220** | 0.0475 | 0.7563 | 0.206 / 0.262 / 0.196 / **0.673** |
| `M1_Full_Ref` | 5 | 0.7674 | 0.6648 | 0.9375 | 0.7102 | **0.9137** | **0.0141** | 0.3618 | 0.6470 / 0.6153 | **0.0176** | **0.8500** | **0.187 / 0.043 / 0.040** / 0.494 |
| `M2_ShotCal` | 5 | 0.8108 | 0.7157 | 0.9365 | 0.8192 | 0.8601 | 0.0318 | 0.4666 | 0.6874 / 0.7359 | 0.0314 | 0.7500 | 0.228 / 0.108 / 0.110 / 0.777 |
| `M2_ScaleSharp` | 5 | 0.8070 | 0.7105 | 0.9333 | 0.8068 | 0.8716 | 0.0291 | 0.4715 | 0.6837 / 0.7246 | 0.0306 | 0.7500 | 0.216 / 0.108 / 0.108 / 0.765 |
| `M2_LayerComp` | 5 | 0.8066 | 0.7106 | 0.9341 | 0.8059 | 0.8692 | 0.0290 | 0.4653 | 0.6830 / 0.7238 | 0.0308 | 0.7563 | 0.226 / 0.110 / 0.101 / 0.746 |
| `M2_Full` | 5 | 0.8210 | 0.7299 | 0.9362 | 0.8265 | 0.8703 | 0.0278 | 0.5001 | 0.7044 / 0.7663 | 0.0326 | 0.7438 | 0.233 / 0.124 / 0.101 / 0.837 |
| `M2_NoLayer_LOO` | 5 | 0.8207 | 0.7294 | 0.9353 | 0.8268 | 0.8729 | 0.0279 | **0.5048** | 0.7035 / 0.7661 | 0.0326 | 0.7438 | 0.226 / 0.121 / 0.108 / **0.863** |
| `M2_NoScale_LOO` | 5 | **0.8244** | **0.7347** | **0.9389** | **0.8373** | 0.8626 | 0.0300 | 0.4945 | **0.7069 / 0.7766** | 0.0330 | 0.7438 | 0.240 / 0.123 / 0.102 / 0.853 |

### Ablation Verdicts & Explicit Pruning for Iteration 3
1. **KEEP `M2_SelfRescue` (`M2_Full` vs `M2_LayerComp` — UNANIMOUS 4/4-Fold Winner on FN & Dice)**:
   - $K=1$: $\Delta\text{Dice}=+0.0212$ `[0.0159, 0.0265]` (4/4 folds), $\Delta\text{Rec}=+0.0263$, $\Delta\text{Prec}=+0.0047$, $\Delta\text{SmallR}=+0.0476$, $\Delta\text{BF1}=+0.0349$.
   - $K=3$: $\Delta\text{Dice}=+0.0224$ `[0.0161, 0.0292]` (4/4 folds), $\Delta\text{AP}_{512}=+0.0033$ (4/4 folds), $\Delta\text{SmallR}=+0.0402$, $\Delta\text{BF1}=+0.0371$, **`Gate_FN=True`**.
   - $K=5$: $\Delta\text{Dice}=+0.0145$ `[0.0097, 0.0197]` (4/4 folds), $\Delta\text{AP}_{512}=+0.0021$ (4/4 folds), $\Delta\text{SmallR}=+0.0425$, $\Delta\text{BF1}=+0.0348$, **`Gate_FN=True`**.
   - Rescues sparse/adjacent missed farms: e.g. `PV-55` Recall jumped `0.494 → 0.863` ($K=5$), `PV-49` `0.490 → 0.702`, `PV-10` `0.345 → 0.632`, `PV-30` `0.260 → 0.576`.
2. **PRUNE `M2_LayerComp` (`alpha_bg_mid=0.65` + `early_32` damp)**: Fails LOO (`M2_Full` vs `M2_NoLayer_LOO`): hurt $K=1$ `AP512` (`-0.0062`, 0/4 folds) and exploded `NegFP` (`0.0318 → 0.0438`, `-37.8%`), zero benefit at $K=3,5$.
3. **PRUNE `M2_ScaleSharp` penalty (`-0.35*relu(s32-s64)`)**: Fails LOO (`M2_Full` vs `M2_NoScale_LOO`): although it trimmed `NegFP` on $K=1$ (`-0.0040`), it hurt `AP512` (`0/4` folds on $K=5$) and `Dice` (`-0.0049` on $K=1$, `-0.0034` on $K=5$, `0/4` folds). Keep clean multi-scale fusion `0.75*s64 + 0.25*s32` instead.
4. **ONLY Remaining Blocker Keeping `Gate_FP` / `Gate_Net` False at $K=3,5$**:
   - In `M2_NoLayer_LOO` / `M2_NoScale_LOO`, `Dice` already jumped `+0.091..+0.098` and `AP512` `+0.074..+0.077` over `Iter 1 B0_32_Ref`, while `NegFP` sat at `0.0458` ($K=3$) / `0.0326` ($K=5$) vs `M1_Full` `0.0269 / 0.0176` (`87%` of all negative FPs come from `PV-93`, `PV-100`, `PV-96`).
   - Reason: lowering global `w_fg` from `0.45 → 0.28` let negative confounders (`PV-100`, `PV-96`) cross `0.50`, AND `top2_soft` without support-shot hubness debiasing amplified high-cosine hub shots.
5. **Iteration 3 Design Plan**: Combine precision-safe global threshold (`w_fg ≈ 0.36..0.38`) + **Support-Shot Hubness Debiasing (`ShotDehub`)** + **Gated Self-Rescue (`M2_SelfRescue`, recovers Recall inside true PV without lowering negative thresholds)** + **DINO `64×64` Feature-Affinity Propagation** + **`512×512` Fast RGB-Guided Boundary Alignment & Weak-Speckle Guard** (lifting `BF1@2px` and suppressing `PosBgFP`/`NegFP`).


