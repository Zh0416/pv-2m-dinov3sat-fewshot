# Module ① Support-Driven Adaptive-Band FFT Soft-Veto Dev Experiment Report

- **Date**: 2026-11-24
- **Experiment Commit**: `c4be28d` (`phase-boundary-dev`)
- **Kaggle Kernel**: `zh0416/pv-2m-dinov3-sat-m1-fft-soft-veto-cpu` (CPU, `100.3 min`, zero GPU queue)
- **Dataset & Protocol**: `PV-100scence` Dev 79 images (`configs/locked_split/`), sealed 21 test images untouched; $K \in \{1, 3, 5\}$, 4 folds × 10 episodes = **120 paired episodes**; frozen `DINOv3-SAT ViT-L/16` (`dev_features_v1.pt`); original `Self-Rescue` (Module ②) + `Affinity` (Module ③); `DeformGrad` disabled; zero fixed-band fallback.
- **Locked Hyperparameters** (`configs/m1_fft_soft_veto_locked.json`): `window_size=32`, `pv_cov_min=0.75`, `bg_cov_max=0.0`, `safe_r=[4.0, 15.5]`, `min_band_width=3.0`, `q_cand=0.85`, `min_pv_cand=min_bg_cand=16`, `num_bins=6`, `min_nonempty_bins=3`, `min_bin_total=5`, `min_bin_bg=3`, `tau=0.40`, `gamma=0.05`, `cap=0.05`.
- **Verification Smoke Checks**:
  - `R` vs legacy `run_iter5_episode` max abs diff: **`0.00e+00`**
  - Inactive `R_FFTSoftVeto` fallback vs `R_NoMidVeto` max abs diff (incl. `(b, T)`): **`0.00e+00`**
  - Note on Kaggle exit status: All 120 episodes, threshold sweeps, and `m1_fft_soft_veto_summary.json` completed 100%; the script exited only during post-save text formatting (`KeyError: 'pos_bg_fp'` in `format_m1_fft_soft_veto_report`), which has been fixed in `scripts/run_m1_fft_soft_veto_dev.py`.

---

## 1. Executive Verdict

**Verdict: Shot-dependent partial pass — NOT ready for unconditional all-shot replacement of Module ①'s mid-layer Softplus yet; clear positive signal in multi-shot (`K >= 3`), clear failure in `K = 1`.**

1. **Control Baseline Discovery (`R_NoMidVeto` vs `R`)**:
   - Simply removing the original mid-layer Softplus (`beta_veto = 0.0`, retaining `+ 0.35 * S_mid`) **significantly improves `K=1`** (`dDice = +0.0059`, 95% CI `[+0.0024, +0.0094]`, **4/4 folds**, `dAP512 = +0.0058`, `dSm<5R = +0.0122`, equal-recall NegFP flat `0.0341 → 0.0342`, equal-recall PosBgFP `-0.0014`), and is neutral at `K=3/5` (`dDice = +0.0009 / -0.0004`).
2. **Multi-Shot (`K=3` & `K=5`) — Genuine Equal-Recall FP Suppression**:
   - **`K=3` passes all criteria**: Beats both `R_NoMidVeto` and `R` on **Final Dice** (`+0.0066` vs `NoMid`, CI `[+0.0005, +0.0127]`, 3/4 folds; `+0.0075` vs `R`, CI `[+0.0013, +0.0137]`, **4/4 folds**), **Equal-Recall NegFP** (`0.0393 / 0.0389 → 0.0342`, **-13.0% vs R**), **Equal-Recall PosBgFP** (`0.0263 / 0.0259 → 0.0232`, **-11.8% vs R**), and **Small<5% Recall** (`0.6991 / 0.7035 → 0.7147`, **`+0.0156`**). The gain originates cleanly in **Module ①** (`dDice +0.0085`, `EqRec NegFP -0.0048`).
   - **`K=5` improves equal-recall FP cleanly, with slight fixed-threshold conservatism**: Equal-Recall NegFP drops `0.0273 / 0.0272 → 0.0257` (**-5.9%**), PosBgFP drops `0.0229 / 0.0234 → 0.0222`, Tiny<1% Recall rises `+0.0290`, Small<5% Recall holds (`-0.0024`). However, fixed-threshold `0.50` Recall dips `-0.0078` and Dice dips `-0.0027` vs `R` (`-0.0023` vs `NoMid`, 2/4 folds) because `(b, T)` shifts slightly conservative (`matched_thr = 0.487`) and penalizes atypical-spacing positive `PV-55`.
3. **Single-Shot (`K=1`) — Clear Failure**:
   - `R_FFTSoftVeto` **worsens NegFP across 0/4 folds** (`+0.0069` vs `R`, CI `[+0.0029, +0.0119]`; `+0.0057` vs `NoMid`, CI `[+0.0021, +0.0100]`), worsens **Equal-Recall NegFP** (`0.0373` vs `0.0341 / 0.0342`), lowers Dice (`-0.0031`) and AP512 (`-0.0017`) vs `R_NoMidVeto`, and inflates negative-site Self-Rescue triggers (`33.1% → 43.1%`).

---

## 2. Quantitative Results (120 Paired Episodes)

### 2.1 Final Output (`P_coarse`, After Self-Rescue + Affinity)

| Shot | Variant | AP512 | Dice | Recall | Prec | NegFP | PosBgFP | Sm<5% Rec | Tiny<1% Rec | EqRec(R) NegFP | EqRec(R) BgFP | EqRec(NoMid) NegFP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K=1 | `R` | 0.8728 | 0.6817 | 0.7083 | 0.7649 | 0.0341 | 0.0424 | 0.6686 | 0.5947 | 0.0341 | 0.0424 | 0.0353 |
| K=1 | `R_NoMidVeto` | **0.8787** | **0.6876** | 0.7146 | **0.7669** | 0.0354 | 0.0429 | 0.6808 | 0.5971 | 0.0342 | **0.0410** | 0.0354 |
| K=1 | `R_FFTSoftVeto` | 0.8769 | 0.6845 | **0.7270** | 0.7502 | 0.0410 | 0.0481 | **0.6963** | **0.6520** | 0.0373 | 0.0420 | 0.0385 |
| K=3 | `R` | **0.9321** | 0.7680 | 0.7648 | 0.8570 | 0.0393 | 0.0263 | 0.6991 | 0.4534 | 0.0393 | 0.0263 | 0.0395 |
| K=3 | `R_NoMidVeto` | 0.9319 | 0.7688 | 0.7662 | 0.8521 | 0.0391 | 0.0262 | 0.7035 | 0.4478 | 0.0389 | 0.0259 | 0.0391 |
| K=3 | `R_FFTSoftVeto` | 0.9316 | **0.7754** | **0.7664** | **0.8708** | **0.0344** | **0.0235** | **0.7147** | **0.4812** | **0.0342** | **0.0232** | **0.0344** |
| K=5 | `R` | **0.9450** | **0.8231** | **0.8147** | 0.8863 | 0.0273 | 0.0229 | 0.7490 | 0.5109 | 0.0273 | 0.0229 | 0.0273 |
| K=5 | `R_NoMidVeto` | 0.9440 | 0.8227 | **0.8147** | 0.8846 | 0.0272 | 0.0234 | **0.7495** | 0.5037 | 0.0272 | 0.0234 | 0.0272 |
| K=5 | `R_FFTSoftVeto` | 0.9447 | 0.8204 | 0.8069 | **0.8915** | **0.0250** | **0.0210** | 0.7466 | **0.5399** | **0.0257** | **0.0222** | **0.0257** |

### 2.2 Module ① Output (`P_sem` Upsampled to `512×512`)

| Shot | Variant | AP512 | Dice | Recall | Prec | NegFP | PosBgFP | Sm<5% Rec | Tiny<1% Rec | EqRec(R) NegFP | EqRec(R) BgFP | EqRec(NoMid) NegFP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K=1 | `R` | 0.8566 | 0.6486 | 0.6686 | **0.7661** | **0.0308** | **0.0462** | 0.6056 | 0.5165 | **0.0308** | 0.0462 | **0.0316** |
| K=1 | `R_NoMidVeto` | **0.8610** | **0.6529** | 0.6727 | 0.7636 | 0.0318 | 0.0467 | 0.6156 | 0.5247 | 0.0310 | **0.0456** | 0.0318 |
| K=1 | `R_FFTSoftVeto` | 0.8596 | 0.6510 | **0.6887** | 0.7446 | 0.0370 | 0.0536 | **0.6400** | **0.6022** | 0.0329 | 0.0473 | 0.0337 |
| K=3 | `R` | **0.9248** | 0.7376 | 0.7282 | 0.8604 | 0.0349 | 0.0279 | 0.6478 | 0.4339 | 0.0349 | 0.0279 | 0.0348 |
| K=3 | `R_NoMidVeto` | 0.9247 | 0.7376 | 0.7277 | 0.8551 | 0.0348 | 0.0277 | 0.6493 | 0.4286 | 0.0349 | 0.0278 | 0.0348 |
| K=3 | `R_FFTSoftVeto` | 0.9222 | **0.7461** | **0.7287** | **0.8668** | **0.0302** | **0.0250** | **0.6604** | **0.4732** | **0.0301** | **0.0249** | **0.0300** |
| K=5 | `R` | **0.9379** | **0.8007** | **0.7789** | 0.8901 | 0.0248 | 0.0229 | **0.6896** | 0.4892 | 0.0248 | 0.0229 | 0.0246 |
| K=5 | `R_NoMidVeto` | 0.9368 | 0.7998 | 0.7772 | 0.8910 | 0.0247 | 0.0232 | 0.6876 | 0.4806 | 0.0249 | 0.0234 | 0.0247 |
| K=5 | `R_FFTSoftVeto` | 0.9354 | 0.7956 | 0.7686 | **0.8920** | **0.0226** | **0.0208** | 0.6889 | **0.5241** | **0.0238** | **0.0223** | **0.0236** |

### 2.3 Paired Site Bootstrap Summary (Final Output)

| Comparison | Shot | dDice (95% CI) | Pos Folds | dNegFP (95% CI) | Imp Folds | dAP512 | dRec | dSm<5% Rec |
|---|---:|---|---:|---|---:|---:|---:|---:|
| `R_FFTSoftVeto` vs `R_NoMidVeto` | K=1 | `-0.0031 [-0.0091, +0.0032]` | 2/4 | `+0.0057 [+0.0021, +0.0100]` | 0/4 | -0.0017 | +0.0124 | +0.0155 |
| `R_FFTSoftVeto` vs `R_NoMidVeto` | K=3 | **`+0.0066 [+0.0005, +0.0127]`** | 3/4 | `-0.0047 [-0.0117, +0.0004]` | 3/4 | -0.0003 | +0.0002 | +0.0112 |
| `R_FFTSoftVeto` vs `R_NoMidVeto` | K=5 | `-0.0023 [-0.0086, +0.0045]` | 2/4 | `-0.0022 [-0.0062, +0.0007]` | 3/4 | +0.0007 | -0.0078 | -0.0029 |
| `R_FFTSoftVeto` vs `R` | K=1 | `+0.0028 [-0.0051, +0.0111]` | 3/4 | `+0.0069 [+0.0029, +0.0119]` | 0/4 | +0.0041 | +0.0188 | +0.0277 |
| `R_FFTSoftVeto` vs `R` | K=3 | **`+0.0075 [+0.0013, +0.0137]`** | **4/4** | **`-0.0048 [-0.0107, -0.0002]`** | 3/4 | -0.0005 | +0.0016 | +0.0156 |
| `R_FFTSoftVeto` vs `R` | K=5 | `-0.0027 [-0.0084, +0.0027]` | 2/4 | `-0.0023 [-0.0058, +0.0002]` | 3/4 | -0.0003 | -0.0078 | -0.0024 |
| `R_NoMidVeto` vs `R` | K=1 | **`+0.0059 [+0.0024, +0.0094]`** | **4/4** | `+0.0012 [-0.0003, +0.0030]` | 1/4 | +0.0058 | +0.0063 | +0.0122 |
| `R_NoMidVeto` vs `R` | K=3 | `+0.0009 [-0.0018, +0.0033]` | 3/4 | `-0.0001 [-0.0016, +0.0018]` | 3/4 | -0.0002 | +0.0014 | +0.0043 |
| `R_NoMidVeto` vs `R` | K=5 | `-0.0004 [-0.0027, +0.0017]` | 2/4 | `-0.0001 [-0.0009, +0.0007]` | 2/4 | -0.0010 | -0.0000 | +0.0004 |

---

## 3. FFT Veto & Self-Rescue Diagnostics

| Shot | BandSel Rate | Active Ep Rate | Fallback Reasons | Mean Band `[low, high]` | Query PenRate (all / cand) | PenRate (pos_q / neg_q) | MeanPen (penalized) | SelfRescue Neg Rate (`R / NoMid / FFT`) |
|---:|---:|---:|---|---|---:|---:|---:|---|
| K=1 | 92.5% (37/40) | 82.5% (33/40) | `insufficient_band_support_windows: 3`, `insufficient_pv_candidates: 4` | `[9.69, 13.77]` | 22.17% / 45.9% | 22.87% / 19.43% | 0.0428 | `36.9% / 33.1% / 43.1%` |
| K=3 | 100.0% (40/40) | 100.0% (40/40) | None | `[9.38, 13.70]` | 15.45% / 50.3% | 16.45% / 11.50% | 0.0442 | `26.9% / 24.4% / 24.4%` |
| K=5 | 100.0% (40/40) | 100.0% (40/40) | None | `[9.33, 14.30]` | 15.82% / 54.0% | 17.06% / 10.93% | 0.0434 | `21.2% / 21.2% / 20.0%` |

---

## 4. Per-Image Hard-Case Breakdown (Final Output)

| Image | Type | K=1 (`R / NoMid / FFT`) | K=3 (`R / NoMid / FFT`) | K=5 (`R / NoMid / FFT`) | Diagnostic Note |
|---|---|---|---|---|---|
| `负样本/PV-100.tif` | NegFP | `0.0940 / 0.1031 / 0.1077` | `0.2401 / 0.2519 /` **`0.2063`** | `0.1007 / 0.1052 /` **`0.0777`** | Strong multi-shot hit (`CandPenRate 72–84%`, `MeanEbg +2.67/+3.45`) |
| `负样本/PV-96.tif` | NegFP | `0.1108 / 0.1102 / 0.1158` | `0.1457 / 0.1407 /` **`0.1159`** | `0.0760 / 0.0723 / 0.0798` | Strong K=3 win (-20%); flat at K=5 |
| `负样本/PV-93.tif` | NegFP | `0.1733 / 0.1753 / 0.1724` | `0.1827 / 0.1785 / 0.1802` | `0.2266 / 0.2253 / 0.2207` | Weak effect (`MeanEbg +0.64` at K=5; striped confounder overlaps PV band) |
| `负样本/PV-86.tif` | NegFP | `0.0052 / 0.0047 / 0.0072` | `0.0231 / 0.0191 /` **`0.0126`** | `0.0177 / 0.0153 /` **`0.0065`** | Strong multi-shot cut (-45% at K=3, -63% at K=5) |
| `负样本/PV-92.tif` | NegFP | `0.0130 / 0.0140 / 0.0189` | `0.0242 / 0.0266 /` **`0.0159`** | `0.0143 / 0.0154 /` **`0.0137`** | Clean multi-shot cut (`MeanEbg +2.86/+2.90`) |
| `正样本/PV-80.tif` | Dice / BgFP | `0.6345 / 0.6273 / 0.6267` | `0.6928 / 0.6865 /` **`0.7009`** | `0.6808 / 0.6685 /` **`0.7132`** | BgFP cut `0.2306→0.2016` (K=3) and `0.2455→0.2054` (K=5)! |
| `正样本/PV-55.tif` | Dice / Rec | `0.7649 / 0.7624 / 0.7561` | `0.6239 / 0.6059 / 0.6066` | `0.8025 / 0.7976 /` `0.7436` | Main K=5 casualty (Recall `0.7990→0.7002`, atypical PV spacing) |
| `正样本/PV-49.tif` | Dice | `0.6206 / 0.6339 / 0.6045` | `0.6909 / 0.6889 / 0.6780` | `0.7675 / 0.7662 / 0.7494` | Mild recall drag (`0.6513→0.6205` at K=5) |
| `正样本/PV-73.tif` | Dice | `0.1622 / 0.1577 /` **`0.1975`** | `0.0070 / 0.0064 /` **`0.0536`** | `0.1328 / 0.1241 / 0.1110` | Tiny-target recall boosted at K=1/3, slightly lower at K=5 |

---

## 5. Mechanism Diagnosis (Why K=1 Failed While K=3/5 Worked)

1. **1-Shot `bias` Downward Shift (`Q_0.98(bg_s)`)**:
   - In 1-shot mode, `bias = Q_0.98(bg_s) + 0.06` depends solely on background tokens of the single support image. Subtracting `Delta_FFT` (`mean 0.043` on penalized BG tokens) directly pushes `Q_0.98(bg_s)` down, lowering `bias` across the entire episode. Unpenalized query tokens (including negative query images) therefore get inflated `P_sem` (`EqRec(R) thr = 0.533`, Self-Rescue negative trigger rate jumps `33.1% → 43.1%`).
2. **1-Shot Band Variance vs Multi-Shot Band Consensus**:
   - At `K=1`, a single PV array's pitch/GSD scatters selected peaks across `r ∈ {4, 7, 8, 10, 11, 12, 13, 14, 15}` and misses query PV arrays with different spacing, causing equal-recall NegFP itself to degrade (`0.0342 → 0.0373`).
   - At `K=3/5`, pooling 3–5 support scenes stabilizes the radial band (`mean [9.3, 14.0]`) and histogram counts (`100%` active episodes), producing genuine ranking separation against non-periodic/weak-periodic confounders (`PV-100`, `PV-86`, `PV-92`, `PV-80` background).
3. **Why `PV-55` (K=5) and `PV-93` Remain Hard**:
   - `PV-55` has different spatial periodicity/texture from the episode's support consensus band, so ~23% of its pixels receive penalty, pulling K=5 Recall down (`-0.0988`).
   - `PV-93` contains periodic greenhouse structures overlapping PV's radial frequency band (`MeanEbg = +0.642`), where `cap = 0.05` is too small to flip high-confidence DINO false positives.

---

## 6. Recommended Next Decision

- **Do NOT blindly replace mid-layer Softplus with `R_FFTSoftVeto` at `K=1`**.
- **Two clear, evidence-backed options to decide on**:
  1. **Option A (Shot-Gated Module ①)**: Disable mid-layer Softplus (`R_NoMidVeto`) at `K=1` (already gives `+0.0059` Dice across 4/4 folds vs `R`), and enable `R_FFTSoftVeto` only for `K >= 2` (or `K >= 3`), plus investigate why `K=5` `(b, T)` shifts slightly conservative (`thr=0.487`) and how to guard positive-query over-penalization (`PV-55`, e.g., higher `tau` or higher `q_cand` / `S_deep` confidence cap so high-confidence PV interiors are never dragged down).
  2. **Option B (Keep `R_NoMidVeto` as simpler Module ① baseline)**: Accept that frequency-only soft veto helps multi-shot non-periodic confounders (`PV-100`, `PV-80` bg, `PV-86`) by ~6–13% equal-recall FP at `K=3/5` but cannot separate periodic greenhouse confounders (`PV-93`) and risks spacing-mismatch drag on atypical PV (`PV-55`).
