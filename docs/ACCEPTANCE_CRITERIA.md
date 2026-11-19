# Acceptance Criteria & Verification Gates (Pre-Registered)

All candidate methods and ablations are evaluated on the locked 4-fold × 10-episode ($K \in \{1, 3, 5\}$) development suite at the unified pixel decision threshold `0.50` (`512×512`) and CAM ranking (`32×32` & `512×512` AP).

## 1. Metric Definitions (Separating 漏检 FN vs. 误检 FP)

### A. 漏检控制指标 (False-Negative / Recall)
- **`Pos Pixel Recall@0.5`**: Macro-average over the 34 positive sites of per-image pixel recall `TP / (TP + FN)`.
- **`Small PV Recall@0.5` (`<5%` area)**: Macro-average pixel recall over the 14 positive images with target fraction `< 0.05` (with `<1%` extreme tiny subset also logged separately).
- **`Image Det Recall` (`IoU >= 0.10`)**: Fraction of positive query images achieving image IoU $\ge 0.10$ (anti whole-image miss).

### B. 误检控制指标 (False-Positive / Suppression)
- **`Neg Site FP@0.5`**: Macro-average over the 16 pure negative sites of falsely predicted positive pixel fraction `(pred >= 0.5).mean()`.
- **`Neg Clean Rate (<1% FP)`**: Fraction of the 16 negative images whose false-positive pixel fraction stays `< 0.01`.
- **`Neg Max FP`**: Worst single-image false-positive fraction across the 16 negative images (tracks hard confounders such as `PV-93`, `PV-100` greenhouses).
- **`Pos Bg-FP@0.5` & `Pos Precision@0.5`**: Background false-positive rate `FP / (FP + TN)` and precision `TP / (TP + FP)` on the 63 positive images (tracks greenhouses/buildings adjacent to true PV).

### C. 综合分割与排序指标 (Overall Segmentation & Ranking)
- **`Pos Site Dice@0.5`** (Primary Segmentation Metric) & **`Pos Site IoU@0.5`**.
- **`Boundary F1@2px`**: Boundary contour F1 with 2-pixel (4 m) tolerance on positive images.
- **`Site Macro AP` (`512×512` & `32×32`)**: Threshold-free ranking power independent of calibration bias.

---

## 2. Pre-Registered Acceptance Gates (Must Pass to Be Accepted)

Every proposed iteration/module is judged against the prior accepted state (and Baseline `B0`) on **each $K \in \{1, 3, 5\}$**:

### Gate-FP (误检治理专项生效标准 / False-Positive Mitigation Gate)
A module claimed to mitigate **误检 (FP)** passes **Gate-FP** iff on at least two of $\{1, 3, 5\}$-shot (including $K=1$ or $K=3$):
1. `Neg Site FP@0.5` decreases by $\Delta \le -0.005$ (or $\ge 25\%$ relative reduction when control FP $< 0.02$) in **$\ge 3/4$ folds**, **OR** `Pos Precision@0.5` improves by $\ge +0.015$ in $\ge 3/4$ folds;
2. **Recall Guardrail**: `Pos Pixel Recall@0.5` does not degrade by more than `0.015` (and `Pos Site Dice@0.5` does not drop).

### Gate-FN (漏检治理专项生效标准 / False-Negative Mitigation Gate)
A module claimed to mitigate **漏检 (FN)** passes **Gate-FN** iff on at least two of $\{1, 3, 5\}$-shot (including $K=1$ or $K=3$):
1. `Pos Pixel Recall@0.5` increases by $\Delta \ge +0.020$ in **$\ge 3/4$ folds**, **OR** `Small PV Recall@0.5 (<5%)` increases by $\Delta \ge +0.030$;
2. **FP Guardrail**: `Neg Site FP@0.5` does not worsen by more than `+0.003` (and `Pos Site Dice@0.5` improves).

### Gate-Net (综合净收益验收标准 / Overall Net Acceptance)
For the integrated system (vs. Baseline `B0`):
1. `Pos Site Dice@0.5` improves by **$\Delta \ge +0.015$** across **$\ge 3/4$ folds**, with 95% paired-site bootstrap CI lower bound $> 0$ on at least $K=1$ and $K=3$;
2. `Site Macro AP` (`512×512`) improves by **$\Delta \ge +0.015$** across **$\ge 3/4$ folds**;
3. `Neg Site FP@0.5` is strictly lower than or within `+0.002` of `B0` (ideally significantly lower).

### Gate-Ablation (严格消融保留法则)
- Every individual module $M_i$ must pass **Incremental** and **Leave-One-Out** ablation on the exact same episodes. Any component whose removal causes $\Delta \text{Dice} \ge -0.002$ with no FP/FN gain is **pruned** as redundant.
