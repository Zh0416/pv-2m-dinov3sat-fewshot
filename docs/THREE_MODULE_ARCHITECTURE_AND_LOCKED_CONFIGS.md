# M3_Pruned 三模块解耦架构（R）与单模块预锁定实验方案

> **阶段 1（第一批交付）说明**：本文档记录已完成的行为等价三模块重构 `R`、公用接口契约、本地合成与全量 120-episode 等价对拍方案，以及**在任何算法实验运行前预先锁定**的三套单模块独立实验配置。代码结构拆分本身不宣称为算法创新；三项候选改进将严格按阶段 2 独立验证、互不耦合，仅串联通过各自验收门的改动进入阶段 3。

---

## 1. 三模块解耦架构（R vs 原版基线 L）

- **主对照基线 `L`**：[pvfs/iter5_methods.py](../pvfs/iter5_methods.py) 中名称精确为 **`M3_Pruned_Ref`** 的 `Iter5Config` 及原版 `run_iter5_episode`（原文件零改动保留）；同时保留 **`B0_32_Ref`** 作为仓库历史基线。
- **公用设施（不计为算法模块）**：冻结 `DINOv3-SAT`（`ViT-L/16`, `deep_32 / deep_64` 取 blocks 20–23，`mid_64` 取 blocks 12–15）、[pvfs/cache_builder.py](../pvfs/cache_builder.py) 缓存与 [pvfs/metrics.py](../pvfs/metrics.py) 评估。
- **重构实现 `R`**：[pvfs/modules/](../pvfs/modules/__init__.py)（薄模块 + 强类型数据契约 [contracts.py](../pvfs/modules/contracts.py)）。

| 模块编号与位置 | 实现类与文件 | 输入 | 输出与职责边界 | 当前默认状态（R） |
| :--- | :--- | :--- | :--- | :--- |
| **① 定位与排伪** | `SemanticLocalizationModule` ([semantic.py](../pvfs/modules/semantic.py)) | 当前 episode 的 `support_ids` / 掩膜、支持与查询多层特征及成对 Top-K 缓存 | `SemanticCalibration`（`bias`, `temp`, `t_mid`）及每图 `S_sem`, `P_sem`, `S_mid`, `t_mid`。**不做阈值化**。 | **原版**（`0.75 S_64 + 0.25 resize(S_32) + 0.35 S_mid - 0.40 softplus(...)` + 支持留一/邻域排除校准） |
| **② 小目标补漏** | `SmallTargetRecoveryModule` ([recovery.py](../pvfs/modules/recovery.py)) | ① 的输出、查询 `deep_64`（仅裁块实验额外读原 RGB 与同支持集裁块特征/标签） | `S_res`, `P_res`（以 ① 的 `bias/temp` 转概率）、传播前 `2×2` 池化 `scores_32` 及触发/回退诊断 | **原版**（Self-Rescue：`seed_prob_thr=0.66, min_seeds=4, seed_mid_margin=-1.0, qsc_weight=0.40`） |
| **③ 免训练轮廓精修** | `BoundaryRefinementModule` ([boundary.py](../pvfs/modules/boundary.py)) | ② 的 `P_res`、查询亲和权重（或 `deep_64`）；仅像素精修开启时读查询原图 RGB | `P_coarse`（$5\times 5$ 亲和传播 $\eta=0.45$ 两步后双线性上采样 512 连续图）、`M_coarse`（$P_{\text{coarse}} \ge 0.50$）、`M_final`（关精修时 `M_final == M_coarse`） | **原版**（仅亲和传播 + 上采样；像素精修关闭） |
| **总编排器** | `ThreeModulePipeline` ([pipeline.py](../pvfs/modules/pipeline.py)) | 同 `run_iter5_episode` 协议 + 可选单模块候选配置开关 | `ThreeModuleEpisodeOutput`（含各模块中间量）及 `.to_legacy_outputs() -> (probs_512, scores_32)` | **行为等价 R** |

---

## 2. 行为等价验证（L vs R）

- **小规模全变体对拍**：[tests/test_three_module_equivalence.py](../tests/test_three_module_equivalence.py) 覆盖全部 7 个 `ITER5_VARIANTS`（含 `M3_Pruned_Ref` 与 `B0_32_Ref`）× $K \in \{1, 3, 5\}$，逐查询比对 `bias / temp / t_mid`、`S_sem / P_sem / S_mid`、`S_res / P_res`、`probs_512`、`scores_32` 及 `0.50` 掩膜，要求 **0 像素翻转、逐元素完全一致**。
- **全量 120-episode Dev 对拍入口**：[scripts/verify_three_module_equivalence.py](../scripts/verify_three_module_equivalence.py)（Kaggle 云端脚本 [kaggle/three_module_equiv/run_kaggle_equiv.py](../kaggle/three_module_equiv/run_kaggle_equiv.py)），在锁定的 79 张 Dev 图 × 4 折 × 10 episodes × $K \in \{1, 3, 5\}$ 上对 `M3_Pruned_Ref` 与 `B0_32_Ref` 逐查询核验全部中间量与最终掩膜。

---

## 3. 预先锁定的三套单模块独立实验配置（阶段 2 待逐一验证）

以下三份配置文件在运行任何阶段 2 实验前已预先锁定；每一项均在其余两个模块严格冻结为 `R` 原版的前提下单独验证：

1. **模块 ①（定位与排伪）— [configs/three_module_exp_m1_hardneg.json](../configs/three_module_exp_m1_hardneg.json)**：
   - **唯一候选改动**：按支持集光伏混淆度从纯背景筛选高混淆难负例库 $H$（每支持图上限 32 token、最小空间距 2 token、去重余弦 $\le 0.95$、最小混淆相似度 $0.30$、全库至少 12 token 才启用，否则精确退回原 $S_{\text{sem}}$），以 $m(q) = F_{64}(q) - \operatorname{TopMean}_3(\cos(q, H))$ 复核并施加受限惩罚 $S'_{\text{sem}} = S_{\text{sem}} - 0.35\,\operatorname{clip}([t_H - m(q)]_+, 0, 0.25)$，随后用含复核的支持留出分数重算 `bias/temp`。
   - **必做三组消融**：`M1_Base_R`（无复核）vs `M1_Ctrl_EqualBudgetRandomBG`（等预算普通纯背景复核）vs `M1_Cand_HardNegH`（高混淆背景 $H$ 复核）。
   - **核心门槛**：固定 0.50 复现 `Gate-FP` + **同正样本召回率诊断曲线**下 `Neg Site FP` 与 `Pos Bg-FP` 仍须真实下降（防止全局压低概率假象）。

2. **模块 ②（小目标补漏）— [configs/three_module_exp_m2_cropzoom.json](../configs/three_module_exp_m2_cropzoom.json)**：
   - **唯一候选改动**：在不依赖强前景种子的前提下，于未阈值化 $S_{\text{sem}}$ 上搜索局部相对显著度峰（$S_{\text{sem}} - \operatorname{median}_{9\times 9}(S_{\text{sem}}) \ge 0.08$ 且现有 $P_{\text{sem}} \le 0.65$，每图最多 4 窗、$128 \to 512$ 裁块放大并用共享冻结 DINOv3-SAT 重编码），以同尺度支持集正/负裁块留出标定局部 logit 并对齐写回 $\widehat{S}_{\text{crop}} = \text{bias} + \text{temp}\,\ell_{\text{crop}}$（重叠窗取最大增量，上限 $0.30$）。
   - **前置审查门（Stage 0）**：正式重编码前先做离线 GT 候选窗覆盖率审计（至少覆盖 $\ge 25\%$ 原模型完全漏掉的小组件，并报告纯负图触发率；覆盖无实际收益则停止）。
   - **必做五组消融**：`M2_NoRecovery` / `M2_SelfRescue_R` / `M2_Ctrl_FullTokenReverify`（同窗仅用全图 token 复核）/ `M2_Cand_CropZoomOnly` / `M2_Cand_SelfRescuePlusCropZoom`。

3. **模块 ③（免训练轮廓精修）— [configs/three_module_exp_m3_deformgrad.json](../configs/three_module_exp_m3_deformgrad.json)**：
   - **唯一候选改动**：保持固定上游 $P_{\text{coarse}} / M_{\text{coarse}}$，在 $\pm 8\,\text{px}$ 窄带内构建多控制点可变形轮廓 $x_i(u; a, b) = x_i^0(u) + ((1-u)a + u b)\,n_i(u)$（共享控制点位置与法向，$(0,0)$ 逐像素保留原掩膜，相邻斜率差 $\le 4\,\text{px}$），以原弧固定权重上的 Scharr 梯度增量 $G_i(a,b)-G_i(0,0)$ 结合闭合轮廓动态规划求解，并执行语义外扩护栏与拓扑/对岸/相邻组件最终校验。
   - **前置几何审查（Stage 0）**：先在合成斜边、圆角/尖角、细长阵列上验收变形自由度，并离线对比刚性平移与可变形路径在相同 GT 像素收益下的可达参考。
   - **必做三组消融**：`M3_Off_R`（关精修）vs `M3_Rigid_Grad`（旧刚性 + 同梯度）vs `M3_Deformable_Grad`（可变形 DP + 同梯度）。
   - **核心门槛**：5-shot `BF1@2px >= +0.015`、95% 站点配对 CI 下界 $> 0$、$\ge 3/4$ 折正向、`BF1@1px > 0`，且三档 $K$ 均满足 `Dice >= -0.005`、`Small<5% Rec >= -0.010`、`NegFP <= +0.002`。
