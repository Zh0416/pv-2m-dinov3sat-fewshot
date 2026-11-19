# 补充研究方案 V2：面向高分辨率遥感分布式光伏少样本分割的三模块协同架构（`FreqBoundPV-FS`）

> **定位**：紧扣高分辨率（0.5 m–2 m）遥感分布式光伏少样本分割任务的核心痛点——**(1) 纯背景强干扰（大棚/工业彩钢瓦/城市屋顶）导致的红色整图误检（NegFP）**、**(2) 碎小分布式光伏阵列漏检（Small-PV FN）**、**(3) $64\times 64$ 网格上采样导致的 $8\text{ px}$ 块状边缘模糊（Boundary F1 偏低）**——在严格冻结 `DINOv3-SAT ViT-L/16` 主干、零查询泄露的前提下，将原有探索凝聚为**三大原创核心模块**，并通过严格增量/留一消融与“**有/无轻量可训练解码器**”对照实验，同时在 `PV-100scence`（Dev 79 + Sealed Test 21）与全新跨数据集 **`PV-512-MERGED-185`（185 张）** 上完成自主验收。

---

## 一、三大原创核心模块设计（紧凑三模块论文级结构）

### 模块一 (`M_A`)：多尺度光谱-核心原型匹配与中层材质约束（`MSCP`）
**针对问题**：常规 ViT 步长导致的 32 m 粗糙网格、大棚边界混杂像素污染以及高亮金属/水体误匹配。
- **核心做法**：
  1. 提取冻结 `DINOv3-SAT` 在双分辨率下的层组表征：`deep_64`（`1024×1024` RoPE，Block 20–23）、`mid_64`（`1024×1024` RoPE，Block 12–15）、`deep_32`（`512×512`，Block 20–23）。
  2. 支持集前景库按空间纯净度 $\ge 0.85$ 选取**内部核心 Token**，消除支持集边界混杂背景污染，构造 $K$-shot 前景/背景 Top-$m$ 双极对比与跨尺度融合：
     $$S_{\text{sem}}(x) = w_{64} S_{64}^{\text{deep}}(x) + (1 - w_{64}) S_{32\uparrow 64}^{\text{deep}}(x) + \lambda_{\text{mid}} S_{64}^{\text{mid}}(x) - \beta_{\text{veto}} \cdot \text{softplus}_{10}\!\left(r_{\text{mid}} - S_{64}^{\text{mid}}(x)\right)$$
  3. 仅使用当前 $K$ 张支持集做留一支持图（$K \ge 2$）或自排邻域对比（$K=1$）动态校准阈值偏置 $b_{\text{sem}}$ 与温度 $T_{\text{sem}}$，输出语义先验对数几率 $Z_{\text{sem}}(x) = (S_{\text{sem}}(x) - b_{\text{sem}})/T_{\text{sem}}$ 及语义亲和传播先验 $P_{\text{sem}}(x)$。

---

### 模块二 (`M_B`)：面向光伏阵列的显式定向频域验证与负场景门控（`OFG-PV`）
**原创核心动机**：分布式光伏组件是由硅晶串/子板与规则间距阴影组成的**高各向异性周期条带栅格结构**，在 2D 傅里叶频谱中表现为**沿特定主方向 $\theta^*$ 及其正交方向 $\theta^* + 90^\circ$ 的尖锐能量脊线**；而光滑工业彩钢屋顶缺少周期高频条带能量、水体与荒地高各向同性、普通塑料大棚频带纹理对比度更弥散或缺乏刚性细栅各向异性。`DINOv3` 的 $16\times 16$ Patch 化会抹平这类高频周期相位，因此必须引入**像素级显式频域物理特征**。

1. **局部滑动窗 2D-FFT 径向带通自适应匹配（`FFT-Band`）**  
   在 $512\times 512$ 亮度图 $L$ 上以步长 $8\text{ px}$ 提取加 Hann 窗的 $32\times 32$ 局部窗口 $W_x$，计算方差归一化 2D 傅里叶功率谱：
   $$\mathcal{P}_x(\omega_u,\omega_v) = \frac{|\mathcal{F}_{2D}(W_x)|^2}{N_w^2 \cdot \operatorname{Var}(W_x) + \epsilon}$$
   划分三个物理波长频带：低频宏观结构带 $B_1$（$r\in[1,3)$，$\lambda\approx 11\text{–}32\text{ px}$）、**光伏组串周期核心带 $B_2$（$r\in[3,9)$，$\lambda\approx 3.5\text{–}10.7\text{ px}$）**、超细微网格边缘带 $B_3$（$r\in[9,15)$，$\lambda\approx 2.1\text{–}3.5\text{ px}$），求各带能量占比 $e_k(x)$。
   - **支持集频带自适应加权**：仅利用支持集真实掩膜统计支持集前景与背景的三频带均值差 $\Delta e_k = \bar{e}_k^{\text{fg}} - \bar{e}_k^{\text{bg}}$（归一化为方向权重 $\hat{\mathbf{w}}_{\text{rad}}$），得到传感器/分辨率自适应的径向频谱匹配分 $E_{\text{fft}}^{\text{sup}}(x) = \sum_{k=1}^3 \hat{w}_{\text{rad},k} e_k(x)$（即使迁移到分辨率不同的 `PV-512-MERGED-185` 也能自动适配波长带）。

2. **频域正交方向集中度与纹理相干性（`Spectral Orth-Peakness`）**  
   对中高频域（$r\in[2.5,14]$）按角度划分为 $N_\theta = 8$ 个扇区（利用实信号功率谱中心对称性 $\theta \in [0,\pi)$），联合每对正交角度能量 $P_{\text{orth}}(m) = P(\theta_m) + P(\theta_{(m+4)\bmod 8})$，定义周期栅格方向尖峰度：
   $$C_{\text{spec}}(x) = \frac{\max_m P_{\text{orth}}(m)}{\frac{1}{4}\sum_{j=1}^4 P_{\text{orth}}(j) + \epsilon} \cdot \left(1 - \frac{H(P_\theta)}{\ln N_\theta}\right)$$
   其中 $H(P_\theta)$ 为角度功率分布熵。**仅当局部高频能量集中于单主向/正交栅格轴时 $C_{\text{spec}}$ 才显著偏高**，能直接区分整齐光伏串阵与无序城市建筑碎屑、漫反射屋顶。

3. **高分辨率多尺度正交 Gabor + 结构张量特征金字塔（$512\times 512$）**  
   在 $512\times 512$ 原图上计算 4 方向 $\times$ 3 尺度正交 Gabor 幅度响应均值与各向异性比 $\frac{G_{\max}-G_{\min}}{G_{\max}+G_{\min}+\epsilon}$、结构张量相干度 $A_{\text{coh}} = \frac{\sqrt{(J_{xx}-J_{yy})^2 + 4J_{xy}^2}}{J_{xx}+J_{yy}+\epsilon}$ 与高对比暗硅蓝-亮反射谱差特征，形成**全分辨率频域-结构特征图 $F_{\text{freq}} \in \mathbb{R}^{8\times 512\times 512}$**。

4. **双域一致性负样本门控（`Spectral-Semantic NegGate`）与小目标保护**  
   将频域可信度 $Z_{\text{freq}}(x)$（支持集背景分位与前景均值归一化）与模块一语义边距 $Z_{\text{sem}}(x)$ 联合构建**图像级与连通域级双域门控**：
   - 在候选最高响应区（Top-8/Top-12 峰值核心区，保证 `<0.5%` 极小分布式光伏不被分位点吞没）计算双域联合证据 $E_{\text{img}}(Q) = \overline{Z_{\text{sem}}}_{\text{top8}} + \gamma_f \cdot \overline{Z_{\text{freq}}}_{\text{top12}}$，若图像无稳定语义峰值或频域周期结构一致度低于支持集基准线 $\tau_{\text{scene}}$，则通过连续 Sigmoid 门控对全图弱疑似区域进行衰减，大幅扫除纯负样本的整块红色误检；
   - 对面积较小（$<5\%$）的高语义核心且具高 $C_{\text{spec}}$ 频域特征的紧凑集群保留高权重并叠加局部频域-纹理置信补偿，防止误杀小光伏阵列。

---

### 模块三 (`M_C`)：轻量级支持条件频域边界解码器（`SCFB-Decoder`，约 48K 参数）
**直接回应“是否需要引入可训练解码器”的关键科学问题**：  
单纯的 $64\times 64$ 相似图双线性上采样存在不可避免的 **$8\text{ px}$ 块状阶梯误差**，难以贴合屋顶光伏板阵列角点与组串间隙。为此我们设计极轻量级、跨传感器域不变的**支持条件频域边界残差解码器 `SCFB-Decoder`**：
- **严格零域记忆的域不变输入（共 17 通道，不直接输入绝对 RGB 颜色以防跨数据集色偏过拟合）**：
  - 模块一语义先验金字塔（6 通道）：`z_sem`、`s_64`、`s_32_up`、`s_mid`、`s_pos_64`、`s_neg_64`（保留完整语义连续梯度，避免硬阈值误伤边界正样本）；
  - 支持集归一化颜色对比 + 有符号局部高通亮度通道（2 通道）：相对支持集前景/背景原型的颜色距离对比 + 有符号高频亮度差 `(L - avg31(L))/std(L)`（为 $512\times 512$ 头部提供光伏板内外明暗极性边界）；
  - 模块二显式频域-结构金字塔（9 通道：`z_freq` + $8$ 路 $512\times 512$ FFT/Gabor/结构相干图）。
- **解耦双流轻量残差结构（约 38K 参数）**：
  $$L_{512} = Z_{\text{sem}, 512} + \alpha_{\text{res}} \cdot \text{Head}_{512}\!\big(\text{CtxNet}(\text{Sem}_{64\uparrow 512},\ \text{ColorContrast}_{512}),\ \text{FineFreqNet}(Z_{\text{sem}},\ Z_{\text{freq}},\ L_{\text{hp}},\ \text{Freq}_{512})\big)$$
  语义上下文分支（`512 -> 256 -> 128 -> 64`）仅接收语义先验与颜色对比以防止低频傅里叶纹理在 $64\times 64$ 瓶颈过拟合困难样本（如 `PV-55`）；全分辨率频域边界分支直接在 $512\times 512$ 上融合显式 FFT/Gabor/结构张量相干与有符号高通亮度边界，精细贴合光伏边沿并清扫孤立误检斑点。
- **零泄露分段元训练准则**：
  - 在 `PV-100scence` 4 折交叉验证中：第 $f$ 折的解码器**严格仅在其余 3 折训练集中随机抽取 $K\in\{1,3,5\}$-shot Episode 进行元训练**（验证折 $f$、21 张封印测试集、以及外部 `PV-512-MERGED-185` 全程 100% 不可见）；
  - 损失函数 = 保护小目标召回的 Tversky Loss ($\alpha=0.35,\beta=0.65$) + 边界 3px 带强加权 BCE（$1 + 6 \cdot \text{EdgeBand}$）+ 负样本全图与 Top-1% 难例抑制损失（`12.5%` 负样本自然配比）。

---

## 二、消融实验矩阵与预注册验收标准（`Acceptance Criteria V2`）

### A. 消融实验设计（严格统一相同的 Episode、支持集与评估脚本）
在 `PV-100scence`（Dev 79 张 4 折×10 Episode + Sealed Test 21 张×40 Episode）及跨数据集 `PV-512-MERGED-185` 上对比以下 5 组方案：
1. `Ref_M5_BalCal`：Iter 5 门控稳健基线（无频域、无解码器）
2. `Ref_M3_Pruned`：Iter 5 峰值 Dice 基线（无频域、无解码器）
3. **`V2_FreqOnly_NoDecoder`**（`M_A + M_B`）：加入模块二显式 2D-FFT/Gabor 频域验证与负门控，**不引入可训练解码器**（剥离纯频域贡献）
4. **`V2_Decoder_NoFreq_LOO`**（`M_A + M_C \ M_B`）：**保留可训练解码器，但去掉全部频域/FFT/Gabor 通道**（留一消融验证显式频域对解码器是否真正有效）
5. **`V2_Full_WithDecoder`**（`M_A + M_B + M_C`）：完整三模块架构（**直接判定：引入轻量可训练解码器相对无解码器方案是否带来跨越式综合增益**）

### B. V2 预注册验收标准（全部需在未见过数据上达标）
1. **边缘精度验收门（`Gate-Boundary-V2`，判断可训练解码器必要性）**：
   - `V2_Full_WithDecoder` 的 `Boundary F1@2px` 必须相比 `Ref_M5_BalCal` 提升 **$\Delta \text{BF1} \ge +0.080$**（至少在 3/4 折成立），且相比 `V2_FreqOnly_NoDecoder`（无解码器版）提升 **$\Delta \text{BF1} \ge +0.050$**，否则不能认定引入解码器有效。
2. **负样本误检治理门（`Gate-NegFP-V2`）**：
   - 相比 `Ref_M3_Pruned`（Dev `NegFP = 0.0340 / 0.0393 / 0.0273`），Dev `NegFP` 相对下降 **$\ge 35\%$**（目标压降至 $\le 0.022 / 0.022 / 0.015$，`NegClean Rate >= 80%`），且在 $K=1,3,5$ 均保持严于或优于 `B0 + 0.002` 与 `Ref_M5_BalCal`。
3. **小光伏防漏检与综合分割门（`Gate-FN-Net-V2`）**：
   - `Pos Site Dice@0.5` 相比 `Ref_M5_BalCal` 在至少 3/4 折上提升 $\Delta \ge +0.025$（同时追平或超越 `Ref_M3_Pruned`）；`Small <5% PV Recall@0.5` 相比 `Ref_M5_BalCal` 提升 $\Delta \ge +0.030$，确保降误检不以漏检小目标为代价。
4. **跨数据集鲁棒性门（`Gate-CrossDataset`，`PV-512-MERGED-185`）**：
   - 在完全不同的第三方测试集 **`PV-512-MERGED-185`（185 张 `.bmp` 影像）** 上测试（1）**跨域零样本支持**（支持集取自 `PV-100scence`，直接分割 185 张外部图）与（2）**域内少样本**（$K=1,3,5$ 域内支持集查询剩余集），验证 `V2_Full_WithDecoder` 相较基线保持显著的 `Dice`、`BF1@2px` 与 `AP512` 提升。
