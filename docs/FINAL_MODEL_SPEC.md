# 最终模型结构、原理与完整数学公式（CleanCore + PBDC, v3.1-pbdc-pure）

- **版本标签**：`v3.1-pbdc-pure`（分支 `clean-core-v1`）
- **核心特性**：全程 **Training-Free**（零可训练参数），一套锁定参数跨分辨率同时适用于 `2m`（`PV-100scence`）与 `0.8m`（`PV663`）光伏少样本分割。
- **符号约定**：
  - **动态计算量**（随 episode、查询图、连通域或局部弧段实时计算）：使用数学字母表示；
  - **固定常量与超参数**：直接写出精确数值（与代码库完全一致）。

---

## 一、整体架构

```text
输入：查询图像 I_q (512×512 RGB) + K 张支持图像与掩膜 {(I_s, M_s)}_{s=1}^K
  │
  ▼ [共享冻结骨干：DINOv3-SAT，L2 归一化特征]
  ├─ 深层 32×32 (F_32)、深层 64×64 (F_64)、中层 64×64 (F_mid)
  │
  ▼ [模块 ①：多尺度与中层语义定位 + 支持集零泄漏校准]  (pvfs/modules/semantic.py)
  ├─ 融合语义得分 S_sem (64×64) & 连续概率图 P_sem (64×64)
  │
  ▼ [模块 ②：门控图内自救小目标回捞 Self-Rescue]       (pvfs/modules/recovery.py)
  ├─ 查询图高置信种子动态重打分 → S_res (64×64) & P_res (64×64)
  │
  ▼ [模块 ③：等权特征亲和传播 + PBDC 跨分辨率边界精修]   (pvfs/modules/boundary.py, pbdc_boundary.py)
  ├─ 3A. 5×5 局部特征亲和随机游走 (0.50/0.50) + 上采样 → P_coarse (512×512) & M_coarse
  └─ 3B. PBDC（仅内部核心锚定 + 半像素非对称多尺度阶跃 + 归一化零气球力变分 DP + 8 扇区回退）→ M_final
```

---

## 二、模块 ①：多尺度与中层语义定位及支持集零泄漏校准

### 1. 单层 Top-3 前背景局部匹配
对已 L2 归一化的查询 token $q \in \mathbb{R}^C$，在第 $s$ 张支持图前景核心集合 $\mathcal{F}_s$（`K=1` 纯度 $\ge 0.50$，`K \ge 2` 纯度 $\ge 0.85$，不足 8 个回退 $0.50$）与纯背景集合 $\mathcal{B}_s$（纯度 $= 0$）上计算：

$$
r_s^{+}(q) = \frac{1}{3} \sum_{j \in \text{Top3}(\mathcal{F}_s)} \text{clip}\!\left(q^\top f_{s,j},\, -1,\, 1\right)
$$

将各支持图的前景得分降序排列为 $r_{(1)}^{+} \ge r_{(2)}^{+} \ge \dots \ge r_{(K)}^{+}$，多样本聚合（`top2_soft`）：

$$
s_{\text{pos}}(q) =
\begin{cases}
r_{(1)}^{+}(q), & K = 1 \\
0.75\, r_{(1)}^{+}(q) + 0.25\, r_{(2)}^{+}(q), & K = 2 \\
0.65\, r_{(1)}^{+}(q) + 0.25\, r_{(2)}^{+}(q) + \dfrac{0.10}{K - 2} \sum_{k=3}^{K} r_{(k)}^{+}(q), & K \ge 3
\end{cases}
$$

背景相似度 $s_{\text{neg}}(q)$ 取 $\bigcup_{s=1}^{K} \mathcal{B}_s$ 上的全局 Top-3 余弦均值，单层对比得分与多尺度融合得分为：

$$
S^{(l)}(q) = s_{\text{pos}}^{(l)}(q) - 0.50\, s_{\text{neg}}^{(l)}(q), \qquad
S_{\text{sem}} = 0.75\, S_{64} + 0.25\, \text{Up}(S_{32}) + 0.35\, S_{\text{mid}}
$$

### 2. 支持集零泄漏动态校准 $(b, T)$
- **$K \ge 2$（支持集留一法 LOO）**：取各支持图留一前景中位数的均值 $\mu_{\text{fg}}$ 与全部支持背景的 95% 分位数 $\mu_{\text{bg}} = Q_{0.95}$：

$$
b = 0.38\, \mu_{\text{fg}} + 0.62\, \mu_{\text{bg}}, \qquad
T = \max\!\left(\frac{\mu_{\text{fg}} - \mu_{\text{bg}}}{2 \ln 4},\, 0.04\right)
$$

  （若 $\mu_{\text{fg}} \le \mu_{\text{bg}} + 0.02$，回退为 $b = \mu_{\text{bg}} + 0.05,\; T = 0.06$。）

- **$K = 1$（单图空间排邻背景对比）**：排除邻域（`64×64` 为 $\pm 2$，`32×32` 为 $\pm 1$）计算背景自对比得分 $S_{\text{bg,ex}}$：

$$
b = Q_{0.95}(S_{\text{bg,ex}}) + 0.075, \qquad T = 0.06
$$

最终输出语义概率图：$P_{\text{sem}} = \sigma\!\left((S_{\text{sem}} - b) / T\right)$。

---

## 三、模块 ②：门控图内自救小目标回捞（Self-Rescue）

### 1. 三重触发门控（缺一不触发，直接透传 $S_{\text{res}} = S_{\text{sem}}$）
1. 显著度：$Q_{0.99}(S_{\text{sem}}) - Q_{0.50}(S_{\text{sem}}) \ge 0.14$；
2. 前景种子集 $\mathcal{S}_{\text{fg}} = \{i \mid P_{\text{sem}}(i) \ge 0.66\}$ 满足 $|\mathcal{S}_{\text{fg}}| \ge 4$（上限 256）；
3. 背景种子集 $\mathcal{S}_{\text{bg}} = \{i \mid P_{\text{sem}}(i) \le 0.30\}$ 满足 $|\mathcal{S}_{\text{bg}}| \ge 32$（上限 1024）。

### 2. 图内纯差值自匹配与自然中点锚定
在查询图 $F_{64}$ 上计算每个 token 与 $\mathcal{S}_{\text{fg}}, \mathcal{S}_{\text{bg}}$ 的 Top-3 余弦均值 $t_i^{+}, t_i^{-}$，定义纯前景背景差值：

$$
s_{\text{self}}(i) = t_i^{+} - t_i^{-}
$$

在 $\mathcal{S}_{\text{fg}}$ 内部排除 $\pm 2$ 空间邻域后计算自留一相似度 $t_{\text{loo}}^{+}$ 与差值 $s_{\text{loo}}$，动态确定自然中点锚点与门控下限：

$$
a_{\text{fg}} = Q_{0.25}(s_{\text{loo}}), \qquad
a_{\text{bg}} = Q_{0.95}\!\left(s_{\text{self}}(\mathcal{S}_{\text{bg}})\right), \qquad
a_{\text{mid}} = 0.50\,(a_{\text{fg}} + a_{\text{bg}})
$$

$$
\tau_{\text{sim}} = \max\!\left(0.55,\; 0.80\, Q_{0.20}(t_{\text{loo}}^{+})\right)
$$

### 3. 门控非对称增益更新

$$
\Delta_i = \text{clip}(s_{\text{self}}(i) - a_{\text{mid}},\; -0.15,\; 0.35), \qquad
g_i = \sigma\!\left(\frac{t_i^{+} - \tau_{\text{sim}}}{0.04}\right) \cdot \sigma\!\left(\frac{(t_i^{+} - t_i^{-}) - 0.05}{0.04}\right)
$$

$$
S_{\text{res}}(i) = S_{\text{sem}}(i) +
\begin{cases}
0.40\, \Delta_i\, g_i, & \Delta_i > 0 \\
0.08\, \Delta_i, & \Delta_i \le 0
\end{cases}
\qquad
P_{\text{res}} = \sigma\!\left(\frac{S_{\text{res}} - b}{T}\right)
$$

---

## 四、模块 ③：等权特征亲和传播 + `PBDC` 跨分辨率边界精修

### 3A. 局部 DINO 特征亲和随机游走（`64×64`）
在 $5 \times 5$ 邻域 $\mathcal{N}_5(i)$ 上基于 $F_{64}$ 构建归一化亲和权重：

$$
\tilde{w}_{ij} = \exp\!\left(\frac{\text{clip}(q_i^\top q_j,\, 0,\, 1) - 1}{0.10}\right) \cdot \exp\!\left(-\frac{\Delta y_{ij}^2 + \Delta x_{ij}^2}{4.5}\right), \qquad
w_{ij} = \frac{\tilde{w}_{ij}}{\sum_{k \in \mathcal{N}_5(i)} \tilde{w}_{ik}}
$$

执行 2 步标准等权阻尼随机游走（$P^{(0)} = P_{\text{res}}$）：

$$
P^{(t+1)}(i) = 0.50\, P^{(t)}(i) + 0.50 \sum_{j \in \mathcal{N}_5(i)} w_{ij}\, P^{(t)}(j), \qquad t = 0, 1
$$

双线性上采样至 `512×512` 得 $P_{\text{coarse}}$，阈值 $0.50$ 得粗掩膜 $M_{\text{coarse}}$。

### 3B. `PBDC` 跨分辨率像素级边界精修（`512×512`）
沿每个面积 $\ge 48\text{ px}$ 的连通域外轮廓按 $8\text{ px}$ 间隔（高曲率 $> 35^\circ$ 为 $6\text{ px}$，锐角 $> 65^\circ$ 锁定）布设控制点，沿外法向搜索整数位移 $d \in \{-6, \dots, +6\}\text{ px}$（相邻坡度 $|b - a| \le 3$）。

#### (1) 拮抗色空间与统一 `3:1` 仅内部核心原型
将归一化 RGB 变换为 $C_{\text{sharp}}$ 与 $3 \times 3$ 平滑图 $C_{\text{smooth}}$：

$$
C = \big[\; 0.299 R + 0.587 G + 0.114 B,\;\; 0.70(R - G),\;\; 0.70\big(B - 0.5(R + G)\big) \;\big]^\top
$$

仅从高置信内部核心 $\Omega_{\text{core}} = \{x \in M_{\text{coarse}} \mid \text{EDT}(x) \ge 2.5 \land P_{\text{coarse}}(x) \ge 0.65\}$ 鲁棒估计局部（$24\text{ px}$ 半径）与整块原型，统一以 `3:1` 融合：

$$
\mu_{\text{in}} = 0.75\, \mu_{\text{loc}} + 0.25\, \mu_{\text{comp}}, \qquad
\sigma_{\text{in}} = 0.75\, \sigma_{\text{loc}} + 0.25\, \sigma_{\text{comp}}, \qquad
\tilde{\sigma} = \sigma_{\text{in}} + 0.03
$$

#### (2) 半像素精确界面上的多尺度方向一致非对称阶跃
在真实界面 $x_{\text{iface}}(d) = x + (d + 0.5)\mathbf{n}$ 上，取窄跨距 $1.0\text{ px}$（于 $C_{\text{sharp}}$）与宽跨距 $3.0\text{ px}$（于 $C_{\text{smooth}}$）色差向量 $\delta_1, \delta_2$，及其中点 $2.0\text{ px}$ 处的内外偏离度 $\phi_{\text{in}}, \phi_{\text{out}}$：

$$
s_{\text{coh}}(d) = \frac{\sqrt{\max(0,\; \delta_1^\top \delta_2)}}{\tilde{\sigma}}, \qquad
\Delta\phi(d) = \max\!\big(0,\; \phi_{\text{out}}(d) - \phi_{\text{in}}(d)\big)
$$

$$
w_{\text{inner}}(d) = \exp\!\left(-\big[\max(0,\; \phi_{\text{in}}(d) - 1.50)\big]^2\right), \qquad
e_{\text{bnd}}(x, d) = w_{\text{inner}}(d) \cdot \tanh\!\left(0.50 \sqrt{s_{\text{coh}}(d)\,\Delta\phi(d)}\right)
$$

#### (3) 归一化零正向气球力变分效用函数与闭环 DP
对弧段 $\gamma$（含 $L_\gamma$ 个采样点），边缘净增益：

$$
\Delta E_{a,b} = E_{a,b} - E_0
$$

严格单边区域否决惩罚与语义输运阻力（统一使用中性概率 $0.50$ 与偏离阈值 $1.50$，无任何正向气球力，也无需额外的硬截断不确定带或自锁阈值）：

$$
R_{\text{veto}}(a, b) = \sum_{v \in \Omega_{\text{add}}} \max(0,\; \phi_v - 1.50) \;+\; \sum_{v \in \Omega_{\text{rem}}} 2.0\,\max(0,\; 1.25 - \phi_v)\,\max(0,\; p_v - 0.50)
$$

$$
R_{\text{sem}}(a, b) = \sum_{v \in \Omega_{\text{add}}} \max(0,\; 0.50 - p_v) \;+\; \sum_{v \in \Omega_{\text{rem}}} \max(0,\; p_v - 0.50)
$$

最终弧段效用函数（主项权重归一化为 `1.0`，`K=1` 外扩上限 $4\text{ px}$，`K \ge 2` 为 $6\text{ px}$）：

$$
U_\gamma(a, b) = \Delta E_{a,b} - 0.25\, R_{\text{veto}}(a, b) - 0.85\, R_{\text{sem}}(a, b) - 0.04\,|b - a|\,\frac{L_\gamma}{8} - 0.01\,(|a| + |b|)\,\frac{L_\gamma}{8}
$$

求解闭环精确 DP（全圈均摊净增益需 $\ge 0.04/\text{px}$），并经 **8 扇区局部拓扑守恒回退** 输出最终掩膜 $M_{\text{final}}$。
