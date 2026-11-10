# MTDB 架构与实现说明

**中文** | [English](ARCHITECTURE_EN.md)

> MTDB = **M**ulti-**T**imescale **D**ynamical **B**ase（多时间尺度动力学基底）。
> 本文件说明概念架构如何落到本仓库的代码里：每个模块的公式、实现位置、接口、
> 训练协议、复杂度与局限。面向的读者是想在此基础上继续训练/改架构的工程师。

---

## 0. 定位与边界

MTDB 是混合 AGI 系统中最下面的一层：**感知—直觉—记忆基底**。它：

- 用稀疏、多时间尺度、相位同步的动力学状态承载时间信息；
- 用工作记忆 / 吸引子记忆承载稳定模式；
- 用预测误差（surprise）驱动记忆写入、注意力和局部可塑性；
- 用睡眠回放完成巩固、剪枝和接口重标定；
- 向上层输出压缩表征 `z(t)` 与惊奇/置信度/记忆索引，而不是 token。

它**不是**：

- 一个能单独实现 AGI 的模型；
- 语言/逻辑/规划能力的替代品（这些仍属于上层世界模型、神经符号与智能体框架）；
- Transformer 的“更快版本”——它解决的是另一类问题：持续学习、长时程动力学、
  低功耗稀疏计算、具身 grounding、稳定记忆。

本仓库是一个**可运行的研究原型**（research prototype）：所有训练都可以在笔记本
CPU 上完成，实验规模见 `configs/` 的三个 preset。它的价值在于把概念变成可检验的
代码、指标与消融，而不是宣称已经达到 AGI。

---

## 1. 设计原则

1. **不靠参数规模堆智能**：核心单元稀疏连接（fan-in 固定），单步代价 O(N·K)，
   面向 1–3 W 级神经形态/数字混合硬件的映射。
2. **时间域承载计算**：多时间尺度状态、相位同步、吸引子动力学是记忆与“直觉”的
   载体，而不是只做前馈 token 预测。
3. **局部学习为主**：Hebb/STDP 式 eligibility + 预测误差 + 奖励三因子调制；读出层
   使用最小二乘/RLS，全程不依赖反向传播。
4. **底层不做语言逻辑**：只输出潜在状态、相位绑定、惊奇、置信度、记忆索引。
5. **接口优先**：底层潜在状态必须能被上层世界模型/符号系统消费；接口协议在
   `mtdb/core/interface.py` 中显式定义并可切片寻址。

---

## 2. 总体数据流

```text
                s(t) 感知事件/连续特征 [F]
                  │
                  ▼
        ┌───────────────────────────────┐
        │ PhaseSyncRouter               │
        │ θ_i(t+1)=θ_i+2πΔt(f_i+K_c·同步项+输入项+噪声) │
        └───────────────┬───────────────┘
                        │ θ(t), 相位簇特征 Φ(t)
                        ▼
        ┌───────────────────────────────┐
        │ MultiTimescaleDynamics        │
        │ x(t+1)=α⊙x+(1-α)⊙tanh(Wφ(x)+Us+b+η) │
        │ α_i = exp(-Δt/τ_i), τ∈{10,100,1000}ms │
        └───────────────┬───────────────┘
                        │ x(t)
             ┌──────────┴───────────┐
             ▼                      ▼
    h(t)=P x(t)（压缩潜在态）   self-prediction error
             │                      │ e(t)=||h(t)-ĥ(t)|| 归一化 → surprise
             ▼                      ▼
        ┌──────────────────────────────────────┐
        │ 记忆：WorkingMemory / AttractorMemory │
        │ 读：内容寻址 + top-k 原型检索          │
        │ 写：surprise 或 novelty 触发（冷却）   │
        └───────────────┬──────────────────────┘
                        │
                        ▼
    z(t) = [h | Φ | M_wm | M_ltm | u | Δu | (surprise, sync, confidence)]
                        │
                        ▼
        上层：世界模型 / 神经符号 / 智能体 / 读出层（ridge/RLS）
```

---

## 3. 模块与代码映射

| 概念层（用户方案） | 代码位置 | 关键类/函数 | 说明 |
|---|---|---|---|
| A. 多时间尺度动力学核心 | `mtdb/core/dynamics.py` | `MultiTimescaleDynamics` | 稀疏 fan-in、三组 τ、谱半径初始化、homeostatic bias |
| B. 相位同步与路由 | `mtdb/core/phase.py` | `PhaseSyncRouter` | Kuramoto 相位、同步簇特征、簇成员 |
| C. 工作记忆/吸引子记忆 | `mtdb/core/memory.py` | `WorkingMemory` `AttractorMemory` `SurpriseGate` | 内容寻址槽位、原型吸引子、惊奇门控 |
| D. 局部可塑性与调制 | `mtdb/core/plasticity.py` + `dynamics.plastic_update` | `ThreeFactorPlasticity` `ErrorFeedback` | eligibility/STDP、误差广播、Oja 稳定项、reward 调制 |
| E. 睡眠回放与巩固 | `mtdb/core/consolidation.py` | `ReplayBuffer` `SleepConsolidator` | 按 regime 配额的回放、原型抽象、剪枝+重归一、门控读出重拟合 |
| 接口协议 | `mtdb/core/interface.py` + `system.py` | `InterfaceSpec` `InterfacePacket` `pack_interface` `MTDBSystem` | z(t) 布局、字段切片、向下调制 |
| 无反向传播读出 | `mtdb/training/readout.py` | `RidgeReadout` `RLSReadout` | 闭式最小二乘 / 贝叶斯初始化 RLS |
| 训练三阶段 | `mtdb/training/trainer.py` | `MTDBTrainer` | stage1/stage2/sleep/recall/ablation/stress |
| 合成任务与指标 | `mtdb/tasks/synthetic.py`、`mtdb/eval/metrics.py` | `MultimodalWorld` `DelayedRecallWorld` | 预测、异常、延迟回忆、遗忘压力测试 |

---

## 4. 各模块的数学与实现细节

### 4.1 多时间尺度动力学核心（A）

用户给出的连续时间形式：

```
τ_i dx_i/dt = -x_i + Σ_j W_ij φ(x_j) + Σ_k U_ik s_k(t) + η_i(t)
```

代码里采用指数欧拉离散化（`dt_ms=10`）：

```
α_i = exp(-Δt/τ_i)
x_i(t+1) = α_i x_i(t) + (1-α_i) · tanh( drive_i(t) )
drive_i  = Σ_{j∈pre(i)} W_ij φ(x_j(t)) + Σ_k U_ik s_k(t) + b_i(t) + η_i(t)
φ = tanh（默认；`activation` 字段预留给脉冲/饱和函数扩展）
```

- **稀疏连接**：每个单元固定 fan-in `in_degree`（例如 16/768≈2.1%），
  `W` 用 `[N,K]` 的 gather 实现，单步 O(N·K)，避免 N² 稠密矩阵。
- **谱半径初始化**：`_rescale_to_spectral_radius` 用幂迭代估计稀疏算子的谱半径，
  缩放到 `target_spectral_radius`（默认 1.0），使动力学处于稳定边缘。
- **三组时间常数**：`τ = (10, 100, 1000) ms`，比例 `(0.5, 0.3, 0.2)`。
  快组响应瞬时输入，中组/慢组积分情境；这是延迟回忆实验的主力。
- **稳态平衡与稳定性**：homeostatic bias 按
  `b_i ← b_i - η_homeo(EMA(|x_i|) - target_rate)` 更新，防止长期漂移/饱和；
  `row_norm_cap` 限制可塑性后的行范数；状态范围由 tanh 与衰减保证有界。
- **输入**：`W_in` 为随机固定映射（可在后续实验中改为可塑/脉冲输入）。
  读出层只看 `x` 的稀疏随机投影 `h(t)=P x(t)`（`projection.fan_in` 固定），
  以控制接口宽度。

复杂度：单步 `O(N·K + H·K_h)`，其中 `H=h_dim`，`K_h` 为投影 fan-in；
100k 单元、K=32 时单步约 3.2M 乘加，适合 CPU/神经形态事件驱动实现。

### 4.2 相位同步与路由（B）

每个单元一个相位变量（Kuramoto 型）：

```
dθ_i/dt = 2π f_i + K_c · mean_{j∈pre(i)} sin(θ_j - θ_i) + K_in (U_phase s)_i + noise
```

- 频率 `f_i` 按时间尺度组给基础值（8/6/4 Hz）+ 抖动；
- 输入项让同一感觉事件驱动的单元产生相位重置，倾向于同步；
- 同步簇按相位分箱得到：对簇 m，

```
r_m = | mean_{i∈m} exp(i θ_i) |          （相干性）
mx_m = mean_{i∈m} x_i
f_m = |m| / N
Φ(t) = [r_0..r_{C-1}, mx_0..mx_{C-1}, f_0..f_{C-1}]  （3C 维）
```

- 同步度 `sync = Σ_m r_m f_m` 作为接口标量，可被上层用于事件绑定/路由判断；
- `cluster_membership` 作为 `InterfacePacket` 的附加字段返回，方便做绑定分析。

**当前局限**：相位层尚未反作用于动力学增益（只影响接口与路由统计）；这是下一步
要验证的“相位路由选择模块”的关键扩展点。

### 4.3 工作记忆与吸引子记忆（C）

**工作记忆**：`S=32`（默认）个归一化内容槽位，内容维度 = `h_dim`。

- 读：`sim = slots·normalize(h)`，softmax 权重加权和 → `M_wm`；返回 best-hit 与熵；
- 写：仅在高 surprise 或高 novelty 时触发，并带冷却 `write_cooldown`；
- 驱逐：空槽优先，满则替换 usage 最小槽（usage 随读/写衰减）。

**吸引子长期记忆**：容量 `P=512`（默认）的归一化原型 `ξ^μ`。

- 读：`top-k` 余弦相似度 + softmax 加权（modern-Hopfield 风格的检索近似）；
- 写：novelty gate（与已存原型最大余弦 < `novelty_cos` 才写），环状覆盖；
- 检索评测：加噪原型 → top-k 合成 → 与干净原型余弦，对照随机原型的 chance 水平。

**惊奇门（SurpriseGate）**：对一步自预测误差做指数滑动均值/方差归一化，
`z_t = (e_t - μ)/σ`，仅在 `e_t > μ + kσ` 且 `z_t > floor` 时视为 surprise 事件。
写入门 = `surprise OR novelty`，共享一个写冷却，避免记忆被高频噪声淹没。

注意：**记忆写入属于推理行为**（即使 `learn=False` 也可以写），而“学习”是权重/读出
层面的更新。评估协议会在 §6 单独说明什么时候开/关写。

### 4.4 三因子局部可塑性（D）

对稀疏突触 `(i,j)`：

```
E_ij(t) = λ_E E_ij(t-1) + post_i(t) pre_j(t)              （eligibility/STDP 迹）
ΔW_ij = lr_hebb · (post_i pre_j − post_i² W_ij)           （Oja 稳定 Hebb 项）
       + lr_error · mod(t) · e_i(t) pre_j(t)               （预测误差项）
       + lr_reward · r(t) · E_ij(t)                        （奖励/任务调制）
W_ij  ← clip_per_step( W_ij + ΔW_ij ) − wd·W_ij
```

- `post=tanh(x_i)`，`pre=tanh(x_j)` 来自当前步骤；
- 误差广播 `e = B · error`，`B` 是固定的随机反馈矩阵（feedback alignment 风格），
  不需要对称的转置权重、不做 BPTT；
- `mod(t) = 1 + gain·max(0, surprise)`，由惊奇信号调节写入强度；
- 每步后做逐行范数上限，防止可塑性导致动力学发散；
- 训练日志记录 `delta_abs_mean / hebb_abs_mean / error_abs_mean / ...`。

复杂度：O(N·K) 每步，与动力学同阶；不需要保存整条时间图的激活。

### 4.5 睡眠回放与巩固（E）

`SleepConsolidator.run(buffer, readout)` 的顺序（重要）：

1. **结构巩固**：按行剪掉最弱 `prune_fraction` 的突触；逐行重归一化到原来的平均
   行范数（`normalize_rows`），并记录剪枝数量与权重漂移；
2. **回放**：按 regime 配额从 `ReplayBuffer` 采样完整序列窗口，重置动力学状态后
   重新跑一遍（不更新可塑性）；把每个窗口 top-surprise 的 `h` 作为原型写入吸引子
   记忆与工作记忆；
3. **读出巩固**：用回放窗口上的 `(z_t, s_{t+horizon})` 样本做最小二乘重拟合，
   但**只在回放误差下降时接受**（`replay-mse gating`），防止一次睡眠把接口弄坏；
4. **检索评测**：对吸引子原型加噪检索，报告 cosine vs chance cosine。

回放缓冲区按 regime 分配配额（每 regime 最多 `buffer_size/2`），否则最近的在线
任务会把旧任务窗口全部挤掉，睡眠就无从“巩固旧经验”。

### 4.6 无反向传播的读出（接口训练）

- **RidgeReadout**：先标准化特征，闭式解
  `W = (XᵀX + λnI)⁻¹ Xᵀ(Y-ȳ)`；这是“离线睡眠式”接口拟合，也用于 stage-1。
- **RLSReadout**：递推最小二乘，`P`（逆相关矩阵）与 `θ`（权重）每步秩一更新，
  支持遗忘因子；stage-2 在线使用。
- **贝叶斯初始化**：stage-2 开始时把 stage-1 ridge 的 `(XᵀX+λnI)⁻¹` 作为 RLS 的
  初始 `P`，把 ridge 权重作为 `θ` 初值。这样在线学习从“已经消化过旧数据”的
  后验出发，显著降低早期剧烈漂移（代码：`RidgeReadout.fit` 保存 `precision`，
  `RLSReadout.set_weights(..., precision=...)`）。
- **为什么不用 backprop**：底层的核心假设是“学习是局部的、可以在推理时持续发生”。
  读出层是唯一需要监督通道的部分，用最小二乘/RLS 即可，不需要对动力学做 BPTT。

---

## 5. 接口协议 z(t)

`InterfaceSpec` 定义的布局（默认配置：`h=96`，相位 `3C=24`，WM 32，LTM 32，
感觉通道 20，辅助 3 → 207 维）：

| 字段 | 维度 | 含义 | 上层用途 |
|---|---|---|---|
| `h` | 96 | 慢/中/快混合潜在态压缩 | 世界模型情境输入 |
| `phase` | 3C | 每簇相干性、平均状态、占比 | 事件绑定、路由、对象级表征 |
| `wm` | 32 | 工作记忆 readout（压缩） | 当前情境/任务上下文 |
| `attractor` | 32 | 长期吸引子原型检索结果（压缩） | 长期经验检索 |
| `input` | F | 当前观测 s(t) | 世界模型同步/条件化 |
| `delta` | F | s(t)-s(t-1) | 瞬时变化/速度信息 |
| `aux` | 3 | (surprise, sync, confidence) | 触发注意/记忆写入/探索/调用符号推理 |

- 向下调制（`UpperModulation`）：`reward`、`attention`、`query`、`target_error`
  已定义结构；当前训练脚本使用 `target_error` 驱动三因子可塑性，`reward` 为负预测
  误差；`attention` 作为可选逐单元掩码接入 `ThreeFactorPlasticity.step`。
- 记忆索引 `memory_key` 与 `cluster_membership` 保留在 `InterfacePacket` 中，不强制
  拼进 `z`，供需要符号级追踪的上层使用。
- `include_sensory_channel` 配置可关闭 `input/delta` 两个通道做消融。

---

## 6. 训练与评估协议

### 6.1 Stage 1：无监督预测编码预训练

- 数据：`MultimodalWorld` 的状态流 `s(t)`，目标 `y(t)=s(t+h)`（默认 h=5）。
- 每一轮：`learn=True` 跑完训练流（homeostasis、自预测 RLS、记忆写入都开启），
  收集接口特征 `z` 与目标 `y`，用 `RidgeReadout.fit` 闭式重拟合。
- 记录每轮 eval MSE；这是“只学预测、不学任务标签”的阶段。

### 6.2 Stage 2：任务中在线学习

- 从 stage-1 的 `RidgeReadout` 权重与 precision 初始化 `RLSReadout`；
- 在 regime B 上逐 tick：`z → ŷ`，误差用于
  (a) RLS 读出更新；(b) `ThreeFactorPlasticity` 的误差广播与奖励调制；
- 记录在线 MSE、每轮 A/B 评测；
- 默认 `forget=0.9999`（保守），另设压力测试 `forget=0.999`（激进遗忘）。

### 6.3 Stage 3：睡眠回放

见 §4.5；默认 `replay_passes=2`，`prune_fraction=0.01`，完整序列窗口。

### 6.4 评估协议（重要）

- 预测/异常评测：`learn=False`，**冻结自预测器与惊奇统计**；记忆写入作为推理行为
  **开启**（`allow_memory_write=True`），并在评测前后快照/恢复记忆状态，保证评测不会
  污染主模型。
- 异常检测：`surprise` 是归一化的一步自预测误差；另给出 3 步滑动平均版本与
  “输入变化”基线；用秩和 ROC-AUC。
- 延迟回忆：独立系统在 recall 流上做两轮无监督适应，再拟合 ridge 读出；评测时
  对每个序列恢复同一记忆快照，分别测“写入开启/冻结”两种条件，统计最后 10 步的
  top-1 分类准确率与逐时刻曲线。
- 遗忘压力测试：恢复 stage-1 基座，用 `forget=0.999` 在 regime B 上快速在线学习，
  量化 A 的遗忘，然后执行睡眠回放，量化 A/B 的恢复量。

---

## 7. 复杂度、参数与规模

| 配置 | N | K | h_dim | 接口维度 | 参数量（约） | 用途 |
|---|---|---|---|---|---|---|
| `debug` | 768 | 16 | 96 | 207 | 9.5 万 | 笔记本快速验证 |
| `standard` | 2048 | 24 | 128 | 271 | 30 万级 | 单次完整实验 |
| `mvp` | 16384 | 32 | 256 | 435 | 200 万级 | 更大规模原型（未在 CI 跑） |

- 单步核心开销：`O(N·K + N·H_out)`；相位 `O(N·K)`；读出 RLS `O(D²)`（D=接口维度，
  默认 200 左右，CPU 可承受）。
- 当前实现是“离散 tick + 稀疏 gather”，还不是真正的 event-driven 异步计算；
  但结构上所有运算都只依赖局部连接，可映射到事件驱动神经形态芯片/模拟慢动力学。

---

## 8. 实验结果

见 [RESULTS.md](RESULTS.md)。摘要（以 `configs/standard.yaml` 为准，`demo_standard`）：

- 下一观测预测 MSE 优于 persistence 与 linear-AR(k=4)；
- 异常检测 AUC 高于输入变化基线；
- 延迟回忆（delay=40–48 步）显著高于 chance，并可测出记忆随延迟的衰减曲线；
- 在线适应（保守遗忘）使 A/B 同时改善；压力测试中激进遗忘造成明显 A 遗忘，
  睡眠回放后 A/B 均显著恢复。

（具体数字以 `artifacts/demo_standard/results.json` 为准，README/RESULTS 中同步。）

---

## 9. 局限与风险（诚实清单）

1. **不是 AGI，也不是 Transformer 替代品**：本仓库只验证底层基座的一小部分行为，
   没有语言、规划、符号推理能力。
2. **合成任务为主**：`MultimodalWorld` 是 10 维玩具世界，结论不能直接外推到真实
   视觉/音频/机器人数据。
3. **局部学习的信用分配有限**：三因子规则依赖固定随机反馈与标量调制，能在线适应，
   但无法承担复杂任务梯度；上层仍需世界模型/决策学习。
4. **组件消融不显著**：在短时程合成预测上，去掉相位/记忆/慢时间常数的差异较小
   （见 RESULTS 的消融表）；组件的价值需要更长时程、更复杂绑定/回忆任务来验证。
5. **工作记忆在延迟回忆中未体现优势**：当前 recall 任务主要由慢动力学承载，
   写入冻结甚至略好；WM/吸引子的作用需要在多物体绑定、干扰、情境切换任务中验证。
6. **睡眠读出重拟合是门控的**：只在回放误差下降时接受，避免“睡眠伤害模型”；这也
   说明回放分布与在线分布仍有 gap，是后续可改进点。
7. **RLS 数值稳定性**：float64、precision 初始化、遗忘因子选择都需要调；目前用
   200 维接口尚可，规模更大时需要分块/正则化。
8. **未做功耗/延迟硬件测量**：文中 1–3 W 是设计目标，不是本仓库实测结果。
9. **相位层未闭环**：相位目前主要影响接口特征，没有反向调制动力学/路由增益。

---

## 10. 扩展路线

- **更接近事件驱动**：用稀疏脉冲（spike）阈值与异步事件队列替代密集 tick；
  引入 `torch.sparse`/自写 kernel 或神经形态后端。
- **连续时间/多隔室**：把离散动力学换成 ODE/脉冲神经元，保留多 τ；
  增加树突隔室与局部电压依赖可塑性。
- **相位闭环路由**：让同步簇真的选择“哪些模块/记忆槽参与下一步计算”，
  并加入相位编码的顺序/间隔读出。
- **分层 MTDB**：底层基座输出到上层世界模型（可微分或非可微分），
  接口 `z(t)` 对接 Transformer/SSM/符号推理器。
- **真实多模态数据**：事件相机（DVS）、音频、触觉/本体感觉；先做自监督预测。
- **任务与规模**：多物体绑定、遮挡、持续导航、在线适应/灾难遗忘基准；
  扩展到 10⁴–10⁵ 单元并做功耗/延迟评估。
- **硬件映射**：稀疏 fan-in + 局部三因子规则天然适合存内计算/神经形态；
  但需要量化、噪声鲁棒性与片上学习验证。

---

## 附录 A：复现命令

```bash
# 环境
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt

# 快速实验（约 45s，CPU 笔记本）
.venv/bin/python scripts/train.py --config configs/debug.yaml

# 标准实验（2048 单元，几分钟）
.venv/bin/python scripts/train.py --config configs/standard.yaml

# 单测
.venv/bin/python -m pytest tests -q

# 从 checkpoint 重新评测
.venv/bin/python scripts/eval.py --checkpoint artifacts/demo_standard/checkpoint.pt
```

## 附录 B：关键超参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `dynamics.tau_groups_ms` | (10,100,1000) | 快/中/慢时间尺度 |
| `dynamics.in_degree` | 16/24/32 | 稀疏 fan-in |
| `dynamics.target_spectral_radius` | 1.0 | 动力学稳定边缘 |
| `phase.coupling_gain` | 0.4 | 相位同步强度 |
| `memory.surprise_k` | 2.0 | 惊奇阈值（σ） |
| `memory.wm_novelty_threshold` | 0.55 | WM 内容新颖性阈值 |
| `memory.attr_novelty_threshold` | 0.65 | LTM 原型新颖性阈值 |
| `plasticity.lr_error` | 4e-4 | 三因子误差项学习率 |
| `readout.rls_forget` | 0.9999 | 在线读出有效记忆 |
| `consolidation.prune_fraction` | 0.01 | 每行剪枝比例 |
| `train.horizon` | 5 | 预测目标步长 |
