# MTDB — 多时间尺度动力学基底

[English](README.md) | **中文**

[![tests](https://github.com/cndichengyu/mtdb/actions/workflows/ci.yml/badge.svg)](https://github.com/cndichengyu/mtdb/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/cndichengyu/mtdb)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

面向混合 AGI 系统“感知—直觉—记忆基底”的**非 Transformer 训练底座**：
稀疏多时间尺度动力学 + Kuramoto 相位同步 + 工作/吸引子记忆 + 三因子局部可塑性 +
睡眠回放巩固 + 面向上层世界模型的压缩接口 `z(t)`。

![architecture](assets/architecture.png)

> **研究原型，不是 AGI，也不是 Transformer 替代品。** MTDB 只实现混合架构的底层：
> 感知压缩、时间动态、在线适应、预测误差（surprise）和稳定长期记忆。语言、逻辑、
> 规划仍属于上层世界模型 / 神经符号 / 智能体。当前实验全部是合成小规模验证。

---

## 为什么做这个

具身、持续学习的系统需要与 Transformer 不同的底层性质：

- **能效与稀疏**：局部稀疏连接，而非稠密平方复杂度注意力；
- **连续时间**：多时间尺度状态承载动态，而不只是 token 位置；
- **在线适应**：推理时持续学习，不需要全局反向传播；
- **稳定记忆**：惊奇门控写入 + 吸引子检索，而不是无限增长上下文；
- **具身 grounding**：传感器流与相位绑定是一等公民。

这个仓库把上述想法做成**可在笔记本 CPU 上训练、可复现、可证伪**的代码。

---

## 30 秒结果

`configs/standard.yaml`，2,048 单元、25.7 万参数、CPU 约 4 分钟：

| 任务 | MTDB | 最强非 Transformer 基线 | 说明 |
|---|---|---|---|
| 下一观测预测（horizon=5） | **MSE 0.0486** | persistence 0.0744；linear-AR(4) 0.0532 | 比 linear-AR 低 8.7% |
| 异常检测 | **ROC-AUC 0.722** | 输入变化 0.647 | 3 步平滑后 0.748 vs 0.745（打平） |
| 延迟回忆（8 类） | **1.000** @ delay 48 | chance 0.125 | delay 96：0.231（开写入）vs 0.134（冻结） |
| 在线适应 regime B | **MSE −6.9%** | — | regime A 仅退化 1.7% |
| 遗忘压力测试 + 睡眠回放 | **A: 0.0495 → 0.1147 → 0.0481** | — | 回放几乎完全恢复遗忘 |

![prediction comparison](assets/prediction_comparison.png)

完整表格、曲线、消融与局限见 **[RESULTS.md](RESULTS.md)**（[English](RESULTS_EN.md)）。

---

## v0.3：主动推理 + 校准 + 诚实的负结果

模型可以在行动前**模拟候选探针并选择最有信息量的一个**（expected free energy）。
在"8 类 + 含噪冗余探针"的主动感知任务上，one-step EIG 在小探针预算下显著提升
样本效率。下表为 **3 个 seed 的均值 ± 95% CI**；Transformer 基线同参数量，且
在公平对比中**同样使用主动规划器和同一张学习到的探针表**：

| 探针预算 | **MTDB 主动 d1** | MTDB 固定 | Transformer 被动 | Transformer+主动 | 贝叶斯上界 |
|---:|---:|---:|---:|---:|---:|
| 2 | **0.430 ± 0.114** | 0.230 | 0.193 | 0.390 | 0.410 |
| 3 | **0.633 ± 0.127** | 0.397 | 0.370 | 0.450 | 0.733 |
| 4 | 0.670 ± 0.131 | **0.730** | 0.607 | 0.623 | 0.727 |
| 5 | 0.777 ± 0.165 | **0.793** | 0.603 | 0.650 | 0.780 |

![active inference](assets/active_inference.png)

这**不是**"通用击败 Transformer"：预算变大后固定探针顺序会追平，且拿到同一
主动规划器的 Transformer 会大幅缩小差距。我们也如实报告了两个负结果：

- **多步粒子规划**（depth 2）在该任务上没有超过 one-step EIG；
- **无标签在线探针可靠性适应**在 drift 下会让噪声估计自我强化，反而伤害精度。

校准（temperature scaling + split conformal）降低了 NLL/ECE，conformal coverage
约 0.90；但 ECE 仍约 0.5，距离贝叶斯上界仍有明显空间。

完整协议、逐 seed 数据、负结果与局限见
**[ACTIVE_INFERENCE.md](ACTIVE_INFERENCE.md)**（英文）。

```bash
# CPU 约 70 秒，3 个 seed，含校准、drift 与 Transformer 主动基线
python scripts/active_benchmark.py
```

---

## 包含什么

| 模块 | 代码 | 作用 |
|---|---|---|
| A 多时间尺度动力学 | [`mtdb/core/dynamics.py`](mtdb/core/dynamics.py) | 稀疏 fan-in 循环核；τ ∈ {10,100,1000} ms；谱半径初始化；homeostatic bias |
| B 相位同步路由 | [`mtdb/core/phase.py`](mtdb/core/phase.py) | Kuramoto 相位、同步簇特征、簇成员 |
| C 记忆 | [`mtdb/core/memory.py`](mtdb/core/memory.py) | 内容寻址工作记忆 + 吸引子原型；surprise/novelty 门控 |
| D 局部可塑 | [`mtdb/core/plasticity.py`](mtdb/core/plasticity.py) | eligibility + 误差广播 + reward；Oja/行范数稳定，**无 BPTT** |
| E 睡眠回放 | [`mtdb/core/consolidation.py`](mtdb/core/consolidation.py) | regime 配额回放、原型抽象、剪枝/重归一、门控读出重拟合 |
| 接口 | [`mtdb/core/interface.py`](mtdb/core/interface.py) | 可寻址的 `z(t)` 数据包 |
| 读出 | [`mtdb/training/readout.py`](mtdb/training/readout.py) | 闭式 ridge / 在线 RLS（贝叶斯 precision 初始化），**无 backprop** |
| 训练管线 | [`mtdb/training/trainer.py`](mtdb/training/trainer.py) | 三阶段训练 + 遗忘压力测试 + 消融 + 延迟回忆 |
| 自定义数据 | [`mtdb/tasks/array.py`](mtdb/tasks/array.py) | 把自己的 `[N,T,F]` 特征流转成训练序列 |
| 主动推理（v0.3） | [`mtdb/active/`](mtdb/active/) | one-step EIG + 多步粒子规划、温度/Conformal 校准、在线噪声适应、含公平 Transformer 基线的主动感知基准 |
| Transformer 基线 | [`mtdb/baselines/transformer.py`](mtdb/baselines/transformer.py) | 仅用于主动感知公平对比的小型 causal Transformer |
| 合成世界 | [`mtdb/tasks/synthetic.py`](mtdb/tasks/synthetic.py) | 多模态预测/异常、延迟回忆任务 |

底座的动力学与记忆**不使用反向传播**，只有上层读出用最小二乘 / RLS 拟合。

---

## 快速开始

```bash
git clone https://github.com/cndichengyu/mtdb.git
cd mtdb
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 快速实验（768 单元，约 1 分钟）
python scripts/train.py --config configs/debug.yaml

# 标准实验（2048 单元，CPU 约 4 分钟）
python scripts/train.py --config configs/standard.yaml --run-name demo_standard

# 单测
pytest tests -q
```

---

## 用自己的数据训练

准备一个 `.npz`（至少包含 `train` / `eval`，形状 `[N, T, F]`）：

```python
# 可选：
# train_b / eval_b             第二个 regime（在线适应实验）
# eval_anomaly                带异常的评估流
# eval_anomaly_mask           [N, T] bool 异常标签
```

```bash
python scripts/train.py --config configs/standard.yaml --data data/my_stream.npz
```

特征只用训练集统计量做 z-score；目标自动取 `x[t + horizon]`；输入维度自动推断。
注意睡眠回放按整条序列存储：序列长度最好 ≥ `consolidation.window_len`
（debug 96、standard 128），短序列可调小该配置。也可以直接调 API：

```python
from mtdb.config import ExperimentConfig
from mtdb.tasks.array import make_streams_from_arrays
from mtdb.training.trainer import MTDBTrainer

streams, meta = make_streams_from_arrays(my_train, my_eval, horizon=5)
trainer = MTDBTrainer(ExperimentConfig.from_yaml("configs/standard.yaml"), streams=streams)
trainer.run()
```

完整可运行示例：[`examples/custom_stream.py`](examples/custom_stream.py)。

---

## 训练三阶段

1. **无监督预测编码预训练**：驱动底座，只学习自预测/稳态；ridge 拟合读出。
2. **在线任务学习**：RLS 从 stage-1 后验出发逐 tick 适应；可选三因子局部可塑性；
   推理时 surprise 门控写入记忆。
3. **睡眠回放巩固**：回放旧+新经验，抽象原型，剪枝/重归一，回放误差下降时重拟合读出。

内置两类可检验诊断：**激进遗忘压力测试**（sleep 恢复率）与**组件消融**
（`full / no_phase / no_memory / single_tau / fast_only`）。

---

## 配置与规模

| Preset | 单元 | fan-in | 接口 | 参数 | CPU 时间 |
|---|---:|---:|---:|---:|---:|
| `configs/debug.yaml` | 768 | 16 | 207 | 9.5 万 | 约 1 分钟 |
| `configs/standard.yaml` | 2048 | 24 | 283 | 25.7 万 | 约 4 分钟 |
| `configs/mvp.yaml` | 16384 | 32 | 435 | 约 200 万 | CI 不跑 |

核心单步复杂度 O(N·K)，稀疏 gather；当前是离散 tick 近似的事件驱动底座，
硬件功耗/延迟是设计目标，不是实测结果。

---

## 诚实局限

- 仅合成 10 维玩具世界，尚未在真实视觉/音频/机器人数据验证；
- 短时程预测上相位/记忆消融差异很小；慢时间常数的最清楚证据是延迟回忆曲线；
- 工作记忆只在更长延迟（96 步）才明显优于冻结写入；
- 睡眠巩固主要发生在读出层，循环突触只做剪枝/重归一，没有轨迹级 STDP 回放；
- 没有语言/推理/规划，也没有硬件功耗/延迟实测；
- 异常检测应使用 `checkpoint_stage1.pt`（在线适应会改写 surprise 统计）；
- 主动基准是 8 类合成任务，优势主要来自主动获取证据；Transformer 拿到同样
  规划器后会追回大部分差距，不构成通用"击败 Transformer"结论；
- 多步粒子规划与无标签在线探针适应已实现，但该基准上为负结果；
- 真实事件相机/音频/机器人验证**尚未完成**，当前基准是合成的。

详见 [ARCHITECTURE.md](ARCHITECTURE.md)（[English](ARCHITECTURE_EN.md)）第 9 节。

---

## 路线图

0. **v0.2/v0.3 已完成**：主动推理、多步规划、校准、公平 Transformer 基线、多 seed CI。
   **待解决**：无标签 drift 适应、不确定性校准到贝叶斯上界水平、真实主动感知；
1. 事件驱动/脉冲 kernel，替代密集 tick；
2. 相位簇闭环选择模块/记忆槽/增益；
3. 轨迹级 STDP 回放；
4. 分层 MTDB 对接世界模型 / SSM / 神经符号推理器；
5. 真实多模态前端（事件相机、音频、触觉/本体感觉）；
6. 10⁴–10⁵ 单元并测量功耗、延迟、吞吐。

---

## 引用与许可

```bibtex
@misc{mtdb2026,
  title        = {MTDB: Multi-Timescale Dynamical Base},
  author       = {cndichengyu},
  year         = {2026},
  howpublished = {\url{https://github.com/cndichengyu/mtdb}}
}
```

MIT License。欢迎 reproducible experiments / 负结果 / 明确边界的扩展；
见 [CONTRIBUTING.md](CONTRIBUTING.md)。

如果这个方向对你有帮助，点一个 ⭐ 能让更多研究者找到它。
