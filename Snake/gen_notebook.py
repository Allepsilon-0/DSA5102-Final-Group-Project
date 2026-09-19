# -*- coding: utf-8 -*-
"""生成 DSA5102 Final Project 的 Jupyter notebook 框架。

按五人分工组织章节，每个成员章节标注负责人和要填写的内容。
运行: python gen_notebook.py
"""
import json
from pathlib import Path

try:
    import nbformat
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False


_cell_counter = 0


def md(source: str):
    global _cell_counter
    _cell_counter += 1
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": f"md-{_cell_counter}"}


def code(source: str):
    global _cell_counter
    _cell_counter += 1
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source, "id": f"code-{_cell_counter}"}


cells = []

# ---------------------------------------------------------------------------
# 标题与元信息
# ---------------------------------------------------------------------------
cells.append(md(
"""# DSA5102 Final Group Project
## 用强化学习训练机器玩贪吃蛇（Snake）

- **课程**：DSA5102 - Foundations of Machine Learning
- **选题**：Option 2 —— Train the machine to play a classic game using Reinforcement Learning
- **小组**：`<填写组号/成员姓名/学号>`
- **日期**：2026-11

> **使用说明**：本 notebook 是最终提交物（project report）。开头有 `TRAIN_MODEL = False` 开关，
> 默认加载已训练好的模型，保证 notebook 无需重新训练即可运行。每个章节顶部标注了负责成员。
"""))

cells.append(md(
"""## 0. 项目概述与声明

### 0.1 选题与目标

本项目选择 **Option 2（强化学习）**，训练机器玩经典游戏 **贪吃蛇（Snake）**。
我们实现了三种学习算法（Actor-Critic、DQN、PPO），并引入 DQN 的改进变体 Double DQN，
在统一的随机种子与独立评估环境下比较它们的性能，同时以「随机策略」和「规则教师策略」作为基线（baseline）。

### 0.2 代码来源声明（学术诚信）

> **【重要，全员确认后填写】**
> 本项目的环境与算法基础实现参考了开源教程
> [NocoldBob/RL](https://github.com/NocoldBob/RL)（《RL 从零开始：低算力强化学习实践》）。
> 我们在其基础上做了如下**实质性改动**（示例，按实际填写）：
> - 成员 1：重设计奖励函数 / 将教师策略由 BFS 改为 A*
> - 成员 2：新增「有/无行为克隆预热」对照实验
> - 成员 3：将经验回放改为优先级经验回放（PER）
> - 成员 4：新增 PPO vs Actor-Critic 对照实验
> - 成员 5：补充统一对比可视化与统计检验
"""))

cells.append(md(
"""### 0.3 生成式 AI 使用声明

> **【全员确认后填写】** 请如实声明：是否使用、用什么工具、用于什么目的（如代码注释、图表排版、
> 章节润色、debug 等）。如果未使用也请明确写「未使用生成式 AI」。不实声明一旦被查实，两部分成绩计零分。
"""))

# ---------------------------------------------------------------------------
# 全局设置
# ---------------------------------------------------------------------------
cells.append(md(
"""### 0.4 全局设置

- `TRAIN_MODEL`：最终提交保持 `False`，加载已训练好的模型；需要重训时改为 `True`。
- 保证 notebook 与 `environment.py` / `model.py` / `dqn.py` / `ppo.py` 等文件在同一目录即可直接 import。
"""))

cells.append(code(
"""# ==================== 全局设置 ====================
# 最终提交请保持 False（加载已训练好的模型）；重训时改为 True
TRAIN_MODEL = False

import sys
from pathlib import Path

# 若 notebook 与 .py 源码同目录，可直接 import；否则把源码目录加进 path
HERE = Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import torch
import matplotlib.pyplot as plt

from environment import SnakeEnv
from model import ConvActorCritic
from dqn import DQNAgent, ReplayBuffer
from ppo import PPOAgent

# 确认已训练模型存在
RUNS = Path("runs")
print("cwd:", HERE)
print("runs 目录存在:", RUNS.exists())
if RUNS.exists():
    for p in sorted(RUNS.rglob("*.pt"))[:10]:
        print("  checkpoint:", p)
"""))

# ---------------------------------------------------------------------------
# 第 1 部分：成员 1 —— 游戏介绍与环境建模
# ---------------------------------------------------------------------------
cells.append(md(
"""---
# 1. 游戏介绍与环境建模　【成员 1 负责】

## 1.1 贪吃蛇游戏规则（假设读者从未玩过）

> **【成员 1 填写】** 用一两段话讲清楚规则：棋盘、蛇、食物、移动、吃食物变长、撞墙/撞自己死亡、目标。
> 下面代码画棋盘示意图。

## 1.2 可视化棋盘
"""))

cells.append(code(
"""# 贪吃蛇棋盘示意图（成员 1：用 matplotlib 画一个初始局面，标注蛇头、蛇身、食物）
# 提示：可复用 environment.py 里 _get_observation 的思路，或手画一个 6x6 网格示例。
GRID = 6
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_xlim(0, GRID); ax.set_ylim(GRID, 0); ax.set_aspect("equal")
for i in range(GRID + 1):
    ax.axhline(i, color="#cccccc", lw=0.8)
    ax.axvline(i, color="#cccccc", lw=0.8)
# 示例：蛇头 (3,3)、食物 (3,5)
ax.add_patch(plt.Rectangle((3 + 0.05, 3 + 0.05), 0.9, 0.9, color="#e63946"))   # 蛇头
ax.scatter(5 + 0.5, 3 + 0.5, s=200, marker="*", color="#2a9d8f")               # 食物
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Snake 棋盘示例（6×6）")
plt.show()
"""))

cells.append(md(
"""## 1.3 状态、动作与奖励的建模

> **【成员 1 填写】** 讲清楚三个核心设计（这也是答辩重点）：
> 1. **动作空间**：为什么是「左转/右转/直行」三个相对动作，而不是「上下左右」四个绝对动作。
> 2. **观测空间**：7 通道网格，每个通道编码什么（蛇身、食物、墙体、四个方向通道）。
> 3. **奖励函数**：吃食物 +10、撞墙 −5、靠近食物 +0.2、远离 −0.1、重复访问惩罚 −0.4、达成目标 +100 等，每项的目的。
"""))

cells.append(code(
"""# 初始化环境，查看动作与观测空间（成员 1）
env = SnakeEnv(grid_size=6, end_score=4, max_steps=100)
print("动作空间:", env.action_space)         # Discrete(3)
print("观测空间:", env.observation_space)     # Box(7, 6, 6)

obs, info = env.reset(seed=42)
print("观测 shape:", obs.shape, "dtype:", obs.dtype)
print("reset 返回 info:", info)

# 可视化观测的 7 个通道（蛇身 / 食物 / 墙体 / 方向）
fig, axes = plt.subplots(1, 7, figsize=(14, 2.5))
channel_names = ["蛇身", "食物", "墙体", "上", "下", "左", "右"]
for i, ax in enumerate(axes):
    ax.imshow(obs[i], cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(channel_names[i], fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("7 通道观测编码")
plt.tight_layout()
plt.show()
"""))

cells.append(md(
"""## 1.4 规则教师策略（baseline 之一，成员 1 主讲）

> **【成员 1 填写 + 主讲】** 解释 `teacher_action()` 的实现思路：
> 对三个动作分别试走一步，若安全且 BFS 能找到安全路径，就选离食物最近的那个；否则直行。
> 这是「启发式规则」基线，也是后面行为克隆（成员 2）的演示数据来源。
> **交叉引用**：行为克隆的使用见 §2（成员 2）；作为对比基线参与评测见 §5（成员 5）。
"""))

cells.append(code(
"""# 展示规则教师策略（成员 1）
obs, _ = env.reset(seed=42)
teacher_action = env.teacher_action()
print("教师选择的动作:", teacher_action, "(0=左转, 1=右转, 2=直行)")
# 走一步看看
next_obs, reward, terminated, truncated, info = env.step(teacher_action)
print("奖励:", reward, "终止:", terminated, "截断:", truncated, "info:", info)
env.close()
"""))

# ---------------------------------------------------------------------------
# 第 2 部分：成员 2 —— Actor-Critic 与行为克隆
# ---------------------------------------------------------------------------
cells.append(md(
"""---
# 2. 方法一：Actor-Critic 与行为克隆　【成员 2 负责】

## 2.1 方法原理

> **【成员 2 填写】** 讲清楚：
> 1. **行为克隆预热**：前 `teacher_episodes` 轮用规则教师（§1.4）的示范做监督学习（交叉熵），解决冷启动。
> 2. **Actor-Critic 更新**：`advantage = target − V(s)`，policy gradient 项 + critic 的 value loss + 熵正则。
> 3. 损失函数拆解（参考 `model.py` 的 `update()`）。

## 2.2 构建模型
"""))

cells.append(code(
"""# 构建 Actor-Critic 模型（成员 2）
import environment as _env_mod
env = SnakeEnv(grid_size=6, end_score=4, max_steps=100)
agent_ac = ConvActorCritic(
    input_channels=env.observation_space.shape[0],
    output_dim=env.action_space.n,
    grid_size=6,
    lr=1e-4,
    weight_decay=1e-5,
    entropy_coef=0.01,
)
print(agent_ac)
env.close()
"""))

cells.append(md(
"""## 2.3 训练（或加载模型）

> 训练入口在 `main.py` 的 `train()`。默认 `TRAIN_MODEL=False` 时加载 checkpoint。
> **成员 2 的实质改动**：新增「有/无行为克隆预热」对照，把 `teacher_episodes` 设为 0 跑一版做对比。
"""))

cells.append(code(
"""if TRAIN_MODEL:
    from main import TrainConfig, train
    summary_ac = train(TrainConfig(
        episodes=500,
        teacher_episodes=100,   # 成员 2 改动点：对比 0 vs 100 的预热效果
        grid_size=6,
        end_score=4,
        max_steps=100,
        seed=42,
        device="auto",
        output_dir=Path("runs/snake"),
        tensorboard=False,
    ))
    print(summary_ac)
else:
    # 加载已训练好的模型
    ckpt = Path("runs/snake/checkpoints/best.pt")
    print("加载 AC checkpoint:", ckpt, "存在:", ckpt.exists())
    # agent_ac.load_checkpoint(ckpt)  # 取消注释以加载权重
"""))

cells.append(md(
"""## 2.4 学习曲线

> **【成员 2 填写】** 训练过程会写 TensorBoard 事件；这里从训练返回或 `history` 里取 reward/score 画学习曲线。
> 若用 `TRAIN_MODEL=False`，请把训练时保存的曲线数据（`summary.json` / TensorBoard）读进来画图。
"""))

cells.append(code(
"""# AC 学习曲线（成员 2：填入实际训练曲线数据）
# 示例占位：横轴 episode，纵轴 episode reward
# plt.plot(episodes, rewards); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("AC 学习曲线")
print("TODO: 绘制 AC 学习曲线（reward / score vs episode）")
"""))

cells.append(md(
"""## 2.5 训练进度可视化（成员 2）

> **【成员 2 填写 + 必做】** 这是 Option 2 的明确得分点：**视觉展示玩法随训练进步**。
> 用 `save_interval` 保存的不同阶段 checkpoint（或用成员 4 提供的通用脚本），
> 渲染「早期（乱撞）vs 后期（稳定吃分）」的对比图 / GIF，并写一段文字说明观察到的进步。
"""))

cells.append(code(
"""# AC 训练进度可视化（成员 2：加载早期/后期 checkpoint 分别渲染并排对比）
print("TODO: 渲染 AC 训练早期 vs 后期的对局对比（并排图或 GIF）")
"""))

# ---------------------------------------------------------------------------
# 第 3 部分：成员 3 —— DQN 与 Double DQN
# ---------------------------------------------------------------------------
cells.append(md(
"""---
# 3. 方法二：DQN 与 Double DQN　【成员 3 负责】

## 3.1 方法原理

> **【成员 3 填写】** 讲清楚：
> 1. **DQN**：经验回放（`ReplayBuffer`）、目标网络、ε-greedy 衰减。
> 2. **Q 值高估问题**：为什么 max 操作会系统性高估。
> 3. **Double DQN**：用在线网络选动作、目标网络估值，如何缓解高估（参考 `dqn.py` 的 `bootstrap_values`）。
> 本成员不涉及 baseline；专注价值方法本身的改进。

## 3.2 构建模型
"""))

cells.append(code(
"""# 构建 DQN Agent（成员 3）
env = SnakeEnv(grid_size=6, end_score=4, max_steps=100)
agent_dqn = DQNAgent(
    input_channels=env.observation_space.shape[0],
    action_count=env.action_space.n,
    grid_size=6,
    learning_rate=3e-4,
    device="auto",
)
print("online 网络参数:", agent_dqn.online)
env.close()
"""))

cells.append(md(
"""## 3.3 训练（或加载模型）

> 训练入口在 `train_dqn.py` 的 `train_dqn()`；Double DQN 通过 `double_dqn=True` 开关。
> **成员 3 的实质改动**：将经验回放改为优先级经验回放（PER），或改 ε 衰减方式，做对照。
"""))

cells.append(code(
"""if TRAIN_MODEL:
    from train_dqn import DQNTrainConfig, train_dqn
    for double_dqn, name in [(False, "dqn"), (True, "double_dqn")]:
        summary = train_dqn(DQNTrainConfig(
            episodes=1000,
            grid_size=6, end_score=4, max_steps=100,
            double_dqn=double_dqn,
            seed=42, device="auto",
            output_dir=Path("runs") / name,
            tensorboard=False,
        ))
        print(name, summary)
else:
    for name in ("dqn", "double_dqn"):
        ckpt = Path("runs") / name / "checkpoints" / "best.pt"
        print("加载", name, "checkpoint:", ckpt, "存在:", ckpt.exists())
"""))

cells.append(md(
"""## 3.4 DQN vs Double DQN 对比（含 Q 值诊断）

> **【成员 3 填写 + 主讲】** 用 `benchmark_double_dqn.py` 的结果画对比图，重点展示：
> 初始 Q 均值、实际折扣回报、Q–回报 gap、Q 高估率、目标选择差值、动作分歧率。
> 用这些诊断量证明「Double 缓解了高估」，再落到最终 reward/score/success 的对比。
"""))

cells.append(code(
"""# DQN vs Double DQN 对比（成员 3：读 benchmark_double_dqn.py 导出的 results.csv/json 画图）
import pandas as pd
# df = pd.read_csv("runs/double-dqn-benchmark/results.csv")
# 画 Q 高估率、Q-回报 gap、reward/success 的柱状对比（含误差棒）
print("TODO: 绘制 DQN vs Double DQN 的 Q 值诊断与性能对比图")
"""))

cells.append(md(
"""## 3.5 训练进度可视化（成员 3）

> **【成员 3 填写 + 必做】** 同 §2.5，渲染 DQN（或 Double DQN）训练早期 vs 后期的对局对比。
"""))

cells.append(code(
"""print("TODO: 渲染 DQN 训练早期 vs 后期的对局对比")
"""))

# ---------------------------------------------------------------------------
# 第 4 部分：成员 4 —— PPO
# ---------------------------------------------------------------------------
cells.append(md(
"""---
# 4. 方法三：PPO　【成员 4 负责】

## 4.1 方法原理

> **【成员 4 填写】** 讲清楚：
> 1. **on-policy vs off-policy** 的区别（PPO 是 on-policy）。
> 2. **GAE**（广义优势估计）与 λ 的含义（参考 `ppo.py` 的 `generalized_advantage_estimate`）。
> 3. **clipped objective**：importance sampling 的 ratio 用 clip 限制，实现「小步、稳定的策略更新」。
> 本成员不涉及 baseline。

## 4.2 构建模型
"""))

cells.append(code(
"""# 构建 PPO Agent（成员 4）
env = SnakeEnv(grid_size=6, end_score=4, max_steps=100)
agent_ppo = PPOAgent(
    input_channels=env.observation_space.shape[0],
    action_count=env.action_space.n,
    grid_size=6,
    learning_rate=3e-4,
    device="auto",
)
print(agent_ppo.model)
env.close()
"""))

cells.append(md(
"""## 4.3 训练（或加载模型）

> 训练入口在 `train_ppo.py` 的 `train_ppo()`。
> **成员 4 的实质改动**：新增 PPO vs Actor-Critic 对照，或对 `clip_ratio` / `gae_lambda` 做敏感性分析。
"""))

cells.append(code(
"""if TRAIN_MODEL:
    from train_ppo import PPOTrainConfig, train_ppo
    summary_ppo = train_ppo(PPOTrainConfig(
        episodes=1000,
        grid_size=6, end_score=4, max_steps=100,
        seed=42, device="auto",
        output_dir=Path("runs/ppo"),
        tensorboard=False,
    ))
    print(summary_ppo)
else:
    ckpt = Path("runs/ppo/checkpoints/best.pt")
    print("加载 PPO checkpoint:", ckpt, "存在:", ckpt.exists())
"""))

cells.append(md(
"""## 4.4 PPO 分析与训练进度可视化

> **【成员 4 填写 + 牵头】** 成员 4 额外负责编写**「训练进度可视化」通用脚本**
> （输入不同阶段 checkpoint，批量渲染并排对比图/GIF），供成员 2/3 复用。
> 这里先产出 PPO 自己的：① PPO vs AC 对照结果；② PPO 训练早期 vs 后期对局对比。
"""))

cells.append(code(
"""# PPO 分析（成员 4：PPO vs AC 对照 + 训练进度可视化）
print("TODO: 绘制 PPO vs AC 对比；渲染 PPO 训练早期 vs 后期对局对比")
print("TODO: 提供「训练进度可视化」通用脚本，供成员 2/3 复用")
"""))

# ---------------------------------------------------------------------------
# 第 5 部分：成员 5 —— 统一评估与对比
# ---------------------------------------------------------------------------
cells.append(md(
"""---
# 5. 统一评估与算法对比　【成员 5 负责】

## 5.1 公平对比方法

> **【成员 5 填写 + 主讲】** 说明为什么要在**统一种子 + 独立评估环境 + 固定训练预算**下比较：
> 消除随机性影响，保证对比公平。对比对象：random、teacher、AC、DQN、PPO 五种策略
> （由 `benchmark.py` 实现）。**random 是下界基线，teacher 是启发式上界基线**。
"""))

cells.append(code(
"""# 统一 benchmark（成员 5：可直接 import benchmark.py 的 run_benchmark，或读取其导出的 results.json/csv）
import pandas as pd
# 若需要重新跑（耗时），用 TRAIN_MODEL 控制；否则读已导出的结果
# from benchmark import run_benchmark
# result = run_benchmark(seeds=[7, 42, 2026], episodes=1000, eval_episodes=100, ...)
print("TODO: 读取/运行 benchmark.py，整理五策略的 reward/score/success 汇总表")
"""))

cells.append(md(
"""## 5.2 对比结果可视化

> **【成员 5 填写 + 必做】** 画五策略的性能对比图（柱状图/箱线图 + 误差棒），
> 并加一个简单的统计检验（如配对 t 检验）说明差异是否显著。
"""))

cells.append(code(
"""# 五策略对比可视化（成员 5）
print("TODO: 绘制 random/teacher/AC/DQN/PPO 的 reward、score、success_rate 对比图（含误差棒）")
print("TODO: 可选——配对 t 检验比较策略间差异显著性")
"""))

cells.append(md(
"""## 5.3 学习曲线汇总对比

> **【成员 5 填写】** 把各算法的学习曲线画在同一张图上，展示收敛速度与最终水平的差异。
"""))

cells.append(code(
"""print("TODO: 汇总 AC / DQN / PPO 学习曲线到同一图，比较收敛速度与最终水平")
"""))

# ---------------------------------------------------------------------------
# 第 6 部分：成员 5 —— 决策复盘
# ---------------------------------------------------------------------------
cells.append(md(
"""---
# 6. 决策复盘（可解释性）　【成员 5 负责】

> **【成员 5 填写】** 用 `inspect_decisions.py` 让 **4 个训练好的模型（AC / DQN / Double DQN / PPO）**
> 读取同样的固定局面，比较它们的动作选择与碰撞风险。
> 注意：该项目原本是 6 模型（含 Dueling），去掉 Dueling 后需改成 4 模型再运行。
"""))

cells.append(code(
"""# 决策复盘（成员 5：改 4 模型后运行 inspect_decisions，产出对比图和 JSON 报告）
print("TODO: 运行 inspect_decisions.py（4 模型版），展示固定局面下各模型的决策差异")
"""))

# ---------------------------------------------------------------------------
# 第 7 部分：成员 5 —— 结论
# ---------------------------------------------------------------------------
cells.append(md(
"""---
# 7. 结论　【成员 5 负责】

> **【成员 5 填写 + 主讲】** 总结：
> 1. 哪些算法最终最强，Double DQN 相对 DQN 的提升有多大。
> 2. baseline 与学习算法的差距说明了什么（随机≈0、教师稳定但短视、学习算法能反超）。
> 3. 本项目的局限与可改进方向（更大网格、更多训练、PER、连续动作等）。
"""))

# ---------------------------------------------------------------------------
# 第 8 部分：团队贡献与 AI 声明
# ---------------------------------------------------------------------------
cells.append(md(
"""---
# 8. 团队贡献与 AI 使用声明　【全员 → 成员 5 汇总】

## 8.1 团队贡献分工

> **【全员填写，成员 5 汇总】** 按实际贡献填写，例：

| 成员 | 学号 | 负责章节 | 主要贡献 |
|---|---|---|---|
| 成员 1 | | §1 游戏介绍与环境建模 | 环境设计、规则教师、游戏可视化 |
| 成员 2 | | §2 Actor-Critic | AC 训练、行为克隆对照实验 |
| 成员 3 | | §3 DQN 与 Double DQN | DQN/Double 训练、Q 值诊断对比 |
| 成员 4 | | §4 PPO | PPO 训练、训练进度可视化脚本 |
| 成员 5 | | §5–§7 统一对比/复盘/结论 | benchmark、决策复盘、notebook 整合 |

> 若「所有成员贡献相同」，请明确写「all members contributed equally」。
"""))

cells.append(md(
"""## 8.2 生成式 AI 使用声明

> **【全员确认后填写】** 与 §0.3 一致，如实声明生成式 AI 的使用情况。
"""))

# ---------------------------------------------------------------------------
# 附录
# ---------------------------------------------------------------------------
cells.append(md(
"""---
# 附录 A. 复现说明

- **依赖**：`gymnasium`、`torch`、`numpy`、`matplotlib`、`pygame`、`tensorboard`（见 `requirements.txt`）。
- **训练**：`python main.py`（AC）、`python train_dqn.py`（DQN，`--double-dqn` 开 Double）、`python train_ppo.py`（PPO）。
- **对比**：`python benchmark.py`、`python benchmark_double_dqn.py`。
- **演示**：`python play.py runs/snake/checkpoints/best.pt` 等。
- **checkpoint 位置**：各算法输出到 `runs/<algo>/checkpoints/`，最终提交时随 notebook 一起打包上传。
"""))

# ---------------------------------------------------------------------------
# 组装与写出
# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path("DSA5102_final_project_snake.ipynb")

if HAS_NBFORMAT:
    nb = nbformat.from_dict(notebook)
    nbformat.validate(nb)
    nbformat.write(nb, OUT)
else:
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

print("已生成:", OUT.resolve())
print("cell 数量:", len(cells))
