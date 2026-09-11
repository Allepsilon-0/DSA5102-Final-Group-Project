# 实验 3 Tabular Q learning

## 实验目的

在不引入神经网络的情况下验证 Bellman 更新、探索策略和 bootstrap 学习是否能在 11 维观测上形成有效策略，并判断函数逼近是否必要。

## 在实验链中的位置

它连接非学习基线与 DQN。Tabular Q 与 DQN 使用相同的价值学习目标，区别在于前者逐状态存值，后者使用共享网络参数泛化。

## 方法

将 11 个二值特征编码为 `0–2047` 的整数，建立 `2048 × 3` 的 Q 表，使用 Q-learning 更新与 epsilon-greedy 探索。与后续 DQN 使用相同环境步数、训练 seeds 和评估协议。

## 控制变量

固定 O1 11D 观测、Sparse reward、动作空间、训练预算、epsilon 日程、评估 checkpoint 和 seeds。模型特有的学习率可单独调优，但调优预算应记录。

## 希望达成的结果

- 明确看到 Q 值与固定 checkpoint score 随训练发生变化。
- 性能优于随机基线，并检验是否接近 Greedy。
- 若 Tabular Q 与 DQN 接近，说明低维观测下神经网络未必必要；若二者都较弱，优先怀疑 O1 的状态别名，而非简单增加网络深度。

## 必须产出

- checkpoint evaluation learning curve；
- 访问过的状态编码比例；
- TD error 或 Q 值变化摘要；
- 多 seed 最终 score；
- 与 Greedy 和 DQN 的公平对比表。

## 继续条件

先确认更新公式、终止状态 target 与评估时 `epsilon = 0` 均正确，再将同一实验协议迁移到 DQN。
