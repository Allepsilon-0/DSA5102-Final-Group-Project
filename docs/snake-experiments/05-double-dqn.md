# 实验 5 Double DQN

## 实验目的

检验将“选择动作”和“评价动作”分离后，是否能减小 DQN 的乐观过估计并改善训练稳定性或最终游戏表现。

## 在实验链中的位置

该实验紧接 Vanilla DQN，是最干净的单变量算法改进：只改变 Bellman target 的计算方式，其他训练组件保持不变。

## 方法

在线网络选择下一动作，目标网络评价该动作。使用与 DQN 相同的训练 seeds、环境步数、网络规模、replay、epsilon 日程、checkpoint 和测试局面。

## 控制变量

唯一主要变化是 target 计算。若同时改变网络宽度、学习率、奖励或观测，将无法把差异归因于 Double DQN。

## 希望达成的结果

- DDQN 的估计 Q 与实际 Monte Carlo return 之间的正偏差更小。
- 多 seed 学习曲线波动不高于 DQN，训练崩溃次数减少。
- 最终 score 可能提高，但这不是预设结论；即使 score 接近，估计更校准也可构成有效发现。

## 必须产出

- DQN 与 DDQN 的配对学习曲线；
- overestimation gap 或 Q calibration 图；
- 最终 score、方差、死亡类型和训练耗时对比；
- 相同局面的行为录像。

## 继续条件

必须确认两者除 target 公式外使用同一协议，并完成至少 3 个训练 seeds 后再决定默认算法。
