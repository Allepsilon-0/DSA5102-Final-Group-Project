# 实验 4 Vanilla DQN

## 实验目的

检验神经网络函数逼近能否利用不同观测之间的共享结构，学习出超过简单规则与表格方法的策略，并建立深度 RL 基线。

## 在实验链中的位置

DQN 是后续 DDQN 和 Dueling 的共同基准。只有 DQN 训练流程稳定，改进算法的比较才有意义。

## 方法

使用在线 Q 网络、目标网络、经验回放、epsilon-greedy、Huber loss 和梯度裁剪。网络输入为 O1 11D，输出三个相对动作的 Q 值。固定 checkpoint 以 `epsilon = 0` 评估，而不是用带探索的原始训练回报替代学习曲线。

## 控制变量

相对 Tabular Q 固定环境、观测、奖励、动作、训练步数、seeds 与评估协议。记录网络参数量与训练耗时，使性能增益和计算成本同时可见。

## 希望达成的结果

- 在多个 seeds 上出现一致的学习趋势，而不是单次偶然高分。
- 最终 score 至少明显高于 Pure Random，并争取达到或超过 Greedy 与 Tabular Q。
- 暴露 DQN 可能存在的 Q 值过估计、训练波动或灾难性退化，为 DDQN 提供机制动机。

## 必须产出

- 多 seed checkpoint evaluation curve 及不确定性带；
- 最终 100 局/seed 的 score 分布；
- loss、Q 值尺度、参数量和训练耗时；
- 成功与失败录像；
- 可加载的模型权重和完整配置。

## 继续条件

至少完成一个 20k–50k steps smoke test 并确认学习信号、target detach、终止状态处理和模型保存/加载正确，再启动正式多 seed 训练。
