# 实验 10 训练进步可视化

## 实验目的

直接展示智能体的游戏行为如何随训练推进而变化，并把学习曲线中的数值趋势转化为可解释的行为证据。

## 在实验链中的位置

该实验使用正式训练中保存的 checkpoints，并与最终评估共同构成项目结论。它直接对应题目中“visually demonstrate improvement”的硬性要求。

## 方法

- 保存训练进度约 0%、25%、50%、75%、100% 的 checkpoints。
- 每个 checkpoint 使用相同的 3–5 个 showcase seeds，评估时关闭探索。
- 画面叠加 checkpoint/environment steps、score、snake length、当前动作和 seed。
- 对同一 seed 的早、中、晚阶段并排或连续展示。
- 最终模型至少播放一局完整过程，并包含失败案例。

## 控制变量

所有 checkpoints 使用同一环境版本、showcase seeds、评估策略和录制帧率。不得只选择每个阶段最幸运的一局。

## 希望达成的结果

- 早期表现为随机碰撞或难以吃食，中期开始稳定追食和避障，后期在更长蛇身下保持更高 score。
- 录像中的行为改善与多 seed evaluation curve 的总体趋势一致。
- 同时揭示最终模型仍存在的局限，例如局部观测导致的自我包围或距离奖励诱发的短视行为。

## 必须产出

1. 游戏规则示意图；
2. Random/Greedy score 箱线图；
3. 核心算法多 seed 学习曲线；
4. observation 与 reward ablation 曲线；
5. 最终 score 分布与死亡类型图；
6. checkpoint montage；
7. 最终完整一局及典型失败一局。

每张图必须标出横纵轴、单位、seed 数量和阴影含义。学习曲线证明总体趋势，录像解释行为机制，两者不能互相替代。

## 完成条件

可视化覆盖训练全过程、使用固定局面、公平展示成功和失败，并能在 20 分钟五人视频中形成清晰故事线。
