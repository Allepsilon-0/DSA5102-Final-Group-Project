> **GitHub 上传简化版说明**：本版本将所有 `runs/` 实验记录、模型和 `reference/` 参考文件放在单个 `member2_assets.zip` 中。上传时只需上传本文件夹顶层文件，不需要手动上传任何 `seed-*` 文件夹。notebook 首次运行时自动解压，之后下文提到的 `runs/` 等路径即可使用。请保留新增的 `member2_assets.py` 和压缩文件。

# DSA5102 贪吃蛇：成员2交付与使用说明

本包只完成成员2的 **Actor-Critic + 行为克隆**。代码、报告和 notebook 中的成绩来自实际执行；没有将成员1或成员3的结果当作本人的实验结果。

## 先打开哪个文件

- **`DSA5102_snake_member2.ipynb`**：成员2独立完整版，包含方法、完整的模型/训练/玩法代码、实际运行输出、图表、GIF 和结论。建议先阅读这个文件。
- **`DSA5102_final_project_snake_member2_completed.ipynb`**：在你提供的最后一个 notebook 上，只替换成员2的原第2节，另外插入该节所需的设置单元格。其他成员的原始代码和文字保留，原有的待填写内容也保留。它是供小组继续合并的版本，不能视为全组已完成的最终报告。
- **`member2_section2.md`**：与成员1/3说明风格配套的英文第2节报告，包含实际成绩、损失公式、局限和 AI 使用说明，可交给负责整合的成员。

请解压整个文件夹后使用。不要只移动 notebook；`runs/` 中的模型与数据、同目录的 `.py` 文件都属于运行依赖。

## 文件与责任边界

| 文件 | 内容 |
|---|---|
| `model.py` | 原有小型 CNN、Actor/Critic 双头、TD 目标、策略/价值/熵损失和 BC 交叉熵 |
| `main.py` | 分离教师预热与自主 AC 阶段；独立验证；CSV 日志；阶段和最佳模型保存 |
| `play.py` | 无教师辅助的游戏；固定种子的分阶段对局 GIF，可无窗口运行 |
| `member2_experiment.py` | 0 对 100 局预热的五种子配对实验、独立测试、统计和绘图 |
| `gen_notebook.py` | 从现有证据重新生成第2节 MD 和两个 notebook |
| `test_member2.py` | 关键正确性与接口检查 |
| `environment.py` | 从原仓库复制的成员1环境，未修改 |
| `dqn.py`、`ppo.py` | 原 notebook 全局导入所需的原样支持文件，未修改，也不属于成员2的新贡献 |
| `reference/` | 从用户提供的 GitHub 网页提取出的原 notebook，供重建整合版本 |
| `runs/member2/final/` | 完整训练日志、逐局测试数据、选定模型与阶段模型、汇总表和图 |

桌面上的原 `.py` / `.ipynb` 实际是 GitHub HTML 网页。本包使用提取出的真实源码，并与用户指定仓库核对一致；本包里的 notebook 是有效的 Jupyter 文件。

## 默认使用：直接加载已训练模型

1. 将整个 `member2_snake` 文件夹放在一起，进入该目录。
2. 使用 Python 3.11 或更高版本建立环境，再安装 `requirements.txt` 中的依赖。例如：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

3. 在 Jupyter 或 VS Code 打开 `DSA5102_snake_member2.ipynb`，选择上述环境并运行所有单元格。若使用 JupyterLab，可另外安装 `jupyterlab`。
4. 保持 `TRAIN_MODEL = False`。notebook 会真正加载权重，并重新运行选定模型的200局测试来核对保存的结果；不会重新训练。

`requirements-lock.txt` 记录本次实验环境的精确版本，`runs/member2/final/versions.json` 记录 Python、平台与核心库版本。跨系统复现通常使用 `requirements.txt`，不要求强行安装不同平台的全部锁定依赖。

## 想复现或修改实验

```bash
# 独立目录的小规模端到端检查，不代表正式结果
python member2_experiment.py --mode smoke

# 复用已完成的正式实验；若模型缺失，则补跑对应实验
python member2_experiment.py --mode final

# 需要全新复跑时，指定新目录，避免覆盖本包的原始证据
python member2_experiment.py --mode final --output runs/member2/replication

# 不训练，重新汇总、绘图和渲染固定种子 GIF
python member2_experiment.py --mode analysis

# 生成 MD 和两个 notebook；会重置 notebook 执行输出，需要重新运行它们
python gen_notebook.py

# 正确性检查
python test_member2.py
```

每个模型使用50,000个自主交互步。BC 组额外使用100局教师示范；这部分数据量和计算成本在报告里单列。两组的模型结构、学习率、奖励、验证/测试种子相同。测试种子不参与选模型。5个训练种子是统计单位，不把1,000个测试回合当成1,000次独立训练。

`episodes` 仍保留原框架语义：例如 `episodes=500, teacher_episodes=100` 表示100局教师阶段和400局自主阶段。正式对照改用 `rl_steps=50000` 控制自主预算，因此不再用 `episodes` 限制自主阶段。

中断后只复用有 `complete.json` 和 `checkpoints/best.pt` 的完整模型；未完成模型从头跑。精确的中途断点续训没有实现，程序会明确拒绝 `resume`，避免假装恢复了随机数和示范缓冲区。

## 给负责全组整合的同学

1. 把完成后的第2节合并到全组最新 notebook；请优先按章节合并，避免用本包中的整合版覆盖其他人之后的新工作。
2. 一起合并 `model.py`、`main.py`、`play.py`、`member2_experiment.py`，以及 `runs/member2/final/` 中的数据与模型。保留小组现有的 `environment.py`；本包环境与原仓库版本相同。
3. 原来的 `ConvActorCritic`、`TrainConfig`、`train()`、`as_tensor()`、`choose_device()` 及 checkpoint 字典字段保留，便于统一 benchmark 加载。主对照的模型在 `seed-<种子>/ac/checkpoints/best.pt` 和 `seed-<种子>/bc_ac/checkpoints/best.pt`。
4. 成员3的实验使用210,000步，成员2本次使用50,000步，不能直接用两份独立报告的数字宣称算法排名。全组比较应由成员5统一预算和测试流程后重测。
5. 小组应在总声明中合并成员2如实记录的 AI 辅助使用情况。此处不替其他成员声明，也没有填写未经确认的姓名或学号。

## 约4分钟讲解提纲

- **0:00–0:40**：为什么随机探索很难吃到食物；本节比较“有/无教师预热”。教师来自成员1，只在预热时提供标签。
- **0:40–1:30**：一个 CNN、两个输出头。BC 是交叉熵；AC 用优势评价动作，策略损失提高好动作概率，价值损失训练预测，熵鼓励探索。说明 `detach` 的作用。
- **1:30–2:15**：两组各50,000自主步、5个配对种子、独立验证选模型、200局独立测试。BC额外成本需要交代。
- **2:15–3:20**：展示学习曲线、真实成绩和阶段 GIF。区分教师代玩的分数、预热后网络独立玩的分数、最后模型和验证选中的模型。
- **3:20–4:00**：解释配对差异与不确定性；哪些结论有数据支持，哪些只能列为后续改进。不要把一次好看的游戏录像当成整体胜率。

演示时由本人讲解并确认理解算法和结果；本包没有生成视频或合成声音。
