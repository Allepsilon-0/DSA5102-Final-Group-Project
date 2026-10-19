# DSA5102 成员2：便于 GitHub 上传的版本

上传本目录的所有顶层文件即可，不需要上传 `seed-*` 文件夹。

- `member2_assets.zip`：所有训练种子的模型、日志、汇总表、图和参考 notebook，合并成一个文件。
- `member2_assets.py`：首次运行时自动解压；不会重新训练，也不会覆盖已存在的数据文件。
- `DSA5102_snake_member2.ipynb`：成员2独立完整版，建议先看。
- `DSA5102_final_project_snake_member2_completed.ipynb`：供小组整合的版本，其他成员原有待完成内容仍保留。
- `member2_section2.md`：成员2报告。
- 其他 `.py` 与依赖文件：保持与 notebook 在同一目录。

下载仓库后，安装 `requirements.txt` 的依赖，从本目录打开 notebook 并运行。保持 `TRAIN_MODEL=False`，程序会先解压数据，再加载已训练模型。只有第一次运行需要解压。

`seed-*` 是不同随机种子的重复实验记录。默认 notebook 只加载 `seed-7/bc_ac/checkpoints/best.pt` 来核对200局结果；其他模型保留在压缩文件中，用于完整复现和分析。

如果需要让 GitHub 直接显示 Markdown 中的外链图片，可在本地首次解压后额外上传 `runs/member2/final/figures/`。即使不额外上传图片，已执行的 notebook 也包含图表和 GIF 输出；运行 notebook 后报告的图片路径恢复正常。

需要直接运行完整实验命令前，可先执行 `python -c "from member2_assets import ensure_assets; ensure_assets()"`。详细说明及讲解提纲见 `README_成员2使用说明.md`。
