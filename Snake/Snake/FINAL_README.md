# DSA5102 consolidated final project

Open `DSA5102_Final_Project.ipynb` from this complete `Snake` directory in Anaconda Jupyter (Python 3.11 or 3.12 recommended). This is the one consolidated project report, covering game/baselines, AC/BC, DQN/Double DQN, PPO, common evaluation, decision analysis, conclusions and contributions.

If necessary, install dependencies in the active notebook environment with `%pip install -r FINAL_requirements.txt`, then restart the kernel and run all cells in order.

Keep `TRAIN_MODEL=False`, `RUN_COMMON_EVALUATION=False` and `RUN_QUICK_INFERENCE=True`. The default loads all 35 distinct trained models, reads the 5,400-episode common evaluation, draws the game illustrations and plays seven small new runtime-check games. No training starts. Set `RUN_COMMON_EVALUATION=True` only to replay the full common test; it writes new timestamped outputs without retraining or selecting new models.

The notebook needs `final_notebook_support.py`, shared `environment.py`/`model.py`/`dqn.py`/`ppo.py`, `inspect_decisions.py`, Member 2's model code and selected checkpoints/data under `runs/member2/final`, Member 3's final models/results/figures, Member 4's final models/results/source and Member 5's analysis modules/results/figures. Keep the directory structure intact; a notebook by itself is not the complete runnable submission. The complete ZIP contains unchanged copies of Member 2's ten selected weights rather than the original archive, so no old reference notebooks are bundled inside a nested ZIP. In the development repository, the existing archive fallback remains supported.

The complete submission ZIP contains exactly one notebook plus its dependencies, original saved models, data and figures. The GitHub update ZIP contains only the new consolidated notebook/support/README/requirements/verification/assets and should be merged into the existing repository's `Snake` folder.

The team roster was retained from the original group notebook. Confirm the member-number-to-name association in the contributions section. The required human video presentation remains a separate deliverable.

The old member notebooks can remain in the development repository. Use the consolidated notebook as the single report for the final submission. Family-specific tests retain their own original seeds; the common test compares delivered outcomes under different training budgets. AI assistance and the limitations are disclosed in the notebook.
