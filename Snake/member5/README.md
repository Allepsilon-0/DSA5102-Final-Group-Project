# Member 5: Unified Evaluation and Integration

It reuses the shared environment and model code and reads the existing checkpoints delivered by Members 2 , 3 and 4.

## Files and directories

| File or directory | Purpose |
|---|---|
| `README.md` | Setup, execution and upload instructions |
| `member5_report.md` | Research questions, evaluation protocol, metrics and interpretation |
| `member5_unified_evaluation.py` | Evaluation code extracted from the verified notebook; loads models, evaluates, aggregates and saves |
| `Member5_Run.ipynb` | A short Jupyter entry point that calls the Python script |
| `requirements.txt` | Required Python packages |
| `results/smoke/<timestamp>/` | Per-episode data, summaries, protocol and report from a smoke test |
| `figures/smoke/<timestamp>/` | Smoke-test figure |
| `results/evaluation/<timestamp>/` | Created when the full common evaluation is executed |
| `figures/evaluation/<timestamp>/` | Figure from that full evaluation |

The existing Anaconda/Jupyter setup is sufficient; a Dockerfile is optional. The program reads checkpoints from the other members' directories instead of duplicating them.

## First run in Jupyter

1. Download and extract the complete group repository, then place `member5` inside `Snake`.
2. In Anaconda Jupyter, open `Snake/member5/Member5_Run.ipynb` with a Python 3.11 or 3.12 environment.
3. Run the installation cell, `%pip install -r requirements.txt`. Restart the kernel if requested.
4. Keep `MODE="smoke"` and run the remaining cells in order.
5. Each of six configurations plays 10 episodes, giving 60 episodes in total. Results and figures are saved inside this directory.

If the shared source files cannot be found, check the folder placement. Alternatively, set `PROJECT_DIR` in the notebook to the absolute path of your local `Snake` directory.

## Command-line use

From `Snake/member5`:

```bash
python -m pip install -r requirements.txt
python member5_unified_evaluation.py --mode smoke
```

After agreeing on the full evaluation protocol:

```bash
python member5_unified_evaluation.py --mode evaluation
```

The `evaluation` mode still performs inference only. It evaluates five trained models per learned method, 200 episodes per model, plus 200 episodes each for Random and Teacher: 4,400 episodes in total. Every execution creates a new timestamped output directory.

To specify the shared source directory:

```bash
python member5_unified_evaluation.py --mode smoke --project-dir "/path/to/Snake"
```

## Result files

| File | Contents |
|---|---|
| `episodes.csv` | Actual score, reward, steps, success and outcome for each episode |
| `per_model.csv` | Episode averages for each trained model |
| `summary.csv` | Method summaries, training budgets and additional demonstration costs |
| `paired_scores.csv` | BC + AC minus AC, and Double DQN minus DQN, paired by training seed |
| `protocol.json` | Evaluation settings, checkpoint progress, source/checkpoint hashes and versions |
| `report.md` | Report generated for that run, with a relative figure link |

A smoke test verifies the pipeline; it does not establish a final ranking. This package does not contain a full evaluation run or PPO results and is not the complete final project submission.

## Included verified run

The 60-episode smoke test is documented in the [included run report](results/smoke/20261023T145446_150083Z/report.md).
