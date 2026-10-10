# Member 3: Step-Budgeted DQN versus Double DQN

This directory contains the reproducible Member 3 experiment, frozen results, and report-ready
figures. The final notebooks are kept locally and are not included in this directory.

## Research question

Under an equal environment-interaction budget, does Double DQN improve value estimation,
performance, or stability relative to standard DQN on the common 8×8 Snake task?

The comparison changes only the Bellman-target calculation. Architecture, replay settings,
exploration schedule, training seeds, validation protocol, and total environment steps are shared.

## Files

- `member3_step_budget_experiment.py` — complete smoke/final/analysis workflow.
- `requirements.txt` — Python dependencies for the experiment script.
- `Dockerfile` — CPU-only Python and JupyterLab environment used for reproducible execution.
- `results/final/` — frozen five-seed final results, per-run histories, completion markers, and the
  selected best checkpoints.
- `figures/` — report-ready figures regenerated directly from the frozen CSV/JSON artifacts.

Intermediate 50k/100k/150k/200k and `latest.pt` checkpoints are intentionally omitted to avoid
unnecessary repository size. Each selected `best.pt` is included; a full training run regenerates
all intermediate checkpoints.

## Frozen protocol

- Board: 8×8; target length 8; maximum 300 steps per episode.
- Training seeds: 7, 42, 2026, 31415, 27182.
- Budget: 210,000 environment steps per model.
- Validation: 30 greedy episodes every 20,000 steps on fixed held-out seeds.
- Final test: 200 greedy episodes per selected model on an untouched seed range.
- Primary outcomes: score and success rate.
- Mechanism outcomes: initial-state Q-return calibration and shared-state online/target diagnostics.
- Uncertainty: paired bootstrap intervals with training seed as the experimental unit.

## Main result

Double DQN increases mean score from 5.388 to 5.534 and success rate from 40.7% to 44.4%, while
reducing their cross-seed standard deviations. However, the paired seed-bootstrap intervals for
both gameplay outcomes cross zero. Absolute Q-return error does not improve. The evidence therefore
supports a modest, seed-dependent average improvement rather than a decisive superiority claim.

## Running the experiment

The script defaults to `analysis`, which reloads the frozen experiment. To reproduce the training
pipeline, first use `smoke`. After it completes successfully, use `final` for the frozen five-seed
protocol.

From the repository's `Snake` directory, run:

```powershell
$env:MEMBER3_MODE = 'analysis'
python '.\member 3\member3_step_budget_experiment.py'
```

Replace `analysis` with `smoke` for a short test or `final` for full retraining. A full final run can
take a long time and will update files under `results/final/`.

To use Docker, build the image from this directory, then mount the repository root that contains
`DSA5102-Final-Group-Project`:

```powershell
docker build -t dsa5102-snake-v3 .

$projectRoot = 'D:\path\to\folder-containing-DSA5102-Final-Group-Project'
docker run --rm -it --name dsa5102-member3-v3 -p 127.0.0.1:8888:8888 --mount "type=bind,source=$($projectRoot),target=/workspace" dsa5102-snake-v3
```

In the JupyterLab terminal, run:

```bash
cd /workspace/DSA5102-Final-Group-Project/Snake
MEMBER3_MODE=analysis python "member 3/member3_step_budget_experiment.py"
```

The script finds the parent `Snake` source directory automatically.

## Reproducibility notes

- `results/final/protocol.json` records the frozen protocol.
- `results/final/environment_versions.json` records the actual Python and package versions.
- A run is considered complete only when all ten model/seed directories contain `complete.json`
  and `checkpoints/best.pt`.
- The frozen test seed range must not be reused for further hyperparameter tuning.
