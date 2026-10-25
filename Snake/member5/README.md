# Member 5: Unified Evaluation and Analysis

PPO is now integrated with the existing Random, Teacher, AC, BC + AC, DQN and Double DQN configurations. PPO uses clip=0.2, fixed before common evaluation. All saved checkpoints remain unchanged; this package performs no training.

## Open in Jupyter

1. Download the complete group repository and merge this update into its existing `Snake` directory.
2. Open `Snake/member5/Member5_Run.ipynb` in Anaconda Jupyter, using Python 3.11 or 3.12.
3. Run the dependency-installation cell if needed, then run the cells in order.
4. Leave `TRAIN_MODEL=False`, `RUN_SMOKE=False`, `RUN_EVALUATION=False` and `RUN_ANALYSIS=False` to inspect the included results. The notebook still verifies that all 25 trained checkpoints load and produce finite, repeatable inference.
5. Set `RUN_EVALUATION=True` only when you want to repeat the complete 5,400-episode common evaluation. No models are retrained. New runs get separate timestamped directories.

`RUN_SMOKE=True` runs a separate 70-episode pipeline check. Smoke output is never used as final evidence. `RUN_ANALYSIS=True` regenerates the analysis from the selected common result directory without rerunning games.

## Included verified outputs

- [Full common evaluation](results/evaluation/20261025T094915_718765Z/report.md): five training seeds per learned method, 200 games per model, and 200 games for each baseline; 5,400 games total.
- [Integrated analysis](results/analysis/20261025T095930_719652Z/report.md): final performance, paired improvement comparisons, outcomes, historical learning curves, original PPO sensitivity results and original Member 2 training-stage gameplay.
- [Fixed-state inspection](results/decisions/20261025T095932_908221Z/report.md): five learned methods on four fixed 8 x 8 states, training seed 42.
- [New PPO-inclusive smoke test](results/smoke/20261025T094651_856757Z/report.md): 70 games, for pipeline verification only.
- `member5_report.md`: a readable copy of the integrated analysis with links relative to this folder.

## Files and commands

| File | Purpose |
|---|---|
| `Member5_Run.ipynb` | Executed Jupyter walkthrough; loads included results by default |
| `member5_unified_evaluation.py` | Loads saved weights, plays common evaluation games and saves raw records |
| `member5_analysis.py` | Verifies 25 checkpoints and generates paired statistics, curves, figures and analysis |
| `../inspect_decisions.py` | Fixed-state inference including PPO; no training |
| `LATEST_RUNS.json` | Portable paths to the included verified runs |
| `requirements.txt` | Required Python packages |
| `results/` and `figures/` | Recorded data, protocols, reports and figures |

From `Snake/member5`:

```bash
python -m pip install -r requirements.txt
python member5_unified_evaluation.py --mode smoke
python member5_unified_evaluation.py --mode evaluation
python member5_analysis.py
```

In Jupyter, fixed-state inspection can also be run separately with `%run ../inspect_decisions.py --seed 42`.

## Interpretation

The board is 8 x 8, success is length 8 (seven foods), and episodes last at most 300 steps. Test seeds 1,200,000-1,200,199 are common to all methods. Learned policies use greedy argmax; no collision mask or teacher takeover is applied.

AC/BC + AC use 50,000 autonomous training interactions, DQN/DDQN 210,000, and PPO 500,000. BC + AC has 100 additional demonstration episodes. The overall comparison describes delivered trained outcomes under common evaluation; it does not establish equal-resource algorithm superiority. Original validation-selected checkpoints are retained even if selected before the final training step.

Within-family score comparisons pair five training seeds. Paired t-tests with Holm adjustment and descriptive bootstrap intervals are exploratory, with only five independent training runs. Historical learning curves retain the members' different validation protocols and are not pooled into the common final table. Original PPO clipping results also retain their separate test seeds.

The copied training-stage GIF is Member 2's BC-assisted Actor-Critic demonstration, not a PPO training-progress claim. The final group submission still needs to merge all members' full methods and the required human video presentation.

Original members supplied the game, algorithms, weights, histories and animation. OpenAI Codex assisted Member 5's evaluation, analysis, documentation and verification; students are responsible for understanding and presenting the work.
