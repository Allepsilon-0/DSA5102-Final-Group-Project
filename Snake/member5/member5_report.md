# Member 5: Unified Evaluation Methods and Interpretation

## 1. Research questions

Evaluate the models already completed by the group under common game rules and test seeds. Use the same evaluation data for the overall comparison and two within-family comparisons: AC versus BC + AC, and DQN versus Double DQN.

This evaluation does not retrain models or modify the implementations supplied by Members 2 and 3. It uses their original validation-selected `best.pt` checkpoints and never selects checkpoints using the current test results.

## 2. Compared configurations and existing training costs

| Configuration | Source | Autonomous training budget | Additional demonstrations |
|---|---|---:|---|
| Random | Seeded uniform random actions | No training | None |
| Teacher | Member 1's BFS rule-based policy | No training | The policy itself supplies rule-based actions |
| AC | Member 2, without BC warm-up | 50,000 steps | None |
| BC + AC | Member 2, with BC warm-up | 50,000 steps | 100 teacher episodes |
| DQN | Member 3 | 210,000 steps | None |
| Double DQN | Member 3 | 210,000 steps | None |

PPO, owned by Member 4, has not yet been integrated. Teacher is a strong heuristic reference, not a theoretical upper bound.

Training budgets differ across algorithm families. The overall comparison describes the delivered trained outcomes, rather than isolating algorithm quality under equal training resources. BC also uses additional demonstration data and optimization.

A validation-selected checkpoint may precede the end of the full training run. `protocol.json` records both the full budget and the selected checkpoint's actual training progress.

## 3. Environment, actions and goal

| Setting | Fixed value |
|---|---|
| Board | 8 x 8 |
| Initial snake length | 1 |
| Success condition | Length 8, corresponding to 7 food items eaten |
| Episode step limit | 300 |
| Observation | 7 x 8 x 8: snake, food, wall and four direction channels |
| Actions | 0 = left, 1 = right, 2 = straight |
| Evaluation reward mode | Shared environment's `sparse` mode |
| Teacher mode | `shortest_path`, using BFS |

The current `sparse` mode still contains step costs and a loop penalty; it is not a purely event-only reward. The evaluation preserves the shared reward and termination rules.

Each model receives an independent evaluation environment. Identical reset seeds provide reproducible random starts. Once policies take different actions, their subsequent trajectories and food-generation sequences need not remain identical.

## 4. Loading saved models

An AC checkpoint supplies `model_state_dict`; a DQN/DDQN checkpoint supplies `online_state_dict`. State dictionaries must match the network architecture strictly. Missing or incompatible weights raise an error rather than silently evaluating an untrained model.

Member 2 checkpoints are read from an extracted directory or directly from `member2_assets.zip`. The latter route does not extract or overwrite the original files, and the archive's verified SHA-256 is checked. Member 3 checkpoints are located under `member 3/results/final/seed-*/.../checkpoints/best.pt`.

AC uses the action with the largest actor logit. DQN and DDQN use the action with the largest online-network Q-value; their difference lies in the training target. Learned-agent evaluation has no teacher takeover, collision mask, search or parameter update. Random actions are not filtered for safety.

## 5. Smoke test and full evaluation

| Mode | Training seeds | Episodes per model | Test seeds | Purpose |
|---|---|---:|---|---|
| `smoke` | 7 | 10 | 1,100,000-1,100,009 | Check loading, gameplay, recording and saving |
| `evaluation` | 7, 42, 2026, 31415, 27182 | 200 | 1,200,000-1,200,199 | Common evaluation under a fixed protocol |

The smoke test contains 60 episodes. The full mode contains four learned configurations x five trained models x 200 episodes, plus two baselines x 200 episodes, giving 4,400 episodes.

The full-evaluation seeds must not be used for checkpoint selection or tuning. If future changes are guided by test outcomes, reserve a new, unused test range.

## 6. Metrics and aggregation

- `score`: food items eaten, from 0 to 7; the primary metric.
- `success`: whether the target length is reached.
- `reward`: cumulative return under the common evaluation reward function; a supporting metric.
- `steps`: episode action count; longer survival may reflect circling rather than useful progress.
- `outcome`: success, timeout or collision. The environment does not return separate wall/self-collision causes, so this version does not infer them.

First average evaluation episodes within each trained model. Then average those model means across training seeds and report their sample SD. Two hundred episodes from one model are not 200 independent training runs. A one-seed smoke test cannot estimate cross-training-seed variation; its SD is left as NaN rather than zero.

Random and Teacher have no training replicates. Their episode-level SD and the learned methods' cross-training-seed SD are different quantities and are not presented as interchangeable error bars.

## 7. Paired improvement comparisons

`paired_scores.csv` pairs models by training seed and computes BC + AC minus AC, and Double DQN minus DQN, in mean food score. The same test batch supports these comparisons and the overall summary.

The BC comparison measures the benefit of demonstration-assisted initialization under the same autonomous RL budget; it is not an equal-total-cost comparison. Interpreting the DQN/DDQN difference as an algorithmic improvement also depends on Member 3's internal training controls. The 10-episode, one-seed smoke test is not used for significance claims.

## 8. Current delivery status

The original notebook's 60-episode smoke test was executed with real checkpoints, and all 20 available learned checkpoints passed loading checks. The program was then extracted into a Python module. The included smoke run was regenerated using the English version.

The package contains pipeline evidence, not a final ranking. Every execution saves its CSV files, protocol and run report under `results/<mode>/<timestamp>/`; the corresponding figure is under `figures/<mode>/<timestamp>/`. The run report references the figure using a relative link.

## 9. Remaining integration work

The full common evaluation, PPO integration, learning-curve consolidation, training-stage gameplay, fixed-state decision analysis, final conclusions, final group notebook and human video presentation remain to be completed. This support directory does not replace the required final notebook.

## 10. Contributions and AI usage

The shared environment, algorithm implementations and previously trained weights come from the original group members. OpenAI Codex assisted with evaluation, documentation, file organization, English translation and execution checks. Member 5 is responsible for understanding, verifying, integrating and presenting this work, with accurate disclosure in the final group report.
