# 2. Actor-Critic with Behavior-Cloning Warm-Up

> **Owner:** Member 2. Scope: `model.py`, `main.py`, `play.py`, the BC ablation, and this notebook section.
> Member 1's environment and teacher implementations are reused unchanged. Member 3's DQN results are not included as new Member 2 experiments.

## 2.1 Motivation and Research Question

Snake gives useful positive rewards only when food is reached or the game is won. A randomly initialized policy often collides before reaching food, leaving a weak learning signal. We ask whether **100 episodes of supervised teacher demonstrations improve a one-step Actor-Critic agent compared with no behavior-cloning (BC) warm-up**, given the same subsequent autonomous interaction budget.

The comparison is **AC from scratch versus BC + AC**. BC has additional demonstration and optimization costs, so we describe it as an assisted initialization experiment, not equal-total-compute training. We also evaluate the saved BC-only and last-AC stages to distinguish imitation gains from later RL gains.

## 2.2 Shared Network and Algorithm

Both conditions use exactly the same network and paired initial weights:

```text
observation [batch, 7, 8, 8]
  -> Conv2d(7, 16, kernel=3, padding=1) -> ReLU -> Flatten [batch, 1024]
  -> actor: Linear(1024, 3) -> categorical action distribution
  -> critic: Linear(1024, 1) -> state-value estimate V(s)
```

There are **5,124 trainable parameters**. Actions follow Member 1's convention: 0 = left, 1 = right, 2 = straight. Training samples from the current policy; validation, test and animation use greedy actions. No teacher override, collision mask or search is used in learned-agent evaluation.

### 2.2.1 Behavior cloning

During warm-up, we explicitly ask Member 1's BFS teacher for an action, execute that same action, and store the `(observation, teacher action)` pair in a demonstration buffer with capacity 10,000. At every teacher transition, four mini-batches of up to 64 examples are sampled and optimized using categorical cross-entropy:

$$L_{BC}=-\frac{1}{B}\sum_{i=1}^B \log\pi_\theta(a_i^T\mid s_i).$$

The actor head and shared convolutional features are trained. The critic receives no direct supervised/value loss during BC. This makes the treatment specifically imitation warm-up. The original group's `value_update()` remains available for compatibility, but this experiment does not call it in warm-up. After the 100 teacher episodes, teacher queries and BC updates stop completely.

A high teacher trajectory score is not a learned-policy score: the teacher chooses those actions. We therefore evaluate the neural network independently immediately after BC (`post_bc.pt`).

### 2.2.2 One-step Actor-Critic

For a transition sampled by the current policy, we compute:

$$y_t = r_t+\gamma(1-z_t)\operatorname{stopgrad}[V_\phi(s_{t+1})],\qquad A_t=y_t-V_\phi(s_t).$$

Here $z_t$ is `terminated`: collision or success. An external 300-step limit is a `truncated` episode; it ends the rollout but still bootstraps the value of the final observation, following Gymnasium's time-limit convention. The remaining budget is not observed, so this is a continuing-task value approximation with capped evaluation episodes.

The minimized loss is:

$$L = \underbrace{-\log\pi_\theta(a_t\mid s_t)\operatorname{stopgrad}(A_t)}_{\text{actor}} + \underbrace{\tfrac12(y_t-V_\phi(s_t))^2}_{\text{critic}} - \underbrace{\beta H(\pi_\theta(\cdot\mid s_t))}_{\text{entropy bonus}}.$$

A positive advantage increases the probability of the selected action, while a negative advantage decreases it. Detaching the advantage prevents the actor term from using the critic as a shortcut; detaching the target prevents gradients through the next-state estimate. The critic fits the bootstrapped target. Entropy regularization encourages exploration; gradient norm clipping at 1.0 limits large updates. This is a compact, one-step on-policy AC implementation, without PPO clipping, GAE, or DQN replay.

## 2.3 Experimental Protocol

| Setting | Frozen value |
|---|---|
| Environment | Member 1's unchanged `SnakeEnv`; 8 × 8; target length 8; score ceiling 7 |
| Reward / teacher | sparse / BFS shortest-path |
| Episode time limit | 300 steps |
| Training seeds | 7, 42, 2026, 31415, 27182 |
| Conditions | `teacher_episodes=0` and `teacher_episodes=100` |
| Autonomous RL budget | exactly 50,000 transitions per model; 10 models total |
| Additional demonstrations | 100 teacher episodes only for BC + AC; counted separately |
| Optimizer | Adam, learning rate 1e-4, weight decay 1e-5 |
| Discount / entropy coefficient | gamma = 0.99 / beta = 0.01 |
| Validation | 30 greedy episodes, seeds 500000–500029, every 5,000 AC steps |
| Checkpoint selection | highest validation mean score, then success rate, then reward; earliest exact tie |
| Final test | 200 greedy episodes/model, seeds 900000–900199 |
| Runtime | CPU, one PyTorch thread; versions recorded in `versions.json` |

Both conditions start from the same weights within a seed. Their autonomous phase starts from the same random-number seed and environment reset-seed sequence. Once their actions differ, their visited states and episode lengths naturally differ. We pair by training seed, not by matching every state.

The original `episodes=500, teacher_episodes=100` API means 100 teacher episodes followed by 400 AC episodes. That gives different interaction counts across conditions. We retain that API for compatibility, but **the reported ablation uses `rl_steps=50000`**, so `episodes` does not limit the autonomous phase. The last training trajectory may end at the overall budget before the game finishes; it is marked `budget_partial` in the log and is never counted as a completed test episode.

Validation uses an independent environment and cannot change training weights. The final test is run only after checkpoint selection. The post-BC and latest checkpoints are additional, predeclared stage diagnostics; they are not selected by test performance. Baselines are evaluated on the same 200 test seeds with a separately seeded random-action generator.

## 2.4 Learning Curves and Training Cost

![Validation learning curves](runs/member2/final/figures/learning_curves.png)

*Fixed validation episodes plotted against autonomous environment steps. Shading shows standard deviation across five training seeds, not a confidence interval. Teacher-executed episodes are excluded.*

![Training reward and score](runs/member2/final/figures/training_reward_score.png)

*Autonomous stochastic training reward and score for the predeclared first seed (7), with 25-episode rolling means. These differ from greedy validation outcomes. The final budget-truncated trajectory is marked in the raw log.*

| Condition | Mean autonomous steps | Mean extra demo steps | Mean BC optimizer updates | Mean wall time |
|---|---:|---:|---:|---:|
| AC | 50000 | 0 | 0 | 37.1 s |
| BC + AC | 50000 | 3917.2 | 15668.8 | 52.0 s |

Wall time includes training and periodic validation, excludes the later final-test evaluation, and depends on this computer. BC + AC uses more total environment transitions and optimizer updates. Thus the experiment tests whether demonstration-assisted initialization helps, not whether it is computationally free.

## 2.5 Held-Out Gameplay Results

The following are means ± **sample standard deviation across the five training seeds**. Each seed's value is first averaged over its 200 test episodes.

| Learned method | Score (max 7) | Success rate | Reward |
|---|---:|---:|---:|
| Actor-Critic (no BC) | 0.127 ± 0.039 | 0.0% ± 0.0% | -3.828 ± 0.350 |
| BC + Actor-Critic | 1.668 ± 0.203 | 0.7% ± 0.8% | -2.596 ± 7.777 |

The independently rerun references on the same test seeds are: Random score **0.165**, legacy teacher **6.965**, BFS teacher **7.000**. The teachers are strong heuristic references, not theoretical upper bounds. These are Member 2's reruns rather than copied figures from Members 1 or 3.

AC from scratch does not exceed the independently rerun Random mean score at this budget. The BC + AC mean score is 1.668 out of 7, so the learned policy still has a substantial gap to the BFS teacher. The experiment therefore supports a narrowly defined initialization benefit rather than a claim that this small AC model solves Snake.

![Final performance](runs/member2/final/figures/final_performance.png)

*Only learned methods have cross-training-seed error bars. A missing baseline error bar does not imply zero episode variation; each baseline has one 200-episode evaluation.*

### Paired seed outcomes and uncertainty

| Training seed | AC score | BC + AC score | Difference |
|---:|---:|---:|---:|
| 7 | 0.135 | 1.910 | +1.775 |
| 42 | 0.140 | 1.505 | +1.365 |
| 2026 | 0.060 | 1.675 | +1.615 |
| 31415 | 0.140 | 1.820 | +1.680 |
| 27182 | 0.160 | 1.430 | +1.270 |

We bootstrap the five **paired training-seed differences** 20,000 times, with fixed bootstrap seed 20261019. Test episodes are not treated as 1,000 independent training runs. These small-sample bootstrap intervals describe uncertainty under this experiment and have limited coverage guarantees with only five seeds.

| Outcome | BC + AC minus AC | Paired bootstrap 95% interval |
|---|---:|---:|
| Mean score | +1.541 | [1.371, 1.705] |
| Success rate (percentage points) | +0.700 | [0.100, 1.400] |
| Reward | +1.232 | [-5.072, 7.265] |

The paired score interval is above zero in these runs. This supports a benefit from BC under this fixed protocol; it does not establish general superiority across tasks or hyperparameters.

The win rate nevertheless remains only **0.7%**. Mean reward changes by **+1.232**, with interval **[-5.072, 7.265]**, which includes zero. Thus the score gain is clearer than either reliable winning or improved total reward. Longer survival can accumulate loop and step penalties without eating more food; higher food score alone does not guarantee higher shaped return.

| Condition | Death rate | Time-limit truncation rate | Success rate |
|---|---:|---:|---:|
| AC | 100.0% | 0.0% | 0.0% |
| BC + AC | 63.7% | 35.6% | 0.7% |

![Paired outcomes and cost](runs/member2/final/figures/paired_outcomes_cost.png)

### Does later RL improve the BC initialization?

The independently tested BC-only snapshot has mean score **1.218** and success rate **1.0%** across the five seeds. The last BC + AC snapshot has mean score **1.121** and success rate **0.5%**. The validation-selected BC + AC result above is a third, separately defined outcome. Selected AC steps by seed are **7: 50,000, 42: 35,000, 2026: 40,000, 31415: 40,000, 27182: 25,000**.

These stage results separate learning from demonstrations, continued RL, and validation-based checkpoint selection. Validation-selected best performance must not be described as the guaranteed performance of the last update. In particular, a gain over AC from scratch does not by itself prove that RL improves every BC-only policy.

## 2.6 Visualizing Training Progress

![Fixed-seed gameplay across stages](runs/member2/final/figures/training_progress.gif)

The GIF uses **training seed 7** and **game seed 900000**, fixed before interpreting the result. It compares the randomly initialized network, the post-BC snapshot, the 25,000-step AC midpoint, and the checkpoint selected by validation. Every panel uses only the neural policy; completed games freeze in place. This is an illustrative trajectory, not statistical evidence.

- **Initial**: score 0, 300 steps, time limit.
- **After BC**: score 1, 9 steps, collision.
- **AC midpoint**: score 1, 11 steps, collision.
- **Best validation**: score 1, 300 steps, time limit.

We interpret the actual animation together with the learning curves and multi-episode table. The visual does not guarantee monotonic improvement: the midpoint can be worse than an earlier snapshot, and a good single trajectory can coexist with a low overall win rate.

## 2.7 Limitations and Interpretation

1. **Additional supervision.** BC + AC sees teacher-selected actions and has additional demonstration and optimizer costs. An equal-total-transition or equal-wall-time comparison would answer a different question.
2. **Small number of independent training runs.** Five paired seeds reveal variation but support only limited generalization. Episode-level averages do not replace training-seed replication.
3. **Limited board and budget.** We use one 8 × 8 environment, one parameter setting, and 50,000 RL steps. These numbers cannot be directly compared with Member 3's 210,000-step DQN experiment as a ranking of algorithms. Member 5 should rerun a shared protocol for group-wide comparisons.
4. **Representation and optimization limits.** The inherited seven-channel observation does not explicitly encode head identity, ordered body segments, loop-history or remaining time. It is not a fully Markov state for every reward/termination feature. A one-step bootstrapped critic, shared features and sparse large terminal rewards can produce unstable policy updates. BC also exposes the model mainly to teacher-visited states, so mistakes can move it to unfamiliar states.
5. **No automatic strongest-agent claim.** Learning is judged by autonomous held-out score and success, not teacher trajectories or training loss alone. The teacher remains a demanding reference on this small board.

Possible follow-up work is to test richer observations, longer or multi-step returns, or conservative policy updates on validation seeds, then reserve a new test range. These are future experiments, not completed results in this section.

## 2.8 Reproduction and Integration

- `python member2_experiment.py --mode smoke`: small pipeline check in a separate directory.
- `python member2_experiment.py --mode final`: reproduce or reuse the completed five-seed experiment in `runs/member2/final`.
- `python member2_experiment.py --mode analysis`: regenerate aggregate tables, figures and the fixed-seed GIF from saved logs and checkpoints.
- `python gen_notebook.py`: regenerate this Markdown report, the standalone Member 2 notebook, and the group notebook with only Section 2 replaced.
- `python test_member2.py`: check terminal targets, explicit BC actions, no teacher leakage in AC/evaluation, exact RL budgets, paired initialization and checkpoint loading.

The notebook defaults to `TRAIN_MODEL = False`, loads saved weights and reruns independent inference without training. `history.csv`, `validation.csv`, `test_episodes.csv`, saved checkpoints, `protocol.json`, source hashes and runtime versions accompany the report. Completed runs are reused only with a matching protocol and source hash; incomplete runs are rerun from scratch. Exact mid-run resume is explicitly unsupported.

`ConvActorCritic`, `TrainConfig`, `train`, `as_tensor`, `choose_device` and the original checkpoint dictionary keys remain available for the group benchmark. The group's defaults can still run the episode-based training interface. Files belonging to Members 3 and 4 are unchanged support copies; their report sections remain their responsibility.

## 2.9 Contribution and Generative AI Disclosure

Member 2's deliverables comprise the Actor-Critic/BC implementation, paired ablation, checkpoints and raw evidence, learning and gameplay visualizations, and the completed Section 2 report and notebook cells. Environment and teacher authorship remains with Member 1. No student name or ID has been inferred.

OpenAI Codex (generative AI) assisted with code implementation, debugging, experiment design, notebook integration and report drafting. The included results were produced by executing the supplied programs; they are not illustrative or fabricated numbers. This disclosure covers Member 2's work only. The group must consolidate all members' actual AI use in its final declaration. The human presenter should verify and understand the implementation and results before presenting them.

## References

- [Group source notebook and original framework](https://github.com/Allepsilon-0/DSA5102-Final-Group-Project/blob/main/Snake/DSA5102_final_project_snake.ipynb).
- [OpenAI Spinning Up: policy optimization and advantage functions](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html). Conceptual background; this project implements one-step TD Actor-Critic rather than Spinning Up's GAE-based VPG training loop.
- [Gymnasium: handling time limits](https://gymnasium.farama.org/v1.1.1/tutorials/gymnasium_basics/handling_time_limits/). Source for distinguishing termination from truncation in bootstrap targets.
