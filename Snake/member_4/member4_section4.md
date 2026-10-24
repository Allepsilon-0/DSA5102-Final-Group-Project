# 4. Proximal Policy Optimization and Training Progress

> **Owner:** Member 4  
> **Delivery status:** Implementation and methodological write-up completed; training and end-to-end execution have not been completed in this delivery environment. No PPO performance results or trained PPO weights are claimed. The experiment requires PyTorch, Gymnasium and pygame, which were unavailable and could not be installed. Five numerical GAE tests and Python syntax checks passed. The protocol below is fixed in code but remains unexecuted.

## 4.1 Research question and scope

This section studies Proximal Policy Optimization (PPO) on the same Snake game as Sections 1–3. Our question is: **how does the clipping parameter affect learning and final performance under a fixed interaction budget?** We compare clipping parameters of 0.1, 0.2 and 0.3. Everything else is held fixed.

This is the clipping-sensitivity option specified in the group notebook. It is not a PPO-versus-Actor-Critic experiment. Section 2 already examines Actor-Critic with and without demonstrations, while Section 3 compares DQN and Double DQN. Comparing their published scores directly with this experiment would confound the learning rule with architecture, training budget, demonstrations and evaluation policy. Such results may provide context, but they do not isolate an algorithmic effect.

The shared task uses an 8 × 8 board, three relative actions, and a seven-channel observation. Each episode starts with length one. Reaching length eight is a success, so the maximum food score is seven. Episodes also end on collision or after 300 steps. The environment file is copied without changing its rules.

## 4.2 Policy learning and the actor–critic model

A policy \(\pi_\theta(a\mid o)\) gives the probability of choosing action \(a\) after observing \(o\). The actor produces three logits, which define a categorical distribution over left, right and straight. The critic estimates \(V_\phi(o)\), the expected discounted return under the current policy. Both heads share a convolutional encoder:

\[
(7,8,8)\rightarrow \operatorname{Conv}_{3\times3}(16)
\rightarrow\operatorname{ReLU}\rightarrow\operatorname{Flatten}
\rightarrow\operatorname{Linear}(64)\rightarrow\operatorname{ReLU}.
\]

The actor maps the 64 features to three logits; the critic maps them to one scalar. This is the architecture in the supplied `ppo.py`. It differs from the simpler Actor-Critic network in `model.py`, which has no additional 64-unit layer.

**On-policy learning** updates a policy using trajectories collected by its current or recently frozen behavior policy. PPO first collects one rollout using a fixed policy, then performs several optimization passes over that rollout. The rollout is discarded before collecting the next one. Reusing a fresh rollout for a small number of passes does not make PPO a general replay-buffer method.

**Off-policy learning**, such as DQN, can learn a target policy from transitions collected by earlier or different behavior policies. DQN therefore maintains an experience-replay buffer that spans many policy versions. This is different from PPO's short-lived rollout buffer.

## 4.3 Generalized advantage estimation

An advantage measures whether an action performed better than the critic expected. The temporal-difference residual is

\[
\delta_t=\widetilde r_t+\gamma m_t V_\phi(o_{t+1})-V_\phi(o_t),
\]

where \(\widetilde r_t=0.1r_t\) is the training reward and \(m_t=0\) at a game boundary and one otherwise. Generalized advantage estimation (GAE) combines consecutive residuals:

\[
\widehat A_t=\delta_t+\gamma\lambda m_t\widehat A_{t+1},
\qquad \widehat R_t=\widehat A_t+V_\phi(o_t).
\]

Here \(\gamma=0.99\) discounts future rewards, while \(\lambda=0.95\) controls how much subsequent residuals contribute. At \(\lambda=0\), the estimate is the one-step TD residual. At \(\lambda=1\), a complete terminal trajectory gives a Monte Carlo return minus the baseline. Larger values often reduce reliance on the critic but increase variance; the trade-off depends on critic quality and trajectory length.

There are two distinct boundaries in the implementation. At a collision, success or the 300-step task horizon, the return ends. At a rollout cut inside a live game, the next observation still has future value, so the code bootstraps with the critic. It never bootstraps from the reset observation across an episode boundary. The task horizon is treated as part of the finite-horizon objective; a continuing-task interpretation of Gymnasium truncation would require a different bootstrap convention.

The policy advantages are normalized within each rollout. The critic targets are computed from the unnormalized advantages. Test rewards and food scores remain in the original environment units.

## 4.4 The clipped PPO objective

Let \(\theta_{\mathrm{old}}\) denote the policy that collected the rollout. Define

\[
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid o_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)}.
\]

PPO maximizes the clipped surrogate

\[
L^{\mathrm{clip}}(\theta)=
\mathbb E_t\left[\min\left(
\rho_t\widehat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t
\right)\right].
\]

For a positive advantage, increasing an action's probability is useful, but the clipped term stops rewarding further increases beyond \(1+\epsilon\). For a negative advantage, it stops rewarding excessive probability reductions below \(1-\epsilon\). For example, if the old probability is 0.25 and the new probability is 0.40, the ratio is 1.60. With a positive advantage of 2 and \(\epsilon=0.2\), the surrogate contribution is \(\min(3.2,2.4)=2.4\).

Clipping is **not a hard constraint** on the action probability or on the KL divergence between policies. Shared parameters, value updates and multiple minibatch steps can still change the distribution substantially. We therefore log approximate KL divergence and clipping fraction as diagnostics rather than assuming that clipping guarantees stable learning.

The supplied implementation minimizes

\[
\mathcal L=-L^{\mathrm{clip}}+
0.5\left[\tfrac12\operatorname{MSE}(V_\phi,\widehat R)\right]
-0.01\,\mathcal H(\pi_\theta).
\]

The inner factor of one half is already in `PPOAgent.update`; the outer 0.5 is `value_coef`. Entropy \(\mathcal H\) encourages exploration. Adam uses learning rate \(3\times10^{-4}\), and the gradient norm is clipped at 0.5.

## 4.5 Controlled experimental protocol

| Item | Fixed specification |
|---|---|
| Environment | Original `SnakeEnv`, 8 × 8, target length 8, horizon 300 |
| Reward mode | Original `sparse`, including its existing loop and step penalties |
| Compared parameter | `clip_ratio` ∈ {0.1, 0.2, 0.3} |
| Training seeds | 7, 42, 2026, 31415, 27182 |
| Training budget | Exactly 50,000 environment steps per model |
| Rollout | Up to 256 transitions; shortened at logging/checkpoint boundaries |
| PPO updates | 4 epochs per rollout, minibatch size 64 |
| Discount / GAE | 0.99 / 0.95 |
| Demonstrations | None |
| Validation | 30 greedy episodes, seeds 600000–600029, every 5,000 steps and at initialization |
| Checkpoint selection | Highest validation score; success rate and raw reward break ties; earliest exact tie retained |
| Final test | 200 greedy episodes per selected model, seeds 950000–950199 |
| Visualization | Training seed 7, clip 0.2, game seed 700000, fixed before evaluation |
| Saved stages | 0%, 25%, 50%, 75%, 100% of the training budget |

Each clip setting receives five independently trained models. Training seeds are paired across settings, which aligns initialization and part of the randomness but does not force identical trajectories. All training and checkpoint selection finish before the test set is opened. None of the test results is used to choose the checkpoints.

The primary metric is food score, supplemented by success rate, timeout rate, raw return and episode length. Mean score is aggregated first within each trained model, then across the five training seeds. The reported standard deviation measures variation between training seeds. Paired bootstrap intervals resample the five seed-level differences; these intervals are descriptive and limited by the small number of independent runs. Two exploratory comparisons against clip 0.2 should not be treated as a confirmatory multiple-testing analysis.

Learning curves use validation food score against environment steps, with mean ± one sample standard deviation across seeds. Runtime includes validation and is recorded as such. Matching interaction budgets does not imply matching wall-clock cost with DQN or one-step Actor-Critic.

## 4.6 Execution and training-progress visualization

`member4_experiment.py` supports smoke, final and analysis modes. Smoke mode uses one seed and 1,024 steps per clip to check the pipeline; its output is separate from final results. A smoke run is not evidence of a trained policy's strength. Final mode runs the protocol above. Analysis mode regenerates figures from completed results without retraining.

A saved `trained.json` marker and selected checkpoint are both required to reuse a completed training run. An interrupted run is retrained from its seed rather than resumed from an incomplete optimizer/RNG state. A source-hash check prevents silently mixing outputs from changed code.

`visualize_progress.py` reads existing PPO, Actor-Critic and DQN-family checkpoint formats. It renders policies on the same initial environment seed using greedy actions and produces a side-by-side GIF, a final-frame PNG and a machine-readable summary. Each panel displays score and elapsed steps. A finished panel holds its last frame. Matching initial random seeds improves comparability, although different policies can subsequently generate different food sequences.

The fixed PPO demonstration contains all five training stages. Its role is to illustrate behavior, not establish average performance. If later checkpoints fail or perform worse, that outcome should remain visible rather than being replaced by a favorable game. The script can be reused by Members 2 and 3 with their checkpoint paths.

## 4.7 Results status and interpretation

**No PPO result table is reported at this stage.** The execution environment lacked the training dependencies, and network installation failed. No trained PPO weights, measured PPO learning curves or training-progress GIFs are included in this delivery.

After a successful final run, the script writes `results/final/observed_results.md`, per-episode test records, per-seed metrics, learning curves and progress visualizations. That observed-results insert should be added here. Interpretation should address: whether differences are consistent across training seeds; whether score and success agree; whether KL or clipping diagnostics explain a training pattern; and whether the fixed-seed demonstration is consistent with the aggregate evidence. A smaller clip may slow adaptation, while a larger clip may permit more aggressive updates, but neither outcome is assumed in advance.

## 4.8 Limitations

The seven-channel input is a compressed observation, not a full Markov state: it does not separately identify the head, encode body order, include the recent-visit history used in loop penalties, or expose remaining time. A feed-forward policy may therefore receive the same observation in situations with different future consequences. The experiment retains this representation to match the group environment.

The environment's border channel is a boundary cue: cells on the edge are traversable, while moving outside the array causes a wall collision. Relative actions prevent immediate reversal, but do not make a fixed-coordinate convolutional network rotation invariant. Rule-based teachers are strong empirical references, not a proven upper bound on all policies.

The experiment uses one board, one training budget and greedy evaluation. It cannot establish that a clipping value is best across tasks or that PPO universally outperforms other algorithms. The five-seed experiment also provides limited precision. A sensitivity analysis of GAE or a matched PPO–Actor-Critic comparison would be separate follow-up experiments.

## 4.9 Contribution and AI-use record

This package contains a PPO clipping experiment, an independently tested GAE utility, a shared checkpoint visualization tool, an English report and notebook, and a Chinese study guide. Generative AI assisted with reading the existing project, drafting code and explanations, and designing checks. The current verification is limited to the tests explicitly listed above; full training and runtime validation remain outstanding. The student should verify, understand and adapt the material before claiming personal implementation or experimental contributions. The required presentation must be delivered by the student in their own voice.
