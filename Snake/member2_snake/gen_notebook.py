"""Regenerate the completed Member 2 section and notebooks from saved evidence.

Run from any directory: python /path/to/gen_notebook.py
The integrated notebook replaces ONLY the original Member 2 cells (13:21).
Other members' source cells are preserved verbatim; their placeholders remain.
"""
from __future__ import annotations
import json
from pathlib import Path
import textwrap
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'runs/member2/final'


def md(text):return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text):return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def report_text():
    result=json.loads((RESULTS/'results.json').read_text())
    df=pd.read_csv(RESULTS/'per_seed_results.csv')
    stages=pd.read_csv(RESULTS/'stage_results.csv')
    costs=pd.read_csv(RESULTS/'training_cost.csv')
    base=pd.read_csv(RESULTS/'baseline_episodes.csv').groupby('policy').mean(numeric_only=True)
    a=result['aggregate']['ac'];b=result['aggregate']['bc_ac']
    def line(name,values):
        return f"| {name} | {values['score']['mean']:.3f} ± {values['score']['std']:.3f} | {100*values['success_rate']['mean']:.1f}% ± {100*values['success_rate']['std']:.1f}% | {values['reward']['mean']:.3f} ± {values['reward']['std']:.3f} |"
    table='\n'.join([line('Actor-Critic (no BC)',a),line('BC + Actor-Critic',b)])
    pair_lines=[]
    for metric,scale,label in [('score',1,'Mean score'),('success_rate',100,'Success rate (percentage points)'),('reward',1,'Reward')]:
        p=result['paired'][metric];lo,hi=p['bootstrap_95']
        pair_lines.append(f"| {label} | {p['mean_difference']*scale:+.3f} | [{lo*scale:.3f}, {hi*scale:.3f}] |")
    seed_lines=[]
    for seed in result['protocol']['seeds']:
        d=df[df.seed==seed].set_index('arm')
        seed_lines.append(f"| {seed} | {d.loc['ac','score']:.3f} | {d.loc['bc_ac','score']:.3f} | {d.loc['bc_ac','score']-d.loc['ac','score']:+.3f} |")
    bc_only=stages[(stages.arm=='bc_ac')&(stages.stage=='post_bc')]
    last=stages[(stages.arm=='bc_ac')&(stages.stage=='latest')]
    cost=costs.groupby('arm').mean(numeric_only=True)
    selections=', '.join(f"{int(r.seed)}: {int(r.selected_rl_steps):,}" for r in costs[costs.arm=='bc_ac'].itertuples())
    interval=result['paired']['score']['bootstrap_95']
    inference=("The paired score interval is above zero in these runs. This supports a benefit from BC under this fixed protocol; it does not establish general superiority across tasks or hyperparameters."
               if interval[0]>0 else "The paired score interval includes zero. We report a seed-dependent trend, not decisive evidence of superiority.")
    scratch_note = ('AC from scratch does not exceed the independently rerun Random mean score at this budget.'
                    if a['score']['mean'] <= base.loc['random','score'] else
                    'AC from scratch exceeds the independently rerun Random mean score at this budget.')
    return rf'''# 2. Actor-Critic with Behavior-Cloning Warm-Up

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

$$L_{{BC}}=-\frac{{1}}{{B}}\sum_{{i=1}}^B \log\pi_\theta(a_i^T\mid s_i).$$

The actor head and shared convolutional features are trained. The critic receives no direct supervised/value loss during BC. This makes the treatment specifically imitation warm-up. The original group's `value_update()` remains available for compatibility, but this experiment does not call it in warm-up. After the 100 teacher episodes, teacher queries and BC updates stop completely.

A high teacher trajectory score is not a learned-policy score: the teacher chooses those actions. We therefore evaluate the neural network independently immediately after BC (`post_bc.pt`).

### 2.2.2 One-step Actor-Critic

For a transition sampled by the current policy, we compute:

$$y_t = r_t+\gamma(1-z_t)\operatorname{{stopgrad}}[V_\phi(s_{{t+1}})],\qquad A_t=y_t-V_\phi(s_t).$$

Here $z_t$ is `terminated`: collision or success. An external 300-step limit is a `truncated` episode; it ends the rollout but still bootstraps the value of the final observation, following Gymnasium's time-limit convention. The remaining budget is not observed, so this is a continuing-task value approximation with capped evaluation episodes.

The minimized loss is:

$$L = \underbrace{{-\log\pi_\theta(a_t\mid s_t)\operatorname{{stopgrad}}(A_t)}}_{{\text{{actor}}}} + \underbrace{{\tfrac12(y_t-V_\phi(s_t))^2}}_{{\text{{critic}}}} - \underbrace{{\beta H(\pi_\theta(\cdot\mid s_t))}}_{{\text{{entropy bonus}}}}.$$

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
| AC | {cost.loc['ac','rl_steps']:.0f} | {cost.loc['ac','demo_steps']:.0f} | {cost.loc['ac','bc_updates']:.0f} | {cost.loc['ac','training_seconds']:.1f} s |
| BC + AC | {cost.loc['bc_ac','rl_steps']:.0f} | {cost.loc['bc_ac','demo_steps']:.1f} | {cost.loc['bc_ac','bc_updates']:.1f} | {cost.loc['bc_ac','training_seconds']:.1f} s |

Wall time includes training and periodic validation, excludes the later final-test evaluation, and depends on this computer. BC + AC uses more total environment transitions and optimizer updates. Thus the experiment tests whether demonstration-assisted initialization helps, not whether it is computationally free.

## 2.5 Held-Out Gameplay Results

The following are means ± **sample standard deviation across the five training seeds**. Each seed's value is first averaged over its 200 test episodes.

| Learned method | Score (max 7) | Success rate | Reward |
|---|---:|---:|---:|
{table}

The independently rerun references on the same test seeds are: Random score **{base.loc['random','score']:.3f}**, legacy teacher **{base.loc['legacy','score']:.3f}**, BFS teacher **{base.loc['shortest_path','score']:.3f}**. The teachers are strong heuristic references, not theoretical upper bounds. These are Member 2's reruns rather than copied figures from Members 1 or 3.

{scratch_note} The BC + AC mean score is {b['score']['mean']:.3f} out of 7, so the learned policy still has a substantial gap to the BFS teacher. The experiment therefore supports a narrowly defined initialization benefit rather than a claim that this small AC model solves Snake.

![Final performance](runs/member2/final/figures/final_performance.png)

*Only learned methods have cross-training-seed error bars. A missing baseline error bar does not imply zero episode variation; each baseline has one 200-episode evaluation.*

### Paired seed outcomes and uncertainty

| Training seed | AC score | BC + AC score | Difference |
|---:|---:|---:|---:|
{chr(10).join(seed_lines)}

We bootstrap the five **paired training-seed differences** 20,000 times, with fixed bootstrap seed 20261019. Test episodes are not treated as 1,000 independent training runs. These small-sample bootstrap intervals describe uncertainty under this experiment and have limited coverage guarantees with only five seeds.

| Outcome | BC + AC minus AC | Paired bootstrap 95% interval |
|---|---:|---:|
{chr(10).join(pair_lines)}

{inference}

The win rate nevertheless remains only **{100*b['success_rate']['mean']:.1f}%**. Mean reward changes by **{result['paired']['reward']['mean_difference']:+.3f}**, with interval **[{result['paired']['reward']['bootstrap_95'][0]:.3f}, {result['paired']['reward']['bootstrap_95'][1]:.3f}]**, which includes zero. Thus the score gain is clearer than either reliable winning or improved total reward. Longer survival can accumulate loop and step penalties without eating more food; higher food score alone does not guarantee higher shaped return.

| Condition | Death rate | Time-limit truncation rate | Success rate |
|---|---:|---:|---:|
| AC | {100*a['death_rate']['mean']:.1f}% | {100*a['truncation_rate']['mean']:.1f}% | {100*a['success_rate']['mean']:.1f}% |
| BC + AC | {100*b['death_rate']['mean']:.1f}% | {100*b['truncation_rate']['mean']:.1f}% | {100*b['success_rate']['mean']:.1f}% |

![Paired outcomes and cost](runs/member2/final/figures/paired_outcomes_cost.png)

### Does later RL improve the BC initialization?

The independently tested BC-only snapshot has mean score **{bc_only.score.mean():.3f}** and success rate **{100*bc_only.success_rate.mean():.1f}%** across the five seeds. The last BC + AC snapshot has mean score **{last.score.mean():.3f}** and success rate **{100*last.success_rate.mean():.1f}%**. The validation-selected BC + AC result above is a third, separately defined outcome. Selected AC steps by seed are **{selections}**.

These stage results separate learning from demonstrations, continued RL, and validation-based checkpoint selection. Validation-selected best performance must not be described as the guaranteed performance of the last update. In particular, a gain over AC from scratch does not by itself prove that RL improves every BC-only policy.

## 2.6 Visualizing Training Progress

![Fixed-seed gameplay across stages](runs/member2/final/figures/training_progress.gif)

The GIF uses **training seed 7** and **game seed 900000**, fixed before interpreting the result. It compares the randomly initialized network, the post-BC snapshot, the 25,000-step AC midpoint, and the checkpoint selected by validation. Every panel uses only the neural policy; completed games freeze in place. This is an illustrative trajectory, not statistical evidence.

''' + '\n'.join(f"- **{label}**: score {rec['score']}, {rec['steps']} steps, {rec['outcome']}." for label,rec in result['gameplay'].items()) + r'''

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
'''


def setup_cell():
    return code('''
        TRAIN_MODEL = False  # default: load the supplied trained checkpoints
        import os, sys, json
        from pathlib import Path
        HERE = Path.cwd()
        if not (HERE / 'member2_experiment.py').exists():
            raise FileNotFoundError('Open this notebook from the extracted member2_snake folder.')
        os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import numpy as np
        import pandas as pd
        import torch
        import matplotlib.pyplot as plt
        from IPython.display import Image as NotebookImage, display
        from environment import SnakeEnv
        from main import load_agent, evaluate_model
        torch.set_num_threads(1)
        RESULT_DIR = HERE / 'runs/member2/final'
        assert (RESULT_DIR / 'results.json').is_file(), 'Extract the entire package, including runs/.'
        print('TRAIN_MODEL =', TRAIN_MODEL, '| device: CPU | inference only by default')
    ''')


def section_cells(text):
    # Full model and train implementations are visible executable cells, not placeholders.
    method=text.split('## 2.3 Experimental Protocol')[0]
    protocol='## 2.3 Experimental Protocol'+text.split('## 2.3 Experimental Protocol')[1].split('## 2.4 Learning Curves')[0]
    conclusion='## 2.7 Limitations'+text.split('## 2.7 Limitations')[1]
    code_model=(ROOT/'model.py').read_text()
    code_train=(ROOT/'main.py').read_text().split('\ndef main():')[0]
    code_play=(ROOT/'play.py').read_text().split('\ndef main():')[0]
    return [
        md(method),
        md('### Executable model implementation\n\nThe following is the complete `model.py` implementation, including the actor/critic losses and BC update.'),
        code(code_model),
        code('''
            env_ac = SnakeEnv(grid_size=8, end_score=8, max_steps=300)
            agent_ac = ConvActorCritic(7, 3, 8)
            print(agent_ac)
            print('Trainable parameters:', sum(p.numel() for p in agent_ac.parameters()))
            sample_obs, _ = env_ac.reset(seed=42)
            logits, value = agent_ac(torch.from_numpy(sample_obs).unsqueeze(0))
            print('Actor logits shape:', tuple(logits.shape), '| critic shape:', tuple(value.shape))
            env_ac.close()
        '''),
        md(protocol),
        md('### Executable training and evaluation implementation\n\nThis is the complete library portion of `main.py`. Defining these functions does not train a model. The paired driver below calls the same saved module.'),
        code(code_train),
        code('''
            from member2_experiment import run_experiment
            if TRAIN_MODEL:
                # Equal RL-step budgets, teacher_episodes=0 vs 100, paired seeds.
                # Completed matching runs are reused; select a new output directory for a fresh replication.
                run_experiment(RESULT_DIR, mode='final')
            protocol_ac = json.loads((RESULT_DIR / 'protocol.json').read_text())
            print(json.dumps({k: protocol_ac[k] for k in ['seeds', 'rl_steps', 'teacher_episodes', 'test_episodes']}, indent=2))
            best_ac = RESULT_DIR / 'seed-7/bc_ac/checkpoints/best.pt'
            agent_ac, config_ac = load_agent(best_ac, 'cpu')
            check_ac = evaluate_model(agent_ac, config_ac, seed_base=900000, episodes=200)
            cached_ac = pd.read_csv(RESULT_DIR / 'per_seed_results.csv')
            expected_ac = cached_ac.query("seed == 7 and arm == 'bc_ac'").iloc[0]
            assert np.isclose(check_ac['score'], expected_ac['score'])
            assert np.isclose(check_ac['success_rate'], expected_ac['success_rate'])
            print('Checkpoint loaded; 200 independent greedy episodes reproduce the saved seed-7 result:')
            display(pd.DataFrame([check_ac]))
        '''),
        md('## 2.4 Learning Curves\n\nEvery plotted point comes from saved logs. Validation uses fixed independent episodes; the shaded region is the standard deviation across training seeds. Autonomous episode curves exclude teacher-executed trajectories.'),
        code('''
            validation_ac = pd.read_csv(RESULT_DIR / 'all_validation.csv')
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
            for arm, label in [('ac', 'AC (no BC)'), ('bc_ac', 'BC + AC')]:
                selected = validation_ac.query("arm == @arm and stage == 'actor_critic'")
                for ax, metric in zip(axes, ['score', 'success_rate']):
                    g = selected.groupby('rl_steps')[metric].agg(['mean', 'std'])
                    ax.plot(g.index, g['mean'], label=label)
                    ax.fill_between(g.index, g['mean']-g['std'], g['mean']+g['std'], alpha=.15)
                    ax.set(xlabel='Autonomous AC steps', ylabel='Validation '+metric)
            axes[0].set_ylim(0, 7.2); axes[1].set_ylim(0, 1); axes[0].legend()
            plt.tight_layout(); plt.show()
            display(NotebookImage(filename=str(RESULT_DIR / 'figures/training_reward_score.png')))
        '''),
        md('## 2.5 Evaluation and Paired Uncertainty\n\nEach training seed is one experimental unit. The table and intervals below use saved held-out episodes; test outcomes never select a checkpoint. Read the standalone Markdown report for the complete interpretation.'),
        code('''
            per_seed_ac = pd.read_csv(RESULT_DIR / 'per_seed_results.csv')
            display(per_seed_ac[['seed','arm','score','reward','success_rate','death_rate','truncation_rate']])
            display(per_seed_ac.groupby('arm')[['score','reward','success_rate']].agg(['mean','std']))
            uncertainty_ac = json.loads((RESULT_DIR / 'paired_results.json').read_text())
            display(pd.DataFrame({key: {'mean_difference': value['mean_difference'],
                                      'bootstrap_95_low': value['bootstrap_95'][0],
                                      'bootstrap_95_high': value['bootstrap_95'][1]}
                                  for key, value in uncertainty_ac.items()}).T)
            display(NotebookImage(filename=str(RESULT_DIR / 'figures/final_performance.png')))
            display(NotebookImage(filename=str(RESULT_DIR / 'figures/paired_outcomes_cost.png')))
            display(pd.read_csv(RESULT_DIR / 'training_cost.csv'))
            stage_ac = pd.read_csv(RESULT_DIR / 'stage_results.csv')
            display(stage_ac.groupby(['arm','stage'])[['score','success_rate']].mean())
        '''),
        md('## 2.6 Visualizing Training Progress\n\nThe complete headless gameplay code from `play.py` is included below. The four stages share the predeclared game seed 900000 and training seed 7. Every panel is played solely by the network. A single GIF illustrates behavior; the multi-episode table supports performance claims.'),
        code(code_play),
        code('''
            display(NotebookImage(filename=str(RESULT_DIR / 'figures/training_progress.gif')))
            progress_ac = json.loads((RESULT_DIR / 'figures/training_progress.json').read_text())
            display(pd.DataFrame({label: {k:v for k,v in record.items() if k != 'frames'}
                                  for label,record in progress_ac.items()}).T)
            # To regenerate from actual checkpoints, run:
            # python member2_experiment.py --mode analysis
        '''),
        md(conclusion),
        md('### Complete recorded findings\n\nThe following saved report contains the numerical interpretation and all limitations. It is generated from the same raw files shown above.'),
        code('''
            from IPython.display import Markdown
            report_ac = (HERE / 'member2_section2.md').read_text(encoding='utf-8')
            # Display the numerical findings; images are already displayed above.
            findings_ac = '## 2.5 Held-Out Gameplay Results' + report_ac.split('## 2.5 Held-Out Gameplay Results')[1].split('## 2.6 Visualizing Training Progress')[0]
            import re
            findings_ac = re.sub(r'!\\[[^\\]]*\\]\\([^)]*\\)', '', findings_ac)
            display(Markdown(findings_ac))
        ''')
    ]


def generate():
    text=report_text()
    (ROOT/'member2_section2.md').write_text(text,encoding='utf-8')
    metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
              'language_info':{'name':'python','version':'3.13'}}
    cells=section_cells(text)
    standalone=nbf.v4.new_notebook(metadata=metadata,cells=[
        md('''# DSA5102 Snake — Member 2 Completed Report

        **Actor-Critic and behavior-cloning warm-up.** This independently runnable notebook covers Member 2 only. It includes executable model, training and gameplay implementations, saved experimental evidence and trained checkpoints. `TRAIN_MODEL=False` keeps training disabled; the saved seed-7 checkpoint is actually loaded and evaluated.

        Extract the entire `member2_snake` folder and launch Jupyter inside it. The integrated group notebook is a separate file; other members' pending sections are not represented as completed work here.
        '''),setup_cell(),*cells])
    nbf.validate(standalone)
    nbf.write(standalone,ROOT/'DSA5102_snake_member2.ipynb')
    template=nbf.read(ROOT/'reference/DSA5102_final_project_snake_original.ipynb',as_version=4)
    # Source outside the owner's section remains exactly as supplied.
    integrated=nbf.v4.new_notebook(metadata=metadata,cells=[md('''**Integration status:** Only Member 2 / Section 2 is completed in this copy. Other members’ original sections, including their pending placeholders, are preserved. Use the separate Member 2 notebook for a focused report and merge this section into the group’s latest working notebook before final submission.'''),*template.cells[:13],setup_cell(),*cells,*template.cells[21:]])
    for c in integrated.cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    nbf.validate(integrated)
    nbf.write(integrated,ROOT/'DSA5102_final_project_snake_member2_completed.ipynb')
    print('Generated Markdown + two notebooks. Execute notebooks after generation to retain outputs.')

if __name__=='__main__':generate()
