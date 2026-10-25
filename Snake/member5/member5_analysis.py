"""Analyze the completed common evaluation without training or rerunning games.

Run from member5: python member5_analysis.py
Historical learning curves and Member 4's clip study retain their own validation
and test protocols; they are never pooled into the common evaluation table.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

METHODS = ["Random", "Teacher", "AC", "BC + AC", "DQN", "Double DQN", "PPO"]
COLORS = ["#8e9aaf", "#2a9d8f", "#b39627", "#d97935", "#457b9d", "#8755b4", "#c65463"]


def verify_saved_checkpoints(project_dir=None):
    """Load all 25 delivered models and check frozen, finite greedy inference."""
    import torch
    torch.set_num_threads(1)
    snake = Path(project_dir).resolve() if project_dir else Path(__file__).resolve().parent.parent
    if (snake / "Snake").is_dir():
        snake = snake / "Snake"
    spec = importlib.util.spec_from_file_location("member5_verify_decisions", snake / "inspect_decisions.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    env_class = module.load_source("verify_env", snake / "environment.py").SnakeEnv
    ac_class = module.load_source("verify_ac", snake / "member2_snake/model.py").ConvActorCritic
    q_class = module.load_source("verify_q", snake / "dqn.py").QNetwork
    ppo_class = module.load_source("verify_ppo", snake / "ppo.py").PPOActorCritic
    env = env_class(**module.ENV_CONFIG)
    observation, _ = env.reset(seed=1100000)
    env.close()
    rows = []
    for method in module.METHODS:
        for seed in module.TRAINING_SEEDS:
            model, metadata = module.load_network(method, seed, snake, ac_class, q_class, ppo_class)
            assert not model.training and all(not p.requires_grad for p in model.parameters())
            before = {k: v.clone() for k, v in model.state_dict().items()}
            first = module.inspect_network(model, method, [observation])[0]
            assert first == module.inspect_network(model, method, [observation])[0]
            assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
            rows.append(dict(method=method, training_seed=seed, loaded=True,
                             greedy_action=module.ACTION_NAMES[first["selected_action"]],
                             selected_steps=metadata["selected_checkpoint_steps"],
                             full_training_budget=metadata["training_budget_steps"]))
    return pd.DataFrame(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_learning_curves(snake):
    records, sources = [], []
    path = snake / "runs/member2/final/all_validation.csv"
    sources.append(path)
    data = pd.read_csv(path)
    # At step zero, BC + AC uses its post-demonstration evaluation. Its initial
    # untrained result is not a second independent observation at that step.
    data = data.groupby(["seed", "arm", "rl_steps"], sort=False).tail(1)
    for row in data.itertuples():
        records.append(dict(method="AC" if row.arm == "ac" else "BC + AC",
                            training_seed=int(row.seed), training_steps=int(row.rl_steps),
                            score=float(row.score)))
    for seed in [7, 42, 2026, 31415, 27182]:
        for condition, method in [("dqn", "DQN"), ("double-dqn", "Double DQN")]:
            path = snake / "member 3/results/final" / f"seed-{seed}" / condition / "history.json"
            sources.append(path)
            for row in json.loads(path.read_text())["validation"]:
                records.append(dict(method=method, training_seed=seed,
                                    training_steps=int(row["step"]), score=float(row["score"])))
        path = snake / "member4/results/final" / f"seed-{seed}" / "clip-0.2/validation.json"
        sources.append(path)
        for row in json.loads(path.read_text()):
            records.append(dict(method="PPO", training_seed=seed,
                                training_steps=int(row["training_steps"]), score=float(row["score"])))
    points = pd.DataFrame(records)
    assert not points.duplicated(["method", "training_seed", "training_steps"]).any()
    assert set(points["method"]) == set(METHODS[2:])
    assert points.groupby(["method", "training_steps"])["training_seed"].nunique().eq(5).all()
    return points, sources


def paired_comparisons(per_model):
    wide = per_model[per_model.training_seed.notna()].pivot(
        index="training_seed", columns="method", values="score_mean")
    rng = np.random.default_rng(20261025)
    rows = []
    for base, revised in [("AC", "BC + AC"), ("DQN", "Double DQN")]:
        difference = (wide[revised] - wide[base]).dropna().to_numpy()
        test = stats.ttest_rel(wide[revised], wide[base])
        bootstrap = rng.choice(difference, (10000, len(difference)), replace=True).mean(axis=1)
        low, high = np.quantile(bootstrap, [.025, .975])
        rows.append(dict(comparison=f"{revised} minus {base}", n_training_seeds=len(difference),
                         mean_score_difference=float(difference.mean()),
                         difference_seed_sd=float(difference.std(ddof=1)),
                         positive_seeds=int((difference > 0).sum()),
                         negative_seeds=int((difference < 0).sum()),
                         t_statistic=float(test.statistic), p_value=float(test.pvalue),
                         bootstrap_95_low=float(low), bootstrap_95_high=float(high)))
    # Holm adjustment across the two prespecified within-family comparisons.
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["p_value"])
    previous = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, max(previous, (len(rows) - rank) * rows[index]["p_value"]))
        rows[index]["p_holm"] = adjusted
        previous = adjusted
    return pd.DataFrame(rows)


def plot_analysis(summary, per_model, curves, figure_dir):
    positions = np.arange(len(METHODS))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, metric, sd, label in [
        (axes[0], "score_mean", "score_training_seed_sd", "Mean food score (maximum 7)"),
        (axes[1], "success_rate", "success_training_seed_sd", "Success rate"),
    ]:
        axis.bar(positions, summary[metric], color=COLORS)
        valid = summary[sd].notna().to_numpy()
        axis.errorbar(positions[valid], summary.loc[valid, metric],
                      yerr=summary.loc[valid, sd], fmt="none", color="#222222", capsize=4)
        axis.set_xticks(positions, METHODS, rotation=25, ha="right")
        axis.set_ylabel(label)
        axis.set_ylim(0, 8 if metric == "score_mean" else 1.05)
    figure.suptitle("Existing trained checkpoints under common evaluation", weight="bold")
    figure.text(.5, .015, "PPO: fixed clip=0.2. Error bars: SD across training seeds; baselines have no training replicates. Training budgets differ.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .05, 1, .93))
    figure.savefig(figure_dir / "performance.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    learned = per_model[per_model.training_seed.notna()]
    for axis, pair in zip(axes, [("AC", "BC + AC"), ("DQN", "Double DQN")]):
        wide = learned.pivot(index="training_seed", columns="method", values="score_mean")
        for seed, row in wide.iterrows():
            axis.plot([0, 1], row[list(pair)], "o-", alpha=.8, label=f"seed {int(seed)}")
        axis.set_xticks([0, 1], pair)
        axis.set_ylim(0, 7.4)
        axis.set_ylabel("Mean food score")
        axis.set_title(f"{pair[0]} versus {pair[1]}")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "paired_comparisons.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(METHODS))
    for key, label, color in [("success_rate", "Success", "#2a9d8f"),
                              ("timeout_rate", "Timeout", "#d2ab55"),
                              ("collision_rate", "Collision", "#bd5b5b")]:
        axis.bar(positions, summary[key], bottom=bottom, label=label, color=color)
        bottom += summary[key].to_numpy()
    axis.set_xticks(positions, METHODS)
    axis.set(ylim=(0, 1.05), ylabel="Fraction of episodes")
    figure.suptitle("How common-evaluation episodes ended", y=.99)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(.5, .93), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, .84))
    figure.savefig(figure_dir / "episode_outcomes.png", dpi=160)
    plt.close(figure)

    grouped = curves.groupby(["method", "training_steps"]).score.agg(["mean", "std"]).reset_index()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for axis, family, title in zip(axes, [("AC", "BC + AC"), ("DQN", "Double DQN"), ("PPO",)],
                                    ["Actor-Critic", "DQN family", "PPO (clip=0.2)"]):
        for method in family:
            frame = grouped[grouped.method == method]
            color = COLORS[METHODS.index(method)]
            axis.plot(frame.training_steps, frame["mean"], label=method, color=color)
            axis.fill_between(frame.training_steps, np.maximum(0, frame["mean"] - frame["std"]),
                              np.minimum(7, frame["mean"] + frame["std"]), alpha=.15, color=color)
        axis.set(xlabel="Autonomous training interactions", title=title, ylim=(0, 7.2))
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Historical validation food score")
    figure.text(.5, .015, "Family panels use different x-axis ranges. Original validation sets differ. BC demonstrations are additional to its RL-step axis.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .06, 1, 1))
    figure.savefig(figure_dir / "learning_curves.png", dpi=160)
    plt.close(figure)


def run_analysis(*, result_dir=None, project_dir=None, output_root=None, include_decisions=True):
    root = Path(output_root).resolve() if output_root else Path(__file__).resolve().parent
    snake = Path(project_dir).resolve() if project_dir else Path(__file__).resolve().parent.parent
    if (snake / "Snake").is_dir():
        snake = snake / "Snake"
    if result_dir is None:
        candidates = sorted((root / "results/evaluation").glob("*/protocol.json"))
        if not candidates:
            raise FileNotFoundError("Run the full common evaluation first. Smoke results are not analyzed as final evidence.")
        result_dir = candidates[-1].parent
    result_dir = Path(result_dir).resolve()
    protocol = json.loads((result_dir / "protocol.json").read_text())
    if protocol["mode"] != "evaluation" or protocol["train_model"]:
        raise ValueError("Analysis requires an inference-only full evaluation")
    if protocol["training_seeds"] != [7, 42, 2026, 31415, 27182] or len(protocol["test_seeds"]) != 200:
        raise ValueError("The delivered analysis expects five training seeds and 200 common test games")
    summary = pd.read_csv(result_dir / "summary.csv").set_index("method").loc[METHODS].reset_index()
    per_model = pd.read_csv(result_dir / "per_model.csv")
    episodes = pd.read_csv(result_dir / "episodes.csv")
    if len(episodes) != 5400 or len(per_model) != 27:
        raise ValueError("The common evaluation is incomplete")
    assert np.allclose(summary[["success_rate", "timeout_rate", "collision_rate"]].sum(axis=1), 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result_out = root / "results/analysis" / stamp
    figure_out = root / "figures/analysis" / stamp
    result_out.mkdir(parents=True, exist_ok=False)
    figure_out.mkdir(parents=True, exist_ok=False)
    curves, curve_sources = collect_learning_curves(snake)
    curves.to_csv(result_out / "learning_curve_points.csv", index=False)
    paired = paired_comparisons(per_model)
    paired.to_csv(result_out / "paired_statistics.csv", index=False)
    plot_analysis(summary, per_model, curves, figure_out)

    # Retain original PPO sensitivity results separately: different test seeds.
    ppo_results = snake / "member4/results/final/results.csv"
    ppo = pd.read_csv(ppo_results)
    ppo_summary = ppo.groupby("clip").agg(score_mean=("score", "mean"), score_seed_sd=("score", "std"),
                                           success_rate=("success", "mean")).reset_index()
    ppo_summary.to_csv(result_out / "ppo_original_clip_study.csv", index=False)

    progress_sources = []
    for name in ("training_progress.gif", "training_progress.png", "training_progress.json"):
        source = snake / "runs/member2/final/figures" / name
        progress_sources.append(source)
        shutil.copyfile(source, (result_out if source.suffix == ".json" else figure_out) / name)
    decisions = None
    if include_decisions:
        source = snake / "inspect_decisions.py"
        spec = importlib.util.spec_from_file_location("member5_decision_inspection", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        decisions = module.run_inspection(seed=42, project_dir=snake, output_root=root)
        expected = {r["method"]: r["checkpoint_sha256"] for r in protocol["checkpoints"] if r["training_seed"] == 42}
        assert all(expected[m["method"]] == m["checkpoint_sha256"] for m in decisions["report"]["models"])

    def relative(path):
        return Path(os.path.relpath(path, result_out)).as_posix()
    lines = ["# Member 5: Common evaluation and integrated analysis", "",
             "This report compares the group's delivered trained checkpoints under common game rules. It does not isolate algorithm quality under equal training resources.", "",
             "## Protocol", "",
             "All methods use the shared 8 x 8 environment, target length 8 (maximum food score 7), 300-step limit, sparse reward mode and relative left/right/straight actions.",
             "Five training seeds (7, 42, 2026, 31415, 27182) and 200 common test seeds (1,200,000-1,200,199) produce 5,400 episodes, including 200 each for Random and Teacher. Learned policies use greedy argmax with no teacher takeover or safety mask.",
             "PPO clip=0.2 was fixed before this common test. Checkpoints were selected by their original validation protocols; these test outcomes never choose a checkpoint or clip.", "",
             "## Common final performance", "",
             "| Method | Mean score | Seed SD | Success | Timeout | Training budget | Extra demonstrations |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    indexed = summary.set_index("method")
    for row in summary.to_dict("records"):
        sd = "N/A" if pd.isna(row["score_training_seed_sd"]) else f"{row['score_training_seed_sd']:.3f}"
        lines.append(f"| {row['method']} | {row['score_mean']:.3f} | {sd} | {row['success_rate']:.1%} | {row['timeout_rate']:.1%} | {int(row['budget_steps']):,} | {int(row['extra_teacher_episodes'])} |")
    top = summary[~summary.method.isin(["Random", "Teacher"])].sort_values("score_mean", ascending=False).iloc[0]
    lines.extend(["", f"{top['method']} has the highest observed learned-agent mean score ({top['score_mean']:.3f}) in this common evaluation. Teacher is a heuristic reference, not a mathematical upper bound.",
                  f"Teacher succeeds in {indexed.loc['Teacher','success_rate']:.1%} of episodes, compared with {top['success_rate']:.1%} for the highest-scoring learned configuration. The learned policies have not matched the heuristic's reliability on this test batch. AC's mean score ({indexed.loc['AC','score_mean']:.3f}) is below Random's ({indexed.loc['Random','score_mean']:.3f}); training does not automatically produce useful play.",
                  "AC/BC + AC, DQN/DDQN and PPO have full autonomous training budgets of 50,000, 210,000 and 500,000 steps. BC + AC additionally uses 100 demonstration episodes. Training architectures, optimization and checkpoint-selection protocols also differ. Thus the table ranks delivered outcomes, not the intrinsic superiority of learning algorithms.",
                  "For learned methods, scores are first averaged within each trained model, then across the five training seeds. Seed SD measures training-run variability. Baselines have no independent training replicates; their episode SD is available in the common summary but is not a training-seed SD.", "",
                  f"![Common performance]({relative(figure_out / 'performance.png')})", "",
                  "## Within-family improvement experiments", "",
                  "| Comparison | Mean score difference | Bootstrap 95% interval | Paired t p | Holm-adjusted p | Positive seeds |",
                  "|---|---:|---|---:|---:|---:|"])
    for row in paired.to_dict("records"):
        lines.append(f"| {row['comparison']} | {row['mean_score_difference']:.3f} | [{row['bootstrap_95_low']:.3f}, {row['bootstrap_95_high']:.3f}] | {row['p_value']:.4f} | {row['p_holm']:.4f} | {row['positive_seeds']}/5 |")
    for row in paired.to_dict("records"):
        evidence = "the exploratory paired t-test remains below 0.05 after Holm adjustment" if row["p_holm"] < .05 else "the exploratory paired t-test does not establish a difference at 0.05 after Holm adjustment"
        lines.append(f"\nFor {row['comparison']}, the observed mean change is {row['mean_score_difference']:.3f}; {evidence}.")
    lines.extend(["", "Paired t-tests assume approximately normal seed-level differences; five seeds cannot validate that assumption well. Bootstrap intervals are descriptive and are also limited by five training replicates. Treat these comparisons as small-sample exploratory evidence. Failure to detect a difference does not establish equivalence.",
                  f"BC + AC improves the mean food score but succeeds in only {indexed.loc['BC + AC','success_rate']:.1%} of games. Demonstrations help relative to the weak AC delivery, while reliable task completion remains unresolved. Double DQN's success rate is {indexed.loc['Double DQN','success_rate']:.1%} versus DQN's {indexed.loc['DQN','success_rate']:.1%}; that success-rate difference is descriptive, not an additional significance claim.",
                  "The BC comparison includes the additional demonstration cost and measures demonstration-assisted initialization at the same autonomous budget. The DQN comparison relies on Member 3's internal training controls. Neither comparison is selected or retuned after inspecting these results.", "",
                  f"![Seed-paired comparisons]({relative(figure_out / 'paired_comparisons.png')})", "",
                  "## Episode outcomes", "",
                  f"PPO achieves success in {indexed.loc['PPO','success_rate']:.1%} of games and times out in {indexed.loc['PPO','timeout_rate']:.1%}. Long survival by itself is not task completion; a timeout alone does not prove circling.",
                  "Collision is recorded without inventing separate wall/self-collision causes, because the common environment does not return that distinction.", "",
                  f"![Episode outcomes]({relative(figure_out / 'episode_outcomes.png')})", "",
                  "## Historical learning curves", "",
                  "The curves below reuse the original members' validation histories; all plotted points include five training seeds. Their validation seed sets and, for PPO, validation episode counts differ. The family panels have different x-axis ranges. Use these as learning-history context; use the common final evaluation for cross-family performance comparisons. The BC + AC zero-RL-step point is after BC warm-up; demonstrations are additional to the RL-step axis.", "",
                  "The DQN-family validation scores rise substantially over their recorded training histories. AC remains weak, and BC + AC does not show monotonic gains after its demonstration warm-up. PPO improves after its weak early period but remains variable late in training. These descriptive patterns do not identify causal mechanisms or establish equal-budget convergence-speed differences.", "",
                  f"![Learning curves]({relative(figure_out / 'learning_curves.png')})", "",
                  "## Original PPO clipping study", "",
                  "This separate study uses Member 4's test seeds 2,950,000-2,950,199, not the common test batch. The three clipping groups each contain five models with 500,000 training interactions per model.", "",
                  "| Clip | Original mean score | Seed SD | Original success |", "|---|---:|---:|---:|"])
    for row in ppo_summary.to_dict("records"):
        lines.append(f"| {row['clip']:.1f} | {row['score_mean']:.3f} | {row['score_seed_sd']:.3f} | {row['success_rate']:.1%} |")
    lines.extend(["", "Member 4 reports descriptive seed-paired intervals that include zero for both differences against clip=0.2. These original test results do not change the common evaluation's prespecified PPO setting.", "",
                  "## Training-stage gameplay", "",
                  "The following original Member 2 animation shows Initial, After BC, AC midpoint and Best validation on game seed 900000. It illustrates BC-assisted Actor-Critic, not PPO. In this game food score changes from 0 to 1; later stages still fail to complete the task. Do not describe it as reliable mastery or monotonically improving gameplay.", "",
                  f"![Member 2 training stages]({relative(figure_out / 'training_progress.gif')})", ""])
    if decisions:
        lines.extend(["## Fixed-state decision inspection", "",
                      "The five learned methods use their seed-42 checkpoints on the same four hand-authored 8 x 8 states. These qualitative probes compare action choices and next-step collision risk; they do not establish a population performance ranking. Actor probabilities and Q values are on different scales.", "",
                      f"[Detailed decision report]({relative(decisions['result_dir'] / 'report.md')})", "",
                      f"![Fixed-state actions]({relative(decisions['figure_dir'] / 'selected_actions.png')})", ""])
        for row in decisions["report"]["scenarios"]:
            if row["name"] in {"wall_ahead", "narrow_escape"}:
                choices = []
                for method, action in row["models"].items():
                    risk = "clear on the next step" if action["selected_action_immediately_safe"] else "immediate collision"
                    choices.append(f"{method}: {action['selected_action_name']} ({risk})")
                lines.append(f"{row['title']}: " + "; ".join(choices) + ".\n")
        lines.extend(["A high average test score does not guarantee a sensible action on every synthetic state. The fixed probes expose local failures without retuning any model; they do not establish why those failures occur.", ""])
    lines.extend(["## Conclusions and limitations", "",
                  f"Under the common evaluation protocol, {top['method']} is the strongest delivered learned configuration by mean food score. Preserve the within-family comparisons and the training-cost disclosures when interpreting this observation.",
                  "The task is a single small board with compressed observations that omit explicit head identity, body order, visit history and remaining time. Five training seeds and one test-seed batch limit precision and generalization. No claim is made that these findings transfer to other boards or establish equal-resource algorithm superiority.",
                  "The group still needs to merge the members' full methods, this analysis and the actual contribution record into one runnable final notebook, then prepare the required human presentation. This Member 5 package is not the entire final submission.", "",
                  "## Contributions and AI assistance", "",
                  "Original group members supplied the environment, algorithms, trained weights, validation histories and gameplay animation. Member 5 integrates and evaluates those deliveries. OpenAI Codex assisted evaluation/analysis code, documentation, execution and verification. The students remain responsible for understanding and presenting their work.", ""])
    (result_out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = dict(common_result_dir=str(result_dir.relative_to(root)),
                    common_protocol_sha256=sha256(result_dir / "protocol.json"),
                    analysis_script_sha256=sha256(Path(__file__)),
                    source_sha256={str(p.relative_to(snake)): sha256(p) for p in curve_sources + progress_sources + [ppo_results]},
                    decision_result_dir=str(decisions["result_dir"].relative_to(root)) if decisions else None,
                    trained_models_changed=False, training_performed=False,
                    statistical_tests="Two prespecified paired t-tests with Holm adjustment; descriptive seed bootstrap intervals")
    (result_out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Analysis report:", result_out / "report.md")
    return dict(result_dir=result_out, figure_dir=figure_out, paired_statistics=paired,
                summary=summary, decisions=decisions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    run_analysis(result_dir=args.result_dir, project_dir=args.project_dir, output_root=args.output_root)
