"""Step-budgeted DQN versus Double DQN experiment.

This script is generated from the local Member 3 notebook. Set MEMBER3_MODE to
smoke, final, or analysis. The default is analysis, which loads frozen results.
"""

# Environment setup
import importlib.util
import importlib.metadata
import gc
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

cwd = Path(__file__).resolve().parent

def find_snake_dir(start: Path):
    candidates = [
        start,
        start.parent,
        start.parent.parent,
        start / "Snake",
        start / "DSA5102-Final-Group-Project" / "Snake",
    ]
    for candidate in candidates:
        if (candidate / "environment.py").is_file() and (candidate / "dqn.py").is_file():
            return candidate
    return None

SNAKE_DIR = find_snake_dir(cwd)
if SNAKE_DIR is None:
    raise FileNotFoundError(
        "Snake source files were not found. Mount the folder containing "
        "DSA5102-Final-Group-Project into /workspace."
    )


MEMBER3_DIR = SNAKE_DIR / "member 3"
if not MEMBER3_DIR.is_dir():
    raise FileNotFoundError(f"Member 3 submission directory was not found: {MEMBER3_DIR}")

required = ["torch", "numpy", "pandas", "matplotlib", "gymnasium", "pygame", "tensorboard"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError("Missing packages in the active container: " + ", ".join(missing))

os.chdir(SNAKE_DIR)
if str(SNAKE_DIR) not in sys.path:
    sys.path.insert(0, str(SNAKE_DIR))

print("Python:", sys.executable)
print("Experiment directory:", SNAKE_DIR)


# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display

from environment import SnakeEnv
from dqn import DQNAgent, ReplayBuffer
from benchmark import load_dqn
from benchmark_double_dqn import evaluate_q_diagnostics
from main import choose_device, seed_everything

MODE = os.environ.get("MEMBER3_MODE", "analysis").strip().lower()
REUSE_COMPLETED = True

if MODE not in {"smoke", "final", "analysis"}:
    raise ValueError("MODE must be 'smoke', 'final', or 'analysis'")

GRID_SIZE = 8
END_SCORE = 8
MAX_STEPS_PER_EPISODE = 300
GAMMA = 0.99
TRAINING_SEEDS = [7, 42, 2026, 31415, 27182]
VALIDATION_SEED_BASE = 500_000
TEST_SEED_BASE = 900_000
COMMON_STATE_SEED_BASE = 800_000

if MODE == "smoke":
    ACTIVE_SEEDS = [42]
    TOTAL_ENV_STEPS = 10_000
    VALIDATION_INTERVAL = 2_000
    VALIDATION_EPISODES = 10
    TEST_EPISODES = 20
    PROGRESS_STEPS = [2_000, 5_000, 10_000]
    RUN_DIR = MEMBER3_DIR / "results" / "smoke"
else:
    ACTIVE_SEEDS = TRAINING_SEEDS
    TOTAL_ENV_STEPS = 210_000
    VALIDATION_INTERVAL = 20_000
    VALIDATION_EPISODES = 30
    TEST_EPISODES = 200
    PROGRESS_STEPS = [50_000, 100_000, 150_000, 200_000]
    RUN_DIR = MEMBER3_DIR / "results" / "final"

REPLAY_CAPACITY = 50_000
BATCH_SIZE = 128
LEARNING_STARTS = 1_000
TRAIN_INTERVAL = 4
TARGET_UPDATE_INTERVAL = 500
LEARNING_RATE = 3e-4
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 60_000 if MODE != "smoke" else 5_000
TORCH_THREADS = 1
DEVICE = choose_device("auto")

RUN_DIR.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(TORCH_THREADS)

protocol = {
    "protocol_mode": "smoke" if MODE == "smoke" else "final",
    "execution_mode": MODE,
    "grid_size": GRID_SIZE,
    "end_score": END_SCORE,
    "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
    "training_seeds": ACTIVE_SEEDS,
    "total_environment_steps": TOTAL_ENV_STEPS,
    "validation_interval": VALIDATION_INTERVAL,
    "validation_episodes": VALIDATION_EPISODES,
    "test_episodes": TEST_EPISODES,
    "validation_seed_base": VALIDATION_SEED_BASE,
    "test_seed_base": TEST_SEED_BASE,
    "gamma": GAMMA,
    "epsilon_decay_steps": EPSILON_DECAY_STEPS,
    "device": str(DEVICE),
}
protocol_path = RUN_DIR / "protocol.json"
if MODE == "analysis":
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Analysis mode requires an existing protocol: {protocol_path}")
    stored_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for key in ["grid_size", "end_score", "total_environment_steps", "training_seeds"]:
        if stored_protocol.get(key) != protocol.get(key):
            raise RuntimeError(f"Stored protocol mismatch for {key}: {stored_protocol.get(key)}")
    protocol = stored_protocol
else:
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

environment_versions = {
    "python": sys.version,
    "platform": platform.platform(),
    **{
        package: importlib.metadata.version(package)
        for package in ["torch", "numpy", "pandas", "matplotlib", "gymnasium", "pygame"]
    },
}
environment_path = RUN_DIR / "environment_versions.json"
if MODE != "analysis":
    environment_path.write_text(
        json.dumps(environment_versions, indent=2), encoding="utf-8"
    )
elif not environment_path.is_file():
    raise FileNotFoundError(
        f"Analysis mode requires the frozen training environment record: {environment_path}"
    )
display(pd.Series(protocol, name="value").to_frame())


# -----------------------------------------------------------------------------

def epsilon_at_step(total_steps):
    fraction = min(max(total_steps, 0) / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


@torch.no_grad()
def evaluate_agent(agent, episodes, seed_base):
    env = SnakeEnv(grid_size=GRID_SIZE, end_score=END_SCORE,
                   max_steps=MAX_STEPS_PER_EPISODE)
    rng = random.Random(seed_base)
    rewards, scores, steps = [], [], []
    successes = 0
    was_training = agent.online.training
    agent.online.eval()
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed_base + episode)
            total_reward = 0.0
            while True:
                action = agent.select_action(observation, epsilon=0.0, random_source=rng)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    rewards.append(total_reward)
                    scores.append(int(info["score"]))
                    steps.append(int(info["steps"]))
                    successes += int(info["length"] >= END_SCORE)
                    break
    finally:
        env.close()
        agent.online.train(was_training)
    return {
        "reward": float(np.mean(rewards)),
        "score": float(np.mean(scores)),
        "score_std_episode": float(np.std(scores, ddof=1)) if episodes > 1 else 0.0,
        "success_rate": successes / episodes,
        "mean_steps": float(np.mean(steps)),
    }


def checkpoint_config(double_dqn, seed):
    return {
        "grid_size": GRID_SIZE,
        "end_score": END_SCORE,
        "max_steps": MAX_STEPS_PER_EPISODE,
        "double_dqn": double_dqn,
        "dueling": False,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "gamma": GAMMA,
        "seed": seed,
        "total_environment_steps": TOTAL_ENV_STEPS,
        "epsilon_decay_steps": EPSILON_DECAY_STEPS,
    }


def train_step_budget(seed, double_dqn, run_dir):
    algorithm = "double_dqn" if double_dqn else "dqn"
    complete_path = run_dir / "complete.json"
    best_path = run_dir / "checkpoints" / "best.pt"
    latest_path = run_dir / "checkpoints" / "latest.pt"
    history_path = run_dir / "history.json"

    if REUSE_COMPLETED and complete_path.is_file() and best_path.is_file() and history_path.is_file():
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        compatible = (
            completed.get("seed") == seed
            and completed.get("algorithm") == algorithm
            and completed.get("total_steps") == TOTAL_ENV_STEPS
        )
        if compatible:
            print(f"Reusing completed run: seed={seed}, algorithm={algorithm}")
            return completed
        raise RuntimeError(
            f"Existing completion marker is incompatible with the current protocol: {complete_path}. "
            "Use a new RUN_DIR instead of mixing experiments."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    seed_everything(seed)
    rng = random.Random(seed)
    env = SnakeEnv(grid_size=GRID_SIZE, end_score=END_SCORE,
                   max_steps=MAX_STEPS_PER_EPISODE)
    agent = DQNAgent(
        input_channels=env.observation_space.shape[0],
        action_count=env.action_space.n,
        grid_size=GRID_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
    )
    replay = ReplayBuffer(REPLAY_CAPACITY, seed=seed)
    history = {"train_step": [], "train_reward": [], "train_score": [],
               "epsilon": [], "validation": []}
    total_steps = 0
    episode = 0
    next_validation = VALIDATION_INTERVAL
    pending_progress = list(PROGRESS_STEPS)
    best_key = (-float("inf"), -float("inf"), -float("inf"))
    best_step = None
    started = time.perf_counter()

    try:
        while total_steps < TOTAL_ENV_STEPS:
            observation, _ = env.reset(seed=seed + episode)
            episode_reward = 0.0
            score = 0
            while total_steps < TOTAL_ENV_STEPS:
                epsilon = epsilon_at_step(total_steps)
                action = agent.select_action(observation, epsilon=epsilon, random_source=rng)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                replay.append(observation, action, reward, next_observation, done)
                observation = next_observation
                episode_reward += reward
                score = int(info["score"])
                total_steps += 1

                ready = total_steps >= LEARNING_STARTS and len(replay) >= BATCH_SIZE
                if ready and total_steps % TRAIN_INTERVAL == 0:
                    agent.update(replay.sample(BATCH_SIZE, DEVICE), GAMMA,
                                 double_dqn=double_dqn)
                while pending_progress and total_steps >= pending_progress[0]:
                    milestone = pending_progress.pop(0)
                    agent.save_checkpoint(
                        run_dir / "checkpoints" / f"step-{milestone}.pt",
                        episode=episode, total_steps=total_steps,
                        config=checkpoint_config(double_dqn, seed),
                    )

                if total_steps >= next_validation or total_steps == TOTAL_ENV_STEPS:
                    metrics = evaluate_agent(agent, VALIDATION_EPISODES, VALIDATION_SEED_BASE)
                    metrics["step"] = total_steps
                    history["validation"].append(metrics)
                    key = (metrics["score"], metrics["success_rate"], metrics["reward"])
                    if key > best_key:
                        best_key = key
                        best_step = total_steps
                        agent.save_checkpoint(
                            best_path, episode=episode, total_steps=total_steps,
                            config=checkpoint_config(double_dqn, seed),
                        )
                    print(
                        f"seed={seed} algorithm={algorithm} step={total_steps}/{TOTAL_ENV_STEPS} "
                        f"val_score={metrics['score']:.2f} "
                        f"val_success={metrics['success_rate']:.0%} epsilon={epsilon:.3f}"
                    )
                    next_validation += VALIDATION_INTERVAL
                    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

                # Validate and save the checkpoint before a scheduled target sync.
                # Otherwise every validation checkpoint would contain identical online
                # and target networks, making selection-gap diagnostics identically zero.
                if total_steps % TARGET_UPDATE_INTERVAL == 0:
                    agent.sync_target()

                if done:
                    break

            history["train_step"].append(total_steps)
            history["train_reward"].append(float(episode_reward))
            history["train_score"].append(score)
            history["epsilon"].append(float(epsilon_at_step(total_steps)))
            episode += 1
    finally:
        env.close()

    agent.save_checkpoint(latest_path, episode=episode - 1, total_steps=total_steps,
                          config=checkpoint_config(double_dqn, seed))
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "seed": seed,
        "algorithm": algorithm,
        "total_steps": total_steps,
        "episodes": episode,
        "best_validation_step": best_step,
        "best_validation_score": best_key[0],
        "train_seconds": time.perf_counter() - started,
        "best_checkpoint": str(best_path),
    }
    complete_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# -----------------------------------------------------------------------------

training_summaries = []
for seed in ACTIVE_SEEDS:
    for algorithm, use_double, dirname in [
        ("dqn", False, "dqn"),
        ("double_dqn", True, "double-dqn"),
    ]:
        model_dir = RUN_DIR / f"seed-{seed}" / dirname
        if MODE == "analysis":
            complete_path = model_dir / "complete.json"
            if not complete_path.is_file():
                raise FileNotFoundError(f"Missing completed run: {complete_path}")
            summary = json.loads(complete_path.read_text(encoding="utf-8"))
        else:
            summary = train_step_budget(seed, use_double, model_dir)
        training_summaries.append(summary)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

training_summaries = pd.DataFrame(training_summaries)
display(training_summaries)
training_summaries.to_csv(RUN_DIR / "training_summaries.csv", index=False)


# -----------------------------------------------------------------------------

# We visualize checkpoint quality and compute cost as soon as paired training is available.
training_plot = training_summaries.copy()
wide_training = training_plot.pivot(index="seed", columns="algorithm")
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
for seed in wide_training.index:
    values = [
        wide_training.loc[seed, ("best_validation_score", "dqn")],
        wide_training.loc[seed, ("best_validation_score", "double_dqn")],
    ]
    axes[0].plot([0, 1], values, color="#A0A0A0", lw=1.2, alpha=0.8)
    axes[0].scatter([0, 1], values, s=46, color=["#D55E00", "#0072B2"], zorder=3)
axes[0].set_xticks([0, 1], ["DQN", "Double DQN"])
axes[0].set_ylabel("Best validation score")
axes[0].set_title("Selected checkpoint by paired seed", weight="bold")

time_summary = (training_plot.assign(
    label=training_plot.algorithm.map({"dqn": "DQN", "double_dqn": "Double DQN"})
).groupby("label")["train_seconds"].agg(["mean", "std"]).reindex(["DQN", "Double DQN"]))
axes[1].bar([0, 1], time_summary["mean"], yerr=time_summary["std"], capsize=5,
            color=["#D55E00", "#0072B2"], width=0.65)
axes[1].set_xticks([0, 1], ["DQN", "Double DQN"])
axes[1].set_ylabel("Training time (seconds)")
axes[1].set_title("CPU training cost", weight="bold")
for ax in axes:
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Training overview after all paired runs finish", weight="bold")
fig.tight_layout()
fig.savefig(RUN_DIR / "training_overview.png", dpi=220, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------

def collect_common_states(max_states=1500):
    env = SnakeEnv(grid_size=GRID_SIZE, end_score=END_SCORE,
                   max_steps=MAX_STEPS_PER_EPISODE)
    rng = random.Random(COMMON_STATE_SEED_BASE)
    states = []
    policies = ["random", "legacy", "shortest_path"]
    try:
        episode = 0
        while len(states) < max_states:
            mode = policies[episode % len(policies)]
            env.teacher_mode = mode if mode != "random" else "shortest_path"
            observation, _ = env.reset(seed=COMMON_STATE_SEED_BASE + episode)
            while len(states) < max_states:
                states.append(observation.copy())
                action = rng.randrange(env.action_space.n) if mode == "random" else env.teacher_action()
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            episode += 1
    finally:
        env.close()
    return np.stack(states)


@torch.no_grad()
def common_state_diagnostics(agent, states):
    selection_gaps = []
    disagreements = 0
    for start in range(0, len(states), 512):
        batch = torch.from_numpy(states[start:start + 512]).to(DEVICE, dtype=torch.float32)
        online_actions = agent.online(batch).argmax(dim=1, keepdim=True)
        target_values = agent.target(batch)
        target_actions = target_values.argmax(dim=1, keepdim=True)
        standard_values = target_values.max(dim=1).values
        double_values = target_values.gather(1, online_actions).squeeze(1)
        selection_gaps.extend((standard_values - double_values).cpu().tolist())
        disagreements += int((online_actions != target_actions).sum().item())
    return {
        "common_target_selection_gap": float(np.mean(selection_gaps)),
        "common_action_disagreement_rate": disagreements / len(states),
    }

common_states = collect_common_states()
print("Shared held-out states:", common_states.shape)


# -----------------------------------------------------------------------------

final_rows = []
for seed in ACTIVE_SEEDS:
    for algorithm, dirname in [("dqn", "dqn"), ("double_dqn", "double-dqn")]:
        model_dir = RUN_DIR / f"seed-{seed}" / dirname
        summary = json.loads((model_dir / "complete.json").read_text(encoding="utf-8"))
        agent = load_dqn(model_dir / "checkpoints" / "best.pt", DEVICE)
        diagnostics = evaluate_q_diagnostics(
            agent,
            grid_size=GRID_SIZE,
            end_score=END_SCORE,
            max_steps=MAX_STEPS_PER_EPISODE,
            episodes=TEST_EPISODES,
            seed_base=TEST_SEED_BASE,
            gamma=GAMMA,
        )
        common = common_state_diagnostics(agent, common_states)
        final_rows.append({
            "seed": seed,
            "algorithm": algorithm,
            "best_validation_step": summary["best_validation_step"],
            "train_seconds": summary["train_seconds"],
            **diagnostics,
            **common,
        })
        del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

results = pd.DataFrame(final_rows)
display(results.round(4))
results.to_csv(RUN_DIR / "final_results.csv", index=False)
(RUN_DIR / "final_results.json").write_text(
    json.dumps({"protocol": protocol, "runs": final_rows}, indent=2), encoding="utf-8"
)


# -----------------------------------------------------------------------------

# We inspect paired final outcomes immediately after the frozen test finishes.
wide_results = results.pivot(index="seed", columns="algorithm")
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
for ax, metric, ylabel in zip(
    axes, ["score", "success_rate"], ["Test mean score", "Test success rate"]
):
    for seed in wide_results.index:
        values = [wide_results.loc[seed, (metric, "dqn")],
                  wide_results.loc[seed, (metric, "double_dqn")]]
        ax.plot([0, 1], values, color="#A0A0A0", lw=1.3, alpha=0.8)
        ax.scatter([0, 1], values, s=48, color=["#D55E00", "#0072B2"], zorder=3)
    ax.set_xticks([0, 1], ["DQN", "Double DQN"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_ylim(0.25, 0.55)
fig.suptitle("Paired frozen-test outcomes", weight="bold")
fig.tight_layout()
fig.savefig(RUN_DIR / "paired_final_outcomes.png", dpi=220, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------

def evaluate_baseline(name):
    teacher_mode = "legacy" if name == "teacher_legacy" else "shortest_path"
    env = SnakeEnv(grid_size=GRID_SIZE, end_score=END_SCORE,
                   max_steps=MAX_STEPS_PER_EPISODE, teacher_mode=teacher_mode)
    rng = random.Random(TEST_SEED_BASE)
    rows = []
    try:
        for episode in range(TEST_EPISODES):
            _, _ = env.reset(seed=TEST_SEED_BASE + episode)
            total_reward = 0.0
            while True:
                action = rng.randrange(env.action_space.n) if name == "random" else env.teacher_action()
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    rows.append({
                        "strategy": name,
                        "episode": episode,
                        "reward": total_reward,
                        "score": int(info["score"]),
                        "success": int(info["length"] >= END_SCORE),
                    })
                    break
    finally:
        env.close()
    return pd.DataFrame(rows)

baseline_episodes = pd.concat(
    [evaluate_baseline("random"), evaluate_baseline("teacher_legacy"),
     evaluate_baseline("teacher_bfs")], ignore_index=True
)
baseline_summary = baseline_episodes.groupby("strategy").agg(
    reward=("reward", "mean"), score=("score", "mean"),
    success_rate=("success", "mean"), score_std_episode=("score", "std")
).reset_index()
display(baseline_summary.round(4))
baseline_episodes.to_csv(RUN_DIR / "baseline_test_episodes.csv", index=False)


# -----------------------------------------------------------------------------

# We place the learned agents next to the non-learning references.
learned_summary = results.groupby("algorithm").agg(
    reward=("reward", "mean"), score=("score", "mean"),
    success_rate=("success_rate", "mean")
)
baseline_for_plot = baseline_summary.set_index("strategy")[["reward", "score", "success_rate"]]
order = ["random", "teacher_legacy", "teacher_bfs", "dqn", "double_dqn"]
comparison = pd.concat([baseline_for_plot, learned_summary]).loc[order]
labels = ["Random", "Legacy\nteacher", "BFS\nteacher", "DQN", "Double\nDQN"]
palette = ["#999999", "#56B4E9", "#009E73", "#D55E00", "#0072B2"]
fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
for ax, metric, title in zip(
    axes, ["score", "success_rate", "reward"],
    ["Mean score", "Success rate", "Mean episodic reward"]
):
    values = comparison[metric].to_numpy(float)
    ax.bar(np.arange(len(values)), values, color=palette, width=0.72)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_title(title, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_ylim(0, 1.05)
fig.suptitle("Baseline vs value-based agents", weight="bold")
fig.tight_layout()
fig.savefig(RUN_DIR / "baseline_vs_agents.png", dpi=220, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------

paired = results.pivot(index="seed", columns="algorithm")
metrics = ["score", "success_rate", "q_return_gap_mean",
           "q_return_gap_abs_mean", "q_overestimate_rate",
           "common_target_selection_gap", "common_action_disagreement_rate"]
paired_rows = []
for seed in ACTIVE_SEEDS:
    row = {"seed": seed}
    for metric in metrics:
        row[f"delta_{metric}"] = (
            paired.loc[seed, (metric, "double_dqn")] - paired.loc[seed, (metric, "dqn")]
        )
    paired_rows.append(row)
paired_differences = pd.DataFrame(paired_rows)
display(paired_differences.round(4))
paired_differences.to_csv(RUN_DIR / "paired_differences.csv", index=False)

def bootstrap_mean_ci(values, seed=12345, resamples=20_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))

bootstrap_rows = []
for metric in metrics:
    mean, low, high = bootstrap_mean_ci(paired_differences[f"delta_{metric}"])
    bootstrap_rows.append({"metric": metric, "mean_delta_ddqn_minus_dqn": mean,
                           "bootstrap_95_low": low, "bootstrap_95_high": high})
bootstrap_summary = pd.DataFrame(bootstrap_rows)
display(bootstrap_summary.round(4))
bootstrap_summary.to_csv(RUN_DIR / "paired_bootstrap_summary.csv", index=False)


# -----------------------------------------------------------------------------

# We show the seed pairs instead of hiding them behind aggregate bars.
diagnostics_to_plot = [
    ("q_return_gap_mean", "Signed Q-return gap", 0.0),
    ("q_return_gap_abs_mean", "Absolute Q-return gap", None),
    ("q_overestimate_rate", "Q-overestimate rate", 0.5),
    ("common_action_disagreement_rate", "Common-state action disagreement", None),
]
wide_results = results.pivot(index="seed", columns="algorithm")
fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.1))
for ax, (metric, title, reference) in zip(axes.flat, diagnostics_to_plot):
    for seed in wide_results.index:
        values = [wide_results.loc[seed, (metric, "dqn")],
                  wide_results.loc[seed, (metric, "double_dqn")]]
        ax.plot([0, 1], values, color="#A0A0A0", lw=1.2, alpha=0.75)
        ax.scatter([0, 1], values, s=42, color=["#D55E00", "#0072B2"], zorder=3)
    if reference is not None:
        ax.axhline(reference, color="black", ls="--", lw=1, alpha=0.65)
    ax.set_xticks([0, 1], ["DQN", "Double DQN"])
    ax.set_title(title, weight="bold", fontsize=10.5)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Q-value and target-network diagnostics by paired seed", weight="bold")
fig.tight_layout()
fig.savefig(RUN_DIR / "q_diagnostics.png", dpi=220, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------

validation_frames = []
for seed in ACTIVE_SEEDS:
    for algorithm, dirname in [("DQN", "dqn"), ("Double DQN", "double-dqn")]:
        history = json.loads(
            (RUN_DIR / f"seed-{seed}" / dirname / "history.json").read_text(encoding="utf-8")
        )
        frame = pd.DataFrame(history["validation"])
        frame["seed"] = seed
        frame["algorithm"] = algorithm
        validation_frames.append(frame)
validation = pd.concat(validation_frames, ignore_index=True)

colors = {"DQN": "#D55E00", "Double DQN": "#0072B2"}
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True)
for ax, metric, ylabel in zip(
    axes, ["score", "success_rate"], ["Validation mean score", "Validation success rate"]
):
    for algorithm in ["DQN", "Double DQN"]:
        selected = validation[validation.algorithm == algorithm]
        grouped = selected.groupby("step")[metric].agg(["mean", "std"]).reset_index()
        grouped["std"] = grouped["std"].fillna(0.0)
        x = grouped["step"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        s = grouped["std"].to_numpy(dtype=float)
        ax.plot(x, y, linewidth=2.3, color=colors[algorithm], label=algorithm)
        ax.fill_between(x, y - s, y + s, color=colors[algorithm], alpha=0.17)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
axes[0].set_ylim(0, 7.2)
axes[1].set_ylim(0, 1.02)
axes[0].legend(frameon=False)
fig.suptitle("Greedy validation performance at equal interaction budgets",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(RUN_DIR / "validation_curves.png", dpi=220, bbox_inches="tight")
plt.show()

aggregate = results.groupby("algorithm").agg(
    score_mean=("score", "mean"), score_sd=("score", "std"),
    success_mean=("success_rate", "mean"), success_sd=("success_rate", "std"),
    q_gap_mean=("q_return_gap_mean", "mean"), q_gap_sd=("q_return_gap_mean", "std"),
).reset_index()
display(aggregate.round(4))
aggregate.to_csv(RUN_DIR / "aggregate_results.csv", index=False)


# -----------------------------------------------------------------------------

print("Artifacts written to:", RUN_DIR.resolve())
print("Primary files:")
for name in [
    "protocol.json", "training_summaries.csv", "final_results.csv",
    "paired_differences.csv", "paired_bootstrap_summary.csv",
    "aggregate_results.csv", "training_overview.png",
    "paired_final_outcomes.png", "baseline_vs_agents.png",
    "q_diagnostics.png", "validation_curves.png",
]:
    path = RUN_DIR / name
    print(f"  {name}: {path.is_file()}")
