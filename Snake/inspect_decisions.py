"""Inspect existing Snake checkpoints on four fixed 8 x 8 states.

From Snake: python inspect_decisions.py --seed 42
From Snake/member5 in Jupyter: %run ../inspect_decisions.py --seed 42
This is a qualitative decision probe. It never trains or runs the benchmark.
PPO integration is pending review of Member 4's delivered model and weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

GRID_SIZE = 8
ENV_CONFIG = dict(grid_size=8, end_score=8, max_steps=300,
                  reward_mode="sparse", teacher_mode="shortest_path")
TRAINING_SEEDS = (7, 42, 2026, 31415, 27182)
METHODS = ("AC", "BC + AC", "DQN", "Double DQN")
ACTION_NAMES = ("left", "right", "straight")
ACTION_COLORS = ("#c45b38", "#167d76", "#3568a8")
MEMBER2_ARCHIVE_SHA256 = "2bb0b353c4851142304685e77bb9000891b222ff181d40e78556d76b1921b6a9"


@dataclass(frozen=True)
class Scenario:
    """Coordinates are (row, column), with the head first in snake."""

    name: str
    title: str
    description: str
    snake: tuple[tuple[int, int], ...]
    food: tuple[int, int]
    direction: tuple[int, int]


def build_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario("food_ahead", "Food ahead", "Food is two cells ahead on an open board.",
                 ((4, 4),), (4, 6), (0, 1)),
        Scenario("wall_ahead", "Wall ahead", "Straight immediately hits the right boundary.",
                 ((4, 7), (4, 6)), (0, 0), (0, 1)),
        Scenario("body_turn", "Bent body", "A bent body is visible behind the head.",
                 ((3, 3), (3, 2), (4, 2)), (5, 3), (0, 1)),
        Scenario("narrow_escape", "Body beside head",
                 "Right immediately hits the body; left and straight avoid immediate collision.",
                 ((2, 2), (3, 2), (3, 3), (2, 3), (1, 3)), (0, 7), (-1, 0)),
    )


def validate_scenario(scenario: Scenario) -> None:
    cells = scenario.snake
    if not 1 <= len(cells) < ENV_CONFIG["end_score"]:
        raise ValueError(f"{scenario.name}: snake must be nonempty and below the target length")
    if len(set(cells)) != len(cells) or scenario.food in cells:
        raise ValueError(f"{scenario.name}: overlapping snake or food")
    if any(not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE) for r, c in (*cells, scenario.food)):
        raise ValueError(f"{scenario.name}: coordinates are outside the board")
    if scenario.direction not in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        raise ValueError(f"{scenario.name}: invalid heading")
    if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(cells, cells[1:])):
        raise ValueError(f"{scenario.name}: body segments are not adjacent")
    if len(cells) > 1:
        heading = tuple(a - b for a, b in zip(cells[0], cells[1]))
        if heading != scenario.direction:
            raise ValueError(f"{scenario.name}: heading does not match the neck-to-head direction")


def scenario_environment(scenario: Scenario, env_class):
    validate_scenario(scenario)
    env = env_class(**ENV_CONFIG)
    env.reset(seed=0)
    env.snake = [np.asarray(cell, dtype=np.int64) for cell in scenario.snake]
    env.food_pos = np.asarray(scenario.food, dtype=np.int64)
    env.current_direction = np.asarray(scenario.direction, dtype=np.int64)
    env.score = len(scenario.snake) - 1
    env.step_count = 0
    env.visited = {scenario.snake[0]}
    env.visited_positions = [env.snake[0].copy()]
    env.game_over = False
    return env


def load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def locate_snake_folder(project_dir=None) -> Path:
    root = Path(project_dir).expanduser().resolve() if project_dir else Path(__file__).resolve().parent
    for candidate in (root, root / "Snake"):
        if all((candidate / p).is_file() for p in ("environment.py", "dqn.py", "member2_snake/model.py")):
            return candidate
    raise FileNotFoundError("Use the complete group repository, or pass --project-dir pointing to Snake.")


def load_network(method: str, seed: int, snake_dir: Path, ac_class, q_class):
    """Read original best.pt weights; reject incompatible or missing deliveries."""
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}. PPO integration is pending.")
    if method in ("AC", "BC + AC"):
        condition = "ac" if method == "AC" else "bc_ac"
        relative = Path("runs/member2/final") / f"seed-{seed}" / condition / "checkpoints/best.pt"
        archive_path = snake_dir / "member2_snake/member2_github_upload 2/member2_assets.zip"
        candidates = [base / relative for base in (snake_dir, snake_dir / "member2_snake", archive_path.parent)]
        path = next((p for p in candidates if p.is_file()), None)
        if path is not None:
            payload = path.read_bytes()
            reference = str(path.relative_to(snake_dir))
        else:
            if not archive_path.is_file():
                raise FileNotFoundError(f"Missing {method}, seed={seed}: {relative}")
            if sha256(archive_path.read_bytes()) != MEMBER2_ARCHIVE_SHA256:
                raise ValueError("Member 2's archive changed; review the new delivery before inspection.")
            with zipfile.ZipFile(archive_path) as archive:
                try:
                    payload = archive.read(relative.as_posix())
                except KeyError as exc:
                    raise FileNotFoundError(f"Missing {relative} in Member 2's archive") from exc
            reference = f"{archive_path.relative_to(snake_dir)}!{relative.as_posix()}"
    else:
        condition = "dqn" if method == "DQN" else "double-dqn"
        path = snake_dir / "member 3/results/final" / f"seed-{seed}" / condition / "checkpoints/best.pt"
        payload = path.read_bytes()
        reference = str(path.relative_to(snake_dir))
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    for key in ("grid_size", "end_score", "max_steps", "reward_mode", "teacher_mode"):
        if key in config and config[key] != ENV_CONFIG[key]:
            raise ValueError(f"{method}, seed={seed}: {key} is incompatible with the shared environment")
    if config.get("grid_size") != GRID_SIZE or config.get("seed") != seed:
        raise ValueError(f"{method}: saved grid size or training seed does not match this inspection")
    if method in ("AC", "BC + AC"):
        if config.get("teacher_episodes") != (0 if method == "AC" else 100):
            raise ValueError("The saved AC/BC condition does not match the requested method")
        network = ac_class(input_channels=7, output_dim=3, grid_size=GRID_SIZE)
        network.load_state_dict(checkpoint["model_state_dict"], strict=True)
        budget = config.get("rl_steps")
        selected_steps = config.get("progress_rl_steps")
    else:
        expected = "dqn" if method == "DQN" else "double_dqn"
        if checkpoint.get("algorithm") != expected or config.get("dueling", False):
            raise ValueError("The saved DQN type does not match its method label")
        network = q_class(input_channels=7, action_count=3, grid_size=GRID_SIZE)
        network.load_state_dict(checkpoint["online_state_dict"], strict=True)
        budget = config.get("total_environment_steps")
        selected_steps = checkpoint.get("total_steps")
    network.eval()
    network.requires_grad_(False)
    metadata = dict(method=method, training_seed=seed, checkpoint=reference,
                    checkpoint_sha256=sha256(payload), checkpoint_config=config,
                    training_budget_steps=budget, selected_checkpoint_steps=selected_steps)
    return network, metadata


def inspect_network(network, method: str, observations: list[np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    with torch.inference_mode():
        for observation in observations:
            state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            output = network(state)
            if method in ("AC", "BC + AC"):
                logits, value = output
                scores = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                row = {"score_type": "probability", "state_value": float(value.item())}
            else:
                scores = output.squeeze(0).numpy()
                row = {"score_type": "q_value"}
            if scores.shape != (3,) or not np.isfinite(scores).all():
                raise ValueError(f"{method}: invalid action scores")
            ordered = np.sort(scores)
            row.update(scores=scores.tolist(), selected_action=int(scores.argmax()),
                       top_action_margin=float(ordered[-1] - ordered[-2]))
            rows.append(row)
    return rows


def draw_board(axis, scenario: Scenario) -> None:
    axis.set(xlim=(0, GRID_SIZE), ylim=(GRID_SIZE, 0), aspect="equal")
    for value in range(GRID_SIZE + 1):
        axis.axhline(value, color="#dfe3e8", linewidth=0.7, zorder=0)
        axis.axvline(value, color="#dfe3e8", linewidth=0.7, zorder=0)
    for index, (row, col) in enumerate(scenario.snake):
        axis.add_patch(plt.Rectangle((col + .08, row + .08), .84, .84,
                                    facecolor="#be4747" if index == 0 else "#f0ad65", zorder=2))
    row, col = scenario.food
    axis.scatter(col + .5, row + .5, marker="*", s=160, color="#167d76", zorder=3)
    row, col = scenario.snake[0]
    dr, dc = scenario.direction
    axis.arrow(col + .5, row + .5, dc * .32, dr * .32, width=.035,
               head_width=.2, color="white", length_includes_head=True, zorder=4)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(scenario.title)


def save_figures(scenarios, report, figure_dir: Path) -> None:
    figure, axes = plt.subplots(1, len(scenarios), figsize=(13, 3.7), squeeze=False)
    for axis, scenario in zip(axes.flat, scenarios, strict=True):
        draw_board(axis, scenario)
    figure.suptitle("Four fixed 8 x 8 decision probes", weight="bold")
    figure.text(.5, .02, "Red: head and heading | Orange: body | Green star: food", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .07, 1, .92))
    figure.savefig(figure_dir / "fixed_states.png", dpi=160)
    plt.close(figure)

    selected = np.array([[s["models"][method]["selected_action"] for s in report["scenarios"]] for method in METHODS])
    figure, axis = plt.subplots(figsize=(10, 4.4))
    axis.imshow(selected, cmap=matplotlib.colors.ListedColormap(ACTION_COLORS), vmin=-.5, vmax=2.5, aspect="auto")
    for r, method in enumerate(METHODS):
        for c, scenario in enumerate(report["scenarios"]):
            row = scenario["models"][method]
            label = "clear" if row["selected_action_immediately_safe"] else "collision"
            axis.text(c, r, f"{row['selected_action_name']}\n{label}", ha="center", va="center", color="white", fontsize=10)
    axis.set_xticks(range(len(scenarios)), [s.title for s in scenarios])
    axis.set_yticks(range(len(METHODS)), METHODS)
    axis.tick_params(length=0)
    axis.set_title("Greedy decisions from existing checkpoints")
    figure.text(.5, .02, "Clear = no collision on the next step only. These probes do not establish a performance ranking.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .07, 1, 1))
    figure.savefig(figure_dir / "selected_actions.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(len(METHODS), len(scenarios), figsize=(13, 10), squeeze=False)
    for r, method in enumerate(METHODS):
        for c, scenario in enumerate(report["scenarios"]):
            axis = axes[r, c]
            row = scenario["models"][method]
            bars = axis.bar(ACTION_NAMES, row["scores"], color=ACTION_COLORS)
            for a, bar in enumerate(bars):
                if not scenario["immediate_action_safety"][ACTION_NAMES[a]]:
                    bar.set_hatch("///")
                    bar.set_alpha(.5)
            axis.axhline(0, color="#555555", linewidth=.6)
            if r == 0:
                axis.set_title(scenarios[c].title)
            if c == 0:
                axis.set_ylabel(f"{method}\n{row['score_type'].replace('_', ' ')}")
            axis.tick_params(axis="x", labelsize=8)
    figure.suptitle("Action scores by fixed state and method", weight="bold")
    figure.text(.5, .015, "Hatching marks an immediately colliding action. Compare action choices; score scales differ between methods.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .96))
    figure.savefig(figure_dir / "action_scores.png", dpi=160)
    plt.close(figure)


def run_inspection(*, seed=42, project_dir=None, output_root=None):
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"Select one delivered training seed: {TRAINING_SEEDS}")
    torch.set_num_threads(1)
    snake_dir = locate_snake_folder(project_dir)
    source_paths = [snake_dir / "environment.py", snake_dir / "member2_snake/model.py", snake_dir / "dqn.py"]
    member2_env = snake_dir / "member2_snake/environment.py"
    if member2_env.is_file() and member2_env.read_bytes() != source_paths[0].read_bytes():
        raise ValueError("Member 2's environment differs from the shared environment; review before inspection.")
    env_class = load_source("decision_environment", source_paths[0]).SnakeEnv
    ac_class = load_source("decision_ac", source_paths[1]).ConvActorCritic
    q_class = load_source("decision_q", source_paths[2]).QNetwork
    scenarios = build_scenarios()
    observations, scenario_rows = [], []
    for scenario in scenarios:
        env = scenario_environment(scenario, env_class)
        try:
            observation = env._get_observation()
            if observation.shape != (7, GRID_SIZE, GRID_SIZE):
                raise ValueError("The shared observation shape has changed")
            observations.append(observation)
            safety = [not env._is_collision(env.snake[0] + env._get_direction(a)) for a in range(3)]
            scenario_rows.append(dict(name=scenario.name, title=scenario.title,
                                      description=scenario.description, snake=scenario.snake,
                                      food=scenario.food, direction=scenario.direction,
                                      immediate_action_safety=dict(zip(ACTION_NAMES, safety)),
                                      teacher_action=ACTION_NAMES[env.teacher_action()], models={}))
        finally:
            env.close()
    metadata, flat_rows = [], []
    for method in METHODS:
        network, model_metadata = load_network(method, seed, snake_dir, ac_class, q_class)
        metadata.append(model_metadata)
        for scenario_row, decision in zip(scenario_rows, inspect_network(network, method, observations), strict=True):
            action = decision["selected_action"]
            action_name = ACTION_NAMES[action]
            decision["selected_action_name"] = action_name
            decision["selected_action_immediately_safe"] = scenario_row["immediate_action_safety"][action_name]
            scenario_row["models"][method] = decision
            flat_rows.append(dict(scenario=scenario_row["name"], method=method, training_seed=seed,
                                  selected_action=action_name, immediately_safe=decision["selected_action_immediately_safe"],
                                  score_type=decision["score_type"], left=decision["scores"][0],
                                  right=decision["scores"][1], straight=decision["scores"][2]))
        del network
    report = {
        "inspection_scope": dict(training_seed=seed, environment=ENV_CONFIG, scenario_count=len(scenarios),
                                 model_count=len(METHODS), pending_methods=["PPO"], train_model=False,
                                 full_benchmark_run=False, policy="argmax; no action masking or teacher override",
                                 limitation="Hand-authored states from one training seed are qualitative probes, not a performance or significance test. Immediate safety covers only the next step."),
        "models": metadata,
        "source_sha256": {str(p.relative_to(snake_dir)): sha256(p.read_bytes()) for p in source_paths},
        "inspection_script_sha256": sha256(Path(__file__).read_bytes()),
        "versions": dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__),
        "scenarios": scenario_rows,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = Path(output_root).expanduser().resolve() if output_root else snake_dir / "member5"
    result_dir = root / "results/decisions" / stamp
    figure_dir = root / "figures/decisions" / stamp
    result_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "decisions.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (result_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    save_figures(scenarios, report, figure_dir)
    lines = ["# Fixed-state decision inspection", "",
             f"Training seed: {seed}. Four existing methods, four fixed 8 x 8 states. PPO is pending.", "",
             "This is an inference-only qualitative probe. No training or full benchmark was run.", "",
             "| State | Teacher reference | AC | BC + AC | DQN | Double DQN |",
             "|---|---|---|---|---|---|"]
    for row in scenario_rows:
        cells = [row["title"], row["teacher_action"]]
        for method in METHODS:
            decision = row["models"][method]
            suffix = "clear" if decision["selected_action_immediately_safe"] else "collision"
            cells.append(f"{decision['selected_action_name']} ({suffix})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Clear means only that the next move avoids collision. It does not establish long-term safety.",
                  "All methods see identical observations. Actor probabilities and Q values use different scales and must not be compared numerically across families.",
                  "These hand-authored states and one training seed cannot establish a final ranking or statistical significance.",
                  "Synthetic states have no preceding trajectory; the visit history starts at the specified head position.", ""])
    for name in ("fixed_states", "selected_actions", "action_scores"):
        relative = Path(os.path.relpath(figure_dir / f"{name}.png", start=result_dir)).as_posix()
        lines.extend([f"![{name.replace('_', ' ')}]({relative})", ""])
    lines.extend(["The original members supplied the environment, algorithms and saved checkpoints. OpenAI Codex assisted with adapting this inspection script and checking its outputs.", ""])
    (result_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Inspected {len(METHODS)} models on {len(scenarios)} fixed states; PPO pending.")
    print("Report:", result_dir / "report.md")
    print("Figures:", figure_dir)
    return dict(report=report, result_dir=result_dir, figure_dir=figure_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, default=42)
    parser.add_argument("--project-dir", type=Path, help="Snake folder or repository root")
    parser.add_argument("--output-root", type=Path, help="Output root; defaults to Snake/member5")
    args = parser.parse_args(argv)
    run_inspection(seed=args.seed, project_dir=args.project_dir, output_root=args.output_root)


if __name__ == "__main__":
    main()
