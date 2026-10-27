"""Read-only support for the consolidated DSA5102 Snake project notebook.

The notebook reuses the members' original models and saved evidence. This module
does not import or execute a training entry point and never rewrites checkpoints.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_project(start=None):
    here = Path(start).resolve() if start else Path(__file__).resolve().parent
    for candidate in [here, here / "Snake", here.parent]:
        if (candidate / "environment.py").is_file() and (candidate / "member5/LATEST_RUNS.json").is_file():
            return candidate
    raise FileNotFoundError("Keep the notebook and this support file inside the complete Snake directory.")


def setup(start=None):
    root = find_project(start)
    torch.set_num_threads(1)
    for folder in [root, root / "member5"]:
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    env = load_source("final_shared_env", root / "environment.py")
    ac = load_source("final_member2_model", root / "member2_snake/model.py")
    q = load_source("final_shared_dqn", root / "dqn.py")
    ppo = load_source("final_shared_ppo", root / "ppo.py")
    decisions = load_source("final_decision_inspection", root / "inspect_decisions.py")
    runs = json.loads((root / "member5/LATEST_RUNS.json").read_text())
    paths = {name: root / "member5" / path for name, path in runs.items()}
    paths["analysis_figures"] = root / "member5/figures/analysis" / paths["analysis"].name
    paths["decision_figures"] = root / "member5/figures/decisions" / paths["decisions"].name
    return dict(root=root, env=env, ac=ac, q=q, ppo=ppo, decisions=decisions, paths=paths)


def load_model(context, method, seed=7, clip=0.2):
    if method != "PPO" or clip == 0.2:
        return context["decisions"].load_network(method, seed, context["root"],
            context["ac"].ConvActorCritic, context["q"].QNetwork, context["ppo"].PPOActorCritic)
    if clip not in (0.1, 0.3):
        raise ValueError("Only the three delivered PPO clipping conditions are supported")
    path = context["root"] / "member4/results/final" / f"seed-{seed}" / f"clip-{clip:.1f}/checkpoints/best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if checkpoint["algorithm"] != "ppo" or config["seed"] != seed or config["clip_ratio"] != clip:
        raise ValueError("PPO checkpoint identity does not match its requested condition")
    if config["grid_size"] != 8 or config["steps"] != 500000:
        raise ValueError("PPO checkpoint differs from the final experimental protocol")
    model = context["ppo"].PPOActorCritic(7, 3, 8, learning_rate=.0003, weight_decay=0.)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, dict(method="PPO", training_seed=seed, clip=clip,
                       selected_checkpoint_steps=checkpoint["total_steps"],
                       training_budget_steps=config["steps"])


def verify_all_models(context):
    """Check all 35 distinct trained models, including all three PPO clips."""
    from member5_analysis import verify_saved_checkpoints
    inventory = verify_saved_checkpoints(context["root"])
    inventory["clip"] = np.where(inventory.method == "PPO", .2, np.nan)
    rows = []
    env = context["env"].SnakeEnv()
    observation, _ = env.reset(seed=1300000)
    env.close()
    state = torch.as_tensor(observation).unsqueeze(0)
    for clip in (.1, .3):
        for seed in [7, 42, 2026, 31415, 27182]:
            model, meta = load_model(context, "PPO", seed, clip)
            before = {k: value.clone() for k, value in model.state_dict().items()}
            with torch.inference_mode():
                first = model(state)
                second = model(state)
            assert all(torch.isfinite(x).all() for x in first)
            assert all(torch.equal(a, b) for a, b in zip(first, second))
            assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
            rows.append(dict(method="PPO", training_seed=seed, clip=clip, loaded=True,
                             greedy_action=["left", "right", "straight"][first[0].argmax(-1).item()],
                             selected_steps=meta["selected_checkpoint_steps"],
                             full_training_budget=meta["training_budget_steps"]))
    inventory = pd.concat([inventory, pd.DataFrame(rows)], ignore_index=True)
    assert len(inventory) == 35 and inventory.loaded.all()
    return inventory


def source_excerpt(path, class_name=None, function_name=None):
    """Show the actual implementation without importing training programs."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function_name)
    return ast.unparse(node)


def common_tables(context):
    folder = context["paths"]["evaluation"]
    episodes = pd.read_csv(folder / "episodes.csv")
    per_model = pd.read_csv(folder / "per_model.csv")
    summary = pd.read_csv(folder / "summary.csv")
    paired = pd.read_csv(context["paths"]["analysis"] / "paired_statistics.csv")
    assert len(episodes) == 5400 and len(per_model) == 27
    assert not episodes.duplicated(["method", "training_seed", "test_seed"]).any()
    for method, group in episodes.groupby("method"):
        expected = summary.set_index("method").loc[method]
        assert np.isclose(group.score.mean(), expected.score_mean)
        assert np.isclose(group.success.mean(), expected.success_rate)
    return episodes, per_model, summary, paired


def _board(axis, env, title):
    grid = env.grid_size
    axis.set(xlim=(0, grid), ylim=(grid, 0), aspect="equal", title=title)
    for value in range(grid + 1):
        axis.axhline(value, color="#dce2e8", lw=.7, zorder=0)
        axis.axvline(value, color="#dce2e8", lw=.7, zorder=0)
    for index, cell in enumerate(env.snake):
        row, col = cell
        axis.add_patch(plt.Rectangle((col+.08,row+.08),.84,.84,
                                    facecolor="#bd4646" if index==0 else "#efa85c",zorder=2))
    row, col = env.food_pos
    axis.scatter(col+.5,row+.5,marker="*",s=170,color="#17867d",zorder=3)
    row,col=env.snake[0];dr,dc=env.current_direction
    axis.arrow(col+.5,row+.5,dc*.3,dr*.3,width=.035,head_width=.18,
               color="white",length_includes_head=True,zorder=4)
    axis.set_xticks([]);axis.set_yticks([])


def make_game_figures(context):
    root=context["root"]
    output=root/"final_assets"
    output.mkdir(exist_ok=True)
    scenario=context["decisions"].build_scenarios()[0]
    figure,axes=plt.subplots(1,3,figsize=(11.5,4))
    env=context["decisions"].scenario_environment(scenario,context["env"].SnakeEnv)
    initial=env._get_observation().copy()
    _board(axes[0],env,"Initial: length 1, food score 0")
    env.step(2)
    _board(axes[1],env,"One straight move")
    env.step(2)
    assert len(env.snake)==2 and env.score==1
    _board(axes[2],env,"Food eaten: length 2, score 1")
    env.close()
    figure.suptitle("Snake moves one cell per action and grows after eating",weight="bold")
    figure.text(.5,.02,"Red: head and heading | Orange: body | Green star: food | Moving outside the grid causes a wall collision",ha="center",fontsize=8)
    figure.tight_layout(rect=(0,.06,1,.92))
    board=output/"game_rules.png";figure.savefig(board,dpi=160);plt.close(figure)
    figure,axes=plt.subplots(2,4,figsize=(11,5.5))
    labels=["Snake occupancy","Food","Border cue","Heading up","Heading down","Heading left","Heading right"]
    for index,axis in enumerate(axes.flat):
        if index==7:axis.axis("off");continue
        axis.imshow(initial[index],cmap="Blues",vmin=0,vmax=1,interpolation="nearest")
        axis.set(title=labels[index],xticks=[],yticks=[])
    figure.suptitle("The seven binary observation channels",weight="bold")
    figure.text(.5,.02,"White = 0; blue = 1. Border cells are traversable; the border channel is a cue, not a map of blocked cells.",ha="center",fontsize=8)
    figure.tight_layout(rect=(0,.05,1,.94))
    channels=output/"observation_channels.png";figure.savefig(channels,dpi=160);plt.close(figure)
    return board,channels


def quick_inference_check(context, game_seed=1300000):
    """One fixed, new game per configuration; a runtime check, not a new study."""
    rows=[]
    for method in ["Random","Teacher","AC","BC + AC","DQN","Double DQN","PPO"]:
        model=None if method in ["Random","Teacher"] else load_model(context,method,7)[0]
        env=context["env"].SnakeEnv()
        obs,_=env.reset(seed=game_seed)
        rng=np.random.default_rng(game_seed+8000000)
        reward_sum=0.
        try:
            while True:
                if method=="Random":action=int(rng.integers(3))
                elif method=="Teacher":action=env.teacher_action()
                else:
                    with torch.inference_mode():
                        prediction=model(torch.as_tensor(obs).unsqueeze(0))
                        scores=prediction[0] if isinstance(prediction,tuple) else prediction
                        action=int(scores.argmax(-1).item())
                obs,reward,terminal,timeout,info=env.step(action)
                reward_sum+=reward
                if terminal or timeout:
                    rows.append(dict(method=method,game_seed=game_seed,score=info["score"],
                                     success=info["length"]>=8,steps=info["steps"],reward=reward_sum))
                    break
        finally:env.close()
    return pd.DataFrame(rows)


def ppo_clip_results(context):
    data=pd.read_csv(context["root"]/"member4/results/final/results.csv")
    assert len(data)==15 and data.groupby("clip").size().eq(5).all()
    return data.groupby("clip").agg(mean_score=("score","mean"),seed_sd=("score","std"),
                                    success_rate=("success","mean"),timeout_rate=("timeout","mean")).reset_index()
