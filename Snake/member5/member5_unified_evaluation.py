"""Member 5: inference-only evaluation of the group's existing Snake checkpoints.

Run: python member5_unified_evaluation.py --mode smoke
The script never imports a training entry point or modifies saved checkpoints.
Existing models were trained under different budgets; comparisons must state that.
"""
from pathlib import Path


def run_benchmark(*, project_dir=None, mode="smoke", output_root=None, show_plots=False):
    """Evaluate six configurations and save a new timestamped results directory.

    project_dir: Snake folder or repository root; inferred from this file by default.
    mode: smoke = one training seed / 10 games, evaluation = five seeds / 200 games.
    output_root: output package root; defaults to the directory containing this script.
    show_plots: True for inline notebook output, False for command-line use.
    """
    # 1. Dependencies and Python environment
    import os
    import sys
    import io
    import json
    import hashlib
    import importlib.util
    import zipfile
    from pathlib import Path
    from datetime import datetime, timezone

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    try:
        from IPython.display import display
    except ImportError:
        def display(value):
            print(value.to_string(index=False) if hasattr(value, "to_string") else value)

    torch.set_num_threads(1)  # Use one CPU thread for this small network and reproducible checks.
    DEVICE = torch.device("cpu")
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Kernel Python:", sys.executable)

    # 2. Locate the shared environment and model source files
    PROJECT_OVERRIDE = project_dir  # Supply the Snake directory if automatic discovery fails.

    def locate_snake_folder(override=None):
        if override is not None:
            p = Path(override).expanduser().resolve()
            candidates = [p, p / "Snake"]
        else:
            here = Path.cwd().resolve()
            candidates = [Path(__file__).resolve().parent.parent, here, here / "Snake", here.parent]
            for name in ("DSA5102-Final-Group-Project", "DSA5102-Final-Group-Project-main"):
                candidates.append(here / name / "Snake")
        for folder in candidates:
            if all((folder / f).is_file() for f in ("environment.py", "model.py", "dqn.py")):
                return folder
        raise FileNotFoundError(
            "Could not find environment.py, model.py and dqn.py. "
            "Download and extract the complete repository, place member5 inside Snake, "
            "or use --project-dir to specify the Snake directory."
        )

    SNAKE_DIR = locate_snake_folder(PROJECT_OVERRIDE)

    def load_source(module_name, filename):
        # Load the exact file rather than importing an unrelated module with the same name.
        spec = importlib.util.spec_from_file_location(module_name, filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    env_module = load_source("member5_environment", SNAKE_DIR / "environment.py")
    ac_source = SNAKE_DIR / "member2_snake" / "model.py"
    if not ac_source.is_file():
        ac_source = SNAKE_DIR / "model.py"
    ac_module = load_source("member5_ac_model", ac_source)
    dqn_module = load_source("member5_dqn_model", SNAKE_DIR / "dqn.py")

    SnakeEnv = env_module.SnakeEnv
    ConvActorCritic = ac_module.ConvActorCritic
    QNetwork = dqn_module.QNetwork

    print("Snake directory:", SNAKE_DIR)
    print("Environment loaded. No models have been trained or evaluated yet.")

    # 3. Shared evaluation configuration
    MODE = mode  # Keep the default smoke mode for the first run.
    if MODE not in {"smoke", "evaluation"}:
        raise ValueError("MODE must be smoke or evaluation")

    TRAINING_SEEDS = [7] if MODE == "smoke" else [7, 42, 2026, 31415, 27182]
    N_EPISODES = 10 if MODE == "smoke" else 200
    TEST_SEED_BASE = 1_100_000 if MODE == "smoke" else 1_200_000
    TEST_SEEDS = list(range(TEST_SEED_BASE, TEST_SEED_BASE + N_EPISODES))
    ENV_CONFIG = dict(grid_size=8, end_score=8, max_steps=300,
                      reward_mode="sparse", teacher_mode="shortest_path")
    METHOD_ORDER = ["Random", "Teacher", "AC", "BC + AC", "DQN", "Double DQN"]
    METHOD_INFO = {
        "AC": {"budget_steps": 50000, "demo_episodes": 0},
        "BC + AC": {"budget_steps": 50000, "demo_episodes": 100},
        "DQN": {"budget_steps": 210000, "demo_episodes": 0},
        "Double DQN": {"budget_steps": 210000, "demo_episodes": 0},
    }

    probe = SnakeEnv(**ENV_CONFIG)
    obs, info = probe.reset(seed=TEST_SEEDS[0])
    assert obs.shape == (7, 8, 8)
    assert probe.action_space.n == 3 and info["length"] == 1
    probe.close()
    print("Mode:", MODE, "| Evaluation episodes per model:", N_EPISODES)
    print("Training seeds:", TRAINING_SEEDS)
    print("Test seeds:", TEST_SEEDS[0], "to", TEST_SEEDS[-1])
    print("Observation shape:", obs.shape, "| Actions: 0 left, 1 right, 2 straight")

    # 4. Locate saved checkpoints; never substitute randomly initialized models
    def sha256_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    MEMBER2_ARCHIVE = (SNAKE_DIR / "member2_snake" / "member2_github_upload 2"
                       / "member2_assets.zip")
    EXPECTED_MEMBER2_ARCHIVE_SHA256 = "2bb0b353c4851142304685e77bb9000891b222ff181d40e78556d76b1921b6a9"
    if MEMBER2_ARCHIVE.is_file():
        if sha256_file(MEMBER2_ARCHIVE) != EXPECTED_MEMBER2_ARCHIVE_SHA256:
            raise ValueError("Member 2 archive differs from the verified version. Check the updated delivery before proceeding.")
        with zipfile.ZipFile(MEMBER2_ARCHIVE) as archive:
            member2_names = set(archive.namelist())
    else:
        member2_names = set()

    member2_env = SNAKE_DIR / "member2_snake" / "environment.py"
    if member2_env.is_file() and sha256_file(member2_env) != sha256_file(SNAKE_DIR / "environment.py"):
        raise ValueError("The shared environment differs from Member 2's environment. Check the differences before evaluation.")

    def checkpoint_ref(method, training_seed):
        if method in {"AC", "BC + AC"}:
            condition = "ac" if method == "AC" else "bc_ac"
            relative = Path("runs/member2/final") / f"seed-{training_seed}" / condition / "checkpoints/best.pt"
            for base in (SNAKE_DIR, SNAKE_DIR / "member2_snake", MEMBER2_ARCHIVE.parent):
                path = base / relative
                if path.is_file():
                    return {"file": path, "zip_member": None}
            if relative.as_posix() in member2_names:
                return {"file": MEMBER2_ARCHIVE, "zip_member": relative.as_posix()}
            raise FileNotFoundError(f"Missing trained checkpoint for {method}, seed={training_seed}: {relative}")
        condition = "dqn" if method == "DQN" else "double-dqn"
        path = (SNAKE_DIR / "member 3" / "results/final" / f"seed-{training_seed}"
                / condition / "checkpoints/best.pt")
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained checkpoint for {method}, seed={training_seed}: {path}")
        return {"file": path, "zip_member": None}

    MODEL_REFS = {(method, seed): checkpoint_ref(method, seed)
                  for method in METHOD_ORDER[2:] for seed in TRAINING_SEEDS}
    inventory = pd.DataFrame([
        {"method": method, "training_seed": seed,
         "checkpoint": str(ref["file"]) + ("!" + ref["zip_member"] if ref["zip_member"] else ""),
         "status": "found", **METHOD_INFO[method]}
        for (method, seed), ref in MODEL_REFS.items()
    ])
    display(inventory[["method", "training_seed", "status", "budget_steps", "demo_episodes"]])
    print("Found", len(inventory), "saved learned-model checkpoints. Random and Teacher will also be evaluated.")

    # 5. Load networks for inference only
    def read_checkpoint(ref):
        if ref["zip_member"]:
            with zipfile.ZipFile(ref["file"]) as archive:
                data = archive.read(ref["zip_member"])
            checkpoint = torch.load(io.BytesIO(data), map_location=DEVICE, weights_only=True)
            digest = hashlib.sha256(data).hexdigest()
        else:
            checkpoint = torch.load(ref["file"], map_location=DEVICE, weights_only=True)
            digest = sha256_file(ref["file"])
        return checkpoint, digest

    def load_saved_network(method, training_seed):
        ref = MODEL_REFS[(method, training_seed)]
        checkpoint, digest = read_checkpoint(ref)
        config = checkpoint.get("config", {})
        for key in ("grid_size", "end_score", "max_steps", "reward_mode"):
            if key in config and config[key] != ENV_CONFIG[key]:
                raise ValueError(f"{method} seed={training_seed}: {key} does not match the shared environment")
        if "seed" in config and int(config["seed"]) != training_seed:
            raise ValueError("The training seed in the checkpoint does not match its file path")

        if method in {"AC", "BC + AC"}:
            expected_bc = METHOD_INFO[method]["demo_episodes"]
            if int(config.get("teacher_episodes", -1)) != expected_bc:
                raise ValueError("The AC/BC + AC demonstration condition does not match the method label")
            network = ConvActorCritic(input_channels=7, output_dim=3, grid_size=8)
            network.load_state_dict(checkpoint["model_state_dict"], strict=True)
            selected_steps = config.get("progress_rl_steps")
            demo_steps = config.get("progress_demo_steps")
            saved_budget = config.get("rl_steps")
        else:
            expected_algorithm = "dqn" if method == "DQN" else "double_dqn"
            if checkpoint.get("algorithm") != expected_algorithm or config.get("dueling", False):
                raise ValueError("The DQN/DDQN checkpoint type does not match its label, or a Dueling model was supplied")
            network = QNetwork(input_channels=7, action_count=3, grid_size=8)
            network.load_state_dict(checkpoint["online_state_dict"], strict=True)
            selected_steps = checkpoint.get("total_steps")
            demo_steps = 0
            saved_budget = config.get("total_environment_steps")
        if saved_budget is None or int(saved_budget) != METHOD_INFO[method]["budget_steps"]:
            raise ValueError(f"The saved training budget for {method} does not match this evaluation's metadata")
        network.to(DEVICE)
        network.eval()
        network.requires_grad_(False)
        metadata = dict(method=method, training_seed=training_seed,
                        budget_steps=METHOD_INFO[method]["budget_steps"],
                        demo_episodes=METHOD_INFO[method]["demo_episodes"],
                        selected_steps=selected_steps, demo_steps=demo_steps,
                        checkpoint_sha256=digest, checkpoint_config=config,
                        checkpoint=str(ref["file"]), zip_member=ref["zip_member"])
        return network, metadata

    @torch.inference_mode()
    def learned_action(network, method, observation):
        state = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        output = network(state)
        action_scores = output[0] if method in {"AC", "BC + AC"} else output
        return int(action_scores.argmax(dim=-1).item())

    print("Inference-loading functions ready. No training has been performed.")

    # 6. Run independent episodes with the same test seeds
    def evaluate_one(method, training_seed=None, network=None):
        env = SnakeEnv(**ENV_CONFIG)
        records = []
        try:
            for test_seed in TEST_SEEDS:
                observation, _ = env.reset(seed=test_seed)
                random_actions = np.random.default_rng(test_seed + 8_000_000)
                total_reward = 0.0
                for step in range(1, ENV_CONFIG["max_steps"] + 1):
                    if method == "Random":
                        action = int(random_actions.integers(env.action_space.n))
                    elif method == "Teacher":
                        action = int(env.teacher_action())
                    else:
                        if network is None:
                            raise ValueError("A learned model must have its trained weights loaded before evaluation")
                        action = learned_action(network, method, observation)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        success = bool(info["length"] >= ENV_CONFIG["end_score"])
                        outcome = "success" if success else ("timeout" if truncated else "collision")
                        records.append({
                            "method": method, "training_seed": training_seed,
                            "test_seed": test_seed, "score": int(info["score"]),
                            "reward": total_reward, "steps": int(info["steps"]),
                            "success": int(success), "outcome": outcome,
                        })
                        break
                else:
                    raise RuntimeError("The environment did not finish within 300 steps. Check its version.")
        finally:
            env.close()
        return records

    episode_records = []
    checkpoint_metadata = []
    for method in METHOD_ORDER:
        seeds = [None] if method in {"Random", "Teacher"} else TRAINING_SEEDS
        for training_seed in seeds:
            if training_seed is None:
                network = None
            else:
                network, metadata = load_saved_network(method, training_seed)
                checkpoint_metadata.append(metadata)
            records = evaluate_one(method, training_seed, network)
            episode_records.extend(records)
            average_score = np.mean([r["score"] for r in records])
            print(f"{method:12s} seed={str(training_seed):>5s} | "
                  f"{len(records):3d} games | mean score={average_score:.3f}")
            del network

    episodes_df = pd.DataFrame(episode_records)
    episodes_df["training_seed"] = pd.array(episodes_df["training_seed"], dtype="Int64")
    assert len(episodes_df) == (2 + 4 * len(TRAINING_SEEDS)) * N_EPISODES
    assert not episodes_df.duplicated(["method", "training_seed", "test_seed"]).any()
    assert episodes_df["score"].between(0, 7).all()
    print("\nRecorded", len(episodes_df), "episodes. PPO has not been integrated yet.")
    display(episodes_df.head(12))

    # 7. Average episodes within each trained model, then aggregate across training seeds
    grouped = episodes_df.groupby(["method", "training_seed"], dropna=False, sort=False)
    per_model_df = grouped.agg(
        n_games=("score", "size"), score_mean=("score", "mean"),
        score_episode_sd=("score", "std"), reward_mean=("reward", "mean"),
        steps_mean=("steps", "mean"), success_rate=("success", "mean"),
    ).reset_index()

    summary_rows = []
    for method in METHOD_ORDER:
        model_rows = per_model_df[per_model_df["method"] == method]
        baseline = method in {"Random", "Teacher"}
        summary_rows.append({
            "method": method,
            "n_trained_models": 0 if baseline else len(model_rows),
            "games_per_model": N_EPISODES,
            "score_mean": model_rows["score_mean"].mean(),
            "score_training_seed_sd": np.nan if baseline else model_rows["score_mean"].std(ddof=1),
            "baseline_episode_sd": model_rows["score_episode_sd"].iloc[0] if baseline else np.nan,
            "success_rate": model_rows["success_rate"].mean(),
            "reward_mean": model_rows["reward_mean"].mean(),
            "steps_mean": model_rows["steps_mean"].mean(),
            "budget_steps": METHOD_INFO.get(method, {}).get("budget_steps", 0),
            "extra_teacher_episodes": METHOD_INFO.get(method, {}).get("demo_episodes", 0),
        })
    summary_df = pd.DataFrame(summary_rows)
    display(summary_df.round(3))

    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(summary_df))
    ax.bar(positions, summary_df["score_mean"], color=["#8e9aaf", "#2a9d8f", "#e9c46a", "#f4a261", "#457b9d", "#9b5de5"])
    has_sd = summary_df["score_training_seed_sd"].notna().to_numpy()
    if has_sd.any():
        ax.errorbar(positions[has_sd], summary_df.loc[has_sd, "score_mean"],
                    yerr=summary_df.loc[has_sd, "score_training_seed_sd"],
                    fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(positions, summary_df["method"])
    ax.set_ylabel("Mean food score (maximum 7)")
    ax.set_ylim(0, 8)
    ax.set_title("Pipeline smoke test: NOT final evidence" if MODE == "smoke"
                 else "Existing checkpoints under common evaluation")
    fig.text(0.02, 0.01, "Error bars: sample SD across training seeds, when available; baselines have no training-seed error bar.", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # 8. Reuse the same evaluation data for both paired comparisons
    learned_rows = per_model_df[per_model_df["training_seed"].notna()]
    paired_scores = learned_rows.pivot(index="training_seed", columns="method", values="score_mean")
    paired_scores["BC + AC minus AC"] = paired_scores["BC + AC"] - paired_scores["AC"]
    paired_scores["Double DQN minus DQN"] = paired_scores["Double DQN"] - paired_scores["DQN"]
    display(paired_scores.round(3))
    if MODE == "smoke":
        print("Pipeline check only: differences from 10 episodes and one training seed do not support final conclusions.")

    # 9. Save data, figure, protocol and this run's report
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    OUTPUT_ROOT = Path(output_root).expanduser().resolve() if output_root else Path(__file__).resolve().parent
    OUTPUT_DIR = OUTPUT_ROOT / "results" / MODE / run_stamp
    FIGURE_DIR = OUTPUT_ROOT / "figures" / MODE / run_stamp
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    FIGURE_DIR.mkdir(parents=True, exist_ok=False)
    episodes_df.to_csv(OUTPUT_DIR / "episodes.csv", index=False)
    per_model_df.to_csv(OUTPUT_DIR / "per_model.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    paired_scores.to_csv(OUTPUT_DIR / "paired_scores.csv")
    fig.savefig(FIGURE_DIR / "score_comparison.png", dpi=160, bbox_inches="tight")

    protocol = {
        "mode": MODE, "train_model": False, "environment": ENV_CONFIG,
        "training_seeds": TRAINING_SEEDS, "test_seeds": TEST_SEEDS,
        "evaluation_policy": "argmax for learned models; BFS for teacher; seeded random for random",
        "models_selected_by": "original validation protocol; never by this test",
        "interpretation": "existing trained checkpoints; training budgets differ across algorithm families",
        "source_sha256": {str(p.relative_to(SNAKE_DIR)): sha256_file(p)
                           for p in [SNAKE_DIR / "environment.py", ac_source, SNAKE_DIR / "dqn.py"]},
        "checkpoints": checkpoint_metadata,
        "versions": {"python": sys.version, "torch": str(torch.__version__),
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    protocol["evaluation_script_sha256"] = sha256_file(Path(__file__))
    protocol["figure_path"] = str(FIGURE_DIR.relative_to(OUTPUT_ROOT))
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Results saved to:", OUTPUT_DIR)
    print("Complete: existing weights loaded and evaluated; no retraining performed.")

    report_lines = [
        "# Member 5: Unified Evaluation Run", "",
        "**Status: pipeline smoke test; not evidence for a final ranking.**" if MODE == "smoke"
        else "**Status: common evaluation of existing checkpoints with different training budgets.**",
        "", f"Evaluation episodes per model: {N_EPISODES}. Training seeds: {TRAINING_SEEDS}.",
        f"Test seed range: {TEST_SEEDS[0]} to {TEST_SEEDS[-1]}.",
        "", "| Method | Trained models | Mean score | Success rate | Full autonomous training budget | Extra teacher episodes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['method']} | {row['n_trained_models']} | {row['score_mean']:.3f} | "
            f"{row['success_rate']:.1%} | {row['budget_steps']} | {row['extra_teacher_episodes']} |"
        )
    image_relative = Path("../../..") / protocol["figure_path"] / "score_comparison.png"
    report_lines.extend([
        "", f"![Score comparison]({image_relative.as_posix()})", "",
        "For learned methods, average test episodes within each training seed before aggregating across training seeds. A single training seed cannot estimate cross-seed SD.",
        "Random and Teacher each receive one evaluation batch and have no independent training replicates; no training-seed error bars are assigned to them.", "",
        "BC + AC uses additional teacher demonstrations relative to AC. DQN/DDQN and AC have different training budgets, so the overall table compares existing trained outcomes.",
        "The current environment does not return separate wall/self-collision causes; this version records collision without guessing the subtype.", "",
        "No training was performed. Learning curves, training-stage gameplay, PPO, fixed-state decision analysis and final conclusions remain to be integrated.", "",
        "Contributions: the original group authors supplied the environment, algorithm implementations and saved weights. OpenAI Codex assisted with Member 5's evaluation code, documentation and execution checks.",
    ])
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("Figure saved to:", FIGURE_DIR)
    return {"mode": MODE, "result_dir": OUTPUT_DIR, "figure_dir": FIGURE_DIR,
            "episodes": episodes_df, "per_model": per_model_df,
            "summary": summary_df, "paired_scores": paired_scores,
            "protocol": protocol}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "evaluation"), default="smoke")
    parser.add_argument("--project-dir", type=Path, help="Snake directory or repository root")
    parser.add_argument("--output-root", type=Path, help="Root for results/ and figures/")
    args = parser.parse_args(argv)
    run_benchmark(project_dir=args.project_dir, mode=args.mode, output_root=args.output_root)


if __name__ == "__main__":
    main()
