"""Member 2: explicit BC warm-up, then unassisted on-policy Actor-Critic.

`episodes` retains the original TOTAL-episode API when rl_steps is None.
For the paired experiment, rl_steps fixes the AC interaction budget independently
of the optional teacher episodes. CSV files are sufficient to replot every curve.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from environment import SnakeEnv
from model import ConvActorCritic


@dataclass(slots=True)
class TrainConfig:
    episodes: int = 500
    teacher_episodes: int = 100
    imitation_updates: int = 4
    grid_size: int = 8
    end_score: int = 8
    max_steps: int = 300
    rl_steps: int | None = None
    eval_interval: int = 100           # episodes in original API
    eval_every_steps: int = 5000       # step-budget mode
    eval_episodes: int = 30
    validation_seed_base: int = 500000
    save_interval: int = 250
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gamma: float = 0.99
    entropy_coef: float = 0.01
    seed: int = 42
    device: str = 'cpu'
    output_dir: Path = Path('runs/snake')
    resume: Path | None = None
    tensorboard: bool = False
    reward_mode: str = 'sparse'
    teacher_mode: str = 'shortest_path'
    torch_threads: int = 1

    def validate(self):
        if self.episodes < 1 or self.teacher_episodes < 0:
            raise ValueError('episodes must be positive and teacher_episodes nonnegative')
        if self.rl_steps is None and self.teacher_episodes >= self.episodes:
            raise ValueError('leave at least one AC episode after the teacher phase')
        if self.rl_steps is not None and self.rl_steps < 1:
            raise ValueError('rl_steps must be positive')
        if self.imitation_updates < 1 or self.eval_episodes < 1 or self.torch_threads < 1:
            raise ValueError('updates, evaluation episodes and threads must be positive')
        if min(self.eval_interval, self.save_interval) < 0 or self.eval_every_steps < 1:
            raise ValueError('invalid interval')
        if not 0 <= self.gamma <= 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError('invalid optimizer/discount configuration')
        if self.resume is not None:
            raise ValueError('Exact mid-run resume is unsupported; rerun incomplete runs in a fresh directory.')

    def checkpoint_payload(self):
        payload = asdict(self)
        payload['output_dir'] = str(self.output_dir)
        payload['resume'] = None
        return payload


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested='auto'):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    return device


def as_tensor(observation, device):
    return torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


def make_env(config, **kwargs):
    return SnakeEnv(grid_size=config.grid_size, end_score=config.end_score,
                    max_steps=config.max_steps, reward_mode=config.reward_mode,
                    teacher_mode=config.teacher_mode, **kwargs)


@torch.no_grad()
def evaluate_model(agent, config, device=None, *, seed_offset=None, seed_base=None,
                   episodes=None, return_episodes=False):
    """Independent greedy rollouts: no teacher, no action masking, no updates."""
    device = device or agent.device
    seed_base = (config.seed + seed_offset if seed_offset is not None else
                 config.validation_seed_base) if seed_base is None else seed_base
    count = episodes or config.eval_episodes
    env = make_env(config)
    was_training = agent.training
    agent.eval()
    rows = []
    try:
        for ep in range(count):
            obs, _ = env.reset(seed=seed_base + ep)
            total_reward = 0.0
            while True:
                action = agent.predict(as_tensor(obs, device), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    won = info['length'] >= config.end_score
                    rows.append(dict(episode_seed=seed_base+ep, reward=total_reward,
                                     score=int(info['score']), steps=int(info['steps']),
                                     success=int(won), death=int(terminated and not won),
                                     truncated=int(truncated)))
                    break
    finally:
        env.close()
        agent.train(was_training)
    result = {key: float(np.mean([r[key] for r in rows]))
              for key in ['reward', 'score', 'steps', 'success', 'death', 'truncated']}
    result['success_rate'] = result.pop('success')
    result['death_rate'] = result.pop('death')
    result['truncation_rate'] = result.pop('truncated')
    if return_episodes:
        return result, rows
    return result


def load_agent(checkpoint_path, device='cpu'):
    device = choose_device(device) if isinstance(device, str) else device
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = payload['config']
    agent = ConvActorCritic(7, 3, int(cfg['grid_size']),
                            lr=cfg.get('learning_rate', 1e-4),
                            weight_decay=cfg.get('weight_decay', 1e-5),
                            entropy_coef=cfg.get('entropy_coef', .01)).to(device)
    agent.load_checkpoint(checkpoint_path)
    agent.eval()
    fields = TrainConfig.__dataclass_fields__
    config = TrainConfig(**{k: v for k, v in cfg.items() if k in fields})
    return agent, config


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(config):
    config.validate()
    config.output_dir = Path(config.output_dir)
    if (config.output_dir/'summary.json').exists():
        raise FileExistsError(f'Completed run exists: {config.output_dir}; choose a fresh output directory')
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)
    device = choose_device(config.device)
    out = config.output_dir
    checkpoints = out/'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(config.checkpoint_payload(), indent=2))
    env = make_env(config)
    agent = ConvActorCritic(7, 3, config.grid_size, config.learning_rate,
                            config.weight_decay, config.entropy_coef).to(device)
    writer = None
    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out/'tensorboard')
    start = time.perf_counter()
    history, validations = [], []
    demo_steps = rl_steps = bc_updates = 0
    best_key = (-float('inf'),)*3
    selected_step = 0
    payload = config.checkpoint_payload()
    def save(name, episode):
        agent.save_checkpoint(checkpoints/name, episode=episode,
                              config={**payload, 'progress_rl_steps': rl_steps,
                                      'progress_demo_steps': demo_steps})
    def validate(stage, episode, selectable=True):
        nonlocal best_key, selected_step
        stats = evaluate_model(agent, config, device)
        row = dict(stage=stage, episode=episode, rl_steps=rl_steps,
                   demo_steps=demo_steps, total_steps=demo_steps+rl_steps, **stats)
        validations.append(row)
        write_csv(out/'validation.csv', validations)
        key = (stats['score'], stats['success_rate'], stats['reward'])
        if selectable and key > best_key:
            best_key, selected_step = key, rl_steps
            save('best.pt', episode)
        print(f"seed={config.seed} BC={config.teacher_episodes} {stage} AC_steps={rl_steps} "
              f"val_score={stats['score']:.3f} success={stats['success_rate']:.1%}", flush=True)
        return stats
    def log_episode(phase, ep, info, reward, metrics, truncated, terminated, seconds):
        row = dict(phase=phase, episode=ep, rl_steps=rl_steps, demo_steps=demo_steps,
                   total_steps=rl_steps+demo_steps, reward=reward, score=int(info['score']),
                   steps=int(info['steps']), success=int(info['length'] >= config.end_score),
                   terminated=int(terminated), truncated=int(truncated),
                   loss=metrics.get('loss', ''), actor_loss=metrics.get('actor_loss', ''),
                   critic_loss=metrics.get('critic_loss', ''), entropy=metrics.get('entropy', ''),
                   bc_loss=metrics.get('bc_loss', ''), seconds=seconds,
                   budget_partial=int(phase == 'actor_critic' and not (terminated or truncated)))
        history.append(row)
        if writer:
            for key in ['score', 'reward']:
                writer.add_scalar(f'{phase}/{key}', row[key], row['total_steps'])
    try:
        save('initial.pt', -1)
        validate('initial', -1, False)
        # Store explicit (state, teacher action) pairs; only CE is used in warm-up.
        demonstrations = []
        demo_rng = random.Random(config.seed)
        for episode in range(config.teacher_episodes):
            obs, _ = env.reset(seed=100000 + config.seed*1000 + episode)
            total, losses = 0.0, []
            ep_start = time.perf_counter()
            while True:
                action = env.teacher_action()
                demonstrations.append((obs.copy(), action))
                if len(demonstrations) > 10000:
                    demonstrations.pop(0)
                nxt, reward, terminated, truncated, info = env.step(action)
                for _ in range(config.imitation_updates):
                    batch = demo_rng.sample(demonstrations, min(64, len(demonstrations)))
                    states = torch.as_tensor(np.stack([x[0] for x in batch]), device=device)
                    actions = torch.tensor([x[1] for x in batch], device=device)
                    losses.append(agent.imitation_update(states, actions))
                    bc_updates += 1
                demo_steps += 1
                total += reward
                obs = nxt
                if terminated or truncated:
                    break
            log_episode('teacher', episode, info, total,
                        {'bc_loss': float(np.mean(losses))}, truncated, terminated,
                        time.perf_counter()-ep_start)
        bc_seconds = time.perf_counter()-start
        save('post_bc.pt', config.teacher_episodes-1)
        if config.teacher_episodes:
            validate('post_bc', config.teacher_episodes-1, False)
        # Give the two arms the same RL environment seed sequence and RNG start.
        seed_everything(config.seed + 10000)
        episode = 0
        next_eval = config.eval_every_steps
        halfway_saved = False
        while config.rl_steps is not None or episode < config.episodes-config.teacher_episodes:
            obs, _ = env.reset(seed=config.seed*1000 + episode)
            total, metrics_list = 0.0, []
            ep_start = time.perf_counter()
            while True:
                state = as_tensor(obs, device)
                action = agent.predict(state)        # sample current policy, NO teacher
                nxt, reward, terminated, truncated, info = env.step(action)
                # Gymnasium time limits bootstrap V(next); true wins/deaths do not.
                metrics_list.append(agent.update(state, action, reward,
                                                 as_tensor(nxt, device), terminated, config.gamma))
                rl_steps += 1
                total += reward
                obs = nxt
                if config.rl_steps and not halfway_saved and rl_steps >= config.rl_steps//2:
                    save('midpoint.pt', config.teacher_episodes+episode)
                    halfway_saved = True
                if config.rl_steps and rl_steps >= next_eval:
                    validate('actor_critic', config.teacher_episodes+episode)
                    next_eval += config.eval_every_steps
                budget_end = config.rl_steps is not None and rl_steps >= config.rl_steps
                if terminated or truncated or budget_end:
                    break
            metrics = {k: float(np.mean([r[k] for r in metrics_list])) for k in metrics_list[0]}
            log_episode('actor_critic', config.teacher_episodes+episode, info, total, metrics,
                        truncated, terminated, time.perf_counter()-ep_start)
            episode += 1
            if episode % 25 == 0 or budget_end:
                write_csv(out/'history.csv', history)
            if config.rl_steps is None and config.eval_interval and episode % config.eval_interval == 0:
                validate('actor_critic', config.teacher_episodes+episode-1)
            if config.rl_steps is None and config.save_interval and episode % config.save_interval == 0:
                save(f'episode-{config.teacher_episodes+episode}.pt', config.teacher_episodes+episode-1)
            if budget_end:
                break
        if not validations or validations[-1]['rl_steps'] != rl_steps or selected_step == 0:
            validate('actor_critic', config.teacher_episodes+episode-1)
        save('latest.pt', config.teacher_episodes+episode-1)
        write_csv(out/'history.csv', history)
        write_csv(out/'validation.csv', validations)
        summary = dict(config=config.checkpoint_payload(), rl_steps=rl_steps, demo_steps=demo_steps,
                       total_steps=rl_steps+demo_steps, bc_updates=bc_updates,
                       ac_updates=rl_steps, ac_episodes=episode,
                       best_validation_score=best_key[0], selected_rl_steps=selected_step,
                       bc_seconds=bc_seconds, training_seconds=time.perf_counter()-start,
                       device=str(device), checkpoint=str(checkpoints/'best.pt'))
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        env.close()
        if writer:
            writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--teacher-episodes', type=int, default=100)
    p.add_argument('--rl-steps', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-dir', type=Path, default=Path('runs/snake'))
    p.add_argument('--grid-size', type=int, default=8)
    p.add_argument('--end-score', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=300)
    p.add_argument('--tensorboard', action='store_true')
    p.add_argument('--no-tensorboard', action='store_false', dest='tensorboard')
    args = p.parse_args()
    print(json.dumps(train(TrainConfig(**vars(args))), indent=2))

if __name__ == '__main__':
    main()
