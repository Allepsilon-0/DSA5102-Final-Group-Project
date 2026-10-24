"""Controlled PPO clipping experiment. English outputs; no fabricated results.
Run: python member4_experiment.py --mode smoke
     python member4_experiment.py --mode final
"""
import argparse
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from environment import SnakeEnv
from ppo import PPOAgent, RolloutBatch
from advantages import compute_gae

ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def environment():
    return SnakeEnv(grid_size=8, end_score=8, max_steps=300, reward_mode='sparse')

def evaluate(agent, base, episodes):
    """Greedy policy, same environment seeds for every checkpoint; no RNG sampling."""
    env = environment()
    rows = []
    try:
        for seed in range(base, base + episodes):
            obs, info = env.reset(seed=seed)
            total = 0.
            while True:
                obs, reward, terminal, timeout, info = env.step(agent.predict(obs, deterministic=True))
                total += reward
                if terminal or timeout:
                    rows.append(dict(episode_seed=seed, score=info['score'], steps=info['steps'],
                                     reward=total, success=int(info['length'] >= 8), timeout=int(timeout)))
                    break
    finally:
        env.close()
    return rows

def summary(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in ['score','steps','reward','success','timeout']}

def make_agent():
    return PPOAgent(7, 3, 8, learning_rate=3e-4, device='cpu')

def train(seed, clip, protocol, folder):
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agent, env = make_agent(), environment()
    # Disjoint seed ranges for independent training runs and for held-out evaluation.
    episode = 0
    obs, _ = env.reset(seed=10000000 + seed*100000)
    history, updates = [], []
    total_steps = 0
    best_key = (-float('inf'),)*3
    def save(name):
        agent.save_checkpoint(folder/'checkpoints'/name, episode=episode,
            total_steps=total_steps, config={**protocol,'seed':seed,'clip_ratio':clip})
    def validate():
        nonlocal best_key
        row = {'steps':total_steps, **summary(evaluate(agent, protocol['validation_base'], protocol['validation_episodes']))}
        history.append(row)
        key = (row['score'], row['success'], row['reward'])
        if key > best_key:
            best_key = key
            save('best.pt')
        write_json(folder/'validation.json', history)
        print(f'seed={seed} clip={clip} steps={total_steps} validation_score={row["score"]:.3f}', flush=True)
    save('step-0.pt')
    validate()
    start = time.perf_counter()
    next_validation = protocol['eval_interval']
    milestones = {round(protocol['steps']*f) for f in [.25,.5,.75,1.]}
    try:
        while total_steps < protocol['steps']:
            next_boundary = min([x for x in milestones | {next_validation,protocol['steps']} if x > total_steps])
            size = min(protocol['rollout_steps'], next_boundary-total_steps)
            states,actions,rewards,ends,logs,values = [],[],[],[],[],[]
            for _ in range(size):
                action, log_prob, value = agent.select_action(obs)
                nxt, reward, terminal, timeout, _ = env.step(action)
                states.append(obs.copy()); actions.append(action); rewards.append(.1*reward)
                ends.append(terminal or timeout); logs.append(log_prob); values.append(value)
                total_steps += 1
                obs = nxt
                if terminal or timeout:
                    episode += 1
                    obs, _ = env.reset(seed=10000000 + seed*100000 + episode)
            # Rollout boundaries are NOT episode boundaries. Bootstrap unless the game ended.
            with torch.no_grad():
                next_value = float(agent.model(torch.as_tensor(obs).unsqueeze(0))[1].item())
            advantage, returns = compute_gae(rewards, values, ends, next_value,
                                             gamma=protocol['gamma'],lam=protocol['gae_lambda'])
            a = torch.as_tensor(advantage)
            a = (a-a.mean())/(a.std(unbiased=False)+1e-8)
            batch = RolloutBatch(torch.as_tensor(np.stack(states)), torch.tensor(actions),
                                 torch.tensor(logs), torch.as_tensor(returns), a)
            metrics = agent.update(batch, update_epochs=4, batch_size=64, clip_ratio=clip,
                                   value_coef=.5, entropy_coef=.01,max_grad_norm=.5)
            updates.append({'steps':total_steps,**metrics})
            if total_steps in milestones:
                save(f'step-{total_steps}.pt')
            if total_steps == next_validation or total_steps == protocol['steps']:
                validate()
                next_validation += protocol['eval_interval']
        save('latest.pt')
    finally:
        env.close()
    write_json(folder/'updates.json', updates)
    write_json(folder/'trained.json', {'seed':seed,'clip_ratio':clip,'steps':total_steps,
                                       'seconds_including_validation':time.perf_counter()-start})

def run(mode, output=None):
    torch.set_num_threads(1)
    if mode == 'analysis':
        from make_figures import generate
        generate(Path(output) if output else ROOT/'results'/'final')
        return
    p = dict(mode=mode, seeds=[7] if mode=='smoke' else [7,42,2026,31415,27182],
             clips=[.1,.2,.3], steps=1024 if mode=='smoke' else 50000,
             eval_interval=512 if mode=='smoke' else 5000,
             validation_episodes=3 if mode=='smoke' else 30,
             test_episodes=5 if mode=='smoke' else 200,
             validation_base=600000, test_base=950000, rollout_steps=256,
             gamma=.99, gae_lambda=.95, grid_size=8,end_score=8,max_steps=300,
             reward_mode='sparse',reward_scale=.1,learning_rate=.0003,update_epochs=4,
             batch_size=64,value_coef=.5,entropy_coef=.01,max_grad_norm=.5,
             evaluation_policy='greedy',timeout_semantics='finite-horizon terminal',
             source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                            for n in ['environment.py','ppo.py','advantages.py','member4_experiment.py']})
    output = Path(output) if output else ROOT/'results'/mode
    output.mkdir(parents=True,exist_ok=True)
    protocol_file = output/'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != p:
        raise RuntimeError('Existing protocol differs. Use a new output directory.')
    write_json(protocol_file,p)
    write_json(output/'versions.json',{'python':platform.python_version(),'torch':torch.__version__,
                                     'numpy':np.__version__,'platform':platform.platform()})
    # Complete all training/validation before opening the held-out test set.
    for seed in p['seeds']:
        for clip in p['clips']:
            folder = output/f'seed-{seed}'/f'clip-{clip:.1f}'
            if not ((folder/'trained.json').exists() and (folder/'checkpoints/best.pt').exists()):
                train(seed,clip,p,folder)
    result=[]
    for seed in p['seeds']:
        for clip in p['clips']:
            folder=output/f'seed-{seed}'/f'clip-{clip:.1f}'
            agent=make_agent(); metadata=agent.load_checkpoint(folder/'checkpoints/best.pt')
            if (folder/'test_episodes.json').exists():
                rows=json.loads((folder/'test_episodes.json').read_text())
            else:
                rows=evaluate(agent,p['test_base'],p['test_episodes'])
                write_json(folder/'test_episodes.json',rows)
            result.append({'seed':seed,'clip':clip,'selected_steps':metadata['total_steps'],**summary(rows)})
    write_json(output/'results.json',result)
    with (output/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    from make_figures import generate
    generate(output)
    from visualize_progress import render
    checkpoints = output/'seed-7'/'clip-0.2'/'checkpoints'
    stages = [0] + sorted({round(p['steps']*f) for f in [.25,.5,.75,1.]})
    render([checkpoints/f'step-{step}.pt' for step in stages],
           [f'{step} steps' for step in stages], output/'figures'/'ppo_progress.gif')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['smoke','final','analysis'],default='smoke')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();run(args.mode,args.output)
