"""Paired AC vs BC+AC ablation. No test-based model selection or tuning."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from main import TrainConfig, train, evaluate_model, load_agent, make_env, write_csv
from play import render_progress
from member2_assets import ensure_assets

PROTOCOL = dict(seeds=[7,42,2026,31415,27182], rl_steps=50000,
                teacher_episodes=[0,100], imitation_updates=4,
                grid_size=8, end_score=8, max_steps=300, reward_mode='sparse',
                validation_seed_base=500000, eval_episodes=30, eval_every_steps=5000,
                test_seed_base=900000, test_episodes=200, progress_seed=900000,
                learning_rate=1e-4, weight_decay=1e-5, gamma=.99, entropy_coef=.01,
                device='cpu', torch_threads=1)


def fingerprint():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['environment.py','model.py','main.py']}


def baseline_rows(config, count, seed_base):
    rows = []
    for policy in ['random','legacy','shortest_path']:
        env = make_env(config)
        rng = np.random.default_rng(12345)
        env.teacher_mode = policy if policy != 'random' else 'shortest_path'
        try:
            for ep in range(count):
                obs, _ = env.reset(seed=seed_base+ep)
                total = 0.0
                while True:
                    action = int(rng.integers(3)) if policy == 'random' else env.teacher_action()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    if terminated or truncated:
                        won = info['length'] >= config.end_score
                        rows.append(dict(policy=policy, episode_seed=seed_base+ep, score=info['score'],
                                         reward=total, success=int(won), steps=info['steps'],
                                         death=int(terminated and not won), truncated=int(truncated)))
                        break
        finally:
            env.close()
    return rows


def run_experiment(output=Path('runs/member2/final'), mode='final'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL)
    if mode == 'smoke':
        protocol.update(seeds=[7], rl_steps=200, teacher_episodes=[0,2],
                        eval_episodes=3, test_episodes=5, eval_every_steps=100)
    protocol['source_sha256'] = fingerprint()
    protocol['mode'] = mode
    protocol_path = output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Saved protocol/source differs; choose a fresh output directory.')
    protocol_path.write_text(json.dumps(protocol, indent=2))
    versions = {p:importlib.metadata.version(p) for p in
                ['torch','numpy','pandas','matplotlib','gymnasium','pygame','pillow']}
    (output/'versions.json').write_text(json.dumps(dict(python=platform.python_version(),
                                                       platform=platform.platform(), packages=versions), indent=2))
    for seed in protocol['seeds']:
        for teacher in protocol['teacher_episodes']:
            arm = 'ac' if teacher == 0 else 'bc_ac'
            run = output/f'seed-{seed}'/arm
            complete = run/'complete.json'
            if complete.exists() and (run/'checkpoints/best.pt').exists():
                print('reuse', run, flush=True)
                continue
            if run.exists():
                shutil.rmtree(run)  # only our incomplete, protocol-checked experiment directory
            fields = TrainConfig.__dataclass_fields__
            kwargs = {k:v for k,v in protocol.items() if k in fields and k not in ['seed','teacher_episodes']}
            cfg = TrainConfig(**kwargs, seed=seed, teacher_episodes=teacher,
                              output_dir=run, tensorboard=False)
            summary = train(cfg)
            # Test only after validation has permanently selected the checkpoint.
            test_rows = []
            stage_results = {}
            for stage in ['best','post_bc','latest']:
                agent, saved_config = load_agent(run/f'checkpoints/{stage}.pt')
                stats, records = evaluate_model(agent, saved_config, seed_base=protocol['test_seed_base'],
                                                episodes=protocol['test_episodes'], return_episodes=True)
                stage_results[stage] = stats
                test_rows.extend([dict(stage=stage, **r) for r in records])
            write_csv(run/'test_episodes.csv', test_rows)
            (run/'test_results.json').write_text(json.dumps(stage_results, indent=2))
            complete.write_text(json.dumps(dict(seed=seed, arm=arm, summary=summary,
                                                test=stage_results), indent=2))
    cfg = TrainConfig()
    write_csv(output/'baseline_episodes.csv', baseline_rows(cfg, protocol['test_episodes'], protocol['test_seed_base']))
    return analyze(output)


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261019)
    means = rng.choice(values, size=(20000,len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025,.975])]


def analyze(output=Path('runs/member2/final')):
    ensure_assets(Path(__file__).resolve().parent)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    protocol = json.loads((output/'protocol.json').read_text())
    rows, val, costs, stages = [], [], [], []
    for seed in protocol['seeds']:
        for arm in ['ac','bc_ac']:
            run = output/f'seed-{seed}'/arm
            data = json.loads((run/'complete.json').read_text())
            rows.append(dict(seed=seed, arm=arm, **data['test']['best']))
            costs.append(dict(seed=seed, arm=arm, **{k:data['summary'][k] for k in
                              ['rl_steps','demo_steps','total_steps','bc_updates','training_seconds','selected_rl_steps']}))
            v = pd.read_csv(run/'validation.csv'); v['seed']=seed;v['arm']=arm; val.append(v)
            for stage, result in data['test'].items():
                stages.append(dict(seed=seed,arm=arm,stage=stage,**result))
    df = pd.DataFrame(rows); df.to_csv(output/'per_seed_results.csv', index=False)
    cost_df = pd.DataFrame(costs);cost_df.to_csv(output/'training_cost.csv', index=False)
    stage_df = pd.DataFrame(stages);stage_df.to_csv(output/'stage_results.csv', index=False)
    validation = pd.concat(val);validation.to_csv(output/'all_validation.csv', index=False)
    metrics = ['score','reward','success_rate','death_rate','truncation_rate','steps']
    aggregate = df.groupby('arm')[metrics].agg(['mean','std'])
    aggregate.to_csv(output/'aggregate_results.csv')
    paired = {}
    for metric in ['score','success_rate','reward']:
        pairs = df.pivot(index='seed', columns='arm', values=metric)
        differences = (pairs.bc_ac-pairs.ac).to_numpy()
        paired[metric] = dict(mean_difference=float(differences.mean()),
                              bootstrap_95=paired_interval(differences), differences=differences.tolist())
    (output/'paired_results.json').write_text(json.dumps(paired, indent=2))
    figures = output/'figures';figures.mkdir(exist_ok=True)
    colors = {'ac':'#31688e','bc_ac':'#e67742'}
    labels = {'ac':'AC (no BC)','bc_ac':'BC + AC'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,2,figsize=(11,3.8))
    for arm in ['ac','bc_ac']:
        group = validation[(validation.arm==arm)&(validation.stage=='actor_critic')]
        for ax,metric in zip(axes,['score','success_rate']):
            g = group.groupby('rl_steps')[metric].agg(['mean','std']).fillna(0)
            ax.plot(g.index,g['mean'],label=labels[arm],color=colors[arm])
            ax.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],alpha=.15,color=colors[arm])
            ax.set(xlabel='Unassisted AC environment steps',ylabel=f'Validation {metric}')
    axes[0].set_ylim(0,7.2); axes[1].set_ylim(0,1);axes[0].legend()
    fig.suptitle('Learning curves: fixed validation episodes; shading = training-seed SD')
    fig.tight_layout();fig.savefig(figures/'learning_curves.png', dpi=160);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,3.8))
    baseline = pd.read_csv(output/'baseline_episodes.csv').groupby('policy').mean(numeric_only=True)
    names = ['Random','Legacy teacher','BFS teacher','AC','BC + AC']
    for ax, metric, bmetric in [(axes[0],'score','score'),(axes[1],'success_rate','success')]:
        vals = [baseline.loc[n,bmetric] for n in ['random','legacy','shortest_path']]
        vals += [df[df.arm==a][metric].mean() for a in ['ac','bc_ac']]
        errors = [np.nan,np.nan,np.nan]+[df[df.arm==a][metric].std(ddof=1) if len(protocol['seeds'])>1 else 0 for a in ['ac','bc_ac']]
        ax.bar(names,vals,yerr=errors,color=['#bec8ce']*3+[colors['ac'],colors['bc_ac']],capsize=4)
        ax.set_ylabel('Test '+metric);ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Selected checkpoints: 200 held-out episodes/model (bars: seed means; errors: seed SD)' if protocol['mode']=='final' else 'Smoke checks only')
    fig.tight_layout();fig.savefig(figures/'final_performance.png',dpi=160);plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,3.6))
    for seed in protocol['seeds']:
        pair=df[df.seed==seed].set_index('arm')
        for ax,metric in zip(axes[:2],['score','success_rate']):
            ax.plot([0,1],[pair.loc['ac',metric],pair.loc['bc_ac',metric]],'o-',alpha=.7,label=str(seed))
            ax.set_xticks([0,1],['AC','BC + AC']);ax.set_ylabel('Test '+metric)
    for arm in ['ac','bc_ac']:
        vals=cost_df[cost_df.arm==arm]['training_seconds']
        axes[2].bar(labels[arm],vals.mean(),color=colors[arm])
        axes[2].scatter([labels[arm]]*len(vals),vals,color='black',s=12)
    axes[2].set_ylabel('Training wall time (seconds)'); axes[0].legend(title='Training seed',fontsize=8)
    fig.tight_layout();fig.savefig(figures/'paired_outcomes_cost.png',dpi=160);plt.close(fig)
    # Reward/score vs episode distinguishes teacher actions from autonomous learning.
    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    seed=protocol['seeds'][0]
    for arm in ['ac','bc_ac']:
        h=pd.read_csv(output/f'seed-{seed}'/arm/'history.csv')
        rl=h[h.phase=='actor_critic'].reset_index(drop=True)
        for ax,metric in zip(axes,['score','reward']):
            ax.plot(np.arange(len(rl))+1,rl[metric].rolling(25,min_periods=1).mean(),label=labels[arm],color=colors[arm])
            ax.set(xlabel='Autonomous training episode',ylabel=metric+' (25-episode rolling mean)')
    axes[0].legend();fig.suptitle(f'Stochastic training trajectories: seed {seed}; teacher episodes excluded')
    fig.tight_layout();fig.savefig(figures/'training_reward_score.png',dpi=160);plt.close(fig)
    checkpoints=output/f'seed-{seed}'/'bc_ac'/'checkpoints'
    gameplay=render_progress({'Initial':checkpoints/'initial.pt','After BC':checkpoints/'post_bc.pt',
                              'AC midpoint':checkpoints/'midpoint.pt','Best validation':checkpoints/'best.pt'},
                             figures/'training_progress.gif',protocol['progress_seed'])
    result=dict(protocol=protocol, aggregate={a:{m:dict(mean=float(df[df.arm==a][m].mean()),
                        std=float(df[df.arm==a][m].std(ddof=1)) if len(protocol['seeds'])>1 else 0)
                        for m in metrics} for a in ['ac','bc_ac']},paired=paired,gameplay=gameplay)
    (output/'results.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','final','analysis'],default='analysis')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    output=args.output or Path('runs/member2')/('smoke' if args.mode=='smoke' else 'final')
    result=analyze(output) if args.mode=='analysis' else run_experiment(output,args.mode)
    print(json.dumps(result['aggregate'],indent=2))

if __name__=='__main__':
    main()
