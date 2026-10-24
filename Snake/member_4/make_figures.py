"""Regenerate figures and a results insert from completed experiments only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate(folder):
    folder=Path(folder)
    rows=json.loads((folder/'results.json').read_text())
    p=json.loads((folder/'protocol.json').read_text())
    target=folder/'figures';target.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for clip in p['clips']:
        histories=[json.loads((folder/f'seed-{s}'/f'clip-{clip:.1f}'/'validation.json').read_text()) for s in p['seeds']]
        xs=np.array([r['steps'] for r in histories[0]])
        for ax,metric in zip(axs,['score','success']):
            ys=np.array([[r[metric] for r in h] for h in histories])
            mean=ys.mean(axis=0);std=ys.std(axis=0,ddof=1) if len(ys)>1 else np.zeros(len(xs))
            ax.plot(xs,mean,label=f'clip={clip}')
            ax.fill_between(xs,mean-std,mean+std,alpha=.15)
            ax.set(xlabel='Training environment steps',ylabel='Validation '+metric)
            ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Mean and ±1 sample SD across training seeds')
    fig.tight_layout();fig.savefig(target/'learning_curves.png',dpi=170);plt.close(fig)
    lines=['# Observed results', '', f'Execution mode: **{p["mode"]}**. Smoke results are pipeline checks, not final findings.', '',
           '| Clip | Test score (mean ± seed SD) | Success rate (mean ± seed SD) |','|---|---:|---:|']
    for clip in p['clips']:
        subset=[r for r in rows if r['clip']==clip]
        a=np.array([r['score'] for r in subset]);b=np.array([r['success'] for r in subset])
        sd=lambda x: float(np.std(x,ddof=1)) if len(x)>1 else 0.
        lines.append(f'| {clip} | {a.mean():.3f} ± {sd(a):.3f} | {100*b.mean():.1f}% ± {100*sd(b):.1f} pp |')
    lines += ['', 'The unit of replication is the training seed. Episode-level variation is not a substitute for independent training runs.', '',
              'Paired differences relative to clip=0.2 (descriptive bootstrap intervals; only five seeds in final mode):','']
    rng=np.random.default_rng(761)
    for clip in [.1,.3]:
        differences=[]
        for seed in p['seeds']:
            by={r['clip']:r for r in rows if r['seed']==seed}
            differences.append(by[clip]['score']-by[.2]['score'])
        d=np.array(differences)
        if len(d)>1:
            boots=rng.choice(d,(10000,len(d)),replace=True).mean(axis=1)
            lo,hi=np.quantile(boots,[.025,.975])
            lines.append(f'- clip={clip} minus 0.2: {d.mean():.3f}, paired bootstrap 95% interval [{lo:.3f}, {hi:.3f}].')
        else:
            lines.append(f'- clip={clip}: one training seed; no inferential interval.')
    lines += ['', 'Do not select or retune a configuration on these test results. A follow-up experiment needs new held-out seeds.',
              'A positive mean difference alone does not establish a reliable improvement. Check individual seeds and learning curves.']
    (folder/'observed_results.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(6,4))
    for seed in p['seeds']:
        subset=sorted([r for r in rows if r['seed']==seed],key=lambda r:r['clip'])
        ax.plot([r['clip'] for r in subset],[r['score'] for r in subset],'-o',label=f'seed {seed}')
    ax.set(xlabel='PPO clipping parameter',ylabel='Mean test food score',ylim=(0,7.2))
    ax.legend();fig.tight_layout();fig.savefig(target/'paired_scores.png',dpi=170);plt.close(fig)
    print(f'Wrote observed results and figures to {folder}')
