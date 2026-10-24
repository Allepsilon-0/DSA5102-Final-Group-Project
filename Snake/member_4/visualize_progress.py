"""Reusable checkpoint-to-GIF tool for the project's PPO, AC and DQN models.
Example: python visualize_progress.py --checkpoints results/final/seed-7/clip-0.2/checkpoints/step-0.pt results/final/seed-7/clip-0.2/checkpoints/step-50000.pt --labels Initial Final
For another environment setting, explicitly pass --grid-size/--end-score/--max-steps.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from environment import SnakeEnv


def load_policy(path, grid):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if 'online_state_dict' in ck:
        from dqn import QNetwork,DuelingQNetwork
        state=ck['online_state_dict']
        dueling=ck.get('config',{}).get('dueling',False) or 'dueling' in ck.get('algorithm','')
        net=(DuelingQNetwork if dueling else QNetwork)(7,3,grid)
        net.load_state_dict(state)
        def policy(obs):
            return int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1).item())
    else:
        state=ck['model_state_dict']
        if ck.get('algorithm')=='ppo' or any(k.startswith('encoder.') for k in state):
            from ppo import PPOAgent
            net=PPOAgent(7,3,grid).model
        else:
            from model import ConvActorCritic
            net=ConvActorCritic(7,3,grid)
        net.load_state_dict(state)
        def policy(obs):
            return int(net(torch.as_tensor(obs).unsqueeze(0))[0].argmax(-1).item())
    net.eval()
    return policy,ck

def frame(env,label,ended):
    cell=25;top=52
    im=Image.new('RGB',(env.grid_size*cell,env.grid_size*cell+top),'#111827')
    draw=ImageDraw.Draw(im)
    draw.text((5,5),label,fill='white')
    draw.text((5,21),f'Score {env.score} | step {env.step_count}'+(' | ended' if ended else ''),fill='white')
    for row in range(env.grid_size):
        for col in range(env.grid_size):
            draw.rectangle((col*cell,top+row*cell,(col+1)*cell-1,top+(row+1)*cell-1),outline='#334155')
    def square(pos,color):
        r,c=map(int,pos);draw.rectangle((c*cell+2,top+r*cell+2,(c+1)*cell-3,top+(r+1)*cell-3),fill=color)
    square(env.food_pos,'#22c55e')
    for i,pos in reversed(list(enumerate(env.snake))):square(pos,'#ef4444' if i==0 else '#f59e0b')
    return im

@torch.no_grad()
def render(checkpoints,labels,output,seed=700000,grid_size=8,end_score=8,max_steps=300):
    torch.set_num_threads(1)
    if len(labels)!=len(checkpoints) or not checkpoints:raise ValueError('One label per checkpoint is required')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    all_frames=[];summary=[]
    for path,label in zip(checkpoints,labels):
        policy,ck=load_policy(path,grid_size)
        cfg=ck.get('config',{})
        for key,val in [('grid_size',grid_size),('end_score',end_score),('max_steps',max_steps)]:
            if key in cfg and cfg[key]!=val:raise ValueError(f'{path}: {key} mismatch')
        env=SnakeEnv(grid_size=grid_size,end_score=end_score,max_steps=max_steps,reward_mode=cfg.get('reward_mode','sparse'))
        obs,_=env.reset(seed=seed);frames=[frame(env,label,False)]
        try:
            while True:
                obs,_,term,trunc,info=env.step(policy(obs));frames.append(frame(env,label,term or trunc))
                if term or trunc:break
            summary.append({'checkpoint':str(path),'label':label,'seed':seed,'score':info['score'],
                            'steps':info['steps'],'success':info['length']>=end_score,'timeout':trunc})
            all_frames.append(frames)
        finally:env.close()
    combined=[]
    for t in range(max(map(len,all_frames))):
        panels=[frames[min(t,len(frames)-1)] for frames in all_frames]
        im=Image.new('RGB',(sum(x.width for x in panels),max(x.height for x in panels)),'white')
        x=0
        for panel in panels:im.paste(panel,(x,0));x+=panel.width
        combined.append(im)
    combined[0].save(output,save_all=True,append_images=combined[1:],duration=100,loop=0)
    combined[-1].save(output.with_suffix('.png'))
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',nargs='+',type=Path,required=True)
    p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--output',type=Path,default=Path('progress.gif'))
    p.add_argument('--seed',type=int,default=700000)
    p.add_argument('--grid-size',type=int,default=8);p.add_argument('--end-score',type=int,default=8)
    p.add_argument('--max-steps',type=int,default=300)
    a=p.parse_args();render(a.checkpoints,a.labels,a.output,a.seed,a.grid_size,a.end_score,a.max_steps)
