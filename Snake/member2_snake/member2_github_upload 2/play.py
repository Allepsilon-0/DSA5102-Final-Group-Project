"""Member 2: unassisted gameplay and headless, reproducible progress GIFs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from main import as_tensor, choose_device, load_agent, make_env


def rollout(checkpoint, seed=900000, device='cpu'):
    model, config = load_agent(checkpoint, device)
    env = make_env(config)
    frames = []
    def capture():
        return dict(snake=[x.tolist() for x in env.snake], food=env.food_pos.tolist(), score=env.score)
    try:
        obs, _ = env.reset(seed=seed)
        frames.append(capture())
        total = 0.0
        while True:
            action = model.predict(as_tensor(obs, model.device), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(capture())
            total += reward
            if terminated or truncated:
                return dict(seed=seed, frames=frames, grid_size=config.grid_size,
                            score=info['score'], steps=info['steps'], reward=total,
                            success=info['length'] >= config.end_score,
                            outcome='win' if info['length'] >= config.end_score else
                                    ('time limit' if truncated else 'collision'))
    finally:
        env.close()


def _font(size):
    for name in ['/System/Library/Fonts/Supplemental/Arial.ttf', 'DejaVuSans.ttf']:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def board_image(record, frame_index, title, cell=27):
    grid = record['grid_size']
    width = grid*cell+24
    image = Image.new('RGB', (width, grid*cell+100), '#f4f6f8')
    draw = ImageDraw.Draw(image)
    frame = record['frames'][min(frame_index, len(record['frames'])-1)]
    draw.text((10, 8), title, font=_font(16), fill='#152e41')
    draw.text((10, 32), f"step {min(frame_index, record['steps'])} | score {frame['score']}", font=_font(14), fill='#152e41')
    for r in range(grid):
        for c in range(grid):
            draw.rectangle((12+c*cell, 60+r*cell, 12+(c+1)*cell, 60+(r+1)*cell), fill='white', outline='#dde3e8')
    for i, (r,c) in enumerate(frame['snake']):
        draw.rectangle((14+c*cell, 62+r*cell, 10+(c+1)*cell, 58+(r+1)*cell), fill='#173d59' if i==0 else '#3e9b8a')
    r,c = frame['food']
    draw.ellipse((17+c*cell, 65+r*cell, 7+(c+1)*cell, 55+(r+1)*cell), fill='#e56353')
    if frame_index >= record['steps']:
        draw.text((12, grid*cell+68), f"{record['outcome']} | final score {record['score']}", font=_font(13), fill='#152e41')
    return image


def render_progress(checkpoints, output, seed=900000):
    """Same predeclared seed for every stage; finished games freeze in place."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = {label: rollout(path, seed) for label, path in checkpoints.items()}
    max_steps = max(x['steps'] for x in records.values())
    frames = []
    for step in list(range(0, max_steps+1, 2)) + [max_steps]*8:
        panels = [board_image(rec, step, label) for label,rec in records.items()]
        canvas = Image.new('RGB', (sum(p.width for p in panels), max(p.height for p in panels)), 'white')
        pos = 0
        for p in panels:
            canvas.paste(p, (pos, 0)); pos += p.width
        frames.append(canvas)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=120, loop=0)
    frames[-1].save(output.with_suffix('.png'))
    output.with_suffix('.json').write_text(json.dumps(records, indent=2))
    return {k:{n:v for n,v in rec.items() if n != 'frames'} for k,rec in records.items()}


def play_game(model_path, *, device_name='auto', seed=42, fps=15):
    model, config = load_agent(model_path, choose_device(device_name))
    env = make_env(config, render_mode='human')
    try:
        obs, _ = env.reset(seed=seed)
        while not env.closed:
            action = model.predict(as_tensor(obs, model.device), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            env.render(fps=fps)
            if terminated or truncated:
                print(info)
                break
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--seed', type=int, default=900000)
    p.add_argument('--device', default='cpu')
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--gif', type=Path)
    args = p.parse_args()
    if args.gif:
        print(render_progress({'Actor-Critic':args.checkpoint}, args.gif, args.seed))
    else:
        play_game(args.checkpoint, device_name=args.device, seed=args.seed, fps=args.fps)

if __name__ == '__main__':
    main()
