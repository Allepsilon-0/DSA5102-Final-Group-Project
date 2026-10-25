"""Command-line entry point for the saved Snake PPO experiment."""
from pathlib import Path
import argparse
import json
import snake_ppo as m

BASE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['demo', 'evaluate', 'analysis'])
    p.add_argument('--checkpoint', type=Path, default=BASE / 'results/final/seed-7/clip-0.2/checkpoints/best.pt')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=2950000)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.episodes < 1:
        p.error('--episodes must be positive')
    if a.command == 'analysis':
        m.run('analysis', output=a.output or BASE / 'results/final')
    elif a.command == 'demo':
        target = a.output or BASE / 'demo/gameplay.gif'
        result = m.render([a.checkpoint], ['PPO saved policy'], target, seed=a.seed)
        print(json.dumps(result, indent=2))
        print('Saved:', target.resolve())
    else:
        agent = m.make_agent()
        metadata = agent.load_checkpoint(a.checkpoint)
        rows = m.evaluate(agent, a.seed, a.episodes)
        result = {'checkpoint': str(a.checkpoint), 'metadata': metadata,
                  'episodes': a.episodes, 'summary': m.summary(rows), 'episode_results': rows}
        target = a.output or BASE / 'demo/evaluation.json'
        m.write_json(target, result)
        print(json.dumps(result['summary'], indent=2))
        print('Saved:', target.resolve())

if __name__ == '__main__':
    main()
