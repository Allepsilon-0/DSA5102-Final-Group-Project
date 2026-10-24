"""GAE for a finite-horizon Snake episode; independently testable with NumPy."""
import numpy as np

def compute_gae(rewards, values, ends, next_value, gamma=.99, lam=.95):
    rewards, values, ends = map(lambda x: np.asarray(x, dtype=np.float32), (rewards, values, ends))
    if not (rewards.ndim == values.ndim == ends.ndim == 1 and len(rewards) == len(values) == len(ends)):
        raise ValueError('Expected equal-length vectors')
    advantages = np.zeros_like(rewards)
    running = 0.
    for t in reversed(range(len(rewards))):
        continuation = 1. - ends[t]
        following = next_value if t == len(rewards)-1 else values[t+1]
        delta = rewards[t] + gamma * continuation * following - values[t]
        running = delta + gamma * lam * continuation * running
        advantages[t] = running
    return advantages, advantages + values
