"""DQN (value-based): Q-network, replay buffer, target network, epsilon-greedy.

This is the plain PyTorch trainer; the TorchRL version, used for the final models, is training/torchrl_dqn.py.
"""
import argparse
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from environment.custom_env import N_ACTIONS, OBS_DIM, StarterGuideEnv
from training.common import Evaluator, TEST_SEED, evaluate, greedy, mlp, set_seed


@dataclass
class DQNConfig:
    total_steps: int = 100_000
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 64
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000     # steps over which epsilon falls from start to end
    target_update: int = 500          # steps between copying the online net into the target net
    train_every: int = 2              # one gradient step per this many env steps
    learn_start: int = 1_000
    hidden: int = 64
    eval_every: int = 10_000
    eval_users: int = 200
    seed: int = 0


class ReplayBuffer:
    def __init__(self, size):
        self.obs = np.zeros((size, OBS_DIM), np.float32)
        self.next_obs = np.zeros((size, OBS_DIM), np.float32)
        self.action = np.zeros(size, np.int64)
        self.reward = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.size, self.count = size, 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.count % self.size
        self.obs[i], self.action[i], self.reward[i], self.next_obs[i], self.done[i] = obs, action, reward, next_obs, done
        self.count += 1

    def sample(self, batch_size):
        idx = np.random.randint(0, min(self.count, self.size), batch_size)
        return [torch.as_tensor(a[idx]) for a in (self.obs, self.action, self.reward, self.next_obs, self.done)]


def train_dqn(cfg=DQNConfig(), log_path=None, save_path=None, verbose=True):
    set_seed(cfg.seed)
    env = StarterGuideEnv()
    q = mlp([OBS_DIM, cfg.hidden, cfg.hidden, N_ACTIONS])
    target = mlp([OBS_DIM, cfg.hidden, cfg.hidden, N_ACTIONS])
    target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    buffer, losses = ReplayBuffer(cfg.buffer_size), []
    ev = Evaluator(greedy, cfg.eval_every, cfg.eval_users, log_path, save_path, verbose)

    obs, _ = env.reset(seed=cfg.seed)
    for step in range(1, cfg.total_steps + 1):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / cfg.eps_decay_steps)
        if np.random.rand() < eps:
            action = np.random.randint(N_ACTIONS)
        else:
            with torch.no_grad():
                action = int(q(torch.as_tensor(obs)).argmax())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, float(done))
        obs = env.reset()[0] if done else next_obs

        if step >= cfg.learn_start and step % cfg.train_every == 0:
            o, a, r, o2, d = buffer.sample(cfg.batch_size)
            with torch.no_grad():
                goal = r + cfg.gamma * (1 - d) * target(o2).max(1).values
            loss = nn.functional.smooth_l1_loss(q(o).gather(1, a[:, None])[:, 0], goal)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.item())
        if step % cfg.target_update == 0:
            target.load_state_dict(q.state_dict())

        ev.check(step, q, lambda: {"epsilon": eps, "loss": float(np.mean(losses[-500:])) if losses else 0.0})
    return ev.load_best(q), ev.history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, value in asdict(DQNConfig()).items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    cfg = DQNConfig(**vars(args))
    net, _ = train_dqn(cfg, f"logs/dqn_seed{cfg.seed}.csv", f"models/dqn/dqn_seed{cfg.seed}.pt")
    print("TEST:", evaluate(greedy(net), 300, TEST_SEED))
