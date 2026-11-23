"""Shared pieces for every algorithm: seeding, networks, evaluation, logging.

Three disjoint sets of simulated users are used, so results are not flattered:
  training users   random, fresh every episode
  validation users (VAL_SEED..)  used during training to pick the best checkpoint
  test users       (TEST_SEED..) used once at the end; the SAME users for all algorithms
"""
import csv
import os

import numpy as np
import torch
from torch import nn

from environment.custom_env import StarterGuideEnv

VAL_SEED = 200_000
TEST_SEED = 100_000


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(sizes, activation=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def evaluate(policy, n_users=200, seed0=VAL_SEED, options=None):
    """Run a policy (obs -> action) on n_users fixed simulated users. Returns mean metrics."""
    env = StarterGuideEnv()
    rewards, flags = [], {"plan": [], "quit": [], "scammed": [], "panic": [], "beat_inflation": []}
    for i in range(n_users):
        obs, _ = env.reset(seed=seed0 + i, options=options)
        done, total = False, 0.0
        while not done:
            obs, r, terminated, truncated, info = env.step(policy(obs))
            total, done = total + r, terminated or truncated
        rewards.append(total)
        flags["plan"].append(info["outcome"] == "plan")
        flags["quit"].append(info["outcome"] == "quit")
        flags["scammed"].append(info.get("scammed", False))
        flags["panic"].append(info.get("panic", False))
        flags["beat_inflation"].append(info.get("real_growth", -1.0) > 0)
    out = {"reward": float(np.mean(rewards)), "se": float(np.std(rewards) / np.sqrt(n_users))}
    out.update({k: float(np.mean(v)) for k, v in flags.items()})
    return out


def greedy(net):
    """Turn a network that outputs one score per action into a greedy policy."""
    def act(obs):
        with torch.no_grad():
            return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax())
    return act


class CsvLog:
    """Append one row per evaluation so plots can be drawn later."""

    def __init__(self, path):
        self.path, self.writer, self.file = path, None, None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, row):
        if not self.path:
            return
        if self.writer is None:
            self.file = open(self.path, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()


class Evaluator:
    """Every `every` steps: score on the validation users, log a row, keep the best checkpoint."""

    def __init__(self, make_policy, every, n_users, log_path=None, save_path=None, verbose=True):
        self.make_policy, self.every, self.n_users = make_policy, every, n_users
        self.log, self.save_path, self.verbose = CsvLog(log_path), save_path, verbose
        self.next_at, self.best, self.history = every, -1e9, []

    def check(self, step, net, extras=None):
        """Call after every update. `extras` is a function returning training stats to log."""
        if step < self.next_at:
            return
        self.next_at += self.every
        m = evaluate(self.make_policy(net), self.n_users, VAL_SEED)
        extra = extras() if extras else {}
        row = {"step": step, **extra, **m}
        self.history.append(row)
        self.log.write(row)
        if self.verbose:
            stats = "  ".join(f"{k} {v:.2f}" for k, v in extra.items())
            print(f"step {step:>7}  {stats}  val reward {m['reward']:+.2f} +/- {m['se']:.2f}  plan {m['plan']:.0%}  quit {m['quit']:.0%}", flush=True)
        if m["reward"] > self.best:
            self.best = m["reward"]
            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(net.state_dict(), self.save_path)

    def load_best(self, net):
        if self.save_path and os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
        return net
