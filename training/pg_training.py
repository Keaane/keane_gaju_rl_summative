"""Policy-gradient methods on one shared network: REINFORCE, A2C and PPO.

This is the plain PyTorch trainer; the TorchRL versions, used for the final models, are in training/torchrl_pg.py.

All three learn a policy (which action to take). They differ in WHEN they learn and
HOW they judge an action:
  REINFORCE  waits for whole sessions, then scores each action by what followed it
  A2C        learns every few steps, scoring actions against a learned value estimate
  PPO        like A2C, but reuses each batch several times with a safety clip
"""
import argparse
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from environment.custom_env import N_ACTIONS, OBS_DIM, StarterGuideEnv
from training.common import Evaluator, TEST_SEED, evaluate, set_seed


# ---- Settings -----------------------------------------------------------------
@dataclass
class ReinforceConfig:
    total_steps: int = 100_000
    lr: float = 1e-3
    gamma: float = 0.99
    episodes_per_update: int = 16     # whole sessions collected before each update
    baseline: bool = True             # subtract a learned value estimate to reduce noise
    ent_coef: float = 0.01
    hidden: int = 64
    eval_every: int = 10_000
    eval_users: int = 200
    seed: int = 0


@dataclass
class A2CConfig:
    total_steps: int = 100_000
    lr: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 1.0           # 1.0 = plain n-step returns
    n_steps: int = 32                 # steps collected before each update
    ent_coef: float = 0.01            # bonus for staying exploratory
    vf_coef: float = 0.5
    hidden: int = 64
    eval_every: int = 10_000
    eval_users: int = 200
    seed: int = 0


@dataclass
class PPOConfig:
    total_steps: int = 100_000
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2                 # how far one update may move the policy
    epochs: int = 8                   # passes over each batch of experience
    minibatch: int = 256
    rollout_steps: int = 2048
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    hidden: int = 64
    eval_every: int = 10_000
    eval_users: int = 200
    seed: int = 0


# ---- Network and experience collection ----------------------------------------
class ActorCritic(nn.Module):
    """A policy head (score per action) and a value head (expected outcome)."""

    def __init__(self, hidden):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.policy = nn.Linear(hidden, N_ACTIONS)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.body(obs)
        return self.policy(h), self.value(h)[..., 0]


def greedy_policy(net):
    def act(obs):
        with torch.no_grad():
            return int(net(torch.as_tensor(obs, dtype=torch.float32))[0].argmax())
    return act


def new_data():
    return {k: [] for k in ("obs", "action", "logp", "reward", "done", "value")}


def step_env(env, net, obs, data):
    """Take one step with an action sampled from the policy, and record it."""
    with torch.no_grad():
        logits, value = net(torch.as_tensor(obs))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
    next_obs, reward, terminated, truncated, _ = env.step(int(action))
    done = terminated or truncated
    for key, v in zip(data, (obs, int(action), dist.log_prob(action).item(), reward, float(done), value.item())):
        data[key].append(v)
    return next_obs, done


def collect(env, net, obs, steps):
    """Run for a fixed number of steps (sessions may continue across calls)."""
    data = new_data()
    for _ in range(steps):
        obs, done = step_env(env, net, obs, data)
        if done:
            obs = env.reset()[0]
    with torch.no_grad():
        last_value = net(torch.as_tensor(obs))[1].item()
    return data, obs, last_value


def collect_episodes(env, net, n_episodes):
    """Run whole sessions, start to finish."""
    data, finished, obs = new_data(), 0, env.reset()[0]
    while finished < n_episodes:
        obs, done = step_env(env, net, obs, data)
        if done:
            finished, obs = finished + 1, env.reset()[0]
    return data


def discounted_returns(rewards, dones, gamma):
    """What each action was followed by: its reward plus discounted later rewards."""
    out, running = np.zeros(len(rewards), np.float32), 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running * (1 - dones[t])
        out[t] = running
    return out


def gae(data, last_value, gamma, lam):
    """Advantage = how much better the outcome was than the value head expected."""
    rewards, dones, values = data["reward"], data["done"], data["value"] + [last_value]
    adv, running = np.zeros(len(rewards), np.float32), 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        running = delta + gamma * lam * (1 - dones[t]) * running
        adv[t] = running
    return adv, adv + np.array(values[:-1], np.float32)


def tensors(data):
    return (torch.as_tensor(np.array(data["obs"], np.float32)), torch.as_tensor(data["action"]),
            torch.as_tensor(np.array(data["logp"], np.float32)))


def make_trainer(cfg, log_path, save_path, verbose):
    set_seed(cfg.seed)
    env, net = StarterGuideEnv(), ActorCritic(cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    ev = Evaluator(greedy_policy, cfg.eval_every, cfg.eval_users, log_path, save_path, verbose)
    return env, net, opt, ev


def update(net, opt, loss):
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
    opt.step()


# ---- The three algorithms ------------------------------------------------------
def train_reinforce(cfg=ReinforceConfig(), log_path=None, save_path=None, verbose=True):
    env, net, opt, ev = make_trainer(cfg, log_path, save_path, verbose)
    env.reset(seed=cfg.seed)
    steps_done = 0
    while steps_done < cfg.total_steps:
        data = collect_episodes(env, net, cfg.episodes_per_update)
        steps_done += len(data["reward"])
        o, a, _ = tensors(data)
        returns = torch.as_tensor(discounted_returns(data["reward"], data["done"], cfg.gamma))
        logits, value = net(o)
        dist = torch.distributions.Categorical(logits=logits)
        score = returns - value.detach() if cfg.baseline else returns
        score = (score - score.mean()) / (score.std() + 1e-8)
        value_loss = nn.functional.mse_loss(value, returns) if cfg.baseline else torch.zeros(())
        policy_loss = -(dist.log_prob(a) * score).mean()
        update(net, opt, policy_loss + 0.5 * value_loss - cfg.ent_coef * dist.entropy().mean())
        ev.check(steps_done, net, lambda: {"policy_loss": policy_loss.item(), "entropy": dist.entropy().mean().item()})
    return ev.load_best(net), ev.history


def train_a2c(cfg=A2CConfig(), log_path=None, save_path=None, verbose=True):
    env, net, opt, ev = make_trainer(cfg, log_path, save_path, verbose)
    obs, _ = env.reset(seed=cfg.seed)
    steps_done = 0
    while steps_done < cfg.total_steps:
        data, obs, last_value = collect(env, net, obs, cfg.n_steps)
        steps_done += cfg.n_steps
        o, a, _ = tensors(data)
        adv, returns = map(torch.as_tensor, gae(data, last_value, cfg.gamma, cfg.gae_lambda))
        logits, value = net(o)
        dist = torch.distributions.Categorical(logits=logits)
        policy_loss = -(dist.log_prob(a) * adv).mean()
        value_loss = nn.functional.mse_loss(value, returns)
        update(net, opt, policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * dist.entropy().mean())
        ev.check(steps_done, net, lambda: {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                                           "entropy": dist.entropy().mean().item()})
    return ev.load_best(net), ev.history


def train_ppo(cfg=PPOConfig(), log_path=None, save_path=None, verbose=True):
    env, net, opt, ev = make_trainer(cfg, log_path, save_path, verbose)
    obs, _ = env.reset(seed=cfg.seed)
    steps_done = 0
    while steps_done < cfg.total_steps:
        data, obs, last_value = collect(env, net, obs, cfg.rollout_steps)
        steps_done += cfg.rollout_steps
        o, a, old_logp = tensors(data)
        adv, returns = map(torch.as_tensor, gae(data, last_value, cfg.gamma, cfg.gae_lambda))
        for _ in range(cfg.epochs):
            for idx in torch.randperm(len(a)).split(cfg.minibatch):
                logits, value = net(o[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(a[idx]) - old_logp[idx]).exp()
                advantage = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
                policy_loss = -torch.min(ratio * advantage, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage).mean()
                value_loss = nn.functional.mse_loss(value, returns[idx])
                update(net, opt, policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * dist.entropy().mean())
        ev.check(steps_done, net, lambda: {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                                           "entropy": dist.entropy().mean().item()})
    return ev.load_best(net), ev.history


TRAINERS = {"reinforce": (ReinforceConfig, train_reinforce), "a2c": (A2CConfig, train_a2c), "ppo": (PPOConfig, train_ppo)}


if __name__ == "__main__":
    first = argparse.ArgumentParser()
    first.add_argument("--algo", choices=TRAINERS, default="ppo")
    known, rest = first.parse_known_args()
    config_class, trainer = TRAINERS[known.algo]
    parser = argparse.ArgumentParser()
    for key, value in asdict(config_class()).items():
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(value, bool) else type(value)
        parser.add_argument(f"--{key}", type=kind, default=value)
    cfg = config_class(**vars(parser.parse_args(rest)))
    net, _ = trainer(cfg, f"logs/{known.algo}_seed{cfg.seed}.csv", f"models/pg/{known.algo}_seed{cfg.seed}.pt")
    print("TEST:", evaluate(greedy_policy(net), 300, TEST_SEED))
