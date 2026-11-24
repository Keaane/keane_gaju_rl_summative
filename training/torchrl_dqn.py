"""DQN built from TorchRL components (TorchRL 0.3.1, Python 3.11 side environment; see the README).

Same algorithm and settings as training/dqn_training.py, but the pieces come from TorchRL:
  GymWrapper            wraps StarterGuideEnv, with a categorical action spec (one number, not one-hot)
  QValueActor           turns the Q-network into a policy that picks the action with the highest value
  EGreedyModule         epsilon-greedy exploration, annealed linearly by step
  SyncDataCollector     steps the environment and returns batches of transitions
  ReplayBuffer + LazyTensorStorage   stores transitions and samples training batches
  DQNLoss + HardUpdate  the TD loss with a target network that is copied every 500 environment steps

The Q-network is the same `mlp` as the plain version, so checkpoints have the same format and are scored with
the same validation and test users (training/common.py).
"""
from dataclasses import dataclass

import numpy as np
import torch
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymWrapper
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate

from environment.custom_env import N_ACTIONS, OBS_DIM, StarterGuideEnv
from training.common import Evaluator, greedy, mlp, set_seed


@dataclass
class TorchRLDQNConfig:
    total_steps: int = 100_000
    lr: float = 1e-3                  # same defaults as DQNConfig; the sweep grid applies its changes on top
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 64
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    target_update: int = 500          # environment steps between target-network copies
    train_every: int = 2              # one gradient step per this many environment steps
    learn_start: int = 1_000
    hidden: int = 64
    eval_every: int = 10_000
    eval_users: int = 200
    seed: int = 0


def make_env(seed):
    """StarterGuideEnv inside TorchRL. Categorical actions: the action is one integer from 0 to 14."""
    env = GymWrapper(StarterGuideEnv(), categorical_action_encoding=True)
    env.set_seed(seed)
    return env


def train_dqn_torchrl(cfg=TorchRLDQNConfig(), log_path=None, save_path=None, verbose=True):
    set_seed(cfg.seed)
    env = make_env(cfg.seed)
    q = mlp([OBS_DIM, cfg.hidden, cfg.hidden, N_ACTIONS])               # same network as the plain DQN
    actor = QValueActor(q, in_keys=["observation"], spec=env.action_spec)
    explore = EGreedyModule(env.action_spec, eps_init=cfg.eps_start, eps_end=cfg.eps_end,
                            annealing_num_steps=cfg.eps_decay_steps)
    policy = TensorDictSequential(actor, explore)

    loss_module = DQNLoss(actor, loss_function="smooth_l1", delay_value=True, action_space=env.action_spec)
    # Plain DQN stops bootstrapping at any episode end, including the 12-step cutoff (1 - done). TorchRL's
    # default stops only at "terminated", so point it at "done" to match (checked: identical loss on one batch).
    loss_module.set_keys(terminated="done")
    loss_module.make_value_estimator(gamma=cfg.gamma)
    target_updater = HardUpdate(loss_module, value_network_update_interval=cfg.target_update // cfg.train_every)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg.buffer_size), batch_size=cfg.batch_size)
    collector = SyncDataCollector(env, policy, frames_per_batch=cfg.train_every, total_frames=cfg.total_steps)
    ev = Evaluator(greedy, cfg.eval_every, cfg.eval_users, log_path, save_path, verbose)

    steps, losses = 0, []
    for batch in collector:
        buffer.extend(batch.reshape(-1))
        steps += batch.numel()
        explore.step(batch.numel())                                    # anneal epsilon by the steps just taken
        if steps >= cfg.learn_start:
            loss = loss_module(buffer.sample())["loss"]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(loss_module.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.item())
        target_updater.step()                                          # copies the network every target_update steps
        eps = float(explore.eps)
        ev.check(steps, q, lambda: {"epsilon": eps, "loss": float(np.mean(losses[-500:])) if losses else 0.0})
    collector.shutdown()
    return ev.load_best(q), ev.history
