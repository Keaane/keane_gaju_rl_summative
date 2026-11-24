"""REINFORCE, A2C and PPO built from TorchRL components (TorchRL 0.3.1, Python 3.11 side environment; see README).

Same algorithms, defaults and network as training/pg_training.py. The TorchRL parts:
  GymWrapper (categorical actions), ProbabilisticActor + ValueOperator around the shared ActorCritic network,
  SyncDataCollector (PPO, A2C) and EnvBase.rollout (REINFORCE, whole sessions), GAE (PPO, A2C),
  TD1Estimator (REINFORCE: Monte Carlo returns), ClipPPOLoss, A2CLoss, ReinforceLoss,
  ReplayBuffer + LazyTensorStorage + SamplerWithoutReplacement (PPO minibatches).
Every estimator treats any episode end (including the 12-step cutoff) as the end, like the plain code: the
"terminated" key is pointed at "done". The checkpoint is the plain ActorCritic, so it is scored with the same
validation and test users. Each loss was checked against the plain loss on a fixed batch (tests/test_torchrl.py).
"""
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE, TD1Estimator

from training.common import Evaluator, set_seed
from training.pg_training import A2CConfig, ActorCritic, PPOConfig, ReinforceConfig, greedy_policy
from training.torchrl_dqn import make_env


class PolicyHead(nn.Module):
    """The policy half of the shared ActorCritic: 15 action scores (logits)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        return self.net(obs)[0]


class ValueHead(nn.Module):
    """The value half of the shared ActorCritic, shaped [..., 1] as TorchRL expects."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        return self.net(obs)[1].unsqueeze(-1)


def make_actor_critic(net, env):
    actor = ProbabilisticActor(TensorDictModule(PolicyHead(net), in_keys=["observation"], out_keys=["logits"]),
                               in_keys=["logits"], distribution_class=torch.distributions.Categorical,
                               spec=env.action_spec, return_log_prob=True)
    critic = ValueOperator(ValueHead(net), in_keys=["observation"])
    return actor, critic


def plain_like(estimator):
    """Stop at any episode end, including the 12-step cutoff (the plain code multiplies by 1 - done)."""
    estimator.set_keys(terminated="done")
    return estimator


def update(net, opt, loss):
    """Same update as the plain code: one Adam step with the gradient norm clipped at 0.5."""
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
    opt.step()


def setup(cfg, log_path, save_path, verbose):
    set_seed(cfg.seed)
    env = make_env(cfg.seed)
    net = ActorCritic(cfg.hidden)
    actor, critic = make_actor_critic(net, env)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    ev = Evaluator(greedy_policy, cfg.eval_every, cfg.eval_users, log_path, save_path, verbose)
    return env, net, actor, critic, opt, ev


# ---- PPO ------------------------------------------------------------------------------------
def ppo_loss_module(actor, critic, cfg):
    return ClipPPOLoss(actor, critic, clip_epsilon=cfg.clip, entropy_coef=cfg.ent_coef, critic_coef=cfg.vf_coef,
                       loss_critic_type="l2", normalize_advantage=True)


def train_ppo_torchrl(cfg=PPOConfig(), log_path=None, save_path=None, verbose=True):
    env, net, actor, critic, opt, ev = setup(cfg, log_path, save_path, verbose)
    advantage = plain_like(GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic))
    loss_module = ppo_loss_module(actor, critic, cfg)
    loss_module.set_keys(terminated="done")
    collector = SyncDataCollector(env, actor, frames_per_batch=cfg.rollout_steps, total_frames=cfg.total_steps)
    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg.rollout_steps), sampler=SamplerWithoutReplacement(),
                          batch_size=cfg.minibatch)
    steps = 0
    for batch in collector:
        steps += batch.numel()
        with torch.no_grad():
            advantage(batch)                                    # advantages once per rollout, before the epochs
        buffer.extend(batch.reshape(-1))
        for _ in range(cfg.epochs):
            for _ in range(-(-batch.numel() // cfg.minibatch)):  # one pass over the rollout in minibatches
                out = loss_module(buffer.sample())
                update(net, opt, out["loss_objective"] + out["loss_critic"] + out["loss_entropy"])
        ev.check(steps, net, lambda: {"policy_loss": out["loss_objective"].item(), "value_loss": out["loss_critic"].item(),
                                      "entropy": out["entropy"].item()})
    collector.shutdown()
    return ev.load_best(net), ev.history


# ---- A2C ------------------------------------------------------------------------------------
def a2c_loss_module(actor, critic, cfg):
    return A2CLoss(actor, critic, entropy_coef=cfg.ent_coef, critic_coef=cfg.vf_coef, loss_critic_type="l2")


def a2c_total_loss(out):
    """Policy + value + entropy terms. TorchRL 0.3.1's A2CLoss leaves out the entropy term when its coefficient is
    0 (entropy_bonus = entropy_bonus and entropy_coef), so a missing term counts as 0, which is what it equals."""
    entropy_term = out["loss_entropy"] if "loss_entropy" in out.keys() else torch.zeros(())
    return out["loss_objective"] + out["loss_critic"] + entropy_term


def a2c_entropy_for_log(out, cfg):
    """The policy entropy for the training log (not logged when the entropy coefficient is 0)."""
    if "loss_entropy" not in out.keys():
        return float("nan")
    return -out["loss_entropy"].item() / max(cfg.ent_coef, 1e-12)


def train_a2c_torchrl(cfg=A2CConfig(), log_path=None, save_path=None, verbose=True):
    env, net, actor, critic, opt, ev = setup(cfg, log_path, save_path, verbose)
    advantage = plain_like(GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic))
    loss_module = a2c_loss_module(actor, critic, cfg)
    loss_module.set_keys(terminated="done")
    collector = SyncDataCollector(env, actor, frames_per_batch=cfg.n_steps, total_frames=cfg.total_steps)
    steps = 0
    for batch in collector:
        steps += batch.numel()
        with torch.no_grad():
            advantage(batch)
        out = loss_module(batch)
        update(net, opt, a2c_total_loss(out))
        ev.check(steps, net, lambda: {"policy_loss": out["loss_objective"].item(), "value_loss": out["loss_critic"].item(),
                                      "entropy": a2c_entropy_for_log(out, cfg)})
    collector.shutdown()
    return ev.load_best(net), ev.history


# ---- REINFORCE ------------------------------------------------------------------------------
def reinforce_losses(actor, critic, returns_estimator, batch, cfg):
    """ReinforceLoss gives the policy and value terms; like the plain code, the score (return minus baseline) is
    normalized over the batch, and the entropy bonus is added here because ReinforceLoss 0.3.1 has none."""
    with torch.no_grad():
        returns_estimator(batch)                              # Monte Carlo returns, and returns minus baseline
    score = batch.get("advantage") if cfg.baseline else batch.get("value_target")   # no baseline: the return
    batch.set("advantage", (score - score.mean()) / (score.std() + 1e-8))
    loss_module = ReinforceLoss(actor, critic, loss_critic_type="l2")
    loss_module.set_keys(terminated="done")
    out = loss_module(batch)
    dist = actor.get_dist(batch)
    entropy = dist.entropy().mean()
    value_loss = out["loss_value"] if cfg.baseline else torch.zeros(())
    return out["loss_actor"], value_loss, entropy


def train_reinforce_torchrl(cfg=ReinforceConfig(), log_path=None, save_path=None, verbose=True):
    env, net, actor, critic, opt, ev = setup(cfg, log_path, save_path, verbose)
    returns_estimator = plain_like(TD1Estimator(gamma=cfg.gamma, value_network=critic))
    steps = 0
    while steps < cfg.total_steps:
        sessions = [env.rollout(max_steps=1000, policy=actor) for _ in range(cfg.episodes_per_update)]
        batch = torch.cat(sessions, 0)                          # whole sessions, start to finish
        steps += batch.numel()
        policy_loss, value_loss, entropy = reinforce_losses(actor, critic, returns_estimator, batch, cfg)
        update(net, opt, policy_loss + 0.5 * value_loss - cfg.ent_coef * entropy)
        ev.check(steps, net, lambda: {"policy_loss": policy_loss.item(), "entropy": entropy.item()})
    return ev.load_best(net), ev.history


TORCHRL_PG = {"ppo": (PPOConfig, train_ppo_torchrl, greedy_policy),
              "a2c": (A2CConfig, train_a2c_torchrl, greedy_policy),
              "reinforce": (ReinforceConfig, train_reinforce_torchrl, greedy_policy)}
