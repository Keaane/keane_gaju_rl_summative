"""The TorchRL versions compute the same losses as the plain code (same weights, same batch).

Runs only in the Python 3.11 side environment with TorchRL 0.3.1 (see the README); skipped everywhere else.
"""
import pytest

pytest.importorskip("torchrl")

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torchrl.collectors import SyncDataCollector  # noqa: E402
from torchrl.modules import QValueActor  # noqa: E402
from torchrl.objectives import DQNLoss  # noqa: E402
from torchrl.objectives.value import GAE, TD1Estimator  # noqa: E402

from environment.custom_env import N_ACTIONS, OBS_DIM, StarterGuideEnv  # noqa: E402
from training import pg_training as plain  # noqa: E402
from training import torchrl_pg as trl  # noqa: E402
from training.common import mlp  # noqa: E402
from training.torchrl_dqn import TorchRLDQNConfig, make_env, train_dqn_torchrl  # noqa: E402


def grads(net):
    """All gradients as one vector; a part the loss does not touch (no grad) counts as zeros."""
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in net.parameters()])


def same(net, plain_loss, trl_loss):
    """Equal loss and equal gradients (float32 rounding only)."""
    net.zero_grad(set_to_none=True); plain_loss.backward(); g1 = grads(net).clone()
    net.zero_grad(set_to_none=True); trl_loss.backward(); g2 = grads(net)
    assert abs(plain_loss.item() - trl_loss.item()) < 1e-4 and (g1 - g2).abs().max().item() < 1e-4


def rollout(seed, actor, frames=256):
    return next(iter(SyncDataCollector(make_env(seed), actor, frames_per_batch=frames, total_frames=frames)))


def plain_targets(batch, net, gamma, lam):
    with torch.no_grad():
        values = net(batch["observation"])[1]
        last = net(batch["next", "observation"][-1])[1].item()
    data = {"reward": batch["next", "reward"][:, 0].tolist(), "done": batch["next", "done"][:, 0].float().tolist(),
            "value": values.tolist()}
    return map(torch.as_tensor, plain.gae(data, last, gamma, lam))


def test_wrapper_passes_observations_and_rewards_through():
    env, raw = make_env(7), StarterGuideEnv()
    raw.reset(seed=7)
    obs, _ = raw.reset()                                   # set_seed already resets once
    td = env.reset()
    assert env.action_spec.shape == () and env.action_spec.space.n == N_ACTIONS     # one integer, not one-hot
    for action in (0, 8, 9, 11):
        assert torch.allclose(td["observation"], torch.as_tensor(obs))
        td["action"] = torch.tensor(action)
        td = env.step(td)
        obs, reward, term, trunc, _ = raw.step(action)
        assert abs(td["next", "reward"].item() - reward) < 1e-5
        if term or trunc:
            break
        td = td["next"]


def test_dqn_loss_matches_plain():
    torch.manual_seed(0)
    spec = make_env(0).action_spec
    q = mlp([OBS_DIM, 64, 64, N_ACTIONS])
    actor = QValueActor(q, in_keys=["observation"], spec=spec)
    batch = rollout(1, actor, 64)
    loss = DQNLoss(actor, loss_function="smooth_l1", delay_value=True, action_space=spec)
    loss.set_keys(terminated="done")
    loss.make_value_estimator(gamma=0.99)
    r, d = batch["next", "reward"][:, 0], batch["next", "done"][:, 0].float()
    with torch.no_grad():
        goal = r + 0.99 * (1 - d) * q(batch["next", "observation"]).max(1).values
    plain_loss = nn.functional.smooth_l1_loss(q(batch["observation"]).gather(1, batch["action"][:, None])[:, 0], goal)
    same(q, plain_loss, loss(batch.clone())["loss"])


def test_a2c_loss_matches_plain():
    torch.manual_seed(1)
    cfg, net = plain.A2CConfig(), plain.ActorCritic(64)
    actor, critic = trl.make_actor_critic(net, make_env(1))
    batch = rollout(1, actor)
    adv, returns = plain_targets(batch, net, cfg.gamma, cfg.gae_lambda)
    logits, value = net(batch["observation"])
    dist = torch.distributions.Categorical(logits=logits)
    plain_loss = (-(dist.log_prob(batch["action"]) * adv).mean() + cfg.vf_coef * nn.functional.mse_loss(value, returns)
                  - cfg.ent_coef * dist.entropy().mean())
    with torch.no_grad():
        trl.plain_like(GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic))(batch)
    lm = trl.a2c_loss_module(actor, critic, cfg)
    lm.set_keys(terminated="done")
    out = lm(batch)
    same(net, plain_loss, out["loss_objective"] + out["loss_critic"] + out["loss_entropy"])


def test_ppo_loss_matches_plain():
    torch.manual_seed(2)
    cfg, net = plain.PPOConfig(), plain.ActorCritic(64)
    actor, critic = trl.make_actor_critic(net, make_env(2))
    batch = rollout(2, actor)
    adv, returns = plain_targets(batch, net, cfg.gamma, cfg.gae_lambda)
    with torch.no_grad():
        trl.plain_like(GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic))(batch)
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))                  # later epoch: ratios differ from 1, clipping is active
    idx = torch.randperm(256)[:64]
    logits, value = net(batch["observation"][idx])
    dist = torch.distributions.Categorical(logits=logits)
    ratio = (dist.log_prob(batch["action"][idx]) - batch["sample_log_prob"][idx]).exp()
    nadv = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
    plain_loss = (-torch.min(ratio * nadv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * nadv).mean()
                  + cfg.vf_coef * nn.functional.mse_loss(value, returns[idx]) - cfg.ent_coef * dist.entropy().mean())
    lm = trl.ppo_loss_module(actor, critic, cfg)
    lm.set_keys(terminated="done")
    out = lm(batch[idx].clone())
    same(net, plain_loss, out["loss_objective"] + out["loss_critic"] + out["loss_entropy"])


@pytest.mark.parametrize("baseline", [True, False])
def test_reinforce_loss_matches_plain(baseline):
    torch.manual_seed(3)
    cfg, net = plain.ReinforceConfig(ent_coef=0.05, baseline=baseline), plain.ActorCritic(64)
    env = make_env(3)
    actor, critic = trl.make_actor_critic(net, env)
    batch = torch.cat([env.rollout(max_steps=1000, policy=actor) for _ in range(cfg.episodes_per_update)], 0)
    returns = torch.as_tensor(plain.discounted_returns(batch["next", "reward"][:, 0].tolist(),
                                                       batch["next", "done"][:, 0].float().tolist(), cfg.gamma))
    logits, value = net(batch["observation"])
    dist = torch.distributions.Categorical(logits=logits)
    score = returns - value.detach() if baseline else returns
    score = (score - score.mean()) / (score.std() + 1e-8)
    value_loss = nn.functional.mse_loss(value, returns) if baseline else torch.zeros(())
    plain_loss = -(dist.log_prob(batch["action"]) * score).mean() + 0.5 * value_loss - cfg.ent_coef * dist.entropy().mean()
    est = trl.plain_like(TD1Estimator(gamma=cfg.gamma, value_network=critic))
    pl, vl, ent = trl.reinforce_losses(actor, critic, est, batch, cfg)
    same(net, plain_loss, pl + 0.5 * vl - cfg.ent_coef * ent)


def test_tiny_dqn_run(tmp_path):
    cfg = TorchRLDQNConfig(total_steps=400, learn_start=100, eval_every=200, eval_users=5)
    _, history = train_dqn_torchrl(cfg, str(tmp_path / "dqn.csv"), str(tmp_path / "dqn.pt"), verbose=False)
    assert len(history) == 2 and (tmp_path / "dqn.pt").exists()
