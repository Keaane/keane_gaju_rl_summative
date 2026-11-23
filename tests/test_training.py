import pytest

from training.dqn_training import DQNConfig, train_dqn
from training.pg_training import A2CConfig, PPOConfig, ReinforceConfig, train_a2c, train_ppo, train_reinforce

SMALL = dict(eval_users=5)


def check(tmp_path, name, history, n_rows):
    assert len(history) == n_rows
    assert (tmp_path / f"{name}.csv").exists() and (tmp_path / f"{name}.pt").exists()


def test_dqn(tmp_path):
    cfg = DQNConfig(total_steps=600, learn_start=100, eval_every=300, buffer_size=1000, **SMALL)
    _, history = train_dqn(cfg, str(tmp_path / "dqn.csv"), str(tmp_path / "dqn.pt"), verbose=False)
    check(tmp_path, "dqn", history, 2)


def test_ppo(tmp_path):
    cfg = PPOConfig(total_steps=512, rollout_steps=256, eval_every=256, epochs=2, minibatch=64, **SMALL)
    _, history = train_ppo(cfg, str(tmp_path / "ppo.csv"), str(tmp_path / "ppo.pt"), verbose=False)
    check(tmp_path, "ppo", history, 2)


def test_a2c(tmp_path):
    cfg = A2CConfig(total_steps=256, n_steps=32, eval_every=128, **SMALL)
    _, history = train_a2c(cfg, str(tmp_path / "a2c.csv"), str(tmp_path / "a2c.pt"), verbose=False)
    check(tmp_path, "a2c", history, 2)


@pytest.mark.parametrize("baseline", [True, False])
def test_reinforce(tmp_path, baseline):
    cfg = ReinforceConfig(total_steps=200, episodes_per_update=4, eval_every=100, baseline=baseline, **SMALL)
    _, history = train_reinforce(cfg, str(tmp_path / "rf.csv"), str(tmp_path / "rf.pt"), verbose=False)
    assert len(history) >= 1 and (tmp_path / "rf.pt").exists()
