"""The saved models load strictly into the right networks, and models/best/ holds the chosen TorchRL seeds."""
import filecmp
import json

import pytest
import torch

from environment.custom_env import N_ACTIONS, OBS_DIM
from training.common import mlp
from training.pg_training import ActorCritic

ALGOS = ("dqn", "ppo", "a2c", "reinforce")
CHOSEN = json.load(open("logs/torchrl/chosen.json"))      # the TorchRL final setting and seed per algorithm


def network(algo):
    return mlp([OBS_DIM, 64, 64, N_ACTIONS]) if algo == "dqn" else ActorCritic(64)


@pytest.mark.parametrize("folder", ["models", "models/plain_pytorch"])
@pytest.mark.parametrize("algo", ALGOS)
def test_every_model_loads_strictly(folder, algo):
    """Every key present and no extra keys (strict=True), for the demo model and all 3 seeds."""
    for path in [f"{folder}/best/{algo}.pt"] + [f"{folder}/seeds/{algo}_seed{s}.pt" for s in range(3)]:
        network(algo).load_state_dict(torch.load(path), strict=True)


@pytest.mark.parametrize("algo", ALGOS)
def test_best_is_the_chosen_torchrl_seed(algo):
    seed = CHOSEN[algo]["seed"]
    copy = "models/dqn/dqn.pt" if algo == "dqn" else f"models/pg/{algo}.pt"
    assert filecmp.cmp(f"models/best/{algo}.pt", f"models/seeds/{algo}_seed{seed}.pt", shallow=False)
    assert filecmp.cmp(f"models/best/{algo}.pt", copy, shallow=False)


def test_plain_models_were_moved_not_changed():
    """The plain PyTorch demo models are still the seeds named in analysis/final_eval.py CHOSEN."""
    from analysis.final_eval import CHOSEN as PLAIN
    for algo, (_, seed) in PLAIN.items():
        assert filecmp.cmp(f"models/plain_pytorch/best/{algo}.pt", f"models/plain_pytorch/seeds/{algo}_seed{seed}.pt",
                           shallow=False)
