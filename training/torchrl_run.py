"""Run the TorchRL versions of the four algorithms (Python 3.11 side environment with TorchRL 0.3.1; see the README).

  # one run, the final setting of an algorithm:
  python -m training.torchrl_run --algo dqn --seed 0
  # a sweep, with the same grids and selection rule as the plain versions:
  python -m training.sweep --impl torchrl --algo dqn --seeds 3 --workers 3

TORCHRL_REGISTRY has the same shape as REGISTRY in training/sweep.py, so the sweep runner can use either.
Logs go to logs/torchrl/ and checkpoints to models/torchrl/; plain results and models are never touched.
"""
import argparse

from training.common import TEST_SEED, evaluate, greedy
from training.sweep import GRIDS, train_settings
from training.torchrl_dqn import TorchRLDQNConfig, train_dqn_torchrl

# algo -> (config class, trainer, how to turn the trained network into a policy); the policy-gradient
# versions are added from training/torchrl_pg.py once each one passes its loss check.
TORCHRL_REGISTRY = {"dqn": (TorchRLDQNConfig, train_dqn_torchrl, greedy)}
try:
    from training.torchrl_pg import TORCHRL_PG
    TORCHRL_REGISTRY.update(TORCHRL_PG)
except ImportError:
    pass

FINAL_SETTING = {"dqn": 1, "ppo": 10, "a2c": 3, "reinforce": 9}   # the plain versions' final settings (models/README.md)


def main():
    p = argparse.ArgumentParser(description="One TorchRL run at an algorithm's final setting.")
    p.add_argument("--algo", choices=sorted(FINAL_SETTING), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    args = p.parse_args()
    if args.algo not in TORCHRL_REGISTRY:
        raise SystemExit(f"no TorchRL version of {args.algo} yet")
    config_class, trainer, make_policy = TORCHRL_REGISTRY[args.algo]
    run = FINAL_SETTING[args.algo]
    cfg = config_class(total_steps=args.steps, seed=args.seed, **train_settings(args.algo, GRIDS[args.algo][run], args.steps))
    tag = f"{args.algo}_run{run:02d}_seed{args.seed}"
    net, history = trainer(cfg, f"logs/torchrl/single/{tag}.csv", f"models/torchrl/single/{tag}.pt")
    test = evaluate(make_policy(net), 300, TEST_SEED)
    print(f"{tag}: best validation {max(h['reward'] for h in history):+.2f}, test (300 users) {test['reward']:+.2f}")


if __name__ == "__main__":
    main()
