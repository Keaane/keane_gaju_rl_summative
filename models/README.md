# Trained models

The final models are the **TorchRL** versions, one per algorithm, all trained for **200,000 steps**.
`main.py`, `serve.py` and `recommend.py` load them from `models/best/`. `models/dqn/` and `models/pg/` hold
identical copies in the folder layout the brief asks for. `models/seeds/` holds all 3 seeds of each chosen
200,000-step setting (`<algo>_seed<N>.pt`) for the final evaluation (`analysis/final_eval.py --impl torchrl`).
The choices are recorded in `logs/torchrl/chosen.json`, and these files were copied from it.

| File (and copy) | Algorithm | Setting | Seed | Validation (best) | Test, 500 users (+/- SE) | Mean of 3 seeds, 500 users |
|---|---|---|---|---|---|---|
| `best/ppo.pt` (`pg/ppo.pt`) | PPO | run 5: 16 epochs | 2 | +0.79 | +0.64 +/- 0.09 | +0.57 (spread 0.05) |
| `best/a2c.pt` (`pg/a2c.pt`) | A2C | run 3: update every 8 steps | 0 | +0.22 | +0.18 +/- 0.01 | +0.17 (spread 0.01) |
| `best/reinforce.pt` (`pg/reinforce.pt`) | REINFORCE | run 2: learning rate 3e-3 | 0 | +0.17 | +0.15 +/- 0.01 | +0.15 (spread 0.00) |
| `best/dqn.pt` (`dqn/dqn.pt`) | DQN | run 1: learning rate 3e-4 | 2 | +0.83 | +0.38 +/- 0.09 | +0.32 (spread 0.08) |

Each file is a PyTorch `state_dict` (the TorchRL trainers wrap the same networks as the plain code, so the files
have the same format). PPO, A2C and REINFORCE use `ActorCritic(64)` from `training/pg_training.py`; DQN uses
`mlp([62, 64, 64, 15])` from `training/common.py`. `tests/test_models.py` checks that every file loads strictly.
Rewards are relative to putting everything in T-bills. Comparison lines on the same 500 test users: best profile
rule +0.05, oracle +0.98, random -1.79.

## The plain PyTorch versions

`models/plain_pytorch/` holds the earlier plain PyTorch models in the same layout (`best/`, `dqn/`, `pg/`,
`seeds/`), moved here unchanged (`tests/test_models.py`). They are kept as a cross-check: on the same 500 test
users, each TorchRL algorithm is within 0.15 of its plain version (PPO -0.03, DQN +0.01, A2C -0.01,
REINFORCE -0.02, 3-seed means). Run them with, for example, `uv run main.py --agent ppo --model
models/plain_pytorch/best/ppo.pt`. Their choices are in `analysis/final_eval.py` (`CHOSEN`): PPO run 10 seed 2,
A2C run 3 seed 2, REINFORCE run 9 seed 1, DQN run 1 seed 0.

## How they were chosen

- **Validation users only** (seeds 200000+). Test users (seeds 100000+) played no part in any choice. The test
  columns are for reporting only.
- **The setting:** the best setting of each algorithm in the 100,000-step TorchRL sweep (best mean validation
  over 3 seeds; settings within 0.02 of the top count as tied, and then the smallest spread across seeds wins) was
  rerun at 200,000 steps with 3 seeds (`logs/torchrl/sweeps_200k/`). When the top two settings were within 0.02
  of each other, both were rerun and the same rule picked between them at 200,000 steps (A2C: runs 3 and 2 ->
  run 3; REINFORCE: runs 9 and 2 -> run 2). The plain versions reran their top three settings instead.
- **The seed:** within that setting, the seed with the best validation score. The file is its best-validation
  checkpoint.
- **The budget:** 200,000 steps for all four algorithms, decided in advance. Validation scores were never compared
  across budgets.
- Every choice is logged in `logs/torchrl/selection.log`.

## Read these numbers with care

- **Longer runs flatter validation.** A 200,000-step run checks validation 20 times and keeps the best checkpoint,
  where a 100,000-step run checks 10 times. More tries give a luckier best, so validation scores from the two
  budgets are not comparable.
- **Best of 3 seeds.** Each file is the best of 3 seeds by validation, so its own scores are optimistic. The fair
  summary of an algorithm is the mean over all 3 seeds (last column).
- **DQN's validation-test gap.** The DQN model scores +0.83 on validation but +0.38 on test. Its best checkpoint
  fits the validation users much more than it improves in general.
- **Close calls.** PPO run 5 beat run 10 at 100,000 steps by 0.022, just outside the tie limit (the plain sweep
  ranked them the other way). REINFORCE runs 9 and 2 were tied at 200,000 steps and run 2 won on spread. Apart from the too-low learning rates, every
  REINFORCE and A2C setting ends on the "safe plan" (SACCO savings) and scores about +0.15 to +0.19 on test, so
  their choices matter little.

Simulation only, not financial advice.
