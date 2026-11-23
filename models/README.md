# Trained models

One final model per algorithm, all trained for **200,000 steps**. `main.py` loads them from `models/best/`.
`models/dqn/` and `models/pg/` hold identical copies in the folder layout the brief asks for.
`models/seeds/` holds all 3 seeds of each chosen 200,000-step setting (`<algo>_seed<N>.pt`) for the final evaluation (`analysis/final_eval.py`).

| File (and copy) | Algorithm | Setting | Seed | Validation (best) | Test, 300 users (+/- SE) |
|---|---|---|---|---|---|
| `best/ppo.pt` (`pg/ppo.pt`) | PPO | run 5: 16 epochs | 0 | +0.84 | +0.69 +/- 0.11 |
| `best/a2c.pt` (`pg/a2c.pt`) | A2C | run 3: update every 8 steps | 2 | +0.22 | +0.17 +/- 0.02 |
| `best/reinforce.pt` (`pg/reinforce.pt`) | REINFORCE | run 9: entropy 0.05 | 1 | +0.20 | +0.18 +/- 0.02 |
| `best/dqn.pt` (`dqn/dqn.pt`) | DQN | run 1: learning rate 3e-4 | 0 | +0.84 | +0.39 +/- 0.12 |

Each file is a PyTorch `state_dict`. PPO, A2C and REINFORCE use `ActorCritic(64)` from `training/pg_training.py`.
DQN uses `mlp([62, 64, 64, 15])` from `training/common.py`. Rewards are relative to putting everything in T-bills.
Comparison lines on the same test users: best profile rule +0.12, oracle +0.94, random -1.77.

## How they were chosen

- **Validation users only** (seeds 200000+). Test users (seeds 100000+) played no part in any choice.
  The test column is for reporting only.
- **The setting:** the 3 best settings of each algorithm in the 100,000-step sweep (by validation) were rerun at
  200,000 steps with 3 seeds (`logs/sweeps_200k/`). The final setting is the one with the best mean validation
  score over its 3 seeds. Settings within 0.02 of the top count as tied, and then the one with the smallest
  spread across seeds wins. One run's validation score is uncertain by about +/- 0.10.
- **The seed:** within that setting, the seed with the best validation score. The file is its best-validation checkpoint.
- **The budget:** 200,000 steps for all four algorithms, decided in advance so that every algorithm gets the same budget.
  Validation scores were never compared across budgets.

## Read these numbers with care

- **Longer runs flatter validation.** A 200,000-step run checks validation 20 times and keeps the best checkpoint,
  where a 100,000-step run checks 10 times. More tries give a luckier best, so validation scores from the two budgets are not comparable.
- **Best of 3 seeds.** Each file is the best of 3 seeds by validation, so its own scores are optimistic. The fair summary
  of an algorithm is the mean over all 3 seeds of the chosen setting (200,000 steps, test): PPO +0.66 +/- 0.02,
  DQN +0.32 +/- 0.06, A2C +0.18 +/- 0.02, REINFORCE +0.16 +/- 0.02. The final 500-user evaluation (`analysis/final_eval.py`) reports both.
- **DQN's validation-test gap.** The DQN model scores +0.84 on validation but +0.39 on test. Its best checkpoint fits the
  validation users much more than it improves in general.
- **The REINFORCE choice is close to arbitrary.** Its top settings are within 0.02 of each other on validation (at 100,000
  steps, 7 of 10 settings were). Every REINFORCE setting ends on the "safe plan" and scores about +0.15 to +0.18 on test.
  A2C behaves the same way.
- **The DQN and A2C models are the same at both budgets.** Their 100,000-step and 200,000-step checkpoints are byte-identical.
  With the same seed, a 200,000-step run repeats the first 100,000 steps exactly. The best validation came at 90,000 steps (DQN)
  and 60,000 steps (A2C), and the extra steps never beat it. This also shows the training is reproducible.

## For reference: the 100,000-step choices (same rule)

Not used as final models. They are listed so the two budgets can be compared.

| Algorithm | Setting | Seed | Validation (best) | Test, 300 users (+/- SE) |
|---|---|---|---|---|
| PPO | run 5: 16 epochs (tied with run 8, smaller spread) | 0 | +0.63 | +0.48 +/- 0.10 |
| A2C | run 3: update every 8 steps | 2 | +0.22 | +0.17 +/- 0.02 (same file as the final model) |
| REINFORCE | run 5: update after 4 sessions (7 settings tied) | 1 | +0.16 | +0.15 +/- 0.02 |
| DQN | run 1: learning rate 3e-4 | 0 | +0.84 | +0.39 +/- 0.12 (same file as the final model) |

Simulation only, not financial advice.
