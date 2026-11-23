# Starter Path: a guide agent trained with reinforcement learning

A reinforcement-learning guide that helps a financial beginner in Rwanda find a first
opportunity that fits them. The environment simulates a beginner's first Starter Path
session: the guide shows opportunities, reads the user's reactions, can pay for a scam
check, and commits a starter plan that is judged over the user's first simulated year.

> Simulation only, not financial advice. Real data (inflation, T-bill yield and minimum,
> youth unemployment) is used where it exists; everything else is a labelled assumption.
> See `DATA_SOURCES.md`.

## Run

```bash
uv sync
uv run main.py                       # best PPO agent walking the street scene (pygame window)
uv run main.py --no-gui              # text transcript instead of a window
uv run main.py --view dashboard      # list dashboard instead of the street scene
uv run main.py --gif assets/session.gif --agent adaptive   # save the walking animation
uv run main.py --agent adaptive      # ppo | a2c | reinforce | dqn | adaptive | rule | oracle | random
uv run main.py --screenshot assets/session.png --agent adaptive --seed 100249
uv run pytest                        # tests
```

## Train

```bash
uv run python -m training.dqn_training --seed 0
uv run python -m training.pg_training --algo ppo --seed 0       # ppo | a2c | reinforce
uv run python -m training.reference                             # baseline scores on the test users
uv run python -m training.sweep --algo ppo --seeds 3 --workers 4   # 10 settings x 3 seeds
uv run python -m training.sweep --table ppo                     # summary table
```

Simulated users are split three ways: random training users, validation users (pick the best
checkpoint) and test users (final score, identical for every algorithm). Curves go to `logs/`.

## Layout

```
main.py                     run an agent with the dashboard
environment/custom_env.py   StarterGuideEnv (Gymnasium environment)
environment/world_data.py   every number, tagged REAL or SYNTHETIC
environment/baselines.py    random, profile rule, hand-written adaptive guide, oracle
environment/scene.py        animated street scene (default view)
environment/rendering.py    list dashboard, fonts and colours
training/common.py          seeds, networks, evaluation, logging
training/dqn_training.py    DQN
training/pg_training.py     REINFORCE, A2C, PPO
training/sweep.py           hyperparameter sweeps
analysis/                   feasibility checks and a learnability probe
assets/                     Poppins font, street-scene picture, walkthrough GIF
models/best/                trained agent used by main.py
logs/  tests/
```

## Environment in brief

- **Actions (15):** show an opportunity (10), commit a small / medium / large plan (3),
  check for scams (1), end the session (1).
- **Guide sees:** the intake (savings, stated risk, time, interest), the user's reactions,
  noisy trend and testimonial signals, scam-check results, an engagement estimate.
- **Hidden:** the user's true risk tolerance, interests and patience; which offers are scams.
- **Reward:** the plan's simulated first-year result versus all-in T-bills, with penalties for
  scams, panic exits and users who decline or quit, plus small signals from reactions.
- **Session ends:** when a plan is committed, the user quits, or after 12 steps.
