# Starter Path: a guide agent trained with reinforcement learning

A reinforcement-learning guide that helps a financial beginner in Rwanda find a first
opportunity that fits them. The environment simulates a beginner's first Starter Path
session: the guide shows opportunities, reads the user's reactions, can pay for a scam
check, and commits a starter plan that is judged over the user's first simulated year.

> Simulation only, not financial advice. Real data (inflation, T-bill yield and minimum,
> youth unemployment) is used where it exists; everything else is a labelled assumption.
> See `DATA_SOURCES.md`.

## Run

Repository: https://github.com/Keaane/keane_gaju_rl_summative

```bash
git clone https://github.com/Keaane/keane_gaju_rl_summative.git
cd keane_gaju_rl_summative
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

## How this plugs into Starter Path

The trained guide can sit behind the Starter Path app as a small HTTP service. It is a demo path,
not a production service: no login, local use only, and simulation only, not financial advice.

**The policy file.** `models/best/ppo.pt` (the default) is a PyTorch `state_dict` for `ActorCritic(64)` in
`training/pg_training.py`. The other trained guides are `a2c.pt`, `reinforce.pt` and `dqn.pt`; choose one with
`--agent ppo|a2c|reinforce|dqn`, for example `uv run serve.py --agent dqn`. Any other name is refused.

```bash
uv run serve.py                                   # http://127.0.0.1:8000, PPO
uv run recommend.py --savings 500000 --risk medium --time some --interest savings
# -> First suggestion (PPO): Show Agri input resale (action 0).
```

**The JSON contract.**

| Call | Request | Reply |
|---|---|---|
| `GET /health` | | `{"status": "ok", "model": "models/best/ppo.pt", "obs_dim": 62, "n_actions": 15}` |
| `POST /next_action` | `{"observation": [62 numbers]}` | `{"action": 9, "label": "Show T-bills", "valid": true, "masked": false}` |

Errors reply `{"error": "..."}` with 400 (not JSON, wrong length, not a number, or a value outside its range),
404 (unknown path), 411 (no `Content-Length`) or 413 (body over 10 KB). Slow clients time out after 5 seconds.

Actions: `0-9` show an opportunity (0 Agri input resale, 1 Crop co-op farming, 2 Local reselling, 3 Import & resale,
4 Mobile money agent, 5 Delivery / moto service, 6 Freelance setup, 7 Online trading group, 8 SACCO savings,
9 T-bills); `10-12` commit a small / medium / large plan (10%, 25%, 50% of savings) in the option shown;
`13` check the option shown for scams; `14` end the session without a plan.

**Masking (a deployment choice).** Before replying, the service removes actions the simulator would reject:
a scam check or a commit when nothing has been shown yet, and a commit whose size is below the shown
option's minimum ticket. It returns the guide's best remaining action. `"masked": true` means the guide's
own first choice was not allowed and was replaced. The trained agents were evaluated without this mask
(an invalid action just cost a small penalty), so the reported scores do not depend on it.

**Which app events fill which observation entries.** The app keeps the 62 numbers for the session, updates
them after every event, and calls `/next_action` again for the guide's next move.

| App event | Entries | How to fill them |
|---|---|---|
| Intake form: savings | 0 | `(ln(savings) - ln(100000)) / (ln(2000000) - ln(100000))`, clipped to 0-1 |
| Intake form: risk comfort | 1 | low 0, medium 0.5, high 1 |
| Intake form: time per week | 2 | little 0, some 0.5, a lot 1 |
| Intake form: interest | 3-8 | one-hot: Agri & supplies, Reselling & importing, Services, Digital work, Savings, not sure yet |
| Each guide turn | 9 | turn number / 12 |
| Engagement | 10 | 0-1; the app's own estimate (for example from dwell time or skips). At the start: 7/9 (assumption) |
| An opportunity card is shown | 11, 52-61 | 11: options shown so far / 10; 52-61: one-hot of the card on screen |
| Card evidence: trend and testimonials | 22-31, 32-41 | -1 to 1 per opportunity. Synthetic in the simulator; a real app needs its own signals on this scale |
| Reaction taps on a card | 12-21 | not shown 0, declines 1/3, hesitates 2/3, accepts 1 |
| Scam-check button result | 42-51 | looks legit +1, flagged -1, not checked 0 |

`recommend.py` builds the first observation from the intake (`first_observation()` in the file), and a test
checks that it matches the simulator's own first observation entry for entry (except engagement).

**Example: one call with curl.** With `uv run serve.py` running in another terminal:

```bash
uv run recommend.py --savings 500000 --risk medium --time some --interest savings --json \
  | curl -s -X POST http://127.0.0.1:8000/next_action -H "Content-Type: application/json" -d @-
# {"action": 0, "label": "Show Agri input resale", "valid": true, "masked": false}

curl -s http://127.0.0.1:8000/health
```
