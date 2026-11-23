"""Turn an intake form into the trained guide's first suggestion, in plain words.

  uv run recommend.py --savings 500000 --risk medium --time some --interest savings
  uv run recommend.py --savings 500000 --risk medium --time some --interest savings --json   # request body for serve.py

The guide is a trained policy (default: PPO from models/best/). It sees the same 62 numbers the simulator
gives it (see the README section "How this plugs into Starter Path"). Before answering, invalid actions are
masked out using the simulator's own rules, so the reply is always something the guide is allowed to do.
Simulation only, not financial advice.
"""
import argparse
import json
import math
import os
import sys
import warnings

import numpy as np

from environment.custom_env import COMMIT, END, N_ACTIONS, OBS_DIM, SIZES, VERIFY
from environment.world_data import AREAS, MIN_TICKET, OPP_NAMES, SAVINGS_MAX, SAVINGS_MIN

AGENTS = ("ppo", "a2c", "reinforce", "dqn")
RISK = {"low": 0.0, "medium": 0.5, "high": 1.0}
TIME = {"little": 0.0, "some": 0.5, "a lot": 1.0}
INTEREST = {a.lower(): i for i, a in enumerate(AREAS)}             # the 5 areas; index 5 = not sure yet
INTEREST.update({"agri": 0, "reselling": 1, "services": 2, "digital": 3, "savings": 4, "not sure": 5})
START_ENGAGEMENT = 7 / 9   # ASSUMPTION: the app has no engagement estimate yet; midpoint of the simulator's start range

# Where each part of the observation lives (62 numbers in total).
FOCUS = slice(52, 62)      # one-hot of the opportunity currently shown


# ---- intake -> observation ------------------------------------------------------------------
def first_observation(savings, risk, time, interest):
    """The 62 numbers at the start of a session, built from the intake form alone."""
    if not SAVINGS_MIN <= savings <= SAVINGS_MAX:
        warnings.warn(f"savings {savings:,.0f} RWF is outside the simulated range "
                      f"{SAVINGS_MIN:,.0f}-{SAVINGS_MAX:,.0f}; using the nearest end")
    s = min(max(float(savings), SAVINGS_MIN), SAVINGS_MAX)
    k = INTEREST[interest.lower()] if isinstance(interest, str) else int(interest)
    lo, hi = math.log(SAVINGS_MIN), math.log(SAVINGS_MAX)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = (math.log(s) - lo) / (hi - lo)                       # savings, log scale 0-1
    obs[1] = RISK[risk] if isinstance(risk, str) else float(risk)   # stated risk comfort
    obs[2] = TIME[time] if isinstance(time, str) else float(time)   # time per week
    obs[3 + k] = 1.0                                                # interest one-hot (6 slots)
    obs[9] = 0.0                                                    # step 0 of 12
    obs[10] = START_ENGAGEMENT                                      # engagement estimate
    return obs                                                      # nothing shown, no reactions, no checks yet


def savings_from(obs):
    """Undo the log scaling of entry 0."""
    lo, hi = math.log(SAVINGS_MIN), math.log(SAVINGS_MAX)
    return math.exp(lo + float(obs[0]) * (hi - lo))


# ---- which actions are allowed (the simulator's own rules) -----------------------------------
def valid_actions(obs):
    """True/False per action. Showing an option or ending is always allowed. A scam check needs something
    shown. A commit needs something shown and a plan size that meets that opportunity's minimum ticket."""
    obs = np.asarray(obs, dtype=float)
    ok = np.ones(N_ACTIONS, dtype=bool)
    shown = np.flatnonzero(obs[FOCUS] > 0.5)
    if shown.size == 0:
        ok[list(COMMIT)] = False
        ok[VERIFY] = False
        return ok
    focus, savings = int(shown[0]), savings_from(obs)
    for a in COMMIT:
        ok[a] = SIZES[a - COMMIT.start] * savings >= MIN_TICKET[focus]
    return ok


def best_valid(scores, obs):
    """Highest-scoring allowed action. Also says whether the guide's own top choice was allowed."""
    scores = np.asarray(scores, dtype=float)
    ok = valid_actions(obs)
    action = int(np.argmax(np.where(ok, scores, -np.inf)))
    return action, bool(ok[int(np.argmax(scores))])


def action_label(a):
    a = int(a)
    if 0 <= a < len(OPP_NAMES):
        return f"Show {OPP_NAMES[a]}"
    if a in COMMIT:
        return f"Commit a {('small', 'medium', 'large')[a - COMMIT.start]} plan"
    if a == VERIFY:
        return "Check the current option for scams"
    if a == END:
        return "End the session without a plan"
    raise ValueError(f"no action {a}")


# ---- the trained guide ----------------------------------------------------------------------
def load_scores(agent="ppo", models_dir="models/best"):
    """A function obs -> one score per action (policy logits, or Q-values for DQN)."""
    if agent not in AGENTS:
        raise SystemExit(f"unknown agent '{agent}': choose one of {', '.join(AGENTS)}")
    path = os.path.join(models_dir, f"{agent}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"no trained model at {path}")
    import torch
    from training.common import mlp
    from training.pg_training import ActorCritic
    net = mlp([OBS_DIM, 64, 64, N_ACTIONS]) if agent == "dqn" else ActorCritic(64)
    net.load_state_dict(torch.load(path))
    net.eval()

    def scores(obs):
        with torch.no_grad():
            out = net(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
            return (out if agent == "dqn" else out[0]).numpy()
    scores.path = path
    return scores


def suggest(scores, obs):
    """The reply both recommend.py and serve.py give: action, label, valid, and whether masking changed it."""
    action, own_choice_ok = best_valid(scores(obs), obs)
    return {"action": action, "label": action_label(action), "valid": True, "masked": not own_choice_ok}


def main(argv=None):
    p = argparse.ArgumentParser(description="The trained guide's first suggestion for one intake form.")
    p.add_argument("--savings", type=float, required=True, help="savings in RWF")
    p.add_argument("--risk", choices=list(RISK), required=True)
    p.add_argument("--time", choices=list(TIME), required=True, help="time per week")
    p.add_argument("--interest", choices=sorted(INTEREST), default="not sure")
    p.add_argument("--agent", choices=AGENTS, default="ppo")
    p.add_argument("--json", action="store_true", help="print the request body for serve.py and stop")
    args = p.parse_args(argv)
    obs = first_observation(args.savings, args.risk, args.time, args.interest)
    if args.json:
        print(json.dumps({"observation": [round(float(x), 6) for x in obs]}))
        return
    reply = suggest(load_scores(args.agent), obs)
    print(f"First suggestion ({args.agent.upper()}): {reply['label']} (action {reply['action']}).")
    print("Simulation only, not financial advice.")


if __name__ == "__main__":
    sys.exit(main())
