"""Reference guides for StarterGuideEnv. Each takes the env and returns an action.

  random_policy   random actions
  StaticRule      uses ONLY the intake (profile). Shows one option, then commits.
  adaptive        hand-written: shows a few options, reads reactions, verifies scams
  Oracle          cheats by seeing hidden traits and scams: an upper bound
"""
import itertools
from dataclasses import dataclass

import numpy as np

from environment.custom_env import ADOPT_P, COMMIT, N_OPP, NONE_INTEREST, SIZES, VERIFY
from environment.world_data import AREA_OF, MIN_TICKET, MU, SACCO, SCAM_P, SIGMA, TBILLS, TIME_REQ


def random_policy(env):
    return int(env.np_random.integers(env.action_space.n))


def appeal(env, o):
    """Return of an opportunity, shaded down if it needs more time than the user has."""
    return MU[o] * (1 - 0.5 * max(0.0, TIME_REQ[o] - env.time))


# ---- Static rule: profile in, plan out -------------------------------------
@dataclass(frozen=True)
class StaticRule:
    size_idx: int = 1           # 0 small, 1 medium, 2 large
    mode: str = "interest"      # "safe" | "return" | "interest"
    risk_shift: float = 0.0     # trust the stated risk, or shade it up or down
    skip_scammy: bool = True    # skip opportunities with a high public scam rate

    def plan(self, env):
        size = SIZES[self.size_idx]
        risk = float(np.clip(env.stated_risk + self.risk_shift, 0, 1))
        options = [
            o for o in range(N_OPP)
            if MIN_TICKET[o] <= size * env.savings
            and SIGMA[o] <= 0.05 + 0.5 * risk
            and TIME_REQ[o] <= env.time + 0.3
            and not (self.skip_scammy and SCAM_P[o] >= 0.25)
        ] or [SACCO]
        if self.mode == "safe":
            options = [o for o in options if AREA_OF[o] == 4] or [SACCO]
        elif self.mode == "interest" and env.stated_interest != NONE_INTEREST:
            options = [o for o in options if AREA_OF[o] == env.stated_interest] or options
        return max(options, key=lambda o: appeal(env, o)), self.size_idx


STATIC_GRID = [StaticRule(*p) for p in itertools.product((0, 1, 2), ("safe", "return", "interest"), (-0.3, 0.0, 0.3), (True, False))]


def static_policy(rule):
    def act(env):
        opp, size = rule.plan(env)
        if env.focus != opp:
            return opp                          # show it first (sets the focus)
        return COMMIT.start + size
    return act


# ---- Adaptive hand-written guide --------------------------------------------
def adaptive(env):
    small, medium = SIZES[0] * env.savings, SIZES[1] * env.savings
    affordable = [o for o in range(N_OPP) if MIN_TICKET[o] <= medium and TIME_REQ[o] <= env.time + 0.3]

    def prior(o):
        s = appeal(env, o) - 0.6 * max(0.0, SIGMA[o] / 0.5 - env.stated_risk)
        return s + (0.15 if AREA_OF[o] == env.stated_interest else 0.0)

    shortlist = sorted([o for o in affordable if AREA_OF[o] != 4], key=prior, reverse=True)[:3]
    seen = [o for o in range(N_OPP) if env.reaction[o] > 0]

    for o in shortlist:                         # 1. show a few options while the user is engaged
        if o not in seen and len(seen) < 3 and env.patience_est > 2.5:
            return o

    candidates = [o for o in seen if env.verified[o] >= 0 and AREA_OF[o] != 4]   # 2. drop flagged scams
    if not candidates or max(env.reaction[o] for o in candidates) == 1:
        safe = TBILLS if MIN_TICKET[TBILLS] <= small else SACCO
        return safe if env.focus != safe else COMMIT.start
    best = max(candidates, key=lambda o: env.reaction[o] + 0.8 * env.trend_obs[o] + 0.2 * env.testi_obs[o])

    if SCAM_P[best] > 0 and env.verified[best] == 0 and env.patience_est > 1.0:    # 3. check for fraud
        return VERIFY if env.focus == best else best
    if env.focus != best:
        return best
    want = 1 if env.reaction[best] == 3 and not (env.stated_risk < 0.5 and SIGMA[best] > 0.2) else 0
    size = next(i for i in range(want, 3) if SIZES[i] * env.savings >= MIN_TICKET[best])
    return COMMIT.start + size


# ---- Oracle: knows the hidden truth -----------------------------------------
class Oracle:
    """Picks the plan with the best expected reward, counting whether the user will accept."""

    def __init__(self, rollouts=20):
        self.rollouts, self.plan = rollouts, None

    def best_plan(self, env):
        best, best_value = (SACCO, 0), -1e9
        for o in range(N_OPP):
            p_adopt = np.mean([ADOPT_P[env._react(o)] for _ in range(8)])
            for i, size in enumerate(SIZES):
                if size * env.savings < MIN_TICKET[o]:
                    continue
                rng = np.random.default_rng(17)         # same random draws for every plan
                good = np.mean([env.plan_reward(o, size, rng) for _ in range(self.rollouts)])
                value = p_adopt * good + (1 - p_adopt) * -0.5
                if value > best_value:
                    best, best_value = (o, i), value
        return best

    def __call__(self, env):
        if env.step_count == 0:
            self.plan = self.best_plan(env)
        opp, size = self.plan
        return opp if env.focus != opp else COMMIT.start + size
