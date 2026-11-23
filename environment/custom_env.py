"""StarterGuideEnv: a beginner's first Starter Path session.

The agent is the GUIDE. The user is a simulated beginner who has filled in a short
intake (savings, risk comfort, interest, time per week) that the guide can see.
People do not know themselves perfectly, so the user also has HIDDEN traits that
only show through reactions: true risk tolerance, true interests, and patience.
The market has hidden state too: which opportunities are scams, how good each is.

A session ends when the guide commits a starter plan. The plan is then judged by
simulating the user's first 52 weeks, relative to putting everything in T-bills.
Numbers come from environment/world_data.py (REAL vs SYNTHETIC is labelled there).
This is a simulation, not financial advice.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.world_data import (
    AREAS, AREA_OF, OPP_NAMES, IDLE_RETURN, INFLATION, MIN_TICKET, MU, N_OPP, SACCO, SACCO_RETURN,
    SAVINGS_MAX, SAVINGS_MIN, SCAM_P, SIGMA, TBILL_RETURN, TIME_REQ,
)

# ---- Actions ----------------------------------------------------------------
EXPLORE = range(N_OPP)       # 0-9   show opportunity k with its evidence (sets the focus)
COMMIT = range(10, 13)       # 10-12 commit the focus as a small / medium / large plan
VERIFY = 13                  # pay for a scam check on the focus
END = 14                     # end the session without a plan
N_ACTIONS = 15
SIZES = (0.10, 0.25, 0.50)   # share of savings

# ---- Session rules ----------------------------------------------------------
MAX_STEPS = 12
HORIZON = 52                 # weeks simulated after the plan is made
ENTRY_FEE = 0.01
VERIFY_FEE = 0.005           # of savings, per check
VERIFY_ACCURACY = 0.85
NONE_INTEREST = 5            # "no interest yet"
REACTION_REWARD = {3: 0.2, 2: 0.0, 1: -0.1}      # small signal when the user reacts (fix 2)
ADOPT_P = {0: 0.70, 1: 0.15, 2: 0.60, 3: 0.95}   # chance the user follows a plan, by reaction
REWARD_SCALE = 20.0
QUIT_REWARD = -2.5
OBS_DIM = 62
REACTION_WORD = {0: "not shown", 1: "declines", 2: "hesitates", 3: "accepts"}
SIZE_NAMES = ("small", "medium", "large")


class StarterGuideEnv(gym.Env):
    """Observation (62 floats): intake (savings, risk, time, interest one-hot), step,
    engagement estimate, number shown, then per opportunity: reaction, trend signal,
    testimonial score, scam-check result, and a one-hot of the current focus."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, scam_boost=0.0, p_undecided=0.35, render_mode=None, render_fps=2, render_font="poppins", render_view="scene"):
        super().__init__()
        self.render_mode, self.render_fps, self.render_font, self.render_view, self._renderer = (
            render_mode, render_fps, render_font, render_view, None)
        self.scam_boost = scam_boost
        self.p_undecided = p_undecided
        self.action_space = spaces.Discrete(N_ACTIONS)
        low, high = np.zeros(OBS_DIM, np.float32), np.ones(OBS_DIM, np.float32)
        low[22:52] = -1.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # ---- episode setup ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        o, rng = options or {}, self.np_random

        # Visible intake
        self.savings = float(o.get("savings", np.exp(rng.uniform(math.log(SAVINGS_MIN), math.log(SAVINGS_MAX)))))
        self.stated_risk = float(o.get("stated_risk", rng.choice((0.0, 0.5, 1.0))))
        self.time = float(o.get("time", rng.choice((0.0, 0.5, 1.0))))
        undecided = o.get("undecided", rng.random() < o.get("p_undecided", self.p_undecided))

        # Hidden user traits: people misjudge themselves
        self.true_risk = float(np.clip(self.stated_risk + rng.normal(-0.1, 0.3), 0, 1))
        self.aff = rng.random(5)                       # how much they would enjoy each area
        self.stated_interest = NONE_INTEREST
        if not undecided:
            k = int(rng.integers(0, 5))
            self.stated_interest = k
            genuine = rng.random() < 0.6               # else the interest is only aspirational
            self.aff[k] = min(1.0, max(self.aff.max(), 0.7) + 0.1) if genuine else rng.random() * 0.5
        self.patience = float(rng.uniform(5.0, 9.0))

        # Hidden market state
        boost = o.get("scam_boost", self.scam_boost)
        scam_p = np.where(AREA_OF < 4, np.clip(SCAM_P + boost, 0, 0.95), 0.0)
        self.scam = rng.random(N_OPP) < scam_p
        self.collapse = rng.integers(2, 40, size=N_OPP)           # week a scam collapses
        self.market = rng.normal(size=N_OPP)                      # how good each is this month
        self.quality = np.clip((MU * (1 + 0.4 * self.market) - 0.10) * 3.0, -1, 1)

        # What the guide has seen so far
        self.step_count = 0
        self.focus = -1                                           # nothing shown yet
        self.reaction = np.zeros(N_OPP, dtype=int)                # 0 unseen, 1 decline, 2 hesitate, 3 accept
        self.trend_obs, self.testi_obs, self.verified = np.zeros(N_OPP), np.zeros(N_OPP), np.zeros(N_OPP)
        self.n_explored = 0
        self.verify_spend = 0.0
        self.last_text, self.last_reward, self.final, self.last_event = "Session started", 0.0, None, "start"
        self._update_engagement()
        return self._obs(), self._info("start")

    # ---- one guide action ---------------------------------------------------
    def step(self, action):
        result = self._step(action)
        self.last_reward = result[1]
        self.last_text = self._describe(int(action), result[4]["outcome"])
        self.last_event = self._event(int(action), result[4]["outcome"])
        return result

    def _step(self, action):
        action, rng = int(action), self.np_random
        self.step_count += 1
        reward = 0.0

        if action in EXPLORE:
            k = action
            if self.reaction[k] == 0:                              # first time: costs patience, gives evidence
                self.patience -= 1.0 + 0.25 * self.n_explored      # each extra option overloads more
                self.n_explored += 1
                self.trend_obs[k] = np.clip((self.market[k] + rng.normal(0, 0.7)) / 3, -1, 1)
                apparent = 0.8 if self.scam[k] else self.quality[k]   # scam testimonials look great
                self.testi_obs[k] = np.clip(apparent + rng.normal(0, 0.35), -1, 1)
                self.reaction[k] = self._react(k)
                reward += REACTION_REWARD[self.reaction[k]]
            else:
                self.patience -= 0.2                               # revisiting is cheap
            self.focus = k
        elif action == VERIFY:
            if self.focus < 0:
                return self._invalid()
            self.patience -= 0.4
            self.verify_spend += VERIFY_FEE
            truth = -1.0 if self.scam[self.focus] else 1.0
            self.verified[self.focus] = truth if rng.random() < VERIFY_ACCURACY else -truth
        elif action == END:
            return self._finish(0.0, "no_plan")
        else:                                                      # commit
            size = SIZES[action - COMMIT.start]
            if self.focus < 0 or size * self.savings < MIN_TICKET[self.focus]:
                return self._invalid()                             # must show something affordable first (fix 1)
            return self._finish(size, "plan")

        if self.patience <= 0:
            return self._obs(), QUIT_REWARD, True, False, self._info("quit")
        if self.step_count >= MAX_STEPS:
            return self._finish(0.0, "timeout", truncated=True)
        self._update_engagement()
        return self._obs(), reward, False, False, self._info("continue")

    def _invalid(self):
        if self.step_count >= MAX_STEPS:
            return self._finish(0.0, "timeout", truncated=True)
        self._update_engagement()
        return self._obs(), -0.1, False, False, self._info("invalid")

    # ---- the user ------------------------------------------------------------
    def _react(self, k):
        """1 decline, 2 hesitate, 3 accept. Driven by the hidden traits."""
        score = 0.7 if AREA_OF[k] == 4 else self.aff[AREA_OF[k]]
        if MIN_TICKET[k] > 0.5 * self.savings:
            score -= 0.5                                           # feels out of reach
        score -= 0.6 * max(0.0, min(1.0, SIGMA[k] / 0.5) - self.true_risk)   # too risky for them
        score -= 0.4 * max(0.0, TIME_REQ[k] - self.time)           # too time-hungry
        score += self.np_random.normal(0, 0.12)
        return 3 if score > 0.55 else 2 if score > 0.30 else 1

    def _simulate(self, k, amount, rng):
        """One year of one opportunity. Returns (final value, panicked, hit by scam)."""
        z = rng.normal(size=HORIZON)
        if self.scam[k]:
            growth = np.cumprod(1.0 + 0.012 + 0.002 * z)           # pays well, then vanishes
            growth[self.collapse[k]:] = 0.0
        else:
            effort = 1.0 if AREA_OF[k] == 4 else 0.5 + 0.5 * self.aff[AREA_OF[k]]   # bored users do worse
            effort *= 1.0 - 0.5 * max(0.0, TIME_REQ[k] - self.time)
            mu = MU[k] * (1 + 0.4 * self.market[k]) * effort
            weekly = np.maximum(mu / 52 + SIGMA[k] / math.sqrt(52) * z, -0.5)
            growth = np.cumprod(1.0 + weekly)
        path = amount * (1 - ENTRY_FEE) * growth
        tolerance = 0.04 + 0.30 * self.true_risk                   # loss the user will sit through
        breach = np.nonzero(path / amount - 1.0 < -tolerance)[0]
        if breach.size:                                            # panic: sells at the first breach
            final = path[breach[0]] * (1 - ENTRY_FEE)
            scammed = bool(self.scam[k] and final <= 1e-9)
            return float(final), not scammed, scammed
        return float(path[-1]), False, False

    def plan_reward(self, k, size, rng, adopt=True):
        """Reward of a plan vs. all-in-T-bills. Also used by the oracle."""
        S, extra = self.savings, 0.0
        self.last_panic = self.last_scammed = False
        if size == 0.0:                                            # no plan: money sits idle
            value, extra = S * (1 + IDLE_RETURN), -0.5
        elif not adopt:                                            # user declined: saves in a SACCO
            value, extra = S * (1 + SACCO_RETURN), -0.5
        else:
            amount = size * S
            final, panic, scammed = self._simulate(k, amount, rng)
            value = (S - amount) * (1 + TBILL_RETURN) + final      # rest of the savings stays in T-bills
            self.last_panic, self.last_scammed = panic, scammed
            extra = -1.5 * (1 - self.true_risk) * panic - 3.0 * scammed
        value -= self.verify_spend * S
        self.last_real_growth = value / (S * (1 + INFLATION)) - 1.0    # did they beat inflation?
        return REWARD_SCALE * (value - S * (1 + TBILL_RETURN)) / S + extra

    def _finish(self, size, outcome, truncated=False):
        rng = self.np_random
        adopt = size > 0 and rng.random() < ADOPT_P[int(self.reaction[self.focus])]
        reward = self.plan_reward(self.focus, size, rng, adopt)
        if size > 0 and not adopt:
            outcome = "not_adopted"
        info = self._info(outcome)
        info.update(chosen=self.focus, size=size, reward=reward, panic=self.last_panic,
                    scammed=self.last_scammed, real_growth=self.last_real_growth)
        self.final = info
        return self._obs(), float(reward), not truncated, truncated, info

    # ---- display ---------------------------------------------------------------
    def _describe(self, action, outcome):
        """One plain-English line about what just happened (for the dashboard and transcripts)."""
        name = OPP_NAMES[self.focus] if self.focus >= 0 else "nothing"
        if outcome == "quit":
            return "User lost patience and left"
        if outcome == "invalid":
            return "Invalid move (show an affordable option first)"
        if action in EXPLORE:
            return f"Showed {OPP_NAMES[action]}: user {REACTION_WORD[self.reaction[action]]}"
        if action == VERIFY:
            return f"Checked {name}: {'looks legit' if self.verified[self.focus] > 0 else 'flagged as scam'}"
        if action == END or outcome in ("timeout", "no_plan"):
            return "Ended the session without a plan"
        text = f"Committed a {SIZE_NAMES[action - COMMIT.start]} plan in {name}"
        return text + (": user declined" if outcome == "not_adopted" else "")

    def _event(self, action, outcome):
        """What kind of thing just happened, so the scene knows what to animate."""
        if outcome in ("quit", "invalid"):
            return outcome
        if action in EXPLORE:
            return "explore"
        if action == VERIFY:
            return "verify"
        return "commit" if action in COMMIT else "end"

    def _renderer_for_view(self):
        if self._renderer is None:
            # pygame is only loaded when something is actually rendered
            if self.render_view == "scene":
                from environment.scene import SceneRenderer as Renderer
            else:
                from environment.rendering import GuideRenderer as Renderer
            self._renderer = Renderer(self.render_mode, self.render_fps, self.render_font)
        return self._renderer

    def render(self):
        if self.render_mode is None:
            return None
        return self._renderer_for_view().draw(self._render_state())

    def render_frames(self):
        """All animation frames of the last step as RGB arrays (scene view, rgb_array mode)."""
        renderer = self._renderer_for_view()
        return renderer.frames(self._render_state()) if hasattr(renderer, "frames") else [renderer.draw(self._render_state())]

    def _render_state(self):
        return {
            "savings": self.savings, "risk": self.stated_risk, "time": self.time,
            "interest": AREAS[self.stated_interest] if self.stated_interest < NONE_INTEREST else "None yet",
            "engagement": self.patience_est / 9.0, "step": self.step_count, "max_steps": MAX_STEPS,
            "reaction": self.reaction.copy(), "verified": self.verified.copy(), "focus": self.focus,
            "names": OPP_NAMES, "areas": [AREAS[a] for a in AREA_OF],
            "last_text": self.last_text, "last_reward": self.last_reward, "final": self.final,
            "event": self.last_event, "trend": self.trend_obs.copy(), "testi": self.testi_obs.copy(),
        }

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---- observation ---------------------------------------------------------
    def _update_engagement(self):
        self.patience_est = float(np.clip(self.patience + self.np_random.normal(0, 0.6), 0, 9.0))

    def _obs(self):
        lo, hi = math.log(SAVINGS_MIN), math.log(SAVINGS_MAX)
        interest, focus = np.zeros(6), np.zeros(N_OPP)
        interest[self.stated_interest] = 1.0
        if self.focus >= 0:
            focus[self.focus] = 1.0
        return np.clip(np.concatenate([
            [(math.log(self.savings) - lo) / (hi - lo), self.stated_risk, self.time], interest,
            [self.step_count / MAX_STEPS, self.patience_est / 9.0, self.n_explored / N_OPP],
            self.reaction / 3.0, self.trend_obs, self.testi_obs, self.verified, focus,
        ]), self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _info(self, outcome):
        return {"outcome": outcome, "step": self.step_count, "savings": self.savings}
