"""Is StarterGuideEnv worth learning? Run:  uv run python -m analysis.checks

Compares profile-only rules, a hand-written adaptive guide and an oracle, in the normal
world and a shifted one. Rules are tuned on different seeds from the ones they are tested on.
"""
import numpy as np

from environment.baselines import Oracle, STATIC_GRID, adaptive, random_policy, static_policy
from environment.custom_env import StarterGuideEnv

TUNE_SEED, TEST_SEED = 0, 100_000
USER_TYPES = [(r, u) for r in (0.0, 0.5, 1.0) for u in (False, True)]


def run(policy, n, seed0, options=None):
    """Mean reward (and standard error) over n simulated users."""
    env, totals, outcomes = StarterGuideEnv(), [], []
    for i in range(n):
        env.reset(seed=seed0 + i, options=options)
        done, total = False, 0.0
        while not done:
            _, r, term, trunc, info = env.step(policy(env))
            total, done = total + r, term or trunc
        totals.append(total)
        outcomes.append(info["outcome"])
    totals = np.array(totals)
    return totals.mean(), totals.std() / np.sqrt(n), outcomes.count("quit") / n


def best_rule(n, options):
    scores = [run(static_policy(r), n, TUNE_SEED, options)[0] for r in STATIC_GRID]
    return STATIC_GRID[int(np.argmax(scores))]


def main(n_tune=80, n_test=300):
    best = best_rule(n_tune * 4, {})
    per_type = {t: best_rule(n_tune, {"stated_risk": t[0], "undecided": t[1]}) for t in USER_TYPES}

    def tuned_per_type(env):
        return static_policy(per_type[(env.stated_risk, env.stated_interest == 5)])(env)

    guides = [("random", random_policy), ("best single profile rule", static_policy(best)),
              ("profile rules tuned per user type", tuned_per_type),
              ("adaptive hand-written guide", adaptive), ("oracle (sees hidden truth)", Oracle())]
    shifted = {"scam_boost": 0.25, "p_undecided": 0.7}
    print(f"{'guide':<36}{'normal world':>16}{'shifted world':>16}")
    for name, policy in guides:
        m, se, quit_rate = run(policy, n_test, TEST_SEED)
        ms, sse, _ = run(policy, n_test, TEST_SEED, shifted)
        print(f"{name:<36}{m:>+9.2f} +/-{se:.2f}{ms:>+9.2f} +/-{sse:.2f}   quit {quit_rate:.0%}")


if __name__ == "__main__":
    main()
