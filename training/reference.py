"""Reference scores on the shared TEST users, cached in logs/reference.json.

Every learned agent is compared against these. Run once:  uv run python -m training.reference
"""
import json
import os

from analysis.checks import best_rule, run
from environment.baselines import Oracle, adaptive, random_policy, static_policy

PATH = "logs/reference.json"
TEST_SEED, N_TEST = 100_000, 300


def compute():
    rule = best_rule(320, {})
    scores = {}
    for name, policy in [("random", random_policy), ("best profile rule", static_policy(rule)),
                         ("adaptive hand-written", adaptive), ("oracle", Oracle())]:
        mean, se, _ = run(policy, N_TEST, TEST_SEED)
        scores[name] = {"reward": mean, "se": se}
    os.makedirs("logs", exist_ok=True)
    json.dump(scores, open(PATH, "w"), indent=2)
    return scores


def load():
    return json.load(open(PATH)) if os.path.exists(PATH) else compute()


if __name__ == "__main__":
    for name, s in compute().items():
        print(f"{name:<24}{s['reward']:+.2f} +/- {s['se']:.2f}")
