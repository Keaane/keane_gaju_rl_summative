"""Final evaluation (Task 7): every final agent and every reference guide on 500 test users.

  uv run python -m analysis.final_eval                    # full run (500 users, 5 worlds)
  uv run python -m analysis.final_eval --users 20 --out-dir scratch/dry_run    # quick dry run

Agents: all 3 seeds of each algorithm's chosen 200,000-step setting (models/seeds/). The headline for an
algorithm is the mean over its 3 seeds, with the spread between seeds. The chosen checkpoint (models/best/,
the demo model) also gets its own row. Guides: random, the best profile rule, the adaptive guide, the oracle.
The profile rule is tuned on seeds 0-319 (analysis/checks.py), never on any user evaluated here.

Paired comparison: on the same users, agent reward minus profile-rule reward. A difference counts as
"suggestive" if it is 2-3 standard errors from zero and "clear" if it is more than 3. Under 2 is "none".

Worlds: the normal world, a shifted world (more scams, more undecided users), low and high savings
(the same 500 users with their savings fixed), and new users from the same distribution (seeds 300000+).

Writes assets/final_results.csv (one row per policy and world) and assets/final_per_user.csv
(one row per policy, seed and user, normal world only, for the plots). Simulation only, not financial advice.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from analysis.checks import best_rule
from environment.baselines import Oracle, adaptive, random_policy, static_policy
from environment.custom_env import N_ACTIONS, OBS_DIM, VERIFY, StarterGuideEnv
from environment.world_data import OPP_NAMES
from training.common import greedy, mlp
from training.pg_training import ActorCritic, greedy_policy

TEST_SEED = 100_000
# The chosen 200,000-step setting per algorithm, and which seed is the demo model in models/best/.
CHOSEN = {"ppo": ("run 5: 16 epochs", 0), "a2c": ("run 3: update every 8 steps", 2),
          "reinforce": ("run 9: entropy 0.05", 1), "dqn": ("run 1: learning rate 3e-4", 0)}
SEEDS = (0, 1, 2)
# name -> (description, first user seed, reset options)
WORLDS = {
    "normal": ("normal world, test users", TEST_SEED, None),
    "shifted": ("more scams and undecided users (scam_boost 0.25, p_undecided 0.7)", TEST_SEED,
                {"scam_boost": 0.25, "p_undecided": 0.7}),
    "low_savings": ("same test users, savings fixed at 150,000 RWF", TEST_SEED, {"savings": 150_000}),
    "high_savings": ("same test users, savings fixed at 1,800,000 RWF", TEST_SEED, {"savings": 1_800_000}),
    "new_users": ("new users, same distribution (seeds 300000+)", 300_000, None),
}
SAME_USERS_AS_NORMAL = {"shifted", "low_savings", "high_savings"}   # these differences are paired per user


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


REC_COLS = [f"rec_{slug(n)}" for n in OPP_NAMES]


# ---- policies: each one is a function (env, obs) -> action ---------------------------------
def load_agent(algo, path):
    if algo == "dqn":
        net = mlp([OBS_DIM, 64, 64, N_ACTIONS])
        net.load_state_dict(torch.load(path))
        act = greedy(net)
    else:
        net = ActorCritic(64)
        net.load_state_dict(torch.load(path))
        act = greedy_policy(net)
    net.eval()
    return lambda env, obs: act(obs)


def guides():
    rule = best_rule(320, {})          # tuned on seeds 0-319 only
    oracle = Oracle()
    return {"random": lambda env, obs: random_policy(env),
            "best profile rule": lambda env, obs: static_policy(rule)(env),
            "adaptive hand-written": lambda env, obs: adaptive(env),
            "oracle": lambda env, obs: oracle(env)}


# ---- running sessions ----------------------------------------------------------------------
def run_users(policy, n_users, seed0, options):
    """Play one session per user. Returns one dict per user."""
    env, rows = StarterGuideEnv(), []
    for i in range(n_users):
        obs, _ = env.reset(seed=seed0 + i, options=options)
        done, total, steps, verified = False, 0.0, 0, False
        while not done:
            action = policy(env, obs)
            verified = verified or action == VERIFY            # did the guide pay for a scam check?
            obs, r, terminated, truncated, info = env.step(action)
            total, done, steps = total + r, terminated or truncated, steps + 1
        size = float(info.get("size", 0.0))
        chosen = int(info.get("chosen", -1)) if size > 0 else -1
        rows.append({"user_seed": seed0 + i, "reward": total, "outcome": info["outcome"],
                     "chosen": chosen, "chosen_name": OPP_NAMES[chosen] if chosen >= 0 else "",
                     "size": size, "scammed": bool(info.get("scammed", False)),
                     "panic": bool(info.get("panic", False)), "real_growth": float(info.get("real_growth", -1.0)),
                     "verified": verified, "session_steps": steps})
    return rows


def metrics(rows):
    """Rates for one policy in one world. Beating inflation is given two ways, both labelled."""
    n = len(rows)
    plans = [r for r in rows if r["outcome"] == "plan"]
    out = {"plan": len(plans) / n,
           "quit": sum(r["outcome"] == "quit" for r in rows) / n,
           "scammed": sum(r["scammed"] for r in rows) / n,
           "panic": sum(r["panic"] for r in rows) / n,
           "beat_inflation_share_of_plans": (sum(r["real_growth"] > 0 for r in plans) / len(plans)) if plans else 0.0,
           "beat_inflation_share_of_users": sum(r["real_growth"] > 0 for r in rows) / n,
           "verify_rate": sum(r["verified"] for r in rows) / n,      # sessions with at least one scam check
           "session_steps": sum(r["session_steps"] for r in rows) / n}
    for k, col in enumerate(REC_COLS):         # share of users the guide recommended each opportunity to
        out[col] = sum(r["chosen"] == k for r in rows) / n
    return out


def mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def verdict(diff, se):
    """How sure we are that a difference is real: under 2 SE none, 2-3 suggestive, over 3 clear."""
    if se == 0:
        return "n/a"
    z = abs(diff) / se
    return "clear" if z > 3 else "suggestive" if z >= 2 else "none"


def paired(a, b):
    """Same users: mean of (a - b), its standard error, and the share of users where a > b and a < b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m, se = mean_se(d)
    return m, se, float((d > 0).mean()), float((d < 0).mean())


# ---- the evaluation ------------------------------------------------------------------------
def evaluate_all(n_users=500, worlds=tuple(WORLDS), models_dir="models", verbose=True):
    """Returns (summary rows, per-user rows for the normal world)."""
    policies = {}                                              # label -> (kind, algo, seed, policy)
    for algo in CHOSEN:
        for s in SEEDS:
            policies[f"{algo} seed {s}"] = ("seed", algo, s, load_agent(algo, f"{models_dir}/seeds/{algo}_seed{s}.pt"))
    for name, pol in guides().items():
        policies[name] = ("guide", "", "", pol)

    users = {}                                                 # (world, label) -> per-user rows
    for w in worlds:
        _, seed0, options = WORLDS[w]
        for label, (_, _, _, pol) in policies.items():
            start = time.time()
            users[w, label] = run_users(pol, n_users, seed0, options)
            if verbose:
                print(f"{w:<13} {label:<22} {np.mean([r['reward'] for r in users[w, label]]):+.2f}"
                      f"  ({time.time() - start:.1f} s)", flush=True)

    def rewards(w, label):
        return [r["reward"] for r in users[w, label]]

    def seed_avg(w, algo):                                     # per-user reward averaged over the 3 seeds
        return np.mean([rewards(w, f"{algo} seed {s}") for s in SEEDS], axis=0)

    summary, row_rewards = [], {}                              # row_rewards: (world, policy name) -> per-user rewards
    for w in worlds:
        rule = rewards(w, "best profile rule")
        entries = []                                           # (policy name, kind, per-user rewards, metrics, extra)
        for algo, (setting, chosen_seed) in CHOSEN.items():
            seed_means = [float(np.mean(rewards(w, f"{algo} seed {s}"))) for s in SEEDS]
            per_seed = [metrics(users[w, f"{algo} seed {s}"]) for s in SEEDS]
            avg_metrics = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
            extra = {"setting": setting, "seed": "0,1,2", "seed_spread": float(np.std(seed_means)),
                     **{f"seed{s}_reward": seed_means[i] for i, s in enumerate(SEEDS)}}
            entries.append((f"{algo} (3 seeds)", "agent, mean of 3 seeds", seed_avg(w, algo), avg_metrics, extra))
            label = f"{algo} seed {chosen_seed}"
            entries.append((f"{algo} chosen (seed {chosen_seed})", "agent, demo model in models/best",
                            rewards(w, label), metrics(users[w, label]), {"setting": setting, "seed": chosen_seed}))
        for name in ("random", "best profile rule", "adaptive hand-written", "oracle"):
            entries.append((name, "guide", rewards(w, name), metrics(users[w, name]), {}))

        for name, kind, r, m, extra in entries:
            row_rewards[w, name] = r
            mean, se = mean_se(r)
            row = {"world": w, "world_label": WORLDS[w][0], "policy": name, "kind": kind,
                   "n_users": len(r), "reward": mean, "reward_se": se, **extra, **m}
            if name != "best profile rule":
                d, dse, better, worse = paired(r, rule)
                row.update(vs_rule_diff=d, vs_rule_se=dse, vs_rule_better_share=better,
                           vs_rule_worse_share=worse, vs_rule_verdict=verdict(d, dse))
            summary.append(row)

    # Change from the normal world: paired for the same users, unpaired for the new users.
    normal = {row["policy"]: row for row in summary if row["world"] == "normal"}
    for row in summary:
        w, name = row["world"], row["policy"]
        if w == "normal" or name not in normal:
            continue
        if w in SAME_USERS_AS_NORMAL:
            d, dse, _, _ = paired(row_rewards[w, name], row_rewards["normal", name])
        else:
            d = row["reward"] - normal[name]["reward"]
            dse = float(np.hypot(row["reward_se"], normal[name]["reward_se"]))
        row.update(change_vs_normal=d, change_vs_normal_se=dse, change_verdict=verdict(d, dse))

    per_user = []
    if "normal" in worlds:
        for label, (kind, algo, s, _) in policies.items():
            for r in users["normal", label]:
                per_user.append({"policy": label, "algo": algo or label, "seed": s, **r})
    return summary, per_user


SUMMARY_COLS = (["world", "world_label", "policy", "kind", "setting", "seed", "n_users", "reward", "reward_se",
                 "seed_spread", "seed0_reward", "seed1_reward", "seed2_reward",
                 "vs_rule_diff", "vs_rule_se", "vs_rule_better_share", "vs_rule_worse_share", "vs_rule_verdict",
                 "change_vs_normal", "change_vs_normal_se", "change_verdict",
                 "plan", "quit", "scammed", "panic", "beat_inflation_share_of_plans", "beat_inflation_share_of_users",
                 "verify_rate", "session_steps"]
                + REC_COLS)
PER_USER_COLS = ["policy", "algo", "seed", "user_seed", "reward", "outcome", "chosen", "chosen_name", "size",
                 "scammed", "panic", "real_growth", "verified", "session_steps"]


def write_csv(path, rows, cols):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items() if k in cols})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=500)
    p.add_argument("--worlds", default=",".join(WORLDS), help="comma-separated, e.g. normal,shifted")
    p.add_argument("--out-dir", default="assets")
    args = p.parse_args()
    start = time.time()
    summary, per_user = evaluate_all(args.users, tuple(args.worlds.split(",")))
    write_csv(f"{args.out_dir}/final_results.csv", summary, SUMMARY_COLS)
    write_csv(f"{args.out_dir}/final_per_user.csv", per_user, PER_USER_COLS)
    print(f"done in {(time.time() - start) / 60:.1f} min -> {args.out_dir}/final_results.csv, final_per_user.csv")


if __name__ == "__main__":
    main()
