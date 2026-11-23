"""Hyperparameter sweeps: every config x several seeds, one summary row per run.

  uv run python -m training.sweep --algo ppo --steps 100000 --seeds 3 --workers 1
  uv run python -m training.sweep --table ppo        # print the summary table

Each run trains on random users, picks its best checkpoint on validation users, and is
scored once on the shared test users. Finished runs are skipped, so you can stop and resume.
"""
import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from training import dqn_training, pg_training
from training.common import TEST_SEED, evaluate, greedy

TARGET = 0.10    # validation reward that counts as "learned" (about the best simple rule)
OUT = "logs/sweeps"

# algo -> (config class, trainer, how to turn the trained network into a policy)
REGISTRY = {
    "dqn": (dqn_training.DQNConfig, dqn_training.train_dqn, greedy),
    **{name: (cls, fn, pg_training.greedy_policy) for name, (cls, fn) in pg_training.TRAINERS.items()},
}

# Ten settings per algorithm. Run 0 is the default; each other run changes one idea.
GRIDS = {
    "dqn": [{}, {"lr": 3e-4}, {"lr": 3e-3}, {"gamma": 0.95}, {"buffer_size": 10_000}, {"batch_size": 128},
            {"eps_decay_steps": 20_000}, {"eps_decay_steps": 80_000}, {"target_update": 100}, {"target_update": 2000}],
    "reinforce": [{}, {"lr": 3e-4}, {"lr": 3e-3}, {"baseline": False}, {"gamma": 0.95}, {"episodes_per_update": 4},
                  {"episodes_per_update": 64}, {"hidden": 32}, {"hidden": 128}, {"ent_coef": 0.05}],
    "a2c": [{}, {"lr": 2e-4}, {"lr": 2e-3}, {"n_steps": 8}, {"n_steps": 128}, {"ent_coef": 0.0},
            {"ent_coef": 0.05}, {"vf_coef": 0.25}, {"gae_lambda": 0.9}, {"gamma": 0.95}],
    "ppo": [{}, {"clip": 0.1}, {"clip": 0.3}, {"gae_lambda": 0.9}, {"epochs": 3}, {"epochs": 16},
            {"ent_coef": 0.0}, {"ent_coef": 0.05}, {"minibatch": 64}, {"rollout_steps": 512}],
}
FIELDS = ["algo", "run", "seed", "config", "test_reward", "test_se", "val_best", "steps_to_target",
          "plan", "quit", "scammed", "panic", "beat_inflation", "minutes"]


def run_one(job):
    algo, run, overrides, seed, steps = job
    torch.set_num_threads(1)
    config_class, trainer, make_policy = REGISTRY[algo]
    settings = {"eval_every": min(10_000, max(1, steps // 5)), **overrides}
    if algo == "dqn":                      # unless a run sets it, explore for the first half of the budget
        settings.setdefault("eps_decay_steps", steps // 2)
    cfg = config_class(total_steps=steps, seed=seed, **settings)
    tag = f"{algo}_run{run:02d}_seed{seed}"
    start = time.time()
    net, history = trainer(cfg, f"{OUT}/{tag}.csv", f"models/sweeps/{tag}.pt", verbose=False)
    test = evaluate(make_policy(net), 300, TEST_SEED)
    reached = next((h["step"] for h in history if h["reward"] >= TARGET), "")
    return {"algo": algo, "run": run, "seed": seed, "config": json.dumps(overrides),
            "test_reward": test["reward"], "test_se": test["se"], "val_best": max(h["reward"] for h in history),
            "steps_to_target": reached, "plan": test["plan"], "quit": test["quit"], "scammed": test["scammed"],
            "panic": test["panic"], "beat_inflation": test["beat_inflation"], "minutes": (time.time() - start) / 60}


def summary_path(algo):
    return f"{OUT}/{algo}_summary.csv"


def finished(algo):
    path = summary_path(algo)
    return {(int(r["run"]), int(r["seed"])) for r in csv.DictReader(open(path))} if os.path.exists(path) else set()


def print_table(algo):
    rows = list(csv.DictReader(open(summary_path(algo))))
    print(f"{algo}: mean test reward over seeds (+/- spread), best first")
    by_run = {}
    for r in rows:
        by_run.setdefault(int(r["run"]), []).append(r)
    ranked = sorted(by_run.items(), key=lambda kv: -np.mean([float(r["test_reward"]) for r in kv[1]]))
    for run, rs in ranked:
        scores = [float(r["test_reward"]) for r in rs]
        print(f"  run {run:>2}  {np.mean(scores):+.2f} +/- {np.std(scores):.2f}  ({len(rs)} seeds)  {rs[0]['config']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=REGISTRY)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--runs", type=int, default=10, help="how many of the 10 settings to run")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--table", choices=REGISTRY, help="just print the summary table")
    args = p.parse_args()
    if args.table:
        return print_table(args.table)

    os.makedirs(OUT, exist_ok=True)
    done = finished(args.algo)
    jobs = [(args.algo, r, GRIDS[args.algo][r], s, args.steps)
            for r in range(args.runs) for s in range(args.seeds) if (r, s) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already finished)", flush=True)
    new_file = not os.path.exists(summary_path(args.algo))
    with open(summary_path(args.algo), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
        for row in (pool.map(run_one, jobs) if pool else map(run_one, jobs)):
            writer.writerow(row)
            f.flush()
            print(f"run {row['run']:>2} seed {row['seed']}  test {row['test_reward']:+.2f}  {row['minutes']:.1f} min  {row['config']}", flush=True)
    print_table(args.algo)


if __name__ == "__main__":
    main()
