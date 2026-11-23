"""Hyperparameter sweeps: every config x several seeds, one summary row per run.

  uv run python -m training.sweep --algo ppo --steps 100000 --seeds 3 --workers 1
  uv run python -m training.sweep --table ppo        # print the summary table
  uv run python -m training.sweep --algo ppo --steps 200000 --only 0,3,5 --out logs/sweeps_200k

Each run trains on random users, picks its best checkpoint on validation users, and is
scored once on the shared test users. Finished runs are skipped, so you can stop and resume.
--only picks which settings to run; --out keeps a separate log folder (and checkpoint folder)
so an extended run never mixes with, or overwrites, the main one.
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


def checkpoint_dir(out):
    """Main sweep: models/sweeps/. Any other --out folder gets its own subfolder, e.g. models/sweeps/sweeps_200k/."""
    if os.path.normpath(out) == os.path.normpath(OUT):
        return "models/sweeps"
    return os.path.join("models/sweeps", os.path.basename(os.path.normpath(out)))


def run_one(job):
    algo, run, overrides, seed, steps, out = job
    torch.set_num_threads(1)
    config_class, trainer, make_policy = REGISTRY[algo]
    settings = {"eval_every": min(10_000, max(1, steps // 5)), **overrides}
    if algo == "dqn":                      # unless a run sets it, explore for the first half of the budget
        settings.setdefault("eps_decay_steps", steps // 2)
    cfg = config_class(total_steps=steps, seed=seed, **settings)
    tag = f"{algo}_run{run:02d}_seed{seed}"
    start = time.time()
    net, history = trainer(cfg, f"{out}/{tag}.csv", f"{checkpoint_dir(out)}/{tag}.pt", verbose=False)
    test = evaluate(make_policy(net), 300, TEST_SEED)
    reached = next((h["step"] for h in history if h["reward"] >= TARGET), "")
    return {"algo": algo, "run": run, "seed": seed, "config": json.dumps(overrides),
            "test_reward": test["reward"], "test_se": test["se"], "val_best": max(h["reward"] for h in history),
            "steps_to_target": reached, "plan": test["plan"], "quit": test["quit"], "scammed": test["scammed"],
            "panic": test["panic"], "beat_inflation": test["beat_inflation"], "minutes": (time.time() - start) / 60}


def summary_path(algo, out=OUT):
    return f"{out}/{algo}_summary.csv"


def finished(algo, out=OUT):
    """(setting, seed) pairs already in the summary file, so they are skipped on a resume."""
    path = summary_path(algo, out)
    return {(int(r["run"]), int(r["seed"])) for r in csv.DictReader(open(path))} if os.path.exists(path) else set()


def print_table(algo, out=OUT):
    rows = list(csv.DictReader(open(summary_path(algo, out))))
    print(f"{algo}: mean test reward over seeds (+/- spread), best first")
    by_run = {}
    for r in rows:
        by_run.setdefault(int(r["run"]), []).append(r)
    ranked = sorted(by_run.items(), key=lambda kv: -np.mean([float(r["test_reward"]) for r in kv[1]]))
    for run, rs in ranked:
        scores = [float(r["test_reward"]) for r in rs]
        print(f"  run {run:>2}  {np.mean(scores):+.2f} +/- {np.std(scores):.2f}  ({len(rs)} seeds)  {rs[0]['config']}")


def parse_only(text):
    """'0,3,5' -> [0, 3, 5]. Checks each number is a real setting index."""
    runs = sorted({int(x) for x in text.split(",") if x.strip()})
    for r in runs:
        if not 0 <= r < 10:
            raise SystemExit(f"--only: setting {r} does not exist (use 0 to 9)")
    return runs


def plan_jobs(algo, runs, seeds, steps, out):
    """Every (setting, seed) pair still to do. Pairs already in the summary file are skipped."""
    done = finished(algo, out)
    jobs = [(algo, r, GRIDS[algo][r], s, steps, out) for r in runs for s in range(seeds) if (r, s) not in done]
    return jobs, done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=REGISTRY)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--runs", type=int, default=10, help="how many of the 10 settings to run")
    p.add_argument("--only", help="run only these settings, e.g. 0,3,5 (overrides --runs)")
    p.add_argument("--out", default=OUT, help="log folder (default logs/sweeps)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--table", choices=REGISTRY, help="just print the summary table")
    args = p.parse_args()
    if args.table:
        return print_table(args.table, args.out)
    if not args.algo:
        p.error("--algo is required (or use --table)")

    runs = parse_only(args.only) if args.only else list(range(args.runs))
    os.makedirs(args.out, exist_ok=True)
    jobs, done = plan_jobs(args.algo, runs, args.seeds, args.steps, args.out)
    print(f"{len(jobs)} runs to do ({len(done)} already finished) -> {args.out}", flush=True)
    new_file = not os.path.exists(summary_path(args.algo, args.out))
    with open(summary_path(args.algo, args.out), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
        for row in (pool.map(run_one, jobs) if pool else map(run_one, jobs)):
            writer.writerow(row)
            f.flush()
            print(f"run {row['run']:>2} seed {row['seed']}  test {row['test_reward']:+.2f}  {row['minutes']:.1f} min  {row['config']}", flush=True)
    print_table(args.algo, args.out)


if __name__ == "__main__":
    main()
