"""Choose each TorchRL algorithm's final setting with the pre-set rule, and run its 200,000-step rerun.

  python -m training.select_torchrl --algo dqn            # show the choice only (nothing runs)
  python -m training.select_torchrl --algo dqn --rerun    # also rerun it at 200,000 steps (3 seeds, 2 workers)
                                                          # and record the final seed in logs/torchrl/chosen.json

Rule (validation users only, never test): the setting with the best mean validation score over its 3 seeds at
100,000 steps; settings within 0.02 of the top count as tied, and then the smallest spread across seeds wins.
Only that setting is rerun at 200,000 steps, except when the top two settings are within 0.02 of each other: then
both are rerun, and the same rule (best mean validation, tie within 0.02: smaller spread) picks between them at
200,000 steps. The final seed is the one with the best validation score at 200,000 steps. DQN's rerun keeps
exploration at 50,000 steps, like the plain reruns. Every choice is printed and appended to
logs/torchrl/selection.log. Run it with the Python 3.11 side environment (it starts TorchRL training).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from training.sweep import GRIDS, TORCHRL_OUT, summary_path

RERUN_OUT = "logs/torchrl/sweeps_200k"
CHOSEN = "logs/torchrl/chosen.json"
LOG = "logs/torchrl/selection.log"
TIE = 0.02


def log(text):
    print(text, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')}  {text}\n")


def by_setting(path):
    groups = {}
    for r in csv.DictReader(open(path)):
        groups.setdefault(int(r["run"]), []).append(r)
    return groups


def select(algo, seeds=3):
    """The setting the rule picks at 100,000 steps, after checking the sweep is complete."""
    groups = by_setting(summary_path(algo, TORCHRL_OUT))
    missing = [r for r in range(len(GRIDS[algo])) if len(groups.get(r, [])) < seeds]
    if missing:
        raise SystemExit(f"{algo}: sweep not finished (settings {missing} have fewer than {seeds} seeds)")
    stats = {r: (np.mean([float(x["val_best"]) for x in rows]), np.std([float(x["val_best"]) for x in rows]))
             for r, rows in groups.items()}
    order = sorted(stats, key=lambda r: -stats[r][0])
    top = stats[order[0]][0]
    tied = [r for r in order if top - stats[r][0] < TIE]
    choice = min(tied, key=lambda r: stats[r][1])
    log(f"{algo}: TorchRL 100,000-step sweep, best mean validation first:")
    for r in order[:5]:
        mark = "  <- chosen" if r == choice else ("  (tied)" if r in tied else "")
        log(f"   run {r:2d} {json.dumps(GRIDS[algo][r]):24s} validation {stats[r][0]:+.3f} +/- {stats[r][1]:.3f}{mark}")
    log(f"{algo}: rule -> run {choice} {json.dumps(GRIDS[algo][choice])}"
        + (f" (tied with runs {[r for r in tied if r != choice]}; smallest spread)" if len(tied) > 1 else ""))
    if stats[order[0]][0] - stats[order[1]][0] < TIE:
        pair = order[:2]
        log(f"{algo}: the top two (runs {pair[0]} and {pair[1]}) are within {TIE}, so both are rerun at 200,000 steps")
        return pair
    return [choice]


def rerun(algo, runs, workers=2, seeds=3):
    cmd = [sys.executable, "-m", "training.sweep", "--impl", "torchrl", "--algo", algo, "--steps", "200000",
           "--seeds", str(seeds), "--only", ",".join(map(str, runs)), "--out", RERUN_OUT, "--workers", str(workers)]
    if algo == "dqn":
        cmd += ["--dqn-explore-steps", "50000"]
    log(f"{algo}: starting the 200,000-step rerun: {' '.join(cmd[1:])}")
    subprocess.run(cmd, check=True)
    groups = {r: rows for r, rows in by_setting(summary_path(algo, RERUN_OUT)).items() if r in runs}
    if len(runs) > 1:                                    # pick between the two with the same rule, at 200,000 steps
        stats = {r: (np.mean([float(x["val_best"]) for x in g]), np.std([float(x["val_best"]) for x in g]))
                 for r, g in groups.items()}
        top = max(m for m, _ in stats.values())
        tied = [r for r in runs if top - stats[r][0] < TIE]
        run = min(tied, key=lambda r: stats[r][1])
        log(f"{algo}: at 200,000 steps " + ", ".join(f"run {r} {stats[r][0]:+.3f} +/- {stats[r][1]:.3f}" for r in runs)
            + f" -> run {run}")
    else:
        run = runs[0]
    rows = groups[run]
    best = max(rows, key=lambda r: float(r["val_best"]))
    seed = int(best["seed"])
    log(f"{algo}: 200,000-step validation by seed {[round(float(r['val_best']), 3) for r in sorted(rows, key=lambda r: int(r['seed']))]}"
        f" -> final seed {seed}")
    chosen = json.load(open(CHOSEN)) if os.path.exists(CHOSEN) else {}
    chosen[algo] = {"run": run, "seed": seed, "setting": f"run {run}: {json.dumps(GRIDS[algo][run])}"}
    json.dump(chosen, open(CHOSEN, "w"), indent=2)
    os.makedirs("models/torchrl/seeds", exist_ok=True)      # the three seeds, for the final evaluation
    for s in range(seeds):
        shutil.copyfile(f"models/torchrl/sweeps_200k/{algo}_run{run:02d}_seed{s}.pt", f"models/torchrl/seeds/{algo}_seed{s}.pt")
    log(f"{algo}: recorded in {CHOSEN}; seed checkpoints copied to models/torchrl/seeds/")


def main():
    p = argparse.ArgumentParser(description="Apply the selection rule to a finished TorchRL sweep.")
    p.add_argument("--algo", choices=sorted(GRIDS), required=True)
    p.add_argument("--rerun", action="store_true", help="also run the 200,000-step rerun of the chosen setting")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    runs = select(args.algo)
    if args.rerun:
        rerun(args.algo, runs, args.workers)


if __name__ == "__main__":
    main()
