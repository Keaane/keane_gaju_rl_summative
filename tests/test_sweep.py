"""The sweep runner: --only picks settings, --out keeps runs apart, finished runs are skipped."""
import csv

import pytest

from training import sweep


def test_only_and_resume(tmp_path):
    out = tmp_path / "sweeps_200k"
    out.mkdir()
    out = str(out)
    jobs, done = sweep.plan_jobs("ppo", sweep.parse_only("5,0,3"), seeds=2, steps=1000, out=out)
    assert done == set()
    assert [(j[1], j[3]) for j in jobs] == [(0, 0), (0, 1), (3, 0), (3, 1), (5, 0), (5, 1)]
    assert all(j[5] == out for j in jobs)

    # Pretend setting 3, seed 1 already finished: a resume must skip only that pair.
    with open(sweep.summary_path("ppo", out), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep.FIELDS)
        writer.writeheader()
        writer.writerow({"algo": "ppo", "run": 3, "seed": 1})
    jobs, done = sweep.plan_jobs("ppo", [0, 3, 5], seeds=2, steps=1000, out=out)
    assert done == {(3, 1)} and len(jobs) == 5 and (3, 1) not in [(j[1], j[3]) for j in jobs]


def test_bad_only_value():
    with pytest.raises(SystemExit):
        sweep.parse_only("0,12")


def test_extended_runs_get_their_own_checkpoints():
    assert sweep.checkpoint_dir("logs/sweeps") == "models/sweeps"
    assert sweep.checkpoint_dir("logs/sweeps_200k") == "models/sweeps/sweeps_200k"


def test_one_tiny_run_writes_to_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep logs and checkpoints out of the real repo
    job = ("reinforce", 0, {"eval_users": 5, "episodes_per_update": 4}, 0, 200, "logs/extra")
    row = sweep.run_one(job)
    assert row["algo"] == "reinforce" and row["run"] == 0
    assert (tmp_path / "logs/extra/reinforce_run00_seed0.csv").exists()
    assert (tmp_path / "models/sweeps/extra/reinforce_run00_seed0.pt").exists()
