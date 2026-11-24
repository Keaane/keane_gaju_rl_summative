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


def test_ppo_has_two_extra_settings():
    assert len(sweep.GRIDS["ppo"]) == 12 and sweep.GRIDS["ppo"][10:] == [{"lr": 1e-3}, {"gamma": 0.95}]
    assert sweep.parse_only("10,11", len(sweep.GRIDS["ppo"])) == [10, 11]
    with pytest.raises(SystemExit):
        sweep.parse_only("10", len(sweep.GRIDS["dqn"]))     # the other algorithms still have 10 settings


def test_extended_runs_get_their_own_checkpoints():
    assert sweep.checkpoint_dir("logs/sweeps") == "models/sweeps"
    assert sweep.checkpoint_dir("logs/sweeps_200k") == "models/sweeps/sweeps_200k"


def test_dqn_explore_steps():
    # Default: explore for half the budget.
    assert sweep.train_settings("dqn", {}, 200_000)["eps_decay_steps"] == 100_000
    # The flag fixes it, so a 200,000-step run explores as long as a 100,000-step one.
    assert sweep.train_settings("dqn", {}, 200_000, 50_000)["eps_decay_steps"] == 50_000
    # A setting that chooses its own exploration keeps it.
    assert sweep.train_settings("dqn", {"eps_decay_steps": 20_000}, 200_000, 50_000)["eps_decay_steps"] == 20_000
    # Other algorithms have no exploration schedule.
    assert "eps_decay_steps" not in sweep.train_settings("ppo", {}, 200_000, 50_000)
    # The flag reaches every planned job.
    jobs, _ = sweep.plan_jobs("dqn", [1, 6], 1, 200_000, "unused_dir", 50_000)
    assert all(j[6] == 50_000 for j in jobs)


def test_one_tiny_run_writes_to_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep logs and checkpoints out of the real repo
    job = ("reinforce", 0, {"eval_users": 5, "episodes_per_update": 4}, 0, 200, "logs/extra", None)
    row = sweep.run_one(job)
    assert row["algo"] == "reinforce" and row["run"] == 0
    assert (tmp_path / "logs/extra/reinforce_run00_seed0.csv").exists()
    assert (tmp_path / "models/sweeps/extra/reinforce_run00_seed0.pt").exists()
