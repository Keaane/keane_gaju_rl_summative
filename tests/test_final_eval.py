"""The final evaluation: verdict labels, paired differences, and the shape of its output."""
import pytest

from analysis import final_eval as fe


def test_verdict_labels():
    assert fe.verdict(0.10, 0.10) == "none"          # 1 standard error
    assert fe.verdict(0.25, 0.10) == "suggestive"    # 2.5
    assert fe.verdict(-0.35, 0.10) == "clear"        # 3.5, either direction
    assert fe.verdict(0.0, 0.0) == "n/a"


def test_paired_difference():
    d, se, better, worse = fe.paired([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 5.0])
    assert d == pytest.approx(0.5)                   # differences 0, 1, 2, -1
    assert se == pytest.approx(1.2909944 / 2)        # sample std / sqrt(4)
    assert (better, worse) == (0.5, 0.25)


def test_tiny_run_has_every_row_and_column():
    summary, per_user = fe.evaluate_all(n_users=3, worlds=("normal", "new_users"), verbose=False)
    # 4 algorithms x (3-seed mean + chosen model) + 4 guides, in each of the 2 worlds
    assert len(summary) == 2 * (4 * 2 + 4)
    assert all(set(row) <= set(fe.SUMMARY_COLS) for row in summary)
    ppo = next(r for r in summary if r["world"] == "normal" and r["policy"] == "ppo (3 seeds)")
    assert {"seed0_reward", "seed1_reward", "seed2_reward", "seed_spread", "vs_rule_verdict"} <= set(ppo)
    new = next(r for r in summary if r["world"] == "new_users" and r["policy"] == "oracle")
    assert new["world_label"].startswith("new users, same distribution") and "change_verdict" in new
    assert "vs_rule_diff" not in next(r for r in summary if r["policy"] == "best profile rule")
    assert len(per_user) == (12 + 4) * 3             # normal world only: 12 seed models + 4 guides
    assert 0 <= ppo["verify_rate"] <= 1 and ppo["session_steps"] >= 2 and all(u["session_steps"] >= 1 for u in per_user)
