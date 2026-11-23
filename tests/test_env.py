import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import COMMIT, StarterGuideEnv, VERIFY
from environment.world_data import TBILLS, TBILL_MIN


def test_gymnasium_api():
    check_env(StarterGuideEnv())


def test_cannot_commit_before_showing_anything():
    env = StarterGuideEnv()
    env.reset(seed=0)
    _, reward, done, _, info = env.step(COMMIT.start)
    assert info["outcome"] == "invalid" and reward < 0 and not done


def test_cannot_verify_before_showing_anything():
    env = StarterGuideEnv()
    env.reset(seed=0)
    assert env.step(VERIFY)[4]["outcome"] == "invalid"


def test_commit_needs_an_affordable_size():
    env = StarterGuideEnv()
    env.reset(seed=0, options={"savings": 200_000})
    env.step(TBILLS)                                   # T-bills need RWF 100,000
    assert env.step(COMMIT.start)[4]["outcome"] == "invalid"        # 10% = 20,000: too small
    assert env.step(COMMIT.start + 2)[4]["outcome"] in ("plan", "not_adopted")   # 50% = 100,000


def test_session_always_ends():
    env = StarterGuideEnv()
    for seed in range(20):
        env.reset(seed=seed)
        for _ in range(12):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
        else:
            raise AssertionError("session did not end within 12 steps")


def test_real_data_is_what_we_cited():
    assert TBILL_MIN == 100_000


def test_rgb_array_render_and_text():
    env = StarterGuideEnv(render_mode="rgb_array")
    env.reset(seed=3)
    env.step(TBILLS)
    assert "T-bills" in env.last_text
    frame = env.render()
    assert frame.shape == (600, 1000, 3) and frame.dtype == np.uint8 and frame.std() > 5
    env.close()


def test_final_frame_has_outcome():
    env = StarterGuideEnv(render_mode="rgb_array")
    env.reset(seed=3, options={"savings": 1_000_000})
    env.step(TBILLS)
    env.step(COMMIT.start + 1)
    assert env.final is not None and env.render().shape == (600, 1000, 3)
    env.close()


def test_human_mode_runs_headless_for_a_full_session():
    env = StarterGuideEnv(render_mode="human", render_fps=1000)
    env.reset(seed=4)
    env.render()
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        env.render()
        done = term or trunc
    env.close()


def test_scene_and_dashboard_views_both_render():
    for view in ("scene", "dashboard"):
        env = StarterGuideEnv(render_mode="rgb_array", render_view=view)
        env.reset(seed=5)
        env.step(TBILLS)
        env.step(VERIFY)
        frame = env.render()
        assert frame.shape == (600, 1000, 3) and frame.std() > 5
        env.close()


def test_scene_walks_between_buildings_in_window_mode():
    env = StarterGuideEnv(render_mode="human", render_fps=1000, render_view="scene")
    env.reset(seed=6, options={"savings": 1_000_000})
    env.render()
    for action in (0, 9, COMMIT.start + 1):          # walk to two buildings, then commit
        env.step(action)
        env.render()
    env.close()
