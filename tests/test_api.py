"""The intake-to-observation mapping, the action masking, and the HTTP API (serve.py)."""
import http.client
import json
import threading

import numpy as np
import pytest

import recommend as rec
import serve
from environment.custom_env import N_ACTIONS, StarterGuideEnv


def test_first_observation_matches_the_simulator():
    """Built from the intake alone, the observation equals the simulator's own first one (except engagement)."""
    env = StarterGuideEnv()
    for seed in (100_000, 100_001, 100_002, 100_003, 100_004):
        obs, _ = env.reset(seed=seed)
        mine = rec.first_observation(env.savings, env.stated_risk, env.time, env.stated_interest)
        same = np.isclose(mine, obs, atol=1e-6)
        same[10] = True                                   # engagement: the app uses a fixed 7/9 instead
        assert same.all(), np.flatnonzero(~same)


def test_every_action_has_a_label():
    labels = [rec.action_label(a) for a in range(N_ACTIONS)]
    assert labels[9] == "Show T-bills" and labels[13] == "Check the current option for scams"
    assert len(set(labels)) == N_ACTIONS


def with_focus(savings, opportunity):
    obs = rec.first_observation(savings, "medium", "some", "savings")
    obs[52 + opportunity] = 1.0
    return obs


def test_masking_follows_the_simulator_rules():
    nothing_shown = rec.valid_actions(rec.first_observation(500_000, "low", "some", "not sure"))
    assert nothing_shown[:10].all() and nothing_shown[14]             # show anything, or end
    assert not nothing_shown[10:14].any()                             # no commit, no scam check yet
    tbills = 9                                                        # minimum ticket 100,000 RWF
    small_savings = rec.valid_actions(with_focus(150_000, tbills))
    assert small_savings[13] and not small_savings[10:13].any()       # 50% of 150,000 is still too small
    more_savings = rec.valid_actions(with_focus(500_000, tbills))
    assert list(more_savings[10:13]) == [False, True, True]           # 50,000 too small; 125,000 and 250,000 fine


def test_best_valid_skips_a_forbidden_top_choice():
    obs = rec.first_observation(500_000, "low", "some", "not sure")
    scores = np.zeros(N_ACTIONS)
    scores[11], scores[8] = 5.0, 1.0                                  # the guide "wants" to commit before showing
    action, own_choice_ok = rec.best_valid(scores, obs)
    assert action == 8 and own_choice_ok is False


@pytest.fixture(scope="module")
def server():
    srv = serve.make_server("ppo", port=0)                            # any free port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def call(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return resp.status, data


def test_health(server):
    status, data = call(server, "GET", "/health")
    assert status == 200 and data["status"] == "ok" and data["obs_dim"] == 62 and data["model"].endswith("ppo.pt")


def test_next_action(server):
    obs = rec.first_observation(500_000, "medium", "some", "savings")
    status, data = call(server, "POST", "/next_action", json.dumps({"observation": obs.tolist()}))
    assert status == 200 and 0 <= data["action"] < N_ACTIONS and data["valid"] is True
    assert data["label"] == rec.action_label(data["action"])
    assert rec.valid_actions(obs)[data["action"]]                    # never a forbidden action


@pytest.mark.parametrize("body, message", [("not json", "valid JSON"),
                                           (json.dumps({"observation": [0.0] * 61}), "61 numbers"),
                                           (json.dumps({"observation": [0.0] * 61 + [7.0]}), "outside its range")])
def test_bad_requests(server, body, message):
    status, data = call(server, "POST", "/next_action", body)
    assert status == 400 and message in data["error"]


def test_body_size_limit(server):
    status, data = call(server, "POST", "/next_action", "x" * (serve.MAX_BODY + 1))
    assert status == 413 and "limit" in data["error"]
