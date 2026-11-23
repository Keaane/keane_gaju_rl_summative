"""Quick learnability probe: evolution strategies on a tiny numpy MLP.
Not one of the four assignment algorithms, just a cheap test of whether
StarterGuideEnv contains a policy better than the hand-written rules.

Run:  uv run python -m analysis.learnability_probe
"""
import sys, time
import numpy as np
from environment.custom_env import N_ACTIONS, OBS_DIM, StarterGuideEnv

H = 32
SHAPES = [(OBS_DIM, H), (H,), (H, N_ACTIONS), (N_ACTIONS,)]
SIZES = [int(np.prod(s)) for s in SHAPES]
N_PARAMS = sum(SIZES)


def unpack(theta):
    out, i = [], 0
    for s, n in zip(SHAPES, SIZES):
        out.append(theta[i:i + n].reshape(s)); i += n
    return out


def act(theta, obs):
    w1, b1, w2, b2 = unpack(theta)
    return int(np.argmax(np.tanh(obs @ w1 + b1) @ w2 + b2))


def evaluate(theta, env, seeds):
    tot = 0.0
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done = False
        while not done:
            obs, r, t, tr, _ = env.step(act(theta, obs))
            tot += r
            done = t or tr
    return tot / len(seeds)


def main(gens=60, pairs=30, eps=40, budget_s=230, sigma=0.1, lr=0.05):
    rng = np.random.default_rng(0)
    env = StarterGuideEnv()
    theta = rng.normal(0, 0.1, N_PARAMS)
    t0 = time.time()
    test_seeds = np.arange(500_000, 500_300)
    for g in range(gens):
        seeds = rng.integers(0, 10**6, eps)       # same users for every candidate this generation
        eps_noise = rng.normal(size=(pairs, N_PARAMS))
        scores = np.zeros((pairs, 2))
        for i in range(pairs):
            scores[i, 0] = evaluate(theta + sigma * eps_noise[i], env, seeds)
            scores[i, 1] = evaluate(theta - sigma * eps_noise[i], env, seeds)
        ranks = (scores.argsort(axis=None).argsort().reshape(scores.shape) / (scores.size - 1)) - 0.5
        grad = ((ranks[:, 0] - ranks[:, 1])[:, None] * eps_noise).sum(0) / (pairs * sigma)
        theta += lr * grad
        if g % 10 == 0 or g == gens - 1:
            print(f"gen {g:3d}  train mean {scores.mean():+.2f}  held-out {evaluate(theta, env, test_seeds):+.2f}  ({time.time()-t0:.0f}s)", flush=True)
        if time.time() - t0 > budget_s:
            print("time budget reached", flush=True); break
    print(f"FINAL held-out (300 users): {evaluate(theta, env, test_seeds):+.2f}")


if __name__ == "__main__":
    main()
