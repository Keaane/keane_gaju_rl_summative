"""Entry point. Watch a Starter Guide agent run sessions with simulated beginners.

  uv run main.py                          best PPO agent, 3 sessions, pygame window
  uv run main.py --agent adaptive         a hand-written guide (ppo | a2c | reinforce | dqn |
                                          adaptive | rule | oracle | random)
  uv run main.py --no-gui                 text transcript only (no window)
  uv run main.py --view dashboard         list dashboard instead of the street scene
  uv run main.py --screenshot assets/session.png --seed 7     save a still image
  uv run main.py --gif assets/session.gif --agent adaptive    save the walking animation
"""
import argparse
import os

import numpy as np

from environment.baselines import Oracle, StaticRule, adaptive, random_policy, static_policy
from environment.custom_env import N_ACTIONS, OBS_DIM, StarterGuideEnv

TRAINED = ("ppo", "a2c", "reinforce", "dqn")


def load_agent(name, model_path=None):
    """Return a function (env, obs) -> action."""
    if name in TRAINED:
        import torch
        from training.common import mlp
        from training.pg_training import ActorCritic

        path = model_path or f"models/best/{name}.pt"
        if not os.path.exists(path):
            raise SystemExit(f"No trained model at {path}. Train one first (see README) or use --agent adaptive.")
        if name == "dqn":
            net = mlp([OBS_DIM, 64, 64, N_ACTIONS])
            net.load_state_dict(torch.load(path))
            return lambda env, obs: int(net(torch.as_tensor(obs)).argmax())
        net = ActorCritic(64)
        net.load_state_dict(torch.load(path))
        return lambda env, obs: int(net(torch.as_tensor(obs))[0].argmax())
    simple = {"adaptive": adaptive, "random": random_policy, "oracle": Oracle(), "rule": static_policy(StaticRule(2, "interest", -0.3, True))}
    return lambda env, obs: simple[name](env)


def run_session(env, agent, seed, render):
    obs, info = env.reset(seed=seed)
    print(f"\nNew user: {info['savings']:,.0f} RWF to start")
    if render:
        env.render()
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(agent(env, obs))
        print(f"  {env.last_text}  ({reward:+.2f})")
        if render:
            env.render()
        done = terminated or truncated
    final = env.final or {}
    extra = f", real return after inflation {final['real_growth']:+.1%}" if final.get("size") else ""
    print(f"  -> {info['outcome']}{extra}")
    for _ in range(3 if render else 0):      # hold the final screen for a moment
        env.render()
    return info


def save_gif(agent, seed, path, font):
    """Play one session and write every animation frame to a GIF (needs the street scene)."""
    from PIL import Image

    env = StarterGuideEnv(render_mode="rgb_array", render_font=font, render_view="scene")
    obs, _ = env.reset(seed=seed)
    frames = [env.render()] * 4
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(agent(env, obs))
        step_frames = env.render_frames()
        frames += step_frames[:-1] + [step_frames[-1]] * 14        # linger on each resting picture
        done = terminated or truncated
    frames += [frames[-1]] * 20
    images = [Image.fromarray(f).resize((760, 456), Image.LANCZOS).quantize(colors=256, dither=Image.Dither.NONE) for f in frames]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=70, loop=0, optimize=True)
    print(f"Saved {path} ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser(description="Starter Guide")
    p.add_argument("--agent", default="ppo", choices=list(TRAINED) + ["adaptive", "rule", "oracle", "random"])
    p.add_argument("--model", help="path to a saved model (default models/best/<agent>.pt)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=100_000, help="first simulated user")
    p.add_argument("--fps", type=int, default=2, help="speed of the window (steps per second)")
    p.add_argument("--view", default="scene", choices=["scene", "dashboard"], help="street scene or list dashboard")
    p.add_argument("--font", default="poppins", help="poppins | classic")
    p.add_argument("--no-gui", action="store_true", help="print a text transcript instead of opening a window")
    p.add_argument("--gif", metavar="PATH", help="save an animated GIF of one session (street scene)")
    p.add_argument("--screenshot", metavar="PATH", help="save the final dashboard of one session as a PNG")
    args = p.parse_args()
    agent = load_agent(args.agent, args.model)

    if args.gif:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        save_gif(agent, args.seed, args.gif, args.font)
        return
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from environment.rendering import save_png

        env = StarterGuideEnv(render_mode="rgb_array", render_font=args.font, render_view=args.view)
        run_session(env, agent, args.seed, render=False)
        os.makedirs(os.path.dirname(args.screenshot) or ".", exist_ok=True)
        save_png(env.render(), args.screenshot)
        print("Saved", args.screenshot)
        return

    render = not args.no_gui
    env = StarterGuideEnv(render_mode="human" if render else None, render_fps=args.fps, render_font=args.font, render_view=args.view)
    try:
        for i in range(args.episodes):
            run_session(env, agent, args.seed + i, render)
    except Exception as error:               # e.g. no display on a server: fall back to text
        if not render or "display" not in str(error).lower() and "video" not in str(error).lower():
            raise
        print(f"(No window available: {error}. Showing text only.)")
        env = StarterGuideEnv()
        for i in range(args.episodes):
            run_session(env, agent, args.seed + i, False)
    env.close()


if __name__ == "__main__":
    main()
