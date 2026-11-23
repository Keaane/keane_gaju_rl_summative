"""A tiny HTTP API for the trained guide, using only the Python standard library.

  uv run serve.py                      # PPO from models/best/, on http://127.0.0.1:8000
  uv run serve.py --agent dqn --port 8001

  GET  /health        -> {"status": "ok", "model": "models/best/ppo.pt", "obs_dim": 62, "n_actions": 15}
  POST /next_action   {"observation": [62 numbers]}
                      -> {"action": 9, "label": "Show T-bills", "valid": true, "masked": false}

Errors come back as {"error": "..."} with status 400 (bad request), 404 (unknown path), 411 (no length),
or 413 (body over 10 KB). There is no login: this is for local or demo use only.
Simulation only, not financial advice.
"""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from environment.custom_env import N_ACTIONS, OBS_DIM, StarterGuideEnv
from recommend import AGENTS, load_scores, suggest

MAX_BODY = 10 * 1024          # bytes; a 62-number request is well under 2 KB
READ_TIMEOUT = 5              # seconds to wait for a slow client
_space = StarterGuideEnv().observation_space
LOW, HIGH = _space.low.tolist(), _space.high.tolist()


def check_observation(body):
    """Return (observation, None) if the request is usable, else (None, error message)."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None, "body is not valid JSON"
    obs = data.get("observation") if isinstance(data, dict) else None
    if not isinstance(obs, list):
        return None, 'expected {"observation": [62 numbers]}'
    if len(obs) != OBS_DIM:
        return None, f"observation has {len(obs)} numbers, expected {OBS_DIM}"
    for i, x in enumerate(obs):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            return None, f"observation[{i}] is not a finite number"
        if not LOW[i] - 1e-6 <= x <= HIGH[i] + 1e-6:
            return None, f"observation[{i}] = {x} is outside its range {LOW[i]:g} to {HIGH[i]:g}"
    return obs, None


def make_handler(scores):
    class Handler(BaseHTTPRequestHandler):
        timeout = READ_TIMEOUT                      # socket read timeout for each connection

        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {"status": "ok", "model": scores.path, "obs_dim": OBS_DIM, "n_actions": N_ACTIONS})
            self.reply(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            if self.path != "/next_action":
                return self.reply(404, {"error": f"unknown path {self.path}"})
            length = self.headers.get("Content-Length")
            if length is None or not length.isdigit():
                return self.reply(411, {"error": "Content-Length header is required"})
            if int(length) > MAX_BODY:
                self.close_connection = True                     # do not read an oversized body
                return self.reply(413, {"error": f"body is {length} bytes; the limit is {MAX_BODY}"})
            obs, error = check_observation(self.rfile.read(int(length)))
            if error:
                return self.reply(400, {"error": error})
            self.reply(200, suggest(scores, obs))

        def log_message(self, fmt, *args):              # one short line per request
            print(f"{self.address_string()} {fmt % args}")

    return Handler


def make_server(agent="ppo", host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), make_handler(load_scores(agent)))


def main():
    p = argparse.ArgumentParser(description="Serve the trained guide over HTTP (local or demo use).")
    p.add_argument("--agent", choices=AGENTS, default="ppo")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    server = make_server(args.agent, args.host, args.port)
    print(f"Serving {args.agent.upper()} on http://{args.host}:{server.server_port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
