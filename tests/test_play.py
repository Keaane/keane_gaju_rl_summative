"""play.py is an alias for main.py: same options, same output."""
import subprocess
import sys


def test_play_runs_like_main():
    args = ["--no-gui", "--agent", "adaptive"]
    play = subprocess.run([sys.executable, "play.py", *args], capture_output=True, text=True, timeout=120)
    main = subprocess.run([sys.executable, "main.py", *args], capture_output=True, text=True, timeout=120)
    assert play.returncode == 0, play.stderr
    assert "New user" in play.stdout and play.stdout == main.stdout
