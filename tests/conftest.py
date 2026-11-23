import os

# Run pygame without a window so tests work on any machine / CI
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
