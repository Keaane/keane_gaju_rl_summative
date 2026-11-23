"""Pygame dashboard for StarterGuideEnv: a lean dark-minimal view of one Starter Path session.

Left     the user's intake and how engaged they still are
Centre   the 10 opportunities: what the guide showed, how the user reacted, scam checks
Bottom   what the guide just did, and (when the session ends) how the plan turned out
"""
import os

import numpy as np

WIDTH, HEIGHT = 1000, 600
BG, EDGE, TRACK = (18, 18, 20), (48, 48, 54), (38, 38, 42)
TEXT, MUTED = (236, 236, 238), (122, 122, 130)
GOOD, BAD, WARN, ACCENT = (74, 222, 128), (248, 113, 113), (250, 190, 70), (110, 170, 255)

REACTION_COLOR = {0: MUTED, 1: BAD, 2: WARN, 3: GOOD}
REACTION_WORD = {0: "not shown", 1: "declines", 2: "hesitates", 3: "accepts"}
OUTCOMES = {"plan": ("PLAN ACCEPTED", GOOD), "not_adopted": ("USER DECLINED", WARN), "quit": ("USER QUIT", BAD),
            "no_plan": ("NO PLAN", WARN), "timeout": ("OUT OF TIME", WARN)}
TIME_WORD = {0.0: "A little", 0.5: "Some", 1.0: "A lot"}
RISK_WORD = {0.0: "Low", 0.5: "Medium", 1.0: "High"}

# Bundled open-licence font (assets/fonts): name -> (regular file, medium file, size scale).
# "classic" is pygame's built-in font, used as a fallback if the files are missing.
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts")
FONTS = {
    "classic": None,
    "poppins": ("Poppins-Regular.ttf", "Poppins-Medium.ttf", 0.73),
}
DEFAULT_FONT = "poppins"


class GuideRenderer:
    def __init__(self, mode="rgb_array", fps=2, font=DEFAULT_FONT):
        import pygame

        self.pg, self.mode, self.fps = pygame, mode, fps
        pygame.display.init()
        pygame.font.init()
        self.surf = pygame.Surface((WIDTH, HEIGHT))
        if mode == "human":
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Starter Guide")
            self.clock = pygame.time.Clock()
        self.f_title, self.f_big, self.f_med, self.f_small = self._load_fonts(font)

    def _load_fonts(self, name):
        """(title, big, medium, small) fonts. Falls back to pygame's default if a file is missing."""
        if name not in FONTS:
            raise ValueError(f"Unknown font {name!r}. Choose from {sorted(FONTS)}")
        spec = FONTS[name]
        if spec:
            regular, strong, scale = (os.path.join(FONT_DIR, spec[0]), os.path.join(FONT_DIR, spec[1]), spec[2])
            if os.path.exists(regular) and os.path.exists(strong):
                font = self.pg.font.Font
                return (font(strong, round(30 * scale)), font(strong, round(34 * scale)),
                        font(regular, round(22 * scale)), font(regular, round(18 * scale)))
        font = self.pg.font.Font
        return font(None, 30), font(None, 34), font(None, 22), font(None, 18)

    def draw(self, s):
        self.surf.fill(BG)
        self._header(s)
        self._intake(s)
        self._opportunities(s)
        self._footer(s)
        if self.mode == "human":
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:
                    self.close()
                    raise SystemExit
            self.window.blit(self.surf, (0, 0))
            self.pg.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(self.pg.surfarray.array3d(self.surf), (1, 0, 2))

    def close(self):
        self.pg.display.quit()
        self.pg.font.quit()

    # ---- helpers -------------------------------------------------------------
    def text(self, txt, font, color, pos, anchor="topleft"):
        img = font.render(str(txt), True, color)
        rect = img.get_rect(**{anchor: pos})
        self.surf.blit(img, rect)
        return rect

    def line(self, y, x0=30, x1=970):
        self.pg.draw.line(self.surf, EDGE, (x0, y), (x1, y), 1)

    def stat(self, x, y, label, value, color=TEXT):
        self.text(label, self.f_small, MUTED, (x, y))
        self.text(value, self.f_big, color, (x, y + 16))

    def badge(self, cx, cy, verified):
        pg = self.pg
        if verified > 0:
            pg.draw.circle(self.surf, GOOD, (cx, cy), 10)
            pg.draw.lines(self.surf, BG, False, [(cx - 4, cy), (cx - 1, cy + 3), (cx + 5, cy - 4)], 2)
        elif verified < 0:
            pg.draw.circle(self.surf, BAD, (cx, cy), 10)
            pg.draw.line(self.surf, BG, (cx - 4, cy - 4), (cx + 4, cy + 4), 2)
            pg.draw.line(self.surf, BG, (cx - 4, cy + 4), (cx + 4, cy - 4), 2)

    # ---- sections ------------------------------------------------------------
    def _header(self, s):
        pg = self.pg
        self.text("Starter Guide", self.f_title, TEXT, (30, 14))
        self.text(f"Step {s['step']} / {s['max_steps']}", self.f_big, TEXT, (970, 12), "topright")
        pg.draw.rect(self.surf, TRACK, (30, 50, 940, 4), border_radius=2)
        width = int(940 * s["step"] / s["max_steps"])
        if width > 0:
            pg.draw.rect(self.surf, ACCENT, (30, 50, width, 4), border_radius=2)
        if s["final"]:
            label, color = OUTCOMES[s["final"]["outcome"]]
            pill = pg.Rect(0, 0, 220, 34)
            pill.center = (560, 26)
            pg.draw.rect(self.surf, color, pill, width=2, border_radius=17)
            self.text(label, self.f_med, color, pill.center, "center")

    def _intake(self, s):
        self.text("USER", self.f_small, MUTED, (30, 84))
        self.line(104, 30, 250)
        self.stat(30, 120, "Savings", f"{s['savings']:,.0f} RWF")
        self.stat(30, 186, "Risk comfort (stated)", RISK_WORD[s["risk"]])
        self.stat(30, 252, "Time per week", TIME_WORD[s["time"]])
        self.stat(30, 318, "Interest", s["interest"])
        self.text("Engagement", self.f_small, MUTED, (30, 384))
        level = max(0.0, min(1.0, s["engagement"]))
        color = GOOD if level > 0.5 else WARN if level > 0.25 else BAD
        self.pg.draw.rect(self.surf, TRACK, (30, 408, 220, 6), border_radius=3)
        self.pg.draw.rect(self.surf, color, (30, 408, int(220 * level), 6), border_radius=3)

    def _opportunities(self, s):
        pg = self.pg
        self.text("OPPORTUNITIES", self.f_small, MUTED, (290, 84))
        self.text("user reaction", self.f_small, MUTED, (724, 84))
        self.text("scam check", self.f_small, MUTED, (970, 84), "topright")
        self.line(104, 290, 970)
        for k, name in enumerate(s["names"]):
            y = 130 + k * 34
            focus, shown = k == s["focus"], s["reaction"][k] > 0
            if focus:
                pg.draw.polygon(self.surf, ACCENT, [(292, y - 6), (292, y + 6), (302, y)])
            self.text(name, self.f_med, TEXT if shown else MUTED, (312, y), "midleft")
            self.text(s["areas"][k], self.f_small, MUTED, (548, y), "midleft")
            code = int(s["reaction"][k])
            pg.draw.circle(self.surf, REACTION_COLOR[code], (730, y), 5, 0 if shown else 1)
            self.text(REACTION_WORD[code], self.f_small, REACTION_COLOR[code], (744, y), "midleft")
            self.badge(950, y, s["verified"][k])

    def _footer(self, s):
        self.line(484)
        self.text("Last action", self.f_small, MUTED, (30, 496))
        self.text(s["last_text"], self.f_med, TEXT, (30, 514))
        r = s["last_reward"]
        self.text("Last reward", self.f_small, MUTED, (800, 496))
        self.text(f"{r:+.2f}", self.f_med, GOOD if r >= 0 else BAD, (800, 514))
        final = s["final"]
        if final and final.get("size"):
            real = final["real_growth"] * 100
            parts = [("Real return after inflation", f"{real:+.1f}%", GOOD if real >= 0 else BAD),
                     ("Scam", "hit" if final["scammed"] else "avoided", BAD if final["scammed"] else GOOD),
                     ("Panic exit", "yes" if final["panic"] else "no", WARN if final["panic"] else GOOD)]
            x = 30
            for label, value, color in parts:
                self.text(label, self.f_small, MUTED, (x, 548))
                self.text(value, self.f_med, color, (x, 566))
                x += 260


def save_png(frame, path):
    """Save an RGB array (H, W, 3) as a PNG."""
    import pygame

    pygame.image.save(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), path)
