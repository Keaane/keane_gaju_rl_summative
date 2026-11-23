"""Scene view: "Starter Street", a night street where the guide and a beginner walk between
the 10 opportunities. Same data as the dashboard, drawn as a world instead of a list.

  buildings   the opportunities, grouped into 5 districts. Lit windows = shown to the user
  faces       the user's reaction to each building (smile / neutral / frown)
  shield / X  result of a scam check
  walkers     the user (with an engagement bar) and the guide robot
  evidence    a small card for the building being discussed: market trend and reviews
  commit      coins travel from the user to the building; then the result card appears

In a window ("human" mode) the walkers are animated between steps; in "rgb_array" mode
only the final frame of each step is returned.
"""
import math

import numpy as np

from environment.rendering import (
    ACCENT, BAD, BG, GOOD, MUTED, OUTCOMES, REACTION_COLOR, REACTION_WORD, RISK_WORD, TEXT,
    TIME_WORD, WARN, WIDTH, HEIGHT, GuideRenderer,
)

GROUND = 420                                   # building baseline (top of the sidewalk)
SLOT_X = [60, 148, 256, 344, 452, 540, 648, 736, 844, 932]
HEIGHTS = [96, 110, 92, 84, 100, 92, 92, 104, 112, 128]
AREA_COLORS = [(76, 160, 96), (214, 138, 64), (76, 138, 210), (176, 106, 214), (64, 178, 168)]
WARM, DIM = (255, 214, 130), (56, 60, 78)
SKIN, GUIDE = (232, 190, 150), (64, 204, 184)
USER_START_X = 24


def shade(color, f):
    return tuple(max(0, min(255, int(c * f))) for c in color)


class SceneRenderer(GuideRenderer):
    def __init__(self, mode="rgb_array", fps=2, font="poppins"):
        super().__init__(mode, fps, font)
        self.user_x, self.tick_count = USER_START_X, 0
        rng = np.random.default_rng(3)
        self.stars = [(int(rng.integers(0, WIDTH)), int(rng.integers(70, 230)), int(rng.integers(1, 3))) for _ in range(40)]
        self.skyline = [(x, int(rng.integers(40, 120)), int(rng.integers(34, 70))) for x in range(-10, WIDTH, 52)]

    # ---- animation driver ------------------------------------------------------
    def _plan(self, s):
        """The in-between frames for this step as (user x, walking, coin progress), plus where the user ends up."""
        goal = SLOT_X[s["focus"]] if s["focus"] >= 0 else USER_START_X
        plan = []
        if s["event"] in ("explore", "verify", "commit") and abs(goal - self.user_x) > 3:
            steps = max(8, int(abs(goal - self.user_x) / 14))
            for i in range(1, steps + 1):
                t = i / steps
                plan.append((self.user_x + (goal - self.user_x) * (t * t * (3 - 2 * t)), True, None))
        if s["event"] == "commit" and s["final"] and s["final"].get("size"):
            plan += [(goal, False, i / 18) for i in range(1, 19)]
        return plan, goal

    def draw(self, s):
        """Window mode: play the walk and coin animation, then hold the final picture."""
        plan, goal = self._plan(s)
        if self.mode == "human":
            for x, walking, coins in plan:
                self._scene(s, x, walking, coins)
                self._present(max(self.fps, 30))
        self.user_x = goal
        self._scene(s, goal)
        return self._present(self.fps, final=True)

    def frames(self, s):
        """Every frame of this step as RGB arrays (the last one is the resting picture). For GIFs and videos."""
        plan, goal = self._plan(s)
        out = []
        for x, walking, coins in plan:
            self._scene(s, x, walking, coins)
            out.append(self._present(0))
        self.user_x = goal
        self._scene(s, goal)
        out.append(self._present(0))
        return out

    def _present(self, fps, final=False):
        pg = self.pg
        if self.mode != "human":
            return np.transpose(pg.surfarray.array3d(self.surf), (1, 0, 2))
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                raise SystemExit
        self.window.blit(self.surf, (0, 0))
        pg.display.flip()
        self.clock.tick(fps)
        return None

    # ---- the whole picture -----------------------------------------------------
    def _scene(self, s, user_x, walking=False, coins=None):
        self.tick_count += 1
        self._sky()
        self._street()
        self._districts(s)
        for k in range(len(SLOT_X)):
            self._building(s, k)
        self._walkers(s, user_x, walking)
        if coins is not None:
            self._coins(s, user_x, coins)
        if not walking and coins is None:
            self._overlays(s, user_x)
        self._hud(s)

    def _sky(self):
        pg = self.pg
        for i in range(30):                                   # vertical gradient
            c = (14 + i // 2, 16 + i // 2, 32 + i)
            pg.draw.rect(self.surf, c, (0, i * 14, WIDTH, 15))
        for x, y, r in self.stars:
            pg.draw.circle(self.surf, (90, 94, 130), (x, y), r)
        for x, h, w in self.skyline:                          # distant buildings
            pg.draw.rect(self.surf, (24, 26, 44), (x, GROUND - h - 40, w, h + 40))

    def _street(self):
        pg = self.pg
        pg.draw.rect(self.surf, (44, 44, 54), (0, GROUND, WIDTH, 24))           # sidewalk
        pg.draw.rect(self.surf, (30, 30, 36), (0, GROUND + 24, WIDTH, HEIGHT))  # road
        for x in range(0, WIDTH, 70):
            pg.draw.rect(self.surf, (70, 70, 80), (x, 528, 36, 3))

    def _districts(self, s):
        areas = ["Agri & supplies", "Reselling & importing", "Services", "Digital work", "Savings"]
        for a, name in enumerate(areas):
            cx = (SLOT_X[2 * a] + SLOT_X[2 * a + 1]) // 2
            self.text(name.upper(), self.f_small, MUTED, (cx, 100), "midtop")
            self.pg.draw.line(self.surf, shade(AREA_COLORS[a], 0.7), (cx - 70, 122), (cx + 70, 122), 2)

    # ---- buildings -------------------------------------------------------------
    def _building(self, s, k):
        pg, x, h = self.pg, SLOT_X[k], HEIGHTS[k]
        color, seen = AREA_COLORS[k // 2], s["reaction"][k] > 0
        final, focus = s["final"], k == s["focus"]
        closed = bool(final and final.get("scammed") and final["chosen"] == k)
        body = shade(color, 0.28 if closed else 0.55)
        top = GROUND - h
        if focus and not final:                                # light on the building being discussed
            glow = pg.Surface((120, 60), pg.SRCALPHA)
            pg.draw.ellipse(glow, (*ACCENT, 55), glow.get_rect())
            self.surf.blit(glow, (x - 60, GROUND - 20))
        pg.draw.rect(self.surf, body, (x - 38, top, 76, h))
        self._roof(k, x, top, color)
        pg.draw.rect(self.surf, shade(body, 0.6), (x - 9, GROUND - 34, 18, 34))            # door
        window = WARM if seen and not closed else DIM
        for wx in (x - 29, x + 15):
            pg.draw.rect(self.surf, window, (wx, top + 22, 14, 14), border_radius=2)
        self._icon(k, x, top + 56)
        if focus:
            pg.draw.rect(self.surf, ACCENT, (x - 39, top - 1, 78, h + 1), 2)
        words = s["names"][k].split()                                                      # name, two lines
        lines = [" ".join(words[: (len(words) + 1) // 2]), " ".join(words[(len(words) + 1) // 2:])]
        for i, line in enumerate(l for l in lines if l):
            self.text(line, self.f_small, TEXT if focus else MUTED, (x, GROUND + 4 + i * 12), "midtop")
        if seen:                                                                           # reaction face
            self._face(x - 12, top - 22, 9, s["reaction"][k])
        if s["verified"][k] != 0:
            self.badge(x + 14, top - 22, s["verified"][k])
        if closed:
            pg.draw.rect(self.surf, BAD, (x - 44, top + 40, 88, 20))
            self.text("CLOSED", self.f_small, BG, (x, top + 50), "center")
        elif final and final.get("size") and final["chosen"] == k:
            for i in range(3):                                                             # coins at the door
                pg.draw.circle(self.surf, WARM, (x - 20 + i * 4, GROUND - 4 - i * 4), 5)
                pg.draw.circle(self.surf, shade(WARM, 0.7), (x - 20 + i * 4, GROUND - 4 - i * 4), 5, 1)

    def _roof(self, k, x, top, color):
        pg = self.pg
        if k == 1:                                                                         # barn
            pg.draw.polygon(self.surf, (176, 70, 64), [(x - 44, top), (x, top - 26), (x + 44, top)])
        elif k in (2, 4):                                                                  # striped awning
            for i in range(8):
                pg.draw.rect(self.surf, color if i % 2 else (236, 236, 236), (x - 42 + i * 10.5, top - 8, 10.5, 14))
        elif k == 9:                                                                       # pediment
            pg.draw.polygon(self.surf, shade(color, 1.1), [(x - 44, top), (x, top - 22), (x + 44, top)])
        else:
            pg.draw.rect(self.surf, color, (x - 42, top - 8, 84, 10))

    def _icon(self, k, x, y):
        """A tiny pictogram on the front of each building."""
        pg, c = self.pg, (225, 228, 235)
        if k == 0:    # sprout
            pg.draw.line(self.surf, (120, 220, 130), (x, y + 10), (x, y - 4), 2)
            pg.draw.ellipse(self.surf, (120, 220, 130), (x - 9, y - 8, 9, 6))
            pg.draw.ellipse(self.surf, (120, 220, 130), (x, y - 12, 9, 6))
        elif k == 1:  # field rows
            for i in range(3):
                pg.draw.line(self.surf, (120, 220, 130), (x - 14, y - 4 + i * 6), (x + 14, y - 4 + i * 6), 2)
        elif k == 2:  # crates
            for dx in (-12, 2):
                pg.draw.rect(self.surf, (200, 150, 90), (x + dx, y, 11, 10), 2)
        elif k == 3:  # container ribs
            for dx in range(-30, 31, 10):
                pg.draw.line(self.surf, shade(AREA_COLORS[1], 0.9), (x + dx, y - 14), (x + dx, y + 14), 2)
        elif k == 4:  # phone
            pg.draw.rect(self.surf, c, (x - 6, y - 10, 12, 20), 2, border_radius=2)
        elif k == 5:  # scooter
            pg.draw.circle(self.surf, c, (x - 10, y + 6), 5, 2)
            pg.draw.circle(self.surf, c, (x + 10, y + 6), 5, 2)
            pg.draw.line(self.surf, c, (x - 10, y + 6), (x + 6, y - 4), 2)
        elif k == 6:  # laptop
            pg.draw.rect(self.surf, c, (x - 10, y - 8, 20, 12), 2)
            pg.draw.line(self.surf, c, (x - 14, y + 8), (x + 14, y + 8), 2)
        elif k == 7:  # neon "get rich" sign
            pg.draw.rect(self.surf, (255, 80, 170), (x - 28, y - 12, 56, 26), 2, border_radius=4)
            pg.draw.lines(self.surf, (255, 80, 170), False, [(x - 20, y + 6), (x - 6, y - 2), (x + 2, y + 2), (x + 18, y - 8)], 2)
        elif k == 8:  # coin
            pg.draw.circle(self.surf, WARM, (x, y), 9, 2)
            self.text("$", self.f_small, WARM, (x, y), "center")
        elif k == 9:  # columns
            for dx in (-18, -6, 6, 18):
                pg.draw.rect(self.surf, c, (x + dx - 2, y - 14, 4, 28))

    def _face(self, cx, cy, r, mood):
        pg, color = self.pg, REACTION_COLOR[int(mood)]
        pg.draw.circle(self.surf, color, (cx, cy), r)
        for dx in (-r * 0.38, r * 0.38):
            pg.draw.circle(self.surf, BG, (cx + dx, cy - r * 0.2), max(1, r // 5))
        mouth = pg.Rect(cx - r * 0.5, cy - r * 0.1, r, r * 0.8)
        if mood == 3:
            pg.draw.arc(self.surf, BG, mouth, math.pi, 2 * math.pi, 2)
        elif mood == 1:
            pg.draw.arc(self.surf, BG, mouth.move(0, r * 0.45), 0, math.pi, 2)
        else:
            pg.draw.line(self.surf, BG, (cx - r * 0.4, cy + r * 0.45), (cx + r * 0.4, cy + r * 0.45), 2)

    # ---- characters ------------------------------------------------------------
    def _walkers(self, s, x, walking):
        pg = self.pg
        bob = 3 * math.sin(self.tick_count * 0.9) if walking else 0
        swing = 6 * math.sin(self.tick_count * 0.9) if walking else 0
        feet = 538
        # the user
        pg.draw.line(self.surf, (80, 84, 110), (x - 4, feet - 22), (x - 4 + swing, feet), 4)
        pg.draw.line(self.surf, (80, 84, 110), (x + 4, feet - 22), (x + 4 - swing, feet), 4)
        pg.draw.rect(self.surf, ACCENT, (x - 12, feet - 56 + bob, 24, 36), border_radius=8)
        head = (int(x), int(feet - 68 + bob))
        mood = s["reaction"][s["focus"]] if s["focus"] >= 0 else 2
        if s["event"] == "quit":
            mood = 1
        elif s["final"] and s["final"].get("size"):
            mood = 1 if (s["final"]["scammed"] or s["final"]["panic"]) else 3
        pg.draw.circle(self.surf, SKIN, head, 11)
        for dx in (-4, 4):
            pg.draw.circle(self.surf, BG, (head[0] + dx, head[1] - 2), 1)
        mouth = pg.Rect(head[0] - 5, head[1], 10, 8)
        if mood == 3:
            pg.draw.arc(self.surf, BG, mouth, math.pi, 2 * math.pi, 2)
        elif mood == 1:
            pg.draw.arc(self.surf, BG, mouth.move(0, 3), 0, math.pi, 2)
        else:
            pg.draw.line(self.surf, BG, (head[0] - 4, head[1] + 5), (head[0] + 4, head[1] + 5), 2)
        # engagement bar above the head
        level = max(0.0, min(1.0, s["engagement"]))
        color = GOOD if level > 0.5 else WARN if level > 0.25 else BAD
        pg.draw.rect(self.surf, (50, 52, 66), (x - 20, feet - 86, 40, 5), border_radius=2)
        pg.draw.rect(self.surf, color, (x - 20, feet - 86, int(40 * level), 5), border_radius=2)
        # the guide robot, hovering beside the user
        gx, gy = x + 42, feet - 40 + 4 * math.sin(self.tick_count * 0.25)
        pg.draw.line(self.surf, GUIDE, (gx, gy - 14), (gx, gy - 22), 2)
        pg.draw.circle(self.surf, WARM, (gx, gy - 24), 3)
        pg.draw.rect(self.surf, GUIDE, (gx - 11, gy - 14, 22, 18), border_radius=6)
        for dx in (-4, 4):
            pg.draw.rect(self.surf, BG, (gx + dx - 2, gy - 9, 4, 5), border_radius=1)
        pg.draw.rect(self.surf, shade(GUIDE, 0.7), (gx - 8, gy + 6, 16, 12), border_radius=4)
        self.text("guide", self.f_small, GUIDE, (gx, gy - 28), "midbottom")

    def _coins(self, s, x, p):
        k = s["focus"]
        for j in range(8):
            t = max(0.0, min(1.0, p * 1.5 - j * 0.09))
            cx = x + (SLOT_X[k] - x) * t
            cy = 498 + (GROUND - 24 - 498) * t - 46 * math.sin(math.pi * t)
            self.pg.draw.circle(self.surf, WARM, (int(cx), int(cy)), 5)
            self.pg.draw.circle(self.surf, shade(WARM, 0.7), (int(cx), int(cy)), 5, 1)

    # ---- speech bubble, evidence and result cards -------------------------------
    def _card(self, rect, edge):
        self.pg.draw.rect(self.surf, (24, 26, 40), rect, border_radius=8)
        self.pg.draw.rect(self.surf, edge, rect, 1, border_radius=8)

    def _star(self, cx, cy, r, color):
        pts = [(cx + (r if i % 2 == 0 else r * 0.45) * math.sin(i * math.pi / 5),
                cy - (r if i % 2 == 0 else r * 0.45) * math.cos(i * math.pi / 5)) for i in range(10)]
        self.pg.draw.polygon(self.surf, color, pts)

    def _overlays(self, s, user_x):
        pg, k, event, final = self.pg, s["focus"], s["event"], s["final"]
        if k < 0:
            return
        x, top = SLOT_X[k], GROUND - HEIGHTS[k]
        left = max(8, min(WIDTH - 148, x - 70))
        if final and final.get("size"):                                    # result card after a commit
            real = final["real_growth"] * 100
            self._card(pg.Rect(left, top - 124, 140, 70), ACCENT)
            self.text("A year later", self.f_small, MUTED, (left + 10, top - 116))
            if final["scammed"]:
                self.text("Lost to a scam", self.f_med, BAD, (left + 10, top - 98))
            else:
                self.text(f"{real:+.1f}% real", self.f_med, GOOD if real >= 0 else BAD, (left + 10, top - 98))
                self.text("after inflation", self.f_small, MUTED, (left + 10, top - 78))
        elif s["reaction"][k] > 0:                                         # evidence card
            self._card(pg.Rect(left, top - 124, 140, 70), MUTED)
            self.text("Market", self.f_small, MUTED, (left + 10, top - 116))
            trend = s["trend"][k]
            ax, ay = left + 112, top - 108
            if trend > 0.12:
                pg.draw.polygon(self.surf, GOOD, [(ax - 7, ay + 5), (ax + 7, ay + 5), (ax, ay - 7)])
            elif trend < -0.12:
                pg.draw.polygon(self.surf, BAD, [(ax - 7, ay - 5), (ax + 7, ay - 5), (ax, ay + 7)])
            else:
                pg.draw.line(self.surf, MUTED, (ax - 7, ay), (ax + 7, ay), 3)
            self.text("Reviews", self.f_small, MUTED, (left + 10, top - 96))
            stars = 1 + round((float(s["testi"][k]) + 1) / 2 * 4)
            for i in range(5):
                self._star(left + 18 + i * 22, top - 70, 8, WARM if i < stars else (60, 62, 80))
        word = {"explore": REACTION_WORD[int(s["reaction"][k])], "verify": None}.get(event)
        if event == "verify":
            legit = s["verified"][k] > 0
            word, color = ("checked: legit", GOOD) if legit else ("checked: SCAM", BAD)
        elif word:
            color = REACTION_COLOR[int(s["reaction"][k])]
        if word and not final:                                             # speech bubble by the user's head
            bx = max(8, user_x - 138)
            self._card(pg.Rect(bx, 470, 116, 26), color)
            self.text(word, self.f_small, color, (bx + 58, 483), "center")

    # ---- text on top -----------------------------------------------------------
    def _hud(self, s):
        self._header(s)
        profile = (f"{s['savings']:,.0f} RWF   |   {RISK_WORD[s['risk']]} risk   |   "
                   f"time: {TIME_WORD[s['time']].lower()}   |   interest: {s['interest'].lower()}")
        self.text(profile, self.f_small, MUTED, (30, 66))
        self.pg.draw.rect(self.surf, (18, 18, 20), (0, 548, WIDTH, 52))
        self.pg.draw.line(self.surf, (48, 48, 54), (0, 548), (WIDTH, 548), 1)
        self.text("Last action", self.f_small, MUTED, (30, 556))
        self.text(s["last_text"], self.f_med, TEXT, (30, 574))
        r = s["last_reward"]
        self.text("Last reward", self.f_small, MUTED, (880, 556))
        self.text(f"{r:+.2f}", self.f_med, GOOD if r >= 0 else BAD, (880, 574))
