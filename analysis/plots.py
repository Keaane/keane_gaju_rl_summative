"""Figures for the report (Task 8). Every figure is a PNG (dpi 200) in assets/plots/.

  uv run python -m analysis.plots                      # all figures
  uv run python -m analysis.plots --out-dir scratch/plots --skip-screenshots

Reads only files already in the repo: logs/sweeps/ (100,000-step sweep), logs/sweeps_200k/ (200,000-step
reruns), logs/reference.json and assets/final_results.csv. Colours follow one fixed order (a colour-blind-safe
reference palette): each algorithm keeps its colour in every figure, and baselines are grey.
Simulation only, not financial advice.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import matplotlib

matplotlib.use("Agg")                     # draw to files, no window
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib import font_manager, ticker   # noqa: E402

THOUSANDS = ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k")   # 50000 -> "50k"

ALGOS = ["ppo", "a2c", "reinforce", "dqn"]
NAMES = {"ppo": "PPO", "a2c": "A2C", "reinforce": "REINFORCE", "dqn": "DQN"}
CHOSEN_RUN = {"ppo": 5, "a2c": 3, "reinforce": 9, "dqn": 1}       # final 200,000-step settings (models/README.md)
SEEDS = (0, 1, 2)

# Colours: reference categorical palette, slots in fixed order. Slots 1-4 = algorithms, 5-8 = opportunities.
COLOR = {"ppo": "#2a78d6", "a2c": "#eb6834", "reinforce": "#1baf7a", "dqn": "#eda100"}
# Second encoding so every figure still reads in grayscale: line style, marker and bar hatch per algorithm.
LINESTYLE = {"ppo": "-", "a2c": "--", "reinforce": "-.", "dqn": ":"}
MARKER = {"ppo": "o", "a2c": "s", "reinforce": "^", "dqn": "D"}
HATCH = {"ppo": "", "a2c": "///", "reinforce": "...", "dqn": "xxx"}
RULE_HATCH = "---"                                                  # profile rule bars, so they differ from PPO in grey
OPP_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948"]
RULE_GREY, LIGHT_GREY, ORACLE_INK = "#8a8985", "#cfcdc8", "#0b0b0b"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

TARGET_SAFE, TARGET_BEYOND = 0.10, 0.30                         # validation reward targets for "converged"
ASSUMED = "assumed conditions"                                  # label for the shifted worlds


# ---- style ---------------------------------------------------------------------------------
def set_style():
    """White background, Poppins, thin recessive axes and grid."""
    for f in ("assets/fonts/Poppins-Regular.ttf", "assets/fonts/Poppins-Medium.ttf"):
        if os.path.exists(f):
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "Poppins", "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "medium",
        "axes.labelsize": 9, "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False, "lines.linewidth": 2,
        "hatch.color": "white", "hatch.linewidth": 0.8,           # white patterns on coloured bars
    })


def finish(fig, title, sample_note, path):
    """Title that states the finding, a sample-size line at the bottom, then save."""
    w, h = fig.get_size_inches()
    note = textwrap.fill(sample_note, width=int(w * 17))       # about 17 characters per inch at this size
    lines = note.count("\n") + 1
    title = textwrap.fill(title, width=int(w * 10))           # about 10 title characters per inch
    title_lines = title.count("\n") + 1
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="medium")
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=7.5, color=INK_2, linespacing=1.4)
    fig.tight_layout(rect=(0, (0.12 + 0.15 * lines) / h, 1, 1 - (0.3 + 0.22 * (title_lines - 1)) / h))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---- data ----------------------------------------------------------------------------------
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def curve(algo, run, seed, folder="logs/sweeps_200k"):
    """One training run's log: a dict of column -> numpy array (one value per evaluation)."""
    rows = read_csv(f"{folder}/{algo}_run{run:02d}_seed{seed}.csv")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def final_curves(algo):
    return [curve(algo, CHOSEN_RUN[algo], s) for s in SEEDS]


def first_step_at(c, target):
    """First evaluation step where validation reward reached the target, or None if it never did."""
    hit = np.nonzero(c["reward"] >= target)[0]
    return int(c["step"][hit[0]]) if hit.size else None


SHORT = {"lr": "learning rate","gamma": "gamma", "buffer_size": "buffer", "batch_size": "batch",
         "eps_decay_steps": "explore", "target_update": "target every", "baseline": "baseline",
         "episodes_per_update": "sessions/update", "hidden": "hidden", "ent_coef": "entropy",
         "n_steps": "update every", "vf_coef": "value weight", "gae_lambda": "GAE lambda",
         "clip": "clip", "epochs": "epochs", "minibatch": "minibatch", "rollout_steps": "rollout"}


def setting_label(config_json):
    """'{"lr": 0.0003}' -> 'lr 3e-4'; '{}' -> 'default'."""
    cfg = json.loads(config_json)
    if not cfg:
        return "default"
    parts = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            text = "on" if v else "off"
        elif isinstance(v, float) and 0 < v < 0.01:
            text = f"{v:.0e}".replace("e-0", "e-")
        elif isinstance(v, int) and v >= 1000:
            text = f"{v // 1000}k"
        else:
            text = f"{v:g}"
        parts.append(f"{SHORT.get(k, k)} {text}")
    return ", ".join(parts)


def rule_validation_score(n_users=200):
    """The best profile rule on the validation users, so curves and line use the same users."""
    from analysis.checks import best_rule, run
    from environment.baselines import static_policy
    from training.common import VAL_SEED
    return float(run(static_policy(best_rule(320, {})), n_users, VAL_SEED)[0])


# ---- 1. learning curves ---------------------------------------------------------------------
def fig_learning_curves(out, rule_val):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for algo in ALGOS:
        cs = final_curves(algo)
        steps = cs[0]["step"]
        ys = np.array([c["reward"] for c in cs])
        for y in ys:                                            # each seed, thin
            ax.plot(steps, y, color=COLOR[algo], linewidth=0.8, alpha=0.35)
        ax.fill_between(steps, ys.min(0), ys.max(0), color=COLOR[algo], alpha=0.12, linewidth=0)
        ax.plot(steps, ys.mean(0), color=COLOR[algo], linestyle=LINESTYLE[algo], marker=MARKER[algo],
                markevery=4, markersize=5, label=f"{NAMES[algo]} (run {CHOSEN_RUN[algo]})")
        nudge = {"a2c": -7, "reinforce": 7}.get(algo, 0)        # A2C and REINFORCE end at nearly the same value
        ax.annotate(NAMES[algo], (steps[-1], ys.mean(0)[-1]), xytext=(6, nudge), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.axhline(TARGET_SAFE, color=INK_2, linestyle=":", linewidth=1)
    ax.text(105_000, TARGET_SAFE - 0.07, "target 0.10", fontsize=7.5, color=INK_2)
    ax.axhline(rule_val, color=RULE_GREY, linestyle="--", linewidth=1)
    ax.text(105_000, rule_val - 0.09, f"best profile rule on the same users ({rule_val:+.2f})", fontsize=7.5, color=INK_2)
    ax.set_xlabel("training steps")
    ax.set_ylabel("average reward per session (validation users)")
    ax.set_ylim(-1.5, 1.0)                                       # early random-like scores go lower; see the CSVs
    ax.set_xlim(0, 215_000)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.legend(loc="lower right", fontsize=8)
    return finish(fig, "PPO climbs past the safe plan; A2C and REINFORCE level off early",
                  "Final 200,000-step setting per algorithm, 3 seeds each (thin lines; band = min to max; thick = mean). "
                  "Validation: 200 users (seeds 200000+), checked every 10,000 steps; these users are used to pick checkpoints. "
                  f"The profile rule scores {rule_val:+.2f} on them but +0.05 on the 500 test users, so compare against it "
                  "in paired_vs_rule.png, not here. Lower early values are clipped.",
                  f"{out}/learning_curves.png")


# ---- 2. DQN loss and exploration -------------------------------------------------------------
def fig_dqn(out):
    cs = final_curves("dqn")
    steps = cs[0]["step"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.6))
    for s, c in zip(SEEDS, cs):
        a.plot(steps, c["loss"], color=COLOR["dqn"], linewidth=1.2, alpha=0.5 + 0.25 * s, label=f"seed {s}")
    a.axvline(50_000, color=INK_2, linestyle=":", linewidth=1)
    a.set_ylim(0, None)
    a.set_title("training loss (mean of the last 500 updates)", fontsize=9, loc="left")
    a.set_ylabel("loss")
    b.plot(steps, cs[0]["epsilon"], color=COLOR["dqn"])
    b.set_title("exploration: share of random actions (epsilon)", fontsize=9, loc="left")
    b.set_ylabel("epsilon")
    b.set_ylim(0, 1.05)
    for ax in (a, b):
        ax.set_xlabel("training steps")
        ax.xaxis.set_major_formatter(THOUSANDS)
    a.legend(fontsize=7.5)
    return finish(fig, "DQN's loss dips while it explores, then levels off near 0.15 once exploration ends",
                  "DQN, final setting (run 1: learning rate 3e-4), 3 seeds, 200,000 steps; values logged every 10,000 steps. "
                  "Epsilon is the same schedule for every seed.",
                  f"{out}/dqn_loss_epsilon.png")


# ---- 3. policy-gradient entropy and policy loss ----------------------------------------------
def fig_pg(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for algo in ("ppo", "a2c", "reinforce"):
        cs = final_curves(algo)
        steps = cs[0]["step"]
        for ax, key in ((a, "entropy"), (b, "policy_loss")):
            ys = np.array([c[key] for c in cs])
            ax.fill_between(steps, ys.min(0), ys.max(0), color=COLOR[algo], alpha=0.15, linewidth=0)
            ax.plot(steps, ys.mean(0), color=COLOR[algo], linestyle=LINESTYLE[algo], marker=MARKER[algo],
                    markevery=4, markersize=5, label=NAMES[algo])
    a.axhline(np.log(15), color=INK_2, linestyle=":", linewidth=1)
    a.text(60_000, np.log(15) + 0.06, "most random possible (15 actions)", fontsize=7.5, color=INK_2)
    a.set_ylim(0, 3.05)
    a.set_title("entropy: how spread out the action choices are", fontsize=9, loc="left")
    a.set_ylabel("entropy (nats)")
    b.axhline(0, color=INK_2, linewidth=0.8)
    b.set_title("policy loss (one update at each check: noisy)", fontsize=9, loc="left")
    b.set_ylabel("policy loss")
    for ax in (a, b):
        ax.set_xlabel("training steps")
        ax.xaxis.set_major_formatter(THOUSANDS)
    a.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.84))   # empty area under the "most random" line
    return finish(fig, "A2C and REINFORCE narrow their choices early; PPO narrows gradually",
                  "Final 200,000-step setting per algorithm, 3 seeds each (line = mean, band = min to max), logged every "
                  "10,000 steps. Each policy-loss point comes from one update, so it is noisy.",
                  f"{out}/pg_entropy_policy_loss.png")


# ---- 4. convergence: steps to reach two targets ----------------------------------------------
def fig_convergence(out):
    never_y, budget = 112_000, 100_000
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), sharey=True)
    rng = np.random.default_rng(0)                              # fixed jitter so the figure never changes
    for ax, target, label in ((axes[0], TARGET_SAFE, "safe-plan level"), (axes[1], TARGET_BEYOND, "beyond the safe plan")):
        for i, algo in enumerate(ALGOS):
            reached = 0
            for run in range(10):
                for s in SEEDS:
                    step = first_step_at(curve(algo, run, s, "logs/sweeps"), target)
                    x = i + rng.uniform(-0.22, 0.22)
                    final = run == CHOSEN_RUN[algo]
                    if step is None:
                        ax.scatter(x, never_y, marker="x", s=28, color=COLOR[algo], linewidths=1.4)
                    else:
                        reached += 1
                        ax.scatter(x, step, s=40 if final else 22, color=COLOR[algo], marker=MARKER[algo],
                                   edgecolors=INK if final else "white", linewidths=1.2 if final else 0.6, zorder=3)
            ax.text(i, -9_000, f"{reached}/30", ha="center", fontsize=8, color=INK_2)
        ax.axhline(budget, color=GRID, linewidth=1)
        ax.set_xticks(range(len(ALGOS)), [NAMES[a] for a in ALGOS])
        ax.set_title(f"{label}: validation reward {target:.2f} or more", fontsize=9, loc="left")
        ax.set_xlim(-0.6, len(ALGOS) - 0.4)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("training steps to first reach the target")
    axes[0].set_yticks([0, 20_000, 40_000, 60_000, 80_000, 100_000, never_y], ["0", "20k", "40k", "60k", "80k", "100k", "never"])
    axes[0].set_ylim(-16_000, never_y + 8_000)
    return finish(fig, "Most runs of every algorithm reach the safe-plan level; only PPO and DQN sometimes get beyond it",
                  "100,000-step sweep: 10 settings x 3 seeds = 30 runs per algorithm (one dot each; x = never reached; "
                  "dark ring = the final setting). Validation: 200 users, checked every 10,000 steps. n/30 = runs that reached it.",
                  f"{out}/convergence.png")


# ---- 5. hyperparameter effect ---------------------------------------------------------------
def fig_hyperparameters(out):
    rule_test = json.load(open("logs/reference.json"))["best profile rule"]["reward"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), sharex=True)
    for ax, algo in zip(axes.flat, ALGOS):
        by_run = {}
        for r in read_csv(f"logs/sweeps/{algo}_summary.csv"):
            by_run.setdefault(int(r["run"]), []).append(r)
        order = sorted(by_run, key=lambda k: np.mean([float(r["test_reward"]) for r in by_run[k]]))
        for y, run in enumerate(order):
            scores = np.array([float(r["test_reward"]) for r in by_run[run]])
            final = run == CHOSEN_RUN[algo]
            ax.barh(y, scores.mean(), height=0.62, color=COLOR[algo], alpha=1.0 if final else 0.45, linewidth=0)
            ax.errorbar(scores.mean(), y, xerr=scores.std(), color=INK, linewidth=1, capsize=2)
            ax.scatter(scores, [y] * len(scores), s=10, color=INK, zorder=3)
        ax.set_yticks(range(len(order)),
                      [("* " if r == CHOSEN_RUN[algo] else "") + setting_label(by_run[r][0]["config"]) for r in order])
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.axvline(rule_test, color=RULE_GREY, linestyle="--", linewidth=1)
        ax.set_title(NAMES[algo], fontsize=10, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("average reward per session (test users)")
    axes[0, 0].text(rule_test, 9.6, " profile rule", fontsize=7, color=INK_2)
    return finish(fig, "Settings change PPO and DQN a lot; A2C and REINFORCE land near +0.16 unless the learning rate is too low",
                  "100,000-step sweep, 10 settings x 3 seeds per algorithm, each scored on the same 300 test users "
                  "(seeds 100000+). Bar = mean over seeds, line = spread, dots = seeds. * = setting of the final model. "
                  f"Dashed = best profile rule on those users ({rule_test:+.2f}).",
                  f"{out}/hyperparameter_effect.png")


# ---- 6. generalization ----------------------------------------------------------------------
WORLD_LABELS = {"normal": "normal world",
                "shifted": f"more scams and\nundecided users\n({ASSUMED})",
                "low_savings": f"low savings\n150,000 RWF\n({ASSUMED})",
                "high_savings": f"high savings\n1,800,000 RWF\n({ASSUMED})",
                "new_users": "new users,\nsame distribution"}


def final_rows():
    """assets/final_results.csv as {(world, policy): row}."""
    return {(r["world"], r["policy"]): r for r in read_csv("assets/final_results.csv")}


def fig_generalization(out):
    rows, worlds = final_rows(), list(WORLD_LABELS)
    series = ([(f"{a} (3 seeds)", NAMES[a], COLOR[a], HATCH[a]) for a in ALGOS]
              + [("best profile rule", "profile rule", RULE_GREY, RULE_HATCH)])
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for j, (policy, label, color, hatch) in enumerate(series):
        xs = np.arange(len(worlds)) - 0.4 + width * (j + 0.5)
        ys = [float(rows[w, policy]["reward"]) for w in worlds]
        se = [float(rows[w, policy]["reward_se"]) for w in worlds]
        ax.bar(xs, ys, width=width * 0.9, color=color, label=label, hatch=hatch, linewidth=0)
        ax.errorbar(xs, ys, yerr=se, fmt="none", color=INK, linewidth=0.9, capsize=2)
    oracle = [float(rows[w, "oracle"]["reward"]) for w in worlds]
    ax.scatter(np.arange(len(worlds)), oracle, marker="D", s=30, color="white", edgecolors=ORACLE_INK,
               linewidths=1.2, zorder=4, label="oracle (sees hidden truth)")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(worlds)), [WORLD_LABELS[w] for w in worlds], fontsize=8)
    ax.set_ylabel("average reward per session (500 users)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=6, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(None, max(oracle) + 0.5)
    return finish(fig, "PPO and DQN lead in the normal world but lose their edge when scams rise; A2C and REINFORCE are unaffected",
                  "500 users per world: test users (seeds 100000-100499) in the first four worlds, new users (seeds 300000-300499) "
                  "in the last. Agents: mean of 3 seeds. Bars +/- 1 standard error. A2C and REINFORCE recommend SACCO savings to "
                  "everyone, so the scam shift does not touch them (change -0.01 and +0.00). Shifted worlds are assumed conditions, not data.",
                  f"{out}/generalization.png")


# ---- 7. behaviour ---------------------------------------------------------------------------
REC_SHOWN = [("rec_sacco_savings", "SACCO savings"), ("rec_t_bills", "T-bills"),
             ("rec_agri_input_resale", "Agri input resale"), ("rec_mobile_money_agent", "Mobile money agent")]


def fig_behaviour(out):
    rows = final_rows()
    agents = [(f"{a} (3 seeds)", NAMES[a], COLOR[a]) for a in ALGOS] + [("best profile rule", "profile rule", RULE_GREY)]
    fig = plt.figure(figsize=(11, 7.6))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1), width_ratios=(1.15, 1))
    a, b, c = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])

    # (a) which opportunity each guide recommends (normal world)
    guides = agents + [("adaptive hand-written", "adaptive guide", None), ("oracle", "oracle", None)]
    for y, (policy, label, _) in enumerate(guides):
        r, left = rows["normal", policy], 0.0
        shares = [float(r[col]) for col, _ in REC_SHOWN]
        recommended = sum(float(r[k]) for k in r if k.startswith("rec_"))      # users given any plan
        parts = list(zip(shares, OPP_COLORS)) + [(recommended - sum(shares), "#b5b3ad"), (1 - recommended, "#ecebe7")]
        for share, color in parts:
            a.barh(y, share, left=left, color=color, edgecolor="white", linewidth=1, height=0.65)
            left += share
    a.set_yticks(range(len(guides)), [g[1] for g in guides])
    a.invert_yaxis()
    a.set_xlim(0, 1)
    a.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    a.set_xlabel("share of users")
    a.set_title("which opportunity the guide recommends (normal world)", fontsize=9, loc="left")
    a.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in OPP_COLORS + ["#b5b3ad", "#ecebe7"]]
    a.legend(handles, [n for _, n in REC_SHOWN] + ["other opportunities", "no plan recommended"],
             fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)

    # (b) outcomes (normal world)
    metrics = [("plan", "plan made\nand adopted"), ("quit", "user quit"), ("scammed", "scammed"),
               ("panic", "panic sale"), ("beat_inflation_share_of_plans", "plan beat\ninflation")]
    width = 0.8 / len(agents)
    for j, (policy, label, color) in enumerate(agents):
        xs = np.arange(len(metrics)) - 0.4 + width * (j + 0.5)
        algo = policy.split(" ")[0]
        b.bar(xs, [float(rows["normal", policy][m]) for m, _ in metrics], width=width * 0.9, color=color,
              label=label, hatch=HATCH.get(algo, RULE_HATCH), linewidth=0)
    b.set_xticks(range(len(metrics)), [m[1] for m in metrics], fontsize=7.5)
    b.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    b.set_ylabel("share of users (beat inflation: share of plans)")
    b.set_title("outcomes (normal world)", fontsize=9, loc="left")
    b.grid(axis="x", visible=False)
    b.legend(fontsize=7, ncol=3, loc="upper right")

    # (c) scam-check use, normal vs shifted world
    for x, (policy, label, color) in enumerate(agents):
        for k, world in enumerate(("normal", "shifted")):           # normal = filled, shifted = outline only
            rate, xk = float(rows[world, policy]["verify_rate"]), x - 0.18 + 0.36 * k
            c.bar(xk, rate, width=0.34, color=color if world == "normal" else "white", edgecolor=color, linewidth=1.5)
            c.text(xk, rate + 0.006, f"{rate:.0%}", ha="center", fontsize=7, color=INK)   # zero bars stay readable
    c.set_xticks(range(len(agents)), [g[1] for g in agents], fontsize=8)
    c.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    c.set_ylim(0, 0.3)
    c.set_ylabel("sessions with a scam check")
    c.set_title(f"scam checks: normal world (filled) vs more scams\n({ASSUMED}, outline)", fontsize=9, loc="left")
    c.grid(axis="x", visible=False)
    return finish(fig, "Only PPO and DQN look beyond SACCO savings, and neither checks for scams more when scams rise",
                  "500 test users (seeds 100000-100499) per world. Agents: mean of the 3 seeds of the final setting. "
                  "'Plan beat inflation' is the share of adopted plans whose value grew faster than 19.5% inflation.",
                  f"{out}/behaviour.png")


# ---- 8. paired difference from the profile rule ---------------------------------------------
def fig_paired(out):
    rows = final_rows()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for y, algo in enumerate(ALGOS):
        r = rows["normal", f"{algo} (3 seeds)"]
        d, se = float(r["vs_rule_diff"]), float(r["vs_rule_se"])
        ax.plot([d - 2 * se, d + 2 * se], [y, y], color=COLOR[algo], linewidth=1.2)      # +/- 2 SE
        ax.plot([d - se, d + se], [y, y], color=COLOR[algo], linewidth=4, solid_capstyle="round")   # +/- 1 SE
        ax.scatter(d, y, s=60, color=COLOR[algo], marker=MARKER[algo], edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(d + 2 * se + 0.02, y, f"{d:+.2f} +/- {se:.2f}  {r['vs_rule_verdict']} ({abs(d) / se:.1f} SE)",
                va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.text(0.005, len(ALGOS) - 0.45, "same as the profile rule", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(ALGOS)), [NAMES[a] for a in ALGOS])
    ax.set_ylim(len(ALGOS) - 0.3, -0.7)                          # top to bottom: PPO first
    ax.set_xlim(-0.35, 1.25)
    ax.set_xlabel("reward difference per user: agent minus best profile rule (same users)")
    ax.grid(axis="y", visible=False)
    return finish(fig, "PPO and DQN clearly beat the profile rule; A2C and REINFORCE are not clearly better",
                  "Normal world, 500 test users (seeds 100000-100499); agents = mean of 3 seeds per user. Thick line = +/- 1 "
                  "standard error, thin = +/- 2. Verdict: under 2 SE none, 2-3 suggestive, over 3 clear.",
                  f"{out}/paired_vs_rule.png")


# ---- 9. screenshots of the two views ---------------------------------------------------------
SCREEN_SEED = 100249                                            # one fixed test user (as in the README)


def fig_screenshots(out):
    """Street scene and list dashboard from main.py, with a caption. Uses the hand-written guide."""
    paths = []
    for view, name, what in (("scene", "street_scene", "Street scene"), ("dashboard", "dashboard", "List dashboard")):
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, f"{name}.png")
            subprocess.run([sys.executable, "main.py", "--agent", "adaptive", "--seed", str(SCREEN_SEED),
                            "--view", view, "--screenshot", raw], check=True, capture_output=True)
            img = plt.imread(raw)
        h, w = img.shape[:2]
        fig = plt.figure(figsize=(w / 200, h / 200 + 0.55))
        ax = fig.add_axes((0, 0.4 / (h / 200 + 0.55), 1, h / 200 / (h / 200 + 0.55)))
        ax.imshow(img)
        ax.axis("off")
        caption = textwrap.fill(f"{what} at the end of one session, guided by the hand-written adaptive guide "
                                f"(not a trained agent). Test user {SCREEN_SEED}. Simulation only.", width=int(w / 200 * 19))
        fig.text(0.01, 0.01, caption, fontsize=7, color=INK_2, linespacing=1.4)
        os.makedirs(out, exist_ok=True)
        path = f"{out}/{name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


# ---- run everything -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="assets/plots")
    p.add_argument("--skip-screenshots", action="store_true", help="skip the two pygame screenshots")
    args = p.parse_args()
    set_style()
    made = [fig_learning_curves(args.out_dir, rule_validation_score()), fig_dqn(args.out_dir), fig_pg(args.out_dir),
            fig_convergence(args.out_dir), fig_hyperparameters(args.out_dir), fig_generalization(args.out_dir),
            fig_behaviour(args.out_dir), fig_paired(args.out_dir)]
    if not args.skip_screenshots:
        made += fig_screenshots(args.out_dir)
    for path in made:
        print(f"{path}  {os.path.getsize(path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
