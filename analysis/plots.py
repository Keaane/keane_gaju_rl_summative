"""Figures for the report (Task 8). Every figure is a PNG (dpi 200).

  uv run python -m analysis.plots --impl torchrl       # the final (TorchRL) figures -> assets/plots/
  uv run python -m analysis.plots                      # plain PyTorch figures -> assets/plots/plain_pytorch/,
                                                       # plus the two environment screenshots -> assets/plots/
  uv run python -m analysis.plots --out-dir scratch/plots --skip-screenshots

Reads only files already in the repo: logs/torchrl/ or logs/sweeps/ and logs/sweeps_200k/ (100,000-step sweep and
200,000-step reruns), logs/reference.json and assets/torchrl/final_results.csv or assets/final_results.csv.
The plain figures in assets/plots/plain_pytorch/ were drawn before Figure 2 was narrowed to 4.2 inches. Colours follow one fixed order (a colour-blind-safe
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
CHOSEN_RUN = {"ppo": 10, "a2c": 3, "reinforce": 9, "dqn": 1}      # final 200,000-step settings (models/README.md)
# Where the data lives. use_impl("torchrl") switches to the TorchRL results (logs/torchrl/, assets/torchrl/).
SWEEP_DIR, RERUN_DIR, FINAL_CSV = "logs/sweeps", "logs/sweeps_200k", "assets/final_results.csv"


def use_impl(impl="plain"):
    """Point every figure at one implementation's results. Plain is the default and changes nothing."""
    global SWEEP_DIR, RERUN_DIR, FINAL_CSV, CHOSEN_RUN
    if impl == "torchrl":
        SWEEP_DIR, RERUN_DIR, FINAL_CSV = "logs/torchrl/sweeps", "logs/torchrl/sweeps_200k", "assets/torchrl/final_results.csv"
        with open("logs/torchrl/chosen.json") as f:
            CHOSEN_RUN = {algo: c["run"] for algo, c in json.load(f).items()}


def ppo_added_later():
    """The note for the two 10-setting figures, when PPO's final setting is one of the two added later."""
    run = CHOSEN_RUN["ppo"]
    return f" (PPO's, run {run}, was added later)" if run >= 10 else ""
SEEDS = (0, 1, 2)
MAIN_SETTINGS = range(10)   # the original 10 settings per algorithm; PPO settings 10-11 were added after model selection

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
        "font.family": "Poppins", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "medium",
        "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False, "lines.linewidth": 2,
        "hatch.color": "white", "hatch.linewidth": 0.8,           # white patterns on coloured bars
    })


def finish(fig, title, sample_note, path):
    """Title that states the finding, a sample-size line at the bottom, then save."""
    w, h = fig.get_size_inches()
    # Figures are drawn at the width they are placed at in the report, so these sizes are the real sizes on the page.
    note = textwrap.fill(sample_note, width=int(w * 15))       # about 15 characters per inch at 8 pt
    lines = note.count("\n") + 1
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="medium")   # one short line
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, color=INK_2, linespacing=1.3)
    fig.tight_layout(rect=(0, (0.06 + 0.14 * lines) / h, 1, 1))   # tight_layout already leaves room for the title
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---- data ----------------------------------------------------------------------------------
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def curve(algo, run, seed, folder=None):
    """One training run's log: a dict of column -> numpy array (one value per evaluation)."""
    rows = read_csv(f"{folder or RERUN_DIR}/{algo}_run{run:02d}_seed{seed}.csv")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def final_curves(algo):
    return [curve(algo, CHOSEN_RUN[algo], s) for s in SEEDS]


def main_sweep_runs(algo):
    """The 100,000-step sweep summary grouped by setting, keeping only the original 10 settings."""
    by_run = {}
    for r in read_csv(f"{SWEEP_DIR}/{algo}_summary.csv"):
        if int(r["run"]) in MAIN_SETTINGS:
            by_run.setdefault(int(r["run"]), []).append(r)
    return by_run


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
LEARNING_CURVES_WIDTH = 4.2          # inches; placed at this width in the report (Figure 2)


def label_line_ends(fig, ax, ends, gap_pt=9):
    """Name each line at its right end, in the order the lines end, at least gap_pt apart (8 pt text)."""
    fig.canvas.draw()                                          # fixes the axes, so data -> points is known
    to_pt = lambda y: ax.transData.transform((0, y))[1] * 72 / fig.dpi
    order = sorted(ends, key=lambda a: -ends[a][1])            # highest end first
    placed = []
    for algo in order:
        y = to_pt(ends[algo][1])
        if placed and placed[-1] - y < gap_pt:
            y = placed[-1] - gap_pt
        placed.append(y)
        ax.annotate(NAMES[algo], ends[algo], xytext=(6, y - to_pt(ends[algo][1])), textcoords="offset points",
                    va="center", fontsize=8, color=INK)


def fig_learning_curves(out, rule_val):
    fig, ax = plt.subplots(figsize=(LEARNING_CURVES_WIDTH, 2.75))
    ends = {}                                                   # algo -> where its mean line ends
    for algo in ALGOS:
        cs = final_curves(algo)
        steps = cs[0]["step"]
        ys = np.array([c["reward"] for c in cs])
        for y in ys:                                            # each seed, thin
            ax.plot(steps, y, color=COLOR[algo], linewidth=0.8, alpha=0.35)
        ax.fill_between(steps, ys.min(0), ys.max(0), color=COLOR[algo], alpha=0.12, linewidth=0)
        ax.plot(steps, ys.mean(0), color=COLOR[algo], linestyle=LINESTYLE[algo], marker=MARKER[algo],
                markevery=4, markersize=5, label=f"{NAMES[algo]} (run {CHOSEN_RUN[algo]})")
        ends[algo] = (steps[-1], ys.mean(0)[-1])
    ax.axhline(TARGET_SAFE, color=INK_2, linestyle=":", linewidth=1)
    ax.axhline(rule_val, color=RULE_GREY, linestyle="--", linewidth=1)
    ax.set_xlabel("training steps")
    ax.set_ylabel("reward per session\n(validation users)")
    ax.set_ylim(-1.5, 1.0)                                       # early random-like scores go lower; see the CSVs
    ax.set_xlim(0, 215_000)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.legend(loc="lower right", fontsize=8, ncol=2, handlelength=1.8, markerscale=0.7, columnspacing=1.0)
    label_line_ends(fig, ax, ends)
    return finish(fig, "PPO and DQN climb past the safe plan",
                  "Validation scores (up to 200,000 steps) pick checkpoints, so they lean optimistic vs test. "
                  f"Dotted: target 0.10; dashed: rule ({rule_val:+.2f}).",
                  f"{out}/learning_curves.png")


# ---- 2. DQN loss and exploration -------------------------------------------------------------
FIG3_WIDTH = 3.1          # inches; Figure 3's two figures sit side by side in the report, each at this width


def fig_dqn(out):
    cs = final_curves("dqn")
    steps = cs[0]["step"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(FIG3_WIDTH, 3.3), sharex=True)
    for s, c in zip(SEEDS, cs):
        a.plot(steps, c["loss"], color=COLOR["dqn"], linewidth=1.2, alpha=0.5 + 0.25 * s, label=f"seed {s}")
    a.axvline(50_000, color=INK_2, linestyle=":", linewidth=1)
    a.set_ylim(0, None)
    a.set_title("loss (mean of last 500 updates)", loc="left")
    a.set_ylabel("loss")
    b.plot(steps, cs[0]["epsilon"], color=COLOR["dqn"])
    b.set_title("epsilon (share of random actions)", loc="left")
    b.set_ylabel("epsilon")
    b.set_ylim(0, 1.05)
    b.set_xlabel("training steps")
    b.xaxis.set_major_formatter(THOUSANDS)
    a.legend(fontsize=8, loc="lower center", ncol=3, handlelength=1.0, columnspacing=0.8, borderaxespad=0.1)  # empty band under the curves
    return finish(fig, "DQN: loss dips, then levels off",
                  "DQN run 1 (learning rate 3e-4), 3 seeds, 200,000 steps, logged every 10,000 steps.",
                  f"{out}/dqn_loss_epsilon.png")


# ---- 3. policy-gradient entropy and policy loss ----------------------------------------------
def fig_pg(out):
    fig, (a, b) = plt.subplots(2, 1, figsize=(FIG3_WIDTH, 3.3), sharex=True)
    for algo in ("ppo", "a2c", "reinforce"):
        cs = final_curves(algo)
        steps = cs[0]["step"]
        for ax, key in ((a, "entropy"), (b, "policy_loss")):
            ys = np.array([c[key] for c in cs])
            ax.fill_between(steps, ys.min(0), ys.max(0), color=COLOR[algo], alpha=0.15, linewidth=0)
            ax.plot(steps, ys.mean(0), color=COLOR[algo], linestyle=LINESTYLE[algo], marker=MARKER[algo],
                    markevery=4, markersize=5, label=NAMES[algo])
    a.axhline(np.log(15), color=INK_2, linestyle=":", linewidth=1)
    a.text(200_000, np.log(15) - 0.12, "fully random", fontsize=8, color=INK_2, ha="right", va="top")
    a.set_ylim(0, 4.4)                                          # headroom above "fully random" for the legend
    a.set_yticks([0, 1, 2, 3])
    a.set_title("entropy (spread of choices)", loc="left")
    a.set_ylabel("entropy (nats)")
    b.axhline(0, color=INK_2, linewidth=0.8)
    b.set_title("policy loss (noisy)", loc="left")
    b.set_ylabel("policy loss")
    b.set_xlabel("training steps")
    b.xaxis.set_major_formatter(THOUSANDS)
    a.legend(fontsize=8, loc="upper right", ncol=3, markerscale=0.7, handlelength=1.4, columnspacing=0.6,
             borderaxespad=0.1)                                     # one row above the "fully random" line
    return finish(fig, "PPO narrows its choices slowly",
                  "3 seeds each (band = min to max); each policy-loss point is one update, so it is noisy.",
                  f"{out}/pg_entropy_policy_loss.png")


# ---- 4. convergence: steps to reach two targets ----------------------------------------------
def fig_convergence(out):
    never_y, budget = 125_000, 100_000
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.05), sharey=True)
    rng = np.random.default_rng(0)                              # fixed jitter so the figure never changes
    for ax, target, label in ((axes[0], TARGET_SAFE, "safe-plan level"), (axes[1], TARGET_BEYOND, "beyond safe plan")):
        for i, algo in enumerate(ALGOS):
            reached = 0
            for run in MAIN_SETTINGS:
                for s in SEEDS:
                    step = first_step_at(curve(algo, run, s, SWEEP_DIR), target)
                    x = i + rng.uniform(-0.22, 0.22)
                    final = run == CHOSEN_RUN[algo]
                    if step is None:
                        ax.scatter(x, never_y, marker="x", s=14, color=COLOR[algo], linewidths=1.4)
                    else:
                        reached += 1
                        ax.scatter(x, step, s=22 if final else 9, color=COLOR[algo], marker=MARKER[algo],
                                   edgecolors=INK if final else "white", linewidths=1.2 if final else 0.6, zorder=3)
            ax.text(i, -11_000, f"{reached}/30", ha="center", fontsize=8, color=INK_2)
        ax.axhline(budget, color=GRID, linewidth=1)
        ax.set_xticks(range(len(ALGOS)), ["PPO", "A2C", "REIN.", "DQN"])
        ax.set_title(f"{label} ({target:.2f})", loc="left")
        ax.set_xlim(-0.6, len(ALGOS) - 0.4)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("steps to reach")
    axes[0].set_yticks([0, 50_000, 100_000, never_y], ["0", "50k", "100k", "never"])
    axes[0].set_ylim(-20_000, never_y + 8_000)
    return finish(fig, "Most runs reach the safe plan; only PPO and DQN go beyond",
                  f"100,000-step sweep, 30 runs each; x = never; ringed = final setting{ppo_added_later()}.",
                  f"{out}/convergence.png")


# ---- 5. hyperparameter effect ---------------------------------------------------------------
def fig_hyperparameters(out):
    rule_test = json.load(open("logs/reference.json"))["best profile rule"]["reward"]
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.8), sharex=True)
    for ax, algo in zip(axes.flat, ALGOS):
        by_run = main_sweep_runs(algo)
        order = sorted(by_run, key=lambda k: np.mean([float(r["test_reward"]) for r in by_run[k]]))
        for y, run in enumerate(order):
            scores = np.array([float(r["test_reward"]) for r in by_run[run]])
            final = run == CHOSEN_RUN[algo]
            ax.barh(y, scores.mean(), height=0.62, color=COLOR[algo], alpha=1.0 if final else 0.45, linewidth=0)
            ax.errorbar(scores.mean(), y, xerr=scores.std(), color=INK, linewidth=1, capsize=2)
            ax.scatter(scores, [y] * len(scores), s=6, color=INK, zorder=3)
        ax.set_yticks(range(len(order)),
                      [("* " if r == CHOSEN_RUN[algo] else "") + setting_label(by_run[r][0]["config"]) for r in order])
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.axvline(rule_test, color=RULE_GREY, linestyle="--", linewidth=1)
        ax.set_title(NAMES[algo], fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("reward per session (test)")
    axes[0, 0].text(rule_test, 9.6, " rule", fontsize=8, color=INK_2)
    return finish(fig, "Settings matter for PPO and DQN; A2C and REINFORCE stay near +0.16",
                  f"300 test users; bar = mean of 3 seeds, dots = seeds; * = final setting{ppo_added_later()}; "
                  f"dashed = profile rule ({rule_test:+.2f}).",
                  f"{out}/hyperparameter_effect.png")


# ---- 6. generalization ----------------------------------------------------------------------
WORLD_LABELS = {"normal": "normal",
                "shifted": "more scams\n(assumed)",
                "low_savings": "low savings\n(assumed)",
                "high_savings": "high savings\n(assumed)",
                "new_users": "new users"}


def final_rows():
    """assets/final_results.csv as {(world, policy): row}."""
    return {(r["world"], r["policy"]): r for r in read_csv(FINAL_CSV)}


def fig_generalization(out):
    rows, worlds = final_rows(), list(WORLD_LABELS)
    series = ([(f"{a} (3 seeds)", NAMES[a], COLOR[a], HATCH[a]) for a in ALGOS]
              + [("best profile rule", "profile rule", RULE_GREY, RULE_HATCH)])
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(5.0, 2.3))
    for j, (policy, label, color, hatch) in enumerate(series):
        xs = np.arange(len(worlds)) - 0.4 + width * (j + 0.5)
        ys = [float(rows[w, policy]["reward"]) for w in worlds]
        se = [float(rows[w, policy]["reward_se"]) for w in worlds]
        ax.bar(xs, ys, width=width * 0.9, color=color, label=label, hatch=hatch, linewidth=0)
        ax.errorbar(xs, ys, yerr=se, fmt="none", color=INK, linewidth=0.9, capsize=2)
    oracle = [float(rows[w, "oracle"]["reward"]) for w in worlds]
    ax.scatter(np.arange(len(worlds)), oracle, marker="D", s=14, color="white", edgecolors=ORACLE_INK,
               linewidths=1.0, zorder=4, label="oracle")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(worlds)), [WORLD_LABELS[w] for w in worlds])
    ax.set_ylabel("reward per session")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=6, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.7)
    ax.set_ylim(None, max(oracle) + 0.25)
    return finish(fig, "PPO and DQN lose their edge when scams rise",
                  "500 users per world, mean of 3 seeds, +/- 1 SE. A2C and REINFORCE (SACCO only) are unaffected by more scams. "
                  "Shifted worlds are assumed.",
                  f"{out}/generalization.png")


# ---- 7. behaviour ---------------------------------------------------------------------------
REC_SHOWN = [("rec_sacco_savings", "SACCO savings"), ("rec_t_bills", "T-bills"),
             ("rec_agri_input_resale", "Agri input resale"), ("rec_mobile_money_agent", "Mobile money agent")]


def fig_behaviour(out):
    rows = final_rows()
    agents = [(f"{a} (3 seeds)", NAMES[a], COLOR[a]) for a in ALGOS] + [("best profile rule", "profile rule", RULE_GREY)]
    fig = plt.figure(figsize=(6.0, 6.2))                    # not in the report, so it can be taller
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 1))
    a, b, c = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])

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
    a.set_title("recommended opportunity", loc="left")
    a.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in OPP_COLORS + ["#b5b3ad", "#ecebe7"]]
    a.legend(handles, [n for _, n in REC_SHOWN] + ["other opportunities", "no plan recommended"],
             fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=1, handlelength=1.0)

    # (b) outcomes (normal world)
    metrics = [("plan", "plan"), ("quit", "quit"), ("scammed", "scam"),
               ("panic", "panic"), ("beat_inflation_share_of_plans", "beat\ninflation")]
    width = 0.8 / len(agents)
    for j, (policy, label, color) in enumerate(agents):
        xs = np.arange(len(metrics)) - 0.4 + width * (j + 0.5)
        algo = policy.split(" ")[0]
        b.bar(xs, [float(rows["normal", policy][m]) for m, _ in metrics], width=width * 0.9, color=color,
              label=label, hatch=HATCH.get(algo, RULE_HATCH), linewidth=0)
    b.set_xticks(range(len(metrics)), [m[1].replace("\n", " ") for m in metrics], rotation=35, ha="right")
    b.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    b.set_ylabel("share")
    b.set_title("outcomes (normal world)", loc="left")
    b.grid(axis="x", visible=False)
    b.legend(fontsize=8, ncol=1, loc="upper right", handlelength=1.0)

    # (c) scam-check use, normal vs shifted world
    for x, (policy, label, color) in enumerate(agents):
        for k, world in enumerate(("normal", "shifted")):           # normal = filled, shifted = outline only
            rate, xk = float(rows[world, policy]["verify_rate"]), x - 0.18 + 0.36 * k
            c.bar(xk, rate, width=0.34, color=color if world == "normal" else "white", edgecolor=color, linewidth=1.5)
            c.text(xk, rate + 0.004, f"{rate:.0%}", ha="center", fontsize=8, color=INK, rotation=90, va="bottom")   # zero bars stay readable
    c.set_xticks(range(len(agents)), ["PPO", "A2C", "REINFORCE", "DQN", "rule"], rotation=35, ha="right")
    c.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    c.set_ylim(0, max(0.2, 1.3 * max(bar.get_height() for bar in c.patches)))   # room for the rotated labels
    c.set_ylabel("sessions")
    c.set_title("scam checks: normal (filled)\nvs more scams (outline)", loc="left")
    c.grid(axis="x", visible=False)
    return finish(fig, "Only PPO and DQN look beyond SACCO savings",
                  "500 test users, mean of 3 seeds. Beat inflation = share of plans above 19.5%.",
                  f"{out}/behaviour.png")


# ---- 8. paired difference from the profile rule ---------------------------------------------
def fig_paired(out):
    rows = final_rows()
    fig, ax = plt.subplots(figsize=(6.0, 2.88))
    for y, algo in enumerate(ALGOS):
        r = rows["normal", f"{algo} (3 seeds)"]
        d, se = float(r["vs_rule_diff"]), float(r["vs_rule_se"])
        ax.plot([d - 2 * se, d + 2 * se], [y, y], color=COLOR[algo], linewidth=1.2)      # +/- 2 SE
        ax.plot([d - se, d + se], [y, y], color=COLOR[algo], linewidth=4, solid_capstyle="round")   # +/- 1 SE
        ax.scatter(d, y, s=60, color=COLOR[algo], marker=MARKER[algo], edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(d + 2 * se + 0.02, y, f"{d:+.2f} +/- {se:.2f}  {r['vs_rule_verdict']} ({abs(d) / se:.1f} SE)",
                va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.text(0.005, len(ALGOS) - 0.45, "same as the rule", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(ALGOS)), [NAMES[a] for a in ALGOS])
    ax.set_ylim(len(ALGOS) - 0.3, -0.7)                          # top to bottom: PPO first
    ax.set_xlim(-0.35, 1.45)
    ax.set_xlabel("reward difference per user: agent minus profile rule")
    ax.grid(axis="y", visible=False)
    return finish(fig, "PPO and DQN clearly beat the profile rule",
                  "500 test users, paired; thick = 1 SE, thin = 2 SE; over 3 SE = clear.",
                  f"{out}/paired_vs_rule.png")


# ---- 9. screenshots of the two views ---------------------------------------------------------
SCREEN_SEED = 100249                                            # one fixed test user (as in the README)
PLACED = 3.1                                                    # width (inches) the screenshots are placed at


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
        total = PLACED * h / w + 0.75                       # 0.75 in under the picture for a 3-line caption
        fig = plt.figure(figsize=(PLACED, total))
        ax = fig.add_axes((0, 0.72 / total, 1, PLACED * h / w / total))
        ax.imshow(img)
        ax.axis("off")
        caption = textwrap.fill(f"{what} at the end of one session, guided by the hand-written adaptive guide "
                                f"(not a trained agent). Test user {SCREEN_SEED}.", width=int(PLACED * 15))
        fig.text(0.01, 0.01, caption, fontsize=8, color=INK_2, linespacing=1.3)
        os.makedirs(out, exist_ok=True)
        path = f"{out}/{name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


# ---- run everything -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["plain", "torchrl"], default="plain", help="plain (default) or torchrl")
    p.add_argument("--out-dir", help="default: assets/plots (torchrl) or assets/plots/plain_pytorch (plain)")
    p.add_argument("--skip-screenshots", action="store_true", help="skip the two pygame screenshots")
    args = p.parse_args()
    use_impl(args.impl)
    screens_dir = args.out_dir or "assets/plots"         # the environment screenshots stay in assets/plots/
    args.out_dir = args.out_dir or ("assets/plots/plain_pytorch" if args.impl == "plain" else "assets/plots")
    if args.impl == "torchrl":
        args.skip_screenshots = True                     # the environment pictures do not depend on the algorithm
    set_style()
    made = [fig_learning_curves(args.out_dir, rule_validation_score()), fig_dqn(args.out_dir), fig_pg(args.out_dir),
            fig_convergence(args.out_dir), fig_hyperparameters(args.out_dir), fig_generalization(args.out_dir),
            fig_behaviour(args.out_dir), fig_paired(args.out_dir)]
    if not args.skip_screenshots:
        made += fig_screenshots(screens_dir)
    for path in made:
        print(f"{path}  {os.path.getsize(path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
