"""
Rliable plotting of eval metrics. Looks for eval_parquet, requires eval to have run.
Runs for Task A and Task B
"""

import matplotlib

matplotlib.use("Agg")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rliable import library as rly
from rliable import metrics, plot_utils

from src.drones.config import ARMS, run_key
from src.tabular import plots as tp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRONES_ROOT = PROJECT_ROOT / "runs" / "drones"
ANCHORS_ROOT = DRONES_ROOT / "anchors"

WIDTH = 6.3
DPI = 200
REPS = 2000
COLORS = dict(zip(list(ARMS), sns.color_palette("colorblind", len(ARMS))))


def arm_dirs(task_dir: Path):
    return {name: task_dir / run_key(arm, "none") for name, arm in ARMS.items()}


def train_tensor(dirs: dict[str, Path], metric: str):
    """
    Training curve tensor
    """
    scores = {}
    frame_ref = None

    for label, d in dirs.items():
        mats = []
        for r in sorted(d.glob("run_*/train_log.parquet")):
            df = pd.read_parquet(r).sort_values("step")
            frames = df["step"].to_numpy()
            if frame_ref is None:
                frame_ref = frames

            assert np.array_equal(frames, frame_ref), f"train grid mismatch: {r}"

            mats.append(df[metric].to_numpy())

        scores[label] = np.stack(mats)[:, np.newaxis, :]

    return scores, frame_ref


def final_scores(dirs: dict[str, Path], metric: str, normalise: bool = False):
    """
    Scores for the final snapshot
    """
    scores = {}

    for label, d in dirs.items():
        df = tp.task_df(d)
        last = df[df["env_steps"] == df["env_steps"].max()].sort_values("run_no.")
        values = last[metric].to_numpy()

        if normalise:
            v_star, v_random = tp.anchors(df)
            values = (values - v_random) / (v_star - v_random)

        scores[label] = values[:, np.newaxis]

    return scores


def save_fig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def add_legend(fig, ax, ncol=5, drop=-0.16):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, drop),
        ncol=ncol,
        frameon=False,
        fontsize="small",
    )


def plot_sample_efficiency(ax, scores, frames, ylabel: str):
    iqm = lambda x: np.array(  # noqa - silence ruff
        [metrics.aggregate_iqm(x[..., f]) for f in range(x.shape[-1])]
    )

    pts, cis = rly.get_interval_estimates(scores, iqm, reps=REPS)
    plot_utils.plot_sample_efficiency_curve(
        frames / 1e6,
        pts,
        cis,
        list(scores),
        COLORS,
        xlabel="Environment Steps (M)",
        ylabel=ylabel,
        ax=ax,
        labelsize="medium",
        ticklabelsize="small",
        legend=False,
        linewidth=1.2,
        marker=".",
    )


def plot_performance_profile(ax, scores, xlabel, tau_max=1.0):
    taus = np.linspace(0.0, tau_max, 100)
    profiles, cis = rly.create_performance_profile(scores, taus)

    plot_utils.plot_performance_profiles(
        profiles,
        taus,
        cis,
        colors=COLORS,
        xlabel=xlabel,
        ax=ax,
        labelsize="medium",
        ticklabelsize="small",
        legend=False,
    )


def fig_task_a_efficiency(dirs: dict[str, Path], out_dir: Path):
    """
    Task A related curves with normalised return.
    Success rate not used since 7/9 arms failed.
    """
    scores, frames = tp.load_score_tensor(
        dirs, metric="eval_return_mean", normalise=True
    )

    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, 3.0))
    plot_sample_efficiency(ax, scores, frames, "IQM Normalised Return")
    fig.tight_layout()
    add_legend(fig, ax)

    save_fig(fig, out_dir / "r_task_a_efficiency.png")


def fig_task_b_efficiency(dirs: dict[str, Path], out_dir: Path):
    """
    Task B on success rate. not normalised. + targets reached
    """
    targets, _ = tp.load_score_tensor(dirs, metric="targets_reached", normalise=False)
    completion, frames = tp.load_score_tensor(
        dirs, metric="success_rate", normalise=False
    )

    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    plot_sample_efficiency(ax[0], completion, frames, "IQM Course Completion")
    plot_sample_efficiency(ax[1], targets, frames, "IQM Targets Reached (of 4)")
    fig.tight_layout()
    add_legend(fig, ax[0])

    save_fig(fig, out_dir / "r_task_b_efficiency.png")


def fig_task_a_decomposition(dirs: dict[str, Path], out_dir: Path):
    """
    episode length and reward per step for task a. episode length shows survival, reward per step shows hover+balance quality.
    """

    def ratio(dirs, num, denom):
        scores = {}
        frame_ref = None

        for label, d in dirs.items():
            df = tp.task_df(d)
            df = df.assign(_ratio=df[num] / df[denom])
            frames = np.sort(df["env_steps"].unique())
            if frame_ref is None:
                frame_ref = frames

            scores[label] = tp.pivot(df, "_ratio")[:, np.newaxis, :]

        return scores, frame_ref

    length, frames = tp.load_score_tensor(
        dirs, metric="episode_len_mean", normalise=False
    )
    per_step, _ = ratio(dirs, num="eval_return_mean", denom="episode_len_mean")

    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    plot_sample_efficiency(ax[0], length, frames, "IQM Episode Length")
    plot_sample_efficiency(ax[1], per_step, frames, "IQM Reward per Step")
    fig.tight_layout()
    add_legend(fig, ax[0])

    save_fig(fig, out_dir / "r_task_a_decomposition.png")


def plot_termination(ax, shares: pd.DataFrame):
    bottom = np.zeros(len(shares))
    # colors encode outcome, not arm
    colors = sns.color_palette("colorblind", len(shares.columns))

    for c, outcome in zip(colors, shares.columns):
        values = shares[outcome].to_numpy()
        ax.bar(
            shares.index,
            values,
            bottom=bottom,
            label=outcome,
            color=c,
            width=0.7,
        )
        bottom += values

    ax.set_ylabel("Fraction of Episodes", fontsize="medium")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize="small")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize="small")


def fig_task_b_termination(dirs: dict[str, Path], out_dir: Path):
    """
    Discriminate between timeout reasons for different archs.
    """
    termination_map = {
        "term_env_complete": "complete",
        "term_timeout": "timeout",
        "term_obstacle_collision": "obstacle",
        "term_floor_collision": "floor",
        "term_oob": "out of bounds",
    }

    def termination_reasons(dirs):
        rows = {}
        for label, d in dirs.items():
            df = tp.task_df(d)
            last = df[df["env_steps"] == df["env_steps"].max()]
            rows[label] = {
                name: last[col].mean() for col, name in termination_map.items()
            }

        return pd.DataFrame(rows).T[list(termination_map.values())]

    shares = termination_reasons(dirs).sort_values("complete", ascending=False)

    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    plot_termination(ax, shares)
    fig.tight_layout()
    add_legend(fig, ax, drop=-0.05)

    save_fig(fig, out_dir / "r_task_b_termination.png")


def fig_task_b_return_vs_success(dirs: dict[str, Path], out_dir: Path):
    """
    Return and task success are not correlated.
    """
    anchors = json.loads((ANCHORS_ROOT / "waypoint.json").read_text())

    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))

    for label, d in dirs.items():
        df = tp.task_df(d)
        last = df[df["env_steps"] == df["env_steps"].max()]
        x = last["eval_return_mean"].mean()
        y = last["success_rate"].mean()
        ax.scatter(x, y, color=COLORS[label], s=60, zorder=3, label=label)

    ax.scatter(
        anchors["v_star_s0"],
        anchors["pid"]["rate_complete"],
        marker="*",
        s=260,
        color="black",
        zorder=4,
        label="PID V*",
    )
    ax.set_xlabel("Mean Eval Return", fontsize="medium")
    ax.set_ylabel("Course Completion", fontsize="medium")
    ax.tick_params(labelsize="small")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    add_legend(fig, ax)

    save_fig(fig, out_dir / "r_task_b_return_vs_success.png")


def fig_task_cross_interaction(out_dir: Path):
    """
    Rank across different task, to show how the archs rank differently.
    Use rank, not returns, as they vary across tasks
    """
    a = final_scores(
        arm_dirs(DRONES_ROOT / "pole-balance"), "eval_return_mean", normalise=True
    )
    b = final_scores(arm_dirs(DRONES_ROOT / "waypoint"), "success_rate")

    score_a = pd.Series({k: v.mean() for k, v in a.items()})
    score_b = pd.Series({k: v.mean() for k, v in b.items()})

    rank_a = score_a.rank(ascending=False)
    rank_b = score_b.rank(ascending=False)

    tied_offsets = {"transformer": -9, "episodelstm": 9}

    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    for arm in score_a.index:
        ax.plot(
            [0, 1],
            [rank_a[arm], rank_b[arm]],
            marker="o",
            color=COLORS[arm],
            linewidth=2,
        )
        ax.annotate(
            f"{arm} ({score_a[arm]:.2f})",
            (0, rank_a[arm]),
            textcoords="offset points",
            xytext=(-8, tied_offsets.get(arm, 0)),
            ha="right",
            fontsize="small",
        )
        ax.annotate(
            f"({score_b[arm]:.2f}) {arm}",
            (1, rank_b[arm]),
            textcoords="offset points",
            xytext=(8, tied_offsets.get(arm, 0)),
            ha="left",
            fontsize="small",
        )

    ax.set_yinverted(True)
    ax.set_xlim(-0.75, 1.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Task A\n(norm. return)", "Task B\n(completion)"])
    ax.set_ylabel("Rank", fontsize="medium")
    ax.tick_params(axis="y", labelsize="small")
    fig.tight_layout()

    save_fig(fig, out_dir / "r_cross_interaction.png")


def clear_pngs(plot_dir: Path):
    for png in plot_dir.glob("*.png"):
        png.unlink()


def figs_task_a(dirs: dict[str, Path], out_dir: Path):
    fig_task_a_efficiency(dirs, out_dir)
    fig_task_a_decomposition(dirs, out_dir)


def figs_task_b(dirs: dict[str, Path], out_dir: Path):
    fig_task_b_efficiency(dirs, out_dir)
    fig_task_b_termination(dirs, out_dir)
    fig_task_b_return_vs_success(dirs, out_dir)


def main(fresh=False):
    iterable = (("pole-balance", figs_task_a), ("waypoint", figs_task_b))

    for task, graph_builder in iterable:
        task_dir = PROJECT_ROOT / "runs" / "drones" / task
        plot_dir = task_dir / "plots"

        if fresh:
            clear_pngs(plot_dir)

        graph_builder(arm_dirs(task_dir), plot_dir)
        print(f"Wrote {task}")

    cross_dir = DRONES_ROOT / "comparison" / "plots"
    if fresh:
        clear_pngs(cross_dir)

    fig_task_cross_interaction(cross_dir)
    print("Wrote cross-task")


if __name__ == "__main__":
    FRESH = True  # TODO Change after dev done
    main(FRESH)
