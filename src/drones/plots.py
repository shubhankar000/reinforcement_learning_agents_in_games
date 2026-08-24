"""
Rliable plotting of eval metrics. Looks for eval_parquet, requires eval to have run.
Runs for Task A and Task B
"""

import matplotlib

matplotlib.use("Agg")

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


def add_legend(fig, ax, ncol=5):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
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
    final = final_scores(dirs, "eval_return_mean", normalise=True)

    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    plot_sample_efficiency(ax[0], scores, frames, "IQM Normalised Return")
    plot_performance_profile(ax[1], final, r"Normalised Return ($\tau$)")
    fig.tight_layout()
    add_legend(fig, ax[0])

    save_fig(fig, out_dir / "task_a_sample_efficiency_profile.png")


def fig_task_b_efficiency(dirs: dict[str, Path], out_dir: Path):
    """
    Task B on success rate. not normalised - returns ranks them inversely
    """
    scores, frames = tp.load_score_tensor(dirs, metric="success_rate", normalise=False)
    final = final_scores(dirs, "success_rate")

    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    plot_sample_efficiency(ax[0], scores, frames, "IQM Env Completion")
    plot_performance_profile(ax[1], final, r"Env Completion ($\tau$)")
    fig.tight_layout()
    add_legend(fig, ax[0])

    save_fig(fig, out_dir / "task_b_sample_efficiency_profile.png")


def figs_task_a(dirs: dict[str, Path], out_dir: Path):
    fig_task_a_efficiency(dirs, out_dir)


def figs_task_b(dirs: dict[str, Path], out_dir: Path):
    fig_task_b_efficiency(dirs, out_dir)


def main():
    iterable = (("pole-balance", figs_task_a), ("waypoint", figs_task_b))

    for task, fn in iterable:
        task_dir = PROJECT_ROOT / "runs" / "drones" / task
        fn(arm_dirs(task_dir), task_dir / "plots")
        print(f"Wrote {task}")


if __name__ == "__main__":
    main()
