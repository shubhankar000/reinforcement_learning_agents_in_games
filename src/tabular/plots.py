from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics, plot_utils


def task_df(exp_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(exp_dir / "all_evals.parquet")


def pivot(df: pd.DataFrame, metric: str) -> np.ndarray:
    return df.pivot(index="run_no.", columns="env_steps", values=metric).to_numpy()


def anchors(df: pd.DataFrame):
    last = df[df["env_steps"] == df["env_steps"].max()]
    return last["v_star_s0"].iloc[0], last["v_random_s0"].iloc[0]


def load_score_tensor(exp_dirs: dict, metric: str = "success_rate", normalize=False):
    """
    Load per-variant metric data for rliable.

    exp_dirs: {label: leaf experiment dir}. Each label becomes its own line
    (rliable's "algorithm" axis). Variants are NOT pooled — det and slippery
    have different dynamics, so each keeps its own task axis of size 1.
    """
    scores, frame_ref = {}, None
    for label, d in exp_dirs.items():
        df = task_df(d)
        frames = np.sort(df["env_steps"].unique())
        if frame_ref is None:
            frame_ref = frames
        assert np.array_equal(frames, frame_ref), f"Checkpoint grid mismatch: {d}"

        mat = pivot(df, metric)
        if normalize:
            v_star_s0, v_random_s0 = anchors(df)
            mat = (mat - v_random_s0) / (v_star_s0 - v_random_s0)

        scores[label] = mat[:, None, :]  # (n_runs, 1 task, n_frames)

    return scores, frame_ref


def load_train_tensor(exp_dirs: dict, metric: str):
    scores, frame_ref = {}, None
    for label, d in exp_dirs.items():
        runs = sorted(d.glob("run_*/train_log.parquet"))
        mats = []
        for r in runs:
            tdf = pd.read_parquet(r).sort_values("env_steps")
            frames = tdf["env_steps"].to_numpy()

            if frame_ref is None:
                frame_ref = frames

            mats.append(tdf[metric].to_numpy())
        scores[label] = np.stack(mats)[:, None, :]

    return scores, frame_ref


def plot_sample_efficiency(scores, frames, out_path: Path, ylabel="IQM Success Rate"):
    iqm = lambda s: np.array(  # noqa: E731
        [metrics.aggregate_iqm(s[..., f]) for f in range(s.shape[-1])]
    )
    iqm_scores, iqm_cis = rly.get_interval_estimates(scores, iqm, reps=2000)
    plot_utils.plot_sample_efficiency_curve(
        frames,
        iqm_scores,
        iqm_cis,
        algorithms=list(scores.keys()),
        xlabel="Environment steps",
        ylabel=ylabel,
        legend=True,
    )

    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def final_scores_normalized(exp_dirs: dict):
    scores = {}
    for label, d in exp_dirs.items():
        df = task_df(d)
        last = df[df["env_steps"] == df["env_steps"].max()].sort_values("run_no.")
        v_star_s0, v_random_s0 = anchors(df)
        norm = (last["eval_return_mean"] - v_random_s0) / (v_star_s0 - v_random_s0)
        scores[label] = norm.to_numpy()[:, None]

    return scores


def plot_aggregate_metrics(scores, out_path: Path, xlabel="Normalized Return"):
    agg = lambda x: np.array(  # noqa: E731
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x),
        ]
    )
    pt, ci = rly.get_interval_estimates(scores, agg, reps=50000)
    plot_utils.plot_interval_estimates(
        pt,
        ci,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=list(scores.keys()),
        xlabel=xlabel,
        xlabel_y_coordinate=-0.5,
        row_height=1.5,
    )
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_performance_profile(
    scores, out_path: Path, xlabel=r"Normalized Return ($\tau$)"
):
    thresholds = np.linspace(0.0, 1.0, 100)

    profiles, profiles_cis = rly.create_performance_profile(scores, thresholds)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        profiles,
        thresholds,
        performance_profile_cis=profiles_cis,
        xlabel=xlabel,
        legend=True,
        ax=ax,
    )
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


if __name__ == "__main__":
    runs_dir = Path("runs/FrozenLake-v1")

    # Hardcoded for now; the final runner will discover these programmatically.
    # Each variant is its own line — det and slippery have different dynamics,
    # so we don't pool them.
    variants = {
        "det": runs_dir / "2026-07-02-05-36-20-PM-det",
        "slip": runs_dir / "2026-07-02-05-36-36-PM-slip",
    }

    out_dir = runs_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores, frames = load_score_tensor(variants)
    for label, arr in scores.items():
        print(f"{label}: {arr.shape}")
        assert not np.isnan(arr).any()

    plot_sample_efficiency(scores, frames, out_dir / "sample_efficiency.png")
    print("Created sample efficiency plot")

    plot_aggregate_metrics(final_scores_normalized(variants), out_dir / "aggregate.png")
    print("Created aggregate metrics plot")

    plot_performance_profile(
        final_scores_normalized(variants), out_dir / "performance_profile.png"
    )
    print("Created performance profiles")
