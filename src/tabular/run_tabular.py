"""
Run Tabular Q-Learning on all 3 gym ToyText envs. Then call their eval to generate eval parquet files, then call rliable plotting to generate all plots. This is the main runner file for tabular q learning.
"""

from pathlib import Path

import gymnasium as gym
from tqdm.auto import tqdm

from src.config import DEFAULT_SEED
from src.tabular import figures, plots
from src.tabular.config import (
    TabularConfig,
    ToyTextAlgoConfig,
    ToyTextEnvConfig,
    ToyTextRunConfig,
)
from src.tabular.eval import evaluate
from src.tabular.q_learner import run_experiment

ENVS = {
    "FrozenLake-v1": {"base_kwargs": {"map_name": "4x4"}, "variant_key": "is_slippery"},
    "CliffWalking-v1": {"base_kwargs": {}, "variant_key": "is_slippery"},
    "Taxi-v4": {"base_kwargs": {}, "variant_key": "is_rainy"},
}
VARIANTS = {"det": False, "slip": True}
ENV_STEP_BUDGET = {
    "FrozenLake-v1": 100_000,
    "CliffWalking-v1": 100_000,
    "Taxi-v4": 500_000,
}

RUNS_ROOT_DIR = Path("runs")
ALGO = "tabular"  # keep in sync with run_experiment's runs/<env>/<algo>/<ts> layout


def algo_root(env_id: str) -> Path:
    """runs/<env_id>/tabular — parent of every experiment, plus plots/ and videos/."""
    return RUNS_ROOT_DIR / env_id / ALGO

MASTER_SEED = DEFAULT_SEED
LR = 0.1
EPSILON = 0.1
GAMMA = 0.95

N_RUNS = 10
CHECKPOINT_POINTS = 100
CHECKPOINT_TYPE = "log"

EVAL_CURVES = [
    ("eval_return_mean", True, "IQM Normalized Return", "sample_efficiency.png"),
    ("success_rate", False, "IQM Success Rate", "success_rate.png"),
    ("episode_len_mean", False, "IQM Steps per Episode", "steps_per_episode.png"),
    ("eval_return_mean", False, "IQM Eval Return", "eval_return.png"),
    ("q_star_linf", False, r"IQM $\|Q - Q^*\|_\infty$", "q_convergence.png"),
    (
        "weighted_q_linf",
        False,
        r"IQM Visitation-Weighted $|Q - Q^*|$",
        "q_convergence_weighted.png",
    ),
    ("optimality_gap", False, "IQM Optimality Gap", "optimality_gap.png"),
]

TRAIN_CURVES = [
    ("train_return_mean", "IQM Training Return (behavior policy)", "train_return.png"),
    ("mean_td_error", "IQM Mean |TD error|", "td_error.png"),
    ("epsilon", "Epsilon", "epsilon.png"),
]


def build_config(env: gym.Env) -> TabularConfig:
    return TabularConfig(
        run_config=ToyTextRunConfig(
            master_seed=MASTER_SEED,
            lr=LR,
            step_budget=ENV_STEP_BUDGET[env.spec.id],
            n_runs=N_RUNS,
            checkpoint_points=CHECKPOINT_POINTS,
            checkpoint_type=CHECKPOINT_TYPE,
        ),
        env_config=ToyTextEnvConfig.from_gym_env(env),
        algo_config=ToyTextAlgoConfig(
            epsilon=EPSILON,
            gamma=GAMMA,
            decay_steps=ENV_STEP_BUDGET[env.spec.id] // 2,
        ),
    )


def train_and_eval_one(env_id: str, env_kwargs: dict) -> Path:
    """
    Train one variant and eval it
    """
    env = gym.make(env_id, **env_kwargs)
    config = build_config(env)

    exp_dir = run_experiment(env, config, RUNS_ROOT_DIR)
    evaluate(exp_dir)

    return exp_dir


def make_plots(env_id: str, variant_dirs: dict):
    """
    Plot every rliable plot for 1 env, into runs/<env_id>/tabular/plots/
    """
    out_dir = algo_root(env_id) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot train
    for metric, ylabel, fname in tqdm(TRAIN_CURVES, desc="train", leave=False):
        scores, frames = plots.load_train_tensor(variant_dirs, metric=metric)
        plots.plot_sample_efficiency(scores, frames, out_dir / fname, ylabel=ylabel)

    # Plot eval
    for metric, normalize, ylabel, fname in tqdm(
        EVAL_CURVES, desc="curves", leave=False
    ):
        scores, frames = plots.load_score_tensor(
            variant_dirs, metric=metric, normalize=normalize
        )
        plots.plot_sample_efficiency(scores, frames, out_dir / fname, ylabel=ylabel)

    final = plots.final_scores_normalized(variant_dirs)
    plots.plot_aggregate_metrics(final, out_dir / "aggregate.png")
    plots.plot_performance_profile(final, out_dir / "performance_profile.png")


def make_figures(env_id: str, spec: dict, variant_dirs: dict):
    """
    Single best seed figures:
    visitation heatmaps, learned policy arrows and best agent video playing the game
    """
    out_dir = algo_root(env_id) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # state-action heatmap
    figures.plot_visitation_sa(
        variant_dirs, env_id, out_dir / "visitation_heatmap_sa.png"
    )

    # spacial and policy arrows (overlaid on the rendered map; same for det/slip)
    if env_id in figures.GRID:
        figures.plot_visitation_spatial(
            variant_dirs,
            env_id,
            spec["base_kwargs"],
            out_dir / "visitation_spatial.png",
        )
        figures.plot_policy_arrows(
            variant_dirs, env_id, spec["base_kwargs"], out_dir / "policy_arrows.png"
        )

    # Champion rollout video
    vid_dir = algo_root(env_id) / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    for label, value in VARIANTS.items():
        env_kwargs = {**spec["base_kwargs"], spec["variant_key"]: value}
        figures.record_champion(
            variant_dirs[label],
            env_id,
            env_kwargs,
            vid_dir / f"champion_{label}.mp4",
        )


def main():
    for env_id, spec in ENVS.items():
        print(f"\n=== {env_id} ===")
        variant_dirs = {}
        for label, value in VARIANTS.items():
            env_kwargs = {**spec["base_kwargs"], spec["variant_key"]: value}
            print(f"  [{label}] training {N_RUNS} seeds + eval")
            variant_dirs[label] = train_and_eval_one(env_id, env_kwargs)

        print(f"  plotting -> {algo_root(env_id) / 'plots'}")
        make_plots(env_id, variant_dirs)
        make_figures(env_id, spec, variant_dirs)
        print(f"  {env_id} done")


if __name__ == "__main__":
    main()
