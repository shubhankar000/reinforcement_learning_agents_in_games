"""
main DQN runner. Run on 10 seeds on toytext, with eval, plot and video. mirrors run_tabular.py. Runs seeds in parallel
"""

from pathlib import Path

import gymnasium as gym
from tqdm.auto import tqdm

from src.config import DEFAULT_SEED
from src.dqn.config import DQNAlgoConfig, DQNConfig, DQNEnvConfig, DQNRunConfig
from src.dqn.runner import run_experiment
from src.tabular import figures, plots
from src.tabular.eval import evaluate  # tabular evaluation works perfectly

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

MASTER_SEED = DEFAULT_SEED
LR = 1e-3
GAMMA = 0.95

N_RUNS = 10  # seeds
N_JOBS = N_RUNS  # 1 seed per job
CHECKPOINT_POINTS = 100
CHECKPOINT_TYPE = "log"
POLICY = "MlpPolicy"
BUFFER_SIZE = 50_000
LEARNING_STARTS = 1_000

ALGO = "dqn"

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
    ("epsilon", "Epsilon", "epsilon.png"),
]


def build_config(env_id: str, env_kwargs: dict):
    env = gym.make(env_id, **env_kwargs)
    cfg = DQNConfig(
        algo_config=DQNAlgoConfig(
            policy=POLICY,
            learning_rate=LR,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            gamma=GAMMA,
        ),
        run_config=DQNRunConfig(
            master_seed=DEFAULT_SEED,
            step_budget=ENV_STEP_BUDGET[env_id],
            n_runs=N_RUNS,
            checkpoint_points=CHECKPOINT_POINTS,
            checkpoint_type=CHECKPOINT_TYPE,
        ),
        env_config=DQNEnvConfig.from_gym_env(env),
    )
    env.close()
    return cfg


def train_one(env_id: str, env_kwargs: dict) -> Path:
    cfg = build_config(env_id, env_kwargs)
    exp_dir = run_experiment(cfg, out=RUNS_ROOT_DIR)
    eval_dqn(exp_dir)
    return exp_dir


def eval_dqn(exp_dir: Path):
    evaluate(exp_dir)


def algo_root(env_id: str) -> Path:
    """runs/<env_id>/dqn — parent of every experiment, plus plots/ and videos/."""
    return RUNS_ROOT_DIR / env_id / ALGO


def make_plots(env_id: str, variant_dirs: dict):
    """
    Plot every rliable plot for 1 env, into runs/<env_id>/dqn/plots/
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
        print(f"\n=== {env_id} (DQN) ===")
        variant_dirs = {}
        for label, value in VARIANTS.items():
            env_kwargs = {**spec["base_kwargs"], spec["variant_key"]: value}
            print(f"  [{label}] training {N_RUNS} seeds ({N_JOBS} parallel)")
            variant_dirs[label] = train_one(env_id, env_kwargs)

        make_plots(env_id, variant_dirs)
        make_figures(env_id, spec, variant_dirs)
        print(f"  {env_id} done")


if __name__ == "__main__":
    main()
