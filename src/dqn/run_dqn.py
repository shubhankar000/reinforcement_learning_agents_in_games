"""
main DQN runner. Run on 10 seeds, with eval, plot and video. mirrors run_tabular.py.
Runs seeds in parallel.

Serves TWO regimes, keyed off the observation space:
  - "Discrete" (ToyText): a Q-table is extractable from the net, so Q* ground truth
    exists and the tabular eval/figures are reused unchanged.
  - "Box" (classic control): continuous obs -> no enumerable states -> no Q-table and
    no Q*. Eval/curves/figures fork accordingly.
"""

from pathlib import Path

import gymnasium as gym
from tqdm.auto import tqdm

from src.config import DEFAULT_SEED
from src.dqn import figures as dqn_figures
from src.dqn.config import DQNAlgoConfig, DQNConfig, DQNEnvConfig, DQNRunConfig
from src.dqn.eval import evaluate as evaluate_box
from src.dqn.runner import run_experiment
from src.tabular import figures, plots
from src.tabular.eval import evaluate as evaluate_tabular

ENVS = {
    # # === ToyText (Discrete obs). ===
    # "FrozenLake-v1": {
    #     "base_kwargs": {"map_name": "4x4"},
    #     "variants": {"det": {"is_slippery": False}, "slip": {"is_slippery": True}},
    #     "budget": 100_000,
    #     "algo": dict(gamma=0.95, net_arch=[64, 64], learning_rate=1e-3,
    #                  buffer_size=50_000, learning_starts=1_000),
    # },
    # "CliffWalking-v1": {
    #     "base_kwargs": {},
    #     "variants": {"det": {"is_slippery": False}, "slip": {"is_slippery": True}},
    #     "budget": 100_000,
    #     "algo": dict(gamma=0.95, net_arch=[64, 64], learning_rate=1e-3,
    #                  buffer_size=50_000, learning_starts=1_000),
    # },
    # "Taxi-v4": {
    #     "base_kwargs": {},
    #     "variants": {"det": {"is_rainy": False}, "slip": {"is_rainy": True}},
    #     "budget": 500_000,
    #     "algo": dict(gamma=0.95, net_arch=[64, 64], learning_rate=1e-3,
    #                  buffer_size=50_000, learning_starts=1_000),
    # },
    # #  === Box obs space envs ===
    # "CartPole-v1": {
    #     "base_kwargs": {},
    #     "variants": {"main": {}},
    #     "budget": 100_000,
    #     "algo": dict(
    #         gamma=0.99,
    #         net_arch=[64, 64],
    #         learning_rate=1e-3,
    #         batch_size=128,
    #         buffer_size=100_000,
    #         learning_starts=1_000,
    #         train_freq=256,
    #         gradient_steps=128,
    #         target_update_interval=10,
    #         exploration_fraction=0.16,
    #         exploration_final_eps=0.04,
    #     ),
    # },
    "LunarLander-v3": {
        "base_kwargs": {},
        "variants": {"main": {}},
        "budget": 500_000,
        "algo": dict(
            gamma=0.99,
            net_arch=[256, 256],
            learning_rate=5e-4,
            batch_size=128,
            buffer_size=50_000,
            learning_starts=1_000,
            train_freq=4,
            gradient_steps=-1,
            target_update_interval=250,
            exploration_fraction=0.12,
            exploration_final_eps=0.1,
        ),
    },
}

RUNS_ROOT_DIR = Path("runs")

MASTER_SEED = DEFAULT_SEED
N_RUNS = 10  # seeds
N_JOBS = N_RUNS  # 1 seed per job
CHECKPOINT_POINTS = 100
CHECKPOINT_TYPE = "log"
POLICY = "MlpPolicy"

ALGO = "dqn"

# Common eval curves for both ToyText and Box
COMMON_EVAL_CURVES = [
    ("eval_return_mean", True, "IQM Normalized Return", "sample_efficiency.png"),
    ("success_rate", False, "IQM Success Rate", "success_rate.png"),
    ("episode_len_mean", False, "IQM Steps per Episode", "steps_per_episode.png"),
    ("eval_return_mean", False, "IQM Eval Return", "eval_return.png"),
]

# Q* dependent curves: Discrete only
DISCRETE_EVAL_CURVES = COMMON_EVAL_CURVES + [
    ("q_star_linf", False, r"IQM $\|Q - Q^*\|_\infty$", "q_convergence.png"),
    (
        "weighted_q_linf",
        False,
        r"IQM Visitation-Weighted $|Q - Q^*|$",
        "q_convergence_weighted.png",
    ),
    ("optimality_gap", False, "IQM Optimality Gap", "optimality_gap.png"),
]

BOX_EVAL_CURVES = COMMON_EVAL_CURVES

# Train loss curves for td-loss
TRAIN_CURVES = [
    ("train_return_mean", "IQM Training Return (behavior policy)", "train_return.png"),
    ("mean_td_error", "IQM Training Loss (Huber TD)", "td_error.png"),
    ("epsilon", "Epsilon", "epsilon.png"),
]


def obs_type_of(env_id: str, base_kwargs: dict) -> str:
    env = gym.make(env_id, **base_kwargs)
    t = "Box" if isinstance(env.observation_space, gym.spaces.Box) else "Discrete"
    env.close()
    return t


def build_config(env_id: str, env_kwargs: dict, spec: dict):
    env = gym.make(env_id, **env_kwargs)
    cfg = DQNConfig(
        algo_config=DQNAlgoConfig(policy=POLICY, **spec["algo"]),
        run_config=DQNRunConfig(
            master_seed=MASTER_SEED,
            step_budget=spec["budget"],
            n_runs=N_RUNS,
            n_jobs=N_JOBS,
            checkpoint_points=CHECKPOINT_POINTS,
            checkpoint_type=CHECKPOINT_TYPE,
        ),
        env_config=DQNEnvConfig.from_gym_env(env),
    )
    env.close()
    return cfg


def eval_dqn(exp_dir: Path, obs_type: str):
    """Box has no Q*/snapshots.npz -> its own eval; Discrete reuses the tabular one."""
    if obs_type == "Box":
        evaluate_box(exp_dir)
    else:
        evaluate_tabular(exp_dir)


def train_one(env_id: str, env_kwargs: dict, spec: dict, obs_type: str, label) -> Path:
    cfg = build_config(env_id, env_kwargs, spec)
    exp_dir = run_experiment(cfg, label, out=RUNS_ROOT_DIR)
    eval_dqn(exp_dir, obs_type)
    return exp_dir


def algo_root(env_id: str) -> Path:
    """runs/<env_id>/dqn — parent of every experiment, plus plots/ and videos/."""
    return RUNS_ROOT_DIR / env_id / ALGO


def plot_predicted_vs_realized(variant_dirs: dict, out_path: Path):
    """
    Box only. Reuses plot_sample_efficiency by putting the two METRICS on its
    "algorithm" axis instead of the variants: predicted Q(s0) vs the discounted
    return actually achieved from s0. The gap is DQN's overestimation bias.
    """
    qp, frames = plots.load_score_tensor(variant_dirs, metric="q_pred_mean")
    mc, _ = plots.load_score_tensor(variant_dirs, metric="mc_return_mean")

    scores = {}
    for label in variant_dirs:
        scores[f"predicted Q(s0) [{label}]"] = qp[label]
        scores[f"realized MC return [{label}]"] = mc[label]

    plots.plot_sample_efficiency(
        scores, frames, out_path, ylabel="Discounted value from s0"
    )


def make_plots(env_id: str, variant_dirs: dict, obs_type: str):
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
    eval_curves = BOX_EVAL_CURVES if obs_type == "Box" else DISCRETE_EVAL_CURVES
    for metric, normalize, ylabel, fname in tqdm(
        eval_curves, desc="curves", leave=False
    ):
        scores, frames = plots.load_score_tensor(
            variant_dirs, metric=metric, normalize=normalize
        )
        plots.plot_sample_efficiency(scores, frames, out_dir / fname, ylabel=ylabel)

    if obs_type == "Box":
        plot_predicted_vs_realized(variant_dirs, out_dir / "predicted_vs_realized.png")

    final = plots.final_scores_normalized(variant_dirs)
    plots.plot_aggregate_metrics(final, out_dir / "aggregate.png")
    plots.plot_performance_profile(final, out_dir / "performance_profile.png")


def make_figures(env_id: str, spec: dict, variant_dirs: dict, obs_type: str):
    """
    Single best seed figures:
    visitation heatmaps, learned policy arrows and best agent video playing the game.
    All of these consume snapshots.npz (Q-table + discrete visitation), so they are
    Discrete-only.
    """
    if obs_type == "Box":
        out_dir = algo_root(env_id) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        dqn_figures.plot_obs_coverage(
            variant_dirs,
            env_id,
            spec["base_kwargs"],
            out_dir / "obs_coverage.png",
        )

        var_dir = algo_root(env_id) / "videos"
        var_dir.mkdir(parents=True, exist_ok=True)
        for label, extra_kwargs in spec["variants"].items():
            env_kwargs = {**spec["base_kwargs"], **extra_kwargs}
            dqn_figures.record_champion_box(
                variant_dirs[label],
                env_id,
                env_kwargs,
                var_dir / f"champion_{label}.mp4",
            )
            dqn_figures.save_champion_box(variant_dirs[label], env_id, env_kwargs)

        return

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
    for label, extra_kwargs in spec["variants"].items():
        env_kwargs = {**spec["base_kwargs"], **extra_kwargs}
        figures.record_champion(
            variant_dirs[label],
            env_id,
            env_kwargs,
            vid_dir / f"champion_{label}.mp4",
        )


def main():
    for env_id, spec in ENVS.items():
        obs_type = obs_type_of(env_id, spec["base_kwargs"])
        print(f"\n=== {env_id} (DQN, {obs_type}) ===")
        variant_dirs = {}
        for label, extra_kwargs in spec["variants"].items():
            env_kwargs = {**spec["base_kwargs"], **extra_kwargs}
            print(f"  [{label}] training {N_RUNS} seeds ({N_JOBS} parallel)")
            variant_dirs[label] = train_one(env_id, env_kwargs, spec, obs_type, label)

        make_plots(env_id, variant_dirs, obs_type)
        make_figures(env_id, spec, variant_dirs, obs_type)
        print(f"  {env_id} done")


if __name__ == "__main__":
    main()
