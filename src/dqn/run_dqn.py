"""
main DQN runner. Run on 10 seeds on toytext, with eval, plot and video. mirrors run_tabular.py. Runs seeds in parallel
"""

from pathlib import Path

import gymnasium as gym

from src.config import DEFAULT_SEED
from src.dqn.config import DQNAlgoConfig, DQNConfig, DQNEnvConfig, DQNRunConfig
from src.dqn.runner import run_experiment

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
GAMMA = 0.99

N_RUNS = 10  # seeds
N_JOBS = N_RUNS  # 1 seed per job
CHECKPOINT_POINTS = 100
CHECKPOINT_TYPE = "log"
POLICY = "MlpPolicy"
BUFFER_SIZE = 50_000
LEARNING_STARTS = 1_000

# TODO
EVAL_CURVES = []
TRAIN_CURVES = []


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
    eval_dqn(exp_dir)  # TODO
    return exp_dir


def eval_dqn(exp_dir: Path): ...


def make_plots(env_id: str, variant_dirs: dict): ...
def make_figures(env_id: str, spec: dict, variant_dirs: dict): ...


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
