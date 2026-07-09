"""
main DQN runner. Run on 10 seeds on toytext, with eval, plot and video. mirrors run_tabular.py. Runs seeds in parallel
"""

from pathlib import Path
import gymnasium as gym
from tqdm.auto import tqdm

from src.config import DEFAULT_SEED
from src.dqn.config import DQNAlgoConfig, DQNRunConfig
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
CHECKPOINT_POINTS = 100
CHECKPOINT_TYPE = "log"
POLICY='MlpPolicy'
BUFFER_SIZE = 50_000
LEARNING_STARTS = 1_000

# TODO
TRAIN_CURVES = []

def build_config(env: gym.Env):
    return DQNAlgoConfig(policy=POLICY, learning_rate=LR, buffer_size=BUFFER_SIZE, learning_starts=LEARNING_STARTS), DQNRunConfig()
