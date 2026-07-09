from src.config import BaseConfig, DEFAULT_SEED
from dataclasses import dataclass, field

default = lambda x: field(default=x)  # noqa


@dataclass
class DQNRunConfig(BaseConfig):
    master_seed: int = default(DEFAULT_SEED)
    step_budget: int = default(100_000)
    n_runs: int = default(10)
    checkpoint_points: int = default(100)
    checkpoint_type: str = default("log")
    n_envs: int = default(1)
    device: str = default("cpu")
    n_jobs: int = default(10)


@dataclass
class DQNAlgoConfig(BaseConfig):
    policy: str = default("MlpPolicy")
    learning_rate: float = default(1e-3)
    buffer_size: int = default(50_000)
    learning_starts: int = default(1_000)
    batch_size: int = default(128)
    gamma: float = default(0.99)
    train_freq: int = default(4)
    gradient_steps: int = default(1)
    target_update_interval: int = default(500)
    exploration_fraction: float = default(0.2)
    exploration_final_eps: float = default(0.05)

@dataclass
class DQNConfig(BaseConfig):
    run_config = DQNRunConfig
    algo_config = DQNAlgoConfig