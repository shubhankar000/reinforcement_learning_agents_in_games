from src.config import BaseConfig, DEFAULT_SEED
from dataclasses import dataclass, field
import gymnasium as gym

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


class DQNEnvConfig(BaseConfig):
    def __init__(self, env: gym.Env):
        self.action_space = int(env.action_space.n)
        self.obs_space = int(env.observation_space.n)
        self.env_id = env.spec.id
        self.env_kwargs = dict(env.spec.kwargs)

    @classmethod
    def from_gym_env(cls, env: gym.Env):
        return cls(env)

    def __repr__(self):
        return f"{self.action_space=}\n{self.obs_space=}"


@dataclass
class DQNConfig(BaseConfig):
    run_config: DQNRunConfig = field(default=None)
    env_config: DQNEnvConfig = field(default=None)
    algo_config: DQNAlgoConfig = field(default=None)
