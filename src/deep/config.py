from dataclasses import dataclass, field

import gymnasium as gym

from src.config import DEFAULT_SEED, BaseConfig

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
    net_arch: list = field(default_factory=lambda: [64, 64])


class DQNEnvConfig(BaseConfig):
    def __init__(self, env: gym.Env):
        self.action_space = int(env.action_space.n)
        space = env.observation_space
        if isinstance(space, gym.spaces.Box) and len(space.shape) == 3:
            self.obs_space = space.shape  # (96,96,3) raw; informational only
            self.obs_space_low = None
            self.obs_space_high = None
            self.obs_type = "Image"
        elif isinstance(space, gym.spaces.Box):
            self.obs_space = space.shape
            self.obs_space_low = space.low.tolist()
            self.obs_space_high = space.high.tolist()
            self.obs_type = "Box"
        else:
            self.obs_space = int(space.n)
            self.obs_space_low = None
            self.obs_space_high = None
            self.obs_type = "Discrete"

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
