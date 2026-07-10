from dataclasses import dataclass, field

import gymnasium as gym

from src.config import DEFAULT_SEED, BaseConfig


@dataclass
class ToyTextRunConfig(BaseConfig):
    """
    Configuration when running.
    Example:
        seed
    """

    master_seed: int = field(default=DEFAULT_SEED)
    lr: float = field(default_factory=float)
    step_budget: int = field(default=100_000)
    n_runs: int = field(default=10)
    checkpoint_points: int = field(default=100)
    checkpoint_type: str = field(default="linear")


@dataclass
class ToyTextAlgoConfig(BaseConfig):
    epsilon: float = field(default=0.1)
    "Epsilon value for e-greedy. Range [0, 1). 0 greedy, 1 random"

    gamma: float = field(default=0.95)
    "Discount factor, controles sightedness of the agent"

    e_floor: float = field(default=0.01)
    "epsilon never drops below this value while training"

    decay_steps: int = field(default=50_000)
    "Steps ramp from epsilon -> e_floor in decay_steps"


class ToyTextEnvConfig(BaseConfig):
    def __init__(
        self,
        env: gym.Env,
        action_space: gym.spaces.Discrete,
        obs_space: gym.spaces.Discrete,
    ):
        self._env: gym.Env = env
        self.action_space = int(action_space.n)
        self.obs_space = int(obs_space.n)
        self.env_id = env.spec.id
        self.env_kwargs = dict(env.spec.kwargs)

    @classmethod
    def from_gym_env(cls, env: gym.Env):
        action_space = env.action_space
        obs_space = env.observation_space
        return cls(env, action_space, obs_space)

    def __repr__(self):
        return f"{self.action_space=}\n{self.obs_space=}"


@dataclass
class TabularConfig(BaseConfig):
    run_config: ToyTextRunConfig = field(default=None)
    env_config: ToyTextEnvConfig = field(default=None)
    algo_config: ToyTextAlgoConfig = field(default=None)
