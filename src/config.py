"""
Generic config class to be used by all learner classes.
"""

from dataclasses import dataclass, field
import gymnasium as gym
import json


DEFAULT_SEED = 67


class BaseConfig:
    def to_dict(self) -> dict:
        """
        Plain-python dict of all public attributes, recursing into nested configs.
        """
        return {
            key: self._encode(value)
            for key, value in vars(self).items()
            if not key.startswith("_")  # skip _env and friends
        }

    @staticmethod
    def _encode(value):
        if isinstance(value, BaseConfig):  # nested config -> recurse
            return value.to_dict()
        if isinstance(value, (list, tuple)):
            return [BaseConfig._encode(v) for v in value]
        if isinstance(value, dict):
            return {k: BaseConfig._encode(v) for k, v in value.items()}
        return value  # int / float / str / bool / None

    def json(self, **kwargs) -> str:
        """
        Serialize to a JSON string. Extra kwargs pass through to json.dumps
        """
        return json.dumps(self.to_dict(), default=str, **kwargs)


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
    checkpoint_every: int = field(default=1000)


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

    @classmethod
    def from_gym_env(cls, env: gym.Env):
        action_space = env.action_space
        obs_space = env.observation_space
        return cls(env, action_space, obs_space)

    def __repr__(self):
        return f"{self.action_space=}\n{self.obs_space=}"


@dataclass
class Config(BaseConfig):
    run_config: ToyTextRunConfig = field(default_factory=ToyTextRunConfig)
    env_config: ToyTextEnvConfig | None = field(default=None)
    algo_config: ToyTextAlgoConfig | None = field(default=None)


if __name__ == "__main__":
    run_config = ToyTextRunConfig()
    run_config_42 = ToyTextRunConfig(master_seed=42)
    env = gym.make("CartPole-v1")

    config = Config(
        run_config=run_config_42, env_config=ToyTextEnvConfig.from_gym_env(env)
    )

    print(run_config, run_config_42, config)
    print(
        run_config_42.master_seed, run_config.master_seed, config.run_config.master_seed
    )
