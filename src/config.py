"""
Generic config class to be used by all learner classes.
"""

from dataclasses import dataclass, field
import gymnasium as gym

DEFAULT_SEED = 67


@dataclass
class RunConfig:
    """
    Configuration when running.
    Example:
        seed
    """

    seed: int = field(default=DEFAULT_SEED)
    lr: float = field(default_factory=float)
    episodes: int = field(default=10)


@dataclass
class ToyTextAlgoConfig:
    epsilon: float = field(default=0.1)
    "Epsilon value for e-greedy. Range [0, 1). 0 greedy, 1 random"

    gamma: float = field(default=0.95)
    "Discount factor, controles sightedness of the agent"


class ToyTextEnvConfig:
    def __init__(
        self, env: gym.Env, action_space: gym.spaces.Discrete, obs_space: gym.spaces.Discrete
    ):
        self.env: gym.Env = env
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
class Config:
    run_config: RunConfig = field(default_factory=RunConfig)
    env_config: ToyTextEnvConfig | None = field(default=None)
    algo_config: ToyTextAlgoConfig | None = field(default=None)


if __name__ == "__main__":
    run_config = RunConfig()
    run_config_42 = RunConfig(seed=42)
    env = gym.make("CartPole-v1")

    config = Config(
        run_config=run_config_42, env_config=ToyTextEnvConfig.from_gym_env(env)
    )

    print(run_config, run_config_42, config)
    print(run_config_42.seed, run_config.seed, config.run_config.seed)
