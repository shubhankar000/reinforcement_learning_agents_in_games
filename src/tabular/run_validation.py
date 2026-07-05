"""
Runs Validation in a loop for all toytext envs and kwargs.

Ideally only needs to be run once.

If any changes needed, edit the global constants
"""

from pathlib import Path

import gymnasium as gym
import numpy as np

from src.config import ToyTextEnvConfig
from src.tabular.validation import value_iteration, policy_evaluation

ENVS = ["FrozenLake-v1", "CliffWalking-v1", "Taxi-v4"]
ENV_KWARGS = [{"map_name": "4x4", "reward_schedule": [1, 0, 0]}, {}, {}]
ENV_KWARGS_VAR = [
    "is_slippery",
    "is_slippery",
    "is_rainy",
]


def main(out: Path):
    """
    Run 3 env with 2 variants each for optimal Q* and V* through value iteration on toytext gym envs.

    All combinations:
    3 toytext envs {FrozenLake-v1, CliffWalking-v1, Taxi-v4}
    x
    2 varaints respectively {is_slippery, is_slippery, is_rainy}

    taxi-v4 fickle passenger removed, since optimal values cannot be found
    """

    for env_id, env_kwarg, env_kwvar in zip(ENVS, ENV_KWARGS, ENV_KWARGS_VAR):
        for var in [False, True]:
            env = gym.make(
                env_id,
                **env_kwarg,
                **{env_kwvar: var},
            )
            Q, V = value_iteration(env.unwrapped.P, 0.95)
            env_config = ToyTextEnvConfig.from_gym_env(env)
            V_random = policy_evaluation(env.unwrapped.P, 0.95)

            slip = bool(env_config.env_kwargs.get("is_slippery", False)) or bool(
                env_config.env_kwargs.get("is_rainy", False)
            )
            key = f"{env_config.env_id}_{'slip' if slip else 'det'}"

            np.savez(out / f"q_v_star_{key}", q_star=Q, v_star=V, v_random=V_random)


if __name__ == "__main__":
    out = Path("./src/tabular/optimal_values")
    main(out)
