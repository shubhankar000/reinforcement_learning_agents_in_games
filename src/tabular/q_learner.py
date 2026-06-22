"""
Run Tabular Q-learning on gym gridworld (Toy Text) environments

Run with TODO
"""

import gymnasium as gym
import numpy as np
import typer
from tqdm.auto import tqdm, trange

from src.config import (
    DEFAULT_SEED,
    Config,
    RunConfig,
    ToyTextAlgoConfig,
    ToyTextEnvConfig,
)
from src.rng_factory import SeededRNG

app = typer.Typer()


class QTabularLearner:
    def __init__(
        self,
        config: Config,
        rng_factory: SeededRNG,
    ):
        self.config = config
        self.state_size = config.env_config.obs_space
        self.action_size = config.env_config.action_space
        self.learning_rate = config.run_config.lr
        self.gamma = config.algo_config.gamma

        self.rng = rng_factory.next_rng()
        self.reset_qtable()

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

    def update(self, state, action, reward, next_state):
        """
        Update Equation:
        Q(s,a) <- Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)
        """
        delta = (
            reward
            + self.gamma * np.max(self.qtable[next_state, :])
            - self.qtable[state, action]
        )

        self.qtable[state, action] += self.learning_rate * delta

    def action(self, state):
        if self.config.algo_config.epsilon == 0:
            # This is the greedy agent
            return np.argmax(self.qtable[state, :])

        random_num = self.rng.uniform(0, 1)

        # e-greedy strategy (exploration)
        if random_num < self.config.algo_config.epsilon:
            return self.config.env_config.env.action_space.sample()

        # Maximal strategy (exploitation)
        max_idx = np.where(self.qtable[state, :] == self.qtable[state, :].max())[0]

        return self.rng.choice(max_idx)


def learn(agent: QTabularLearner, config: Config):
    env = config.env_config.env

    total_rewards = np.zeros((config.run_config.episodes,))
    total_steps = np.zeros((config.run_config.episodes,))
    success_rate = np.zeros((config.run_config.episodes,))

    for episode in trange(config.run_config.episodes):
        obs, _ = env.reset()
        done = False
        goal_condition = 1  # Reward = 1 for goal condition

        cum_rewards = 0
        cum_steps = 0

        while not done:
            action = agent.action(obs)

            new_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            success = terminated and reward == goal_condition

            agent.update(obs, action, reward, new_obs)

            obs = new_obs

            cum_steps += 1
            cum_rewards += reward

        success_rate[episode] = success

        total_rewards[episode] = cum_rewards
        total_steps[episode] = cum_steps

    return {
        "total_rewards": total_rewards,
        "total_steps": total_steps,
        "success_rate": success_rate,
    }


@app.command()
def main(
    seed: int = typer.Option(
        default=DEFAULT_SEED, help="RNG seed to set the entire run"
    ),
    lr: float = typer.Option(default=0.1, help="Learning rate for q learning agent"),
    epsilon: float = typer.Option(
        default=0.1,
        help="Epsilon value for e-greedy. Range [0, 1). 0 greedy, 1 random",
    ),
    gamma: float = typer.Option(default=0.95, help="Discount Factor"),
    episodes: int = typer.Option(default=10, help="Number of Episodes"),
):
    env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        # render_mode="human",
        reward_schedule=(1, 0, 0),  # [reward_goal,  reward_hole, reward_frozen]
        is_slippery=False,
    )

    # initialize the config
    config = Config(
        run_config=RunConfig(seed=seed, lr=lr, episodes=episodes),
        env_config=ToyTextEnvConfig.from_gym_env(env),
        algo_config=ToyTextAlgoConfig(epsilon=epsilon, gamma=gamma),
    )

    rng_factory = SeededRNG(config.run_config.seed)

    # Set action space sample seed to our seed
    env.action_space.seed(rng_factory.next_seed())

    agent = QTabularLearner(config, rng_factory)

    results = learn(agent, config)

    print(results)

    return results


if __name__ == "__main__":
    typer.run(main)
