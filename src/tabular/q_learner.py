"""
Run Tabular Q-learning on gym gridworld (Toy Text) environments

Run with TODO
"""

import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import typer
from tqdm.auto import tqdm, trange

from src.config import (
    DEFAULT_SEED,
    Config,
    ToyTextAlgoConfig,
    ToyTextEnvConfig,
    ToyTextRunConfig,
)
from src.rng_factory import SeededRNG

app = typer.Typer()


def epsilon_schedule(
    e: float,
    step: int,
    decay_steps: int,
    e_floor: float = 0.01,
) -> float:
    """
    Epsilon greedy annealing schedule based on step
    """
    return max(e_floor, e - (e - e_floor) * step / decay_steps)


class QTabularLearner:
    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
    ):
        self.config = config
        self.state_size = config.env_config.obs_space
        self.action_size = config.env_config.action_space
        self.learning_rate = config.run_config.lr
        self.gamma = config.algo_config.gamma

        self.rng = rng
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

        return delta

    def action(self, state, eps):
        if eps == 0:
            # This is the greedy agent
            return np.argmax(self.qtable[state, :])

        random_num = self.rng.uniform(0, 1)

        # e-greedy strategy (exploration)
        if random_num < eps:
            return self.config.env_config._env.action_space.sample()

        # Maximal strategy (exploitation)
        max_idx = np.where(self.qtable[state, :] == self.qtable[state, :].max())[0]

        return self.rng.choice(max_idx)

    @property
    def policy(self):
        return np.argmax(self.qtable, axis=1)

    @property
    def V(self):
        return np.max(self.qtable, axis=1)


def create_checkpoint_steps(budget, chkpt_type, n_points, start):
    """
    Create checkpoint list based on the budget, type, #of points and start
    """
    if chkpt_type == "linear":
        # linear silently drops start to reach the budget
        every = budget // n_points
        points = list(range(every, budget + 1, every))

        return points

    elif chkpt_type == "log":
        points = np.unique(
            np.round(np.geomspace(start, budget, n_points)).astype("int")
        )
        return list(points)

    else:
        raise ValueError(
            f"Unknown checkpoint type {chkpt_type}. Only `linear` and `log` available"
        )


def run_one_seed(
    agent: QTabularLearner,
    config: Config,
    out: Path,
    env_seed: int,
):
    env = config.env_config._env

    next_chkpt = 0
    td_sum, td_count = 0.0, 0
    snapshots, rows = [], []
    checkpoints = create_checkpoint_steps(
        config.run_config.step_budget,
        config.run_config.checkpoint_type,
        config.run_config.checkpoint_points,
        10,
    )
    episodes = 0
    visitation = np.zeros((config.env_config.obs_space, config.env_config.action_space))

    total_steps = 0
    obs, _ = env.reset(seed=env_seed)
    pbar = tqdm(total=config.run_config.step_budget)

    while total_steps < config.run_config.step_budget:
        eps = epsilon_schedule(
            config.algo_config.epsilon,
            total_steps,
            config.algo_config.decay_steps,
            config.algo_config.e_floor,
        )

        action = agent.action(obs, eps)

        new_obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        delta = agent.update(obs, action, reward, new_obs)

        td_sum += abs(delta)
        td_count += 1

        visitation[obs, action] += 1

        obs = new_obs

        total_steps += 1
        pbar.update(1)

        if next_chkpt < len(checkpoints) and total_steps >= checkpoints[next_chkpt]:
            snapshots.append(agent.qtable.copy())
            rows.append(
                {
                    "env_steps": checkpoints[next_chkpt],
                    "episode": episodes,
                    "mean_td_error": td_sum / max(td_count, 1),
                    "epsilon": eps,
                }
            )
            td_sum, td_count = 0.0, 0
            next_chkpt += 1

        if done:
            obs, _ = env.reset()
            episodes += 1

    pbar.close()

    np.savez(
        out / "snapshots.npz",
        snapshots=np.array(snapshots),
        steps=np.array(checkpoints),
        visitation=np.array(visitation),
    )
    pd.DataFrame(rows).to_parquet(out / "train_log.parquet")


def run_experiment(env: gym.Env, config: Config, out: Path = "."):
    timestamp = datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p")
    experiment_dir = out / "FrozenLake" / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(experiment_dir / "meta.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    rng_factory = SeededRNG(config.run_config.master_seed)

    action_space_seeds = []
    rngs = []
    env_seed = []
    for _ in range(config.run_config.n_runs):
        action_space_seeds.append(rng_factory.next_seed())
        rngs.append(rng_factory.next_rng())
        env_seed.append(rng_factory.next_seed())

    for run_idx in trange(config.run_config.n_runs, desc="Running on random seed"):
        run_dir = experiment_dir / f"run_{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Set action space sample seed to our seed
        env.action_space.seed(action_space_seeds[run_idx])

        agent = QTabularLearner(config, rngs[run_idx])

        run_one_seed(agent, config, run_dir, env_seed[run_idx])


@app.command()
def main(
    master_seed: int = typer.Option(
        default=DEFAULT_SEED, help="RNG seed to set the entire run"
    ),
    lr: float = typer.Option(default=0.1, help="Learning rate for q learning agent"),
    epsilon: float = typer.Option(
        default=0.1,
        help="Epsilon value for e-greedy. Range [0, 1). 0 greedy, 1 random",
    ),
    gamma: float = typer.Option(default=0.95, help="Discount Factor"),
    step_budget: int = typer.Option(
        default=100_000, help="Maximum number of steps, the step budget"
    ),
    n_runs: int = typer.Option(
        default=10, help="How many seeded runs to run the agent in the environment."
    ),
    checkpoint_points: int = typer.Option(
        default=100, help="How often to checkpoint the agent for later analysis"
    ),
    checkpoint_type: str = typer.Option(
        default="linear", help="Options are `linear`, `log`"
    ),
):
    env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        # render_mode="human",
        reward_schedule=(1, 0, 0),  # [reward_goal,  reward_hole, reward_frozen]
        is_slippery=True,
    )

    # initialize the config
    config = Config(
        run_config=ToyTextRunConfig(
            master_seed=master_seed,
            lr=lr,
            step_budget=step_budget,
            n_runs=n_runs,
            checkpoint_points=checkpoint_points,
            checkpoint_type=checkpoint_type,
        ),
        env_config=ToyTextEnvConfig.from_gym_env(env),
        algo_config=ToyTextAlgoConfig(
            epsilon=epsilon, gamma=gamma, decay_steps=step_budget // 2
        ),
    )

    run_experiment(env, config, out=Path("./runs"))


if __name__ == "__main__":
    typer.run(main)

# %%
