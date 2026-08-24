"""
Task A anchors. v_random is measured, but v_star is analytic.

There is no flight mode or 'best agent' for task a, it has to be analysed from the env's reward function to produce a ceiling that no actual policy can reach. This is because it requires the drone to translate with no roll or yaw, which is not possible for a drone.

v_star derivation:
    per agent step: reward=-0.1
    env_step_ratio = pysics_hz/agent_hz = 120/40 = 3
    reward_per_physics_step = 1 - lin_dist - ang_dist - leaningness - 0.01*yaw_rate^2
=> maximum reward per agent step = -0.1 + 3*1.0 = 2.9
   (lin_dist=ang_dist=leaningness=yaw_rate=0)
max_steps = int(agent_hz * max_duration_seconds) = 800

v_star = max_steps * max_reward_per_step = 2.9 * 800 = 2320

Drone cannot have zero roll/pitch and zero pole lean together, especially due to the noise injected to the motors at each step.
"""

import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from src.drones.config import EVAL_SEED
from src.drones.envs import make_env
from src.rng_factory import SeededRNG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "runs" / "drones" / "anchors" / "pole-balance.json"

AGENT_FLIGHT_MODE = 0
N_EPISODES = 100
MAX_PER_AGENT_STEP = 2.9
MAX_EPISODE_LENGTH = 800
V_STAR = MAX_PER_AGENT_STEP * MAX_EPISODE_LENGTH  # 2320


def random_action(env: gym.Env):
    return env.action_space.sample()


def rollout(env: gym.Env, seed: int, policy_fn: Callable):
    env.reset(seed=seed)
    env.action_space.seed(seed)

    while True:
        _, _, term, trunc, info = env.step(policy_fn(env))
        if term or trunc:
            break

    return info["episode"]


def env_facts(env: gym.Env):
    u = env.unwrapped

    return {
        "flight_dome_size": u.flight_dome_size,
        "max_steps": u.max_steps,
        "env_step_ratio": u.env_step_ratio,
        "max_per_agent_step": MAX_PER_AGENT_STEP,
        "max_episode_steps": MAX_EPISODE_LENGTH,
    }


def measure(n_episodes: int, seed: int):
    env = make_env(seed, stripped=True, flight_mode=AGENT_FLIGHT_MODE, k_frames=1)()

    rng = SeededRNG(seed)

    try:
        episodes = [
            rollout(env, rng.next_seed(), random_action) for _ in range(n_episodes)
        ]
        efacts = env_facts(env)

    finally:
        env.close()

    return episodes, efacts


def summarise(episodes: list[dict]):
    returns = np.array([e["r"] for e in episodes], dtype=np.float64)
    lengths = np.array([e["l"] for e in episodes], dtype=np.float64)

    return {
        "n_episodes": len(episodes),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_median": float(np.median(returns)),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
        "length_max": float(lengths.max()),
    }


def main():
    episodes, facts = measure(N_EPISODES, EVAL_SEED)
    stats = summarise(episodes)

    payload = {
        "task": "pole-balance",
        "v_star_s0": V_STAR,
        "v_random_s0": stats["return_mean"],
        "random": stats,
        "env_facts": facts,
        "agent_flight_mode": AGENT_FLIGHT_MODE,
        "n_episodes": N_EPISODES,
        "seed": EVAL_SEED,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"v_star_s0 = {payload['v_star_s0']:.3f}  (analytic supremum)")
    print(f"v_random_s0 = {payload['v_random_s0']:.3f}")
    print(f"\nwritten to {OUT_PATH}\n")
    for k, v in stats.items():
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
