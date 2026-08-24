"""
Task B PID-based measurement of v_star and v_random
This is a function of the task, and the constants that define the env. Any change in env requires this file to be run again
"""

import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np

from src.drones.config import EVAL_SEED
from src.drones.envs import make_env_b
from src.drones.scene import N_OBSTACLES, scene_params
from src.rng_factory import SeededRNG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PID_FLIGHT_MODE = 7
AGENT_FLIGHT_MODE = 0
N_EPISODES = 100
OUT_PATH = PROJECT_ROOT / "runs" / "drones" / "anchors" / "waypoint.json"
MAX_DURATION = 20.0  # mirror make_env_b's
PID_K_FRAMES = 1  # pid never looks at pixels


def pid_action(env: gym.Env):
    """
    Mode 7 setpoint, [x, y, yaw, z]. Waypoint targets are world-frame absolute,
    so the active target is fed straight through as a position setpoint.
    """
    remaining_targets = env.unwrapped.waypoints.targets

    if len(remaining_targets) == 0:
        state = env.unwrapped.env.state(0)
        lin_pos, ang_pos = state[-1], state[1]
        return np.array(
            [lin_pos[0], lin_pos[1], ang_pos[2], lin_pos[2]], dtype=np.float64
        )

    active_target = remaining_targets[0]

    return np.array(
        [active_target[0], active_target[1], 0.0, active_target[2]], dtype=np.float64
    )


def random_action(env: gym.Env):
    return env.action_space.sample()


def rollout(env: gym.Env, seed: int, policy_fn: Callable):
    env.reset(seed=seed)
    env.action_space.seed(seed)

    while True:
        action = policy_fn(env)
        _, _, term, trunc, info = env.step(action)

        if term or trunc:
            break

    return info["episode"]


def env_facts(env: gym.Env):
    u = env.unwrapped

    return {
        "n_obstacles": u.n_obstacles,
        "num_targets": u.waypoints.num_targets,
        "goal_reach_distance": u.waypoints.goal_reach_distance,
        "flight_dome_size": u.flight_dome_size,
        "max_steps": u.max_steps,
        "env_step_ratio": u.env_step_ratio,
        "camera_resolution": list(u.camera_resolution),
    }


def measure(policy: str, n_episodes: int, seed: int, n_obstacles: int):
    flight_mode = PID_FLIGHT_MODE if policy == "pid" else AGENT_FLIGHT_MODE

    env = make_env_b(
        seed,
        n_obstacles,
        k_frames=PID_K_FRAMES,
        flight_mode=flight_mode,
        max_duration=MAX_DURATION,
    )()

    rng = SeededRNG(seed)

    policy_fn: Callable = pid_action if policy == "pid" else random_action

    try:
        episodes = [rollout(env, rng.next_seed(), policy_fn) for _ in range(n_episodes)]
        efacts = env_facts(env)
    finally:
        env.close()

    return episodes, efacts


def summarise(episodes: list[dict]) -> dict:
    returns = np.array([e["r"] for e in episodes], dtype=np.float64)
    lengths = np.array([e["l"] for e in episodes], dtype=np.float64)
    targets = np.array([e["num_targets_reached"] for e in episodes], dtype=np.float64)

    complete = np.array([bool(e["env_complete"]) for e in episodes])
    oob = np.array([bool(e["out_of_bounds"]) for e in episodes])
    any_coll = np.array([bool(e["collision"]) for e in episodes])
    obstacle = np.array([bool(e["obstacle_collision"]) for e in episodes])

    floor = any_coll & ~obstacle  # collision fires for floor AND obstacle
    timeout = ~complete & ~any_coll & ~oob  # trunc is ambiguous, derive as residual

    return {
        "n_episodes": len(episodes),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_median": float(np.median(returns)),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
        "targets_mean": float(targets.mean()),
        "targets_max": float(targets.max()),
        "rate_complete": float(complete.mean()),
        "rate_obstacle_collision": float(obstacle.mean()),
        "rate_floor_collision": float(floor.mean()),
        "rate_out_of_bounds": float(oob.mean()),
        "rate_timeout": float(timeout.mean()),
    }


def main() -> None:
    pid_eps, pid_facts = measure("pid", N_EPISODES, EVAL_SEED, N_OBSTACLES)
    rand_eps, rand_facts = measure("random", N_EPISODES, EVAL_SEED, N_OBSTACLES)

    assert pid_facts == rand_facts, f"env mismatch\n{pid_facts}\n{rand_facts}"

    pid_stats = summarise(pid_eps)
    rand_stats = summarise(rand_eps)

    payload = {
        "task": "waypoint",
        "v_star_s0": pid_stats["return_mean"],
        "v_random_s0": rand_stats["return_mean"],
        "pid": pid_stats,
        "random": rand_stats,
        "env_facts": pid_facts,
        "scene_params": scene_params(),
        "pid_flight_mode": PID_FLIGHT_MODE,
        "agent_flight_mode": AGENT_FLIGHT_MODE,
        "n_episodes": N_EPISODES,
        "seed": EVAL_SEED,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"v_star_s0   = {payload['v_star_s0']:.3f}")
    print(f"v_random_s0 = {payload['v_random_s0']:.3f}")
    print(f"\nwritten to {OUT_PATH}\n")

    for name, stats in (("pid", pid_stats), ("random", rand_stats)):
        print(f"[{name}]")
        for k, v in stats.items():
            print(f"  {k:26s} {v}")
        print()


if __name__ == "__main__":
    main()
