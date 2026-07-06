import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import ToyTextEnvConfig
from src.rng_factory import SeededRNG

# Eval constant for success criteria and fixed starting pos
ENV_EVAL = {
    "FrozenLake-v1": {
        "is_success": lambda term, trunc, r: term and r == 1,
        "fixed_start": True,
    },
    "CliffWalking-v1": {
        "is_success": lambda term, trunc, r: term,
        "fixed_start": True,
    },
    "Taxi-v4": {
        "is_success": lambda term, trunc, r: term,
        "fixed_start": False,
    },
}


def evaluate(exp_dir: Path, eval_master_seed: int = 67):
    """
    Evaluate every seeded run inside a single experiment dir and write a
    per-run eval_log.parquet plus an aggregated all_evals.parquet.

    exp_dir: a single leaf experiment dir, e.g.
        runs/FrozenLake-v1/2026-07-02-05-36-20-PM-det
    """
    with open(exp_dir / "meta.json") as f:
        meta = json.load(f)

    gamma = meta["algo_config"]["gamma"]

    env_kwargs: dict = meta["env_config"]["env_kwargs"]

    env_config = ToyTextEnvConfig.from_gym_env(
        gym.make(meta["env_config"]["env_id"], **env_kwargs)
    )

    slip = bool(env_config.env_kwargs.get("is_slippery", False)) or env_kwargs.get(
        "is_rainy", False
    )
    key = f"{env_config.env_id}_{'slip' if slip else 'det'}"

    q_v_star = np.load(f"./src/tabular/optimal_values/q_v_star_{key}.npz")
    q_star = q_v_star["q_star"]
    v_star = q_v_star["v_star"]
    v_random = q_v_star["v_random"]
    all_evals = []

    for dir_ in tqdm(sorted(exp_dir.glob("run_*")), desc="Evaluating 1 seed"):
        snapshots = np.load(dir_ / "snapshots.npz")

        qtable_snapshots = snapshots["snapshots"]
        steps = snapshots["steps"]
        visitation = snapshots["visitation"]

        eval_env = gym.make(meta["env_config"]["env_id"], **env_kwargs)
        eval_rng = SeededRNG(eval_master_seed)

        rows = []
        cfg = ENV_EVAL[env_config.env_id]
        M = 1 if (not slip and cfg["fixed_start"]) else 100
        cap = eval_env.spec.max_episode_steps or 200

        for qtable, step in zip(qtable_snapshots, steps):
            returns, succ, lens, starts = [], [], [], []
            for _ in range(M):
                obs, _ = eval_env.reset(seed=eval_rng.next_seed())
                s0 = obs
                done = False
                G, L = 0.0, 0
                discount = 1.0
                while not done and L < cap:
                    act = int(np.argmax(qtable[obs]))
                    obs, r, term, trunc, _ = eval_env.step(act)
                    done = term or trunc
                    G += discount * r
                    discount *= gamma
                    L += 1

                returns.append(G)
                succ.append(cfg["is_success"](term, trunc, r))
                lens.append(L)
                starts.append(s0)

            starts = np.array(starts)
            v_star_s0 = float(np.mean(v_star[starts]))
            v_random_s0 = float(np.mean(v_random[starts]))
            optimality_gap = float(v_star_s0 - np.mean(returns))
            weighted_q_linf = (
                visitation * np.abs(qtable - q_star)
            ).sum() / visitation.sum()
            rows.append(
                {
                    "run_no.": dir_.name,
                    "env_steps": step,
                    "eval_return_mean": np.mean(returns),
                    "eval_return_std": np.std(returns),
                    "success_rate": np.mean(succ),
                    "episode_len_mean": np.mean(lens),
                    "q_star_linf": np.max(np.abs(qtable - q_star)),
                    "weighted_q_linf": weighted_q_linf,
                    "optimality_gap": optimality_gap,
                    "v_star_s0": v_star_s0,
                    "v_random_s0": v_random_s0,
                }
            )

        eval_df = pd.DataFrame(rows)
        all_evals.append(eval_df)
        eval_df.to_parquet(dir_ / "eval_log.parquet", index=False)

    all_evals: pd.DataFrame = pd.concat(all_evals).reset_index(drop=True)
    all_evals.to_parquet(exp_dir / "all_evals.parquet")


if __name__ == "__main__":
    root = Path("runs/FrozenLake-v1")
    for exp in sorted(root.iterdir()):
        if exp.is_dir() and (exp / "meta.json").exists():
            evaluate(exp)
