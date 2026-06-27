import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from src.rng_factory import SeededRNG


def evaluate(path: Path, eval_master_seed: int = 67):
    """
    Populate run directories from a given path, by picking out the latest run
    path is the runs/ dir for the specific
    type_: allowed strings: `nonslippery` and `slippery`
    """
    run_dirs = sorted(
        (p for p in path.iterdir() if p.is_dir()),
        reverse=True,
        key=lambda p: p.stat().st_mtime,  # Sort by latest modified
    )
    latest_run_dir = run_dirs[0]

    with open(latest_run_dir / "meta.json") as f:
        meta = json.load(f)

    gamma = meta["algo_config"]["gamma"]

    env_kwargs = meta["env_config"]["env_kwargs"]
    is_slippery = env_kwargs["is_slippery"]
    type_ = "slippery" if is_slippery else "nonslippery"

    q_v_star = np.load(f"./src/tabular/q_v_star_frozenlake_{type_}.npz")
    q_star = q_v_star["q_star"]
    v_star = q_v_star["v_star"]
    all_evals = []

    for dir_ in sorted(latest_run_dir.glob("run_*"), reverse=False):
        snapshots = np.load(dir_ / "snapshots.npz")

        qtable_snapshots = snapshots["snapshots"]
        steps = snapshots["steps"]
        visitation = snapshots["visitation"]

        eval_env = gym.make(meta["env_config"]["env_id"], **env_kwargs)
        eval_rng = SeededRNG(eval_master_seed)

        rows = []

        M = 1 if "non" in type_ else 100

        for qtable, step in zip(qtable_snapshots, steps):
            returns, succ, lens = [], [], []
            for _ in range(M):
                obs, _ = eval_env.reset(seed=eval_rng.next_seed())
                done = False
                G, L = 0.0, 0
                discount = 1.0
                while not done:
                    act = int(np.argmax(qtable[obs]))
                    obs, r, term, trunc, _ = eval_env.step(act)
                    done = term or trunc
                    G += discount * r
                    discount *= gamma
                    L += 1

                returns.append(G)
                succ.append(term and r == 1)
                lens.append(L)

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
                    "optimality_gap": v_star[0] - np.mean(returns),
                }
            )

        eval_df = pd.DataFrame(rows)
        all_evals.append(eval_df)
        eval_df.to_parquet(dir_ / "eval_log.parquet", index=False)

    all_evals: pd.DataFrame = pd.concat(all_evals).reset_index(drop=True)
    all_evals.to_parquet(latest_run_dir / "all_evals.parquet")
    print(all_evals)


if __name__ == "__main__":
    evaluate(Path("./runs/FrozenLake"))
