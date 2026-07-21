import json
from pathlib import Path

import gymnasium as gym
import imageio.v3 as imageio
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN

from src.dqn.eval import ENVS
from src.rng_factory import SeededRNG

VAL_SEED = 67
TEST_SEED = 420


def find_champion(exp_dir, metric="eval_return_mean", smooth=3):
    df = pd.read_parquet(exp_dir / "all_evals.parquet").sort_values(
        ["run_no.", "env_steps"]
    )
    sm_df = df.groupby("run_no.")[metric].transform(
        lambda x: x.rolling(smooth, center=True, min_periods=1).mean()
    )
    best = df.loc[sm_df.idxmax()]
    return best["run_no."], int(best["env_steps"]), float(best[metric])


def load_champion_box(exp_dir, env):
    run_no, champ_step, sel_value = find_champion(exp_dir)
    snaps = torch.load(exp_dir / run_no / "snapshots.pt", weights_only=False)
    idx = list(snaps["steps"]).index(champ_step)
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": snaps["net_arch"]},
        buffer_size=1,
        device="cpu",
    )
    model.q_net.load_state_dict(snaps["snapshots"][idx])
    model.policy.set_training_mode(False)
    return model, run_no, champ_step, sel_value


def rollout(model: DQN, env: gym.Env, seed, cap, render=False):
    obs, _ = env.reset(seed=seed)
    frames = [env.render()] if render else None
    done, L, r, term, trunc, G = False, 0, 0.0, False, False, 0.0

    while not done and L < cap:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(int(act))

        if render:
            frames.append(env.render())

        G += r
        done = term or trunc
        L += 1

    return G, L, r, term, trunc, frames


def record_champion_box(
    exp_dir: Path, env_id: str, env_kwargs: dict, out_path: Path, fps=30, tries=20
):
    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    model, _, _, _ = load_champion_box(exp_dir, env)
    is_success = ENVS[env_id]["is_success"]
    cap = env.spec.max_episode_steps or 1000
    best = None

    # Arbitrary seeds not derived from SeedSequence.
    # This is for the video only, not for measurement.
    for t in range(tries):
        G, L, r, term, trunc, frames = rollout(model, env, t, cap, render=True)
        if best is None:
            best = frames

        if is_success(term, trunc, r, G, L):
            best = frames
            break

    imageio.imwrite(
        out_path.with_suffix(".mp4"), best, fps=fps, macro_block_size=1, plugin="FFMPEG"
    )
    imageio.imwrite(out_path.with_suffix(".gif"), best, duration=1000 / fps, loop=0)
    env.close()


def save_champion_box(exp_dir: Path, env_id, env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    model, run_no, champ_step, sel_value = load_champion_box(exp_dir, env)

    model.save(exp_dir / "champion_model.zip")  # inference only
    meta = {
        "run_no": run_no,
        "env_steps": champ_step,
        "metric": "eval_return_mean",
        "selected_return_mean": sel_value,
        "select_master_seed": VAL_SEED,
        **verify_champion(model, env),
    }
    (exp_dir / "champion_meta.json").write_text(json.dumps(meta, indent=2))
    env.close()


def verify_champion(model: DQN, env: gym.Env, M=200):
    """
    Evaluate champion on seed distinct from selection seed, to prevent selection bias
    """
    cap = env.spec.max_episode_steps or 1000
    is_success = ENVS[env.spec.id]["is_success"]

    rng = SeededRNG(TEST_SEED)
    Gs, succ = [], []
    for _ in range(M):
        G, L, r, term, trunc, _ = rollout(model, env, rng.next_seed(), cap)
        Gs.append(G)
        succ.append(is_success(term, trunc, r, G, L))

    Gs = np.asarray(Gs)
    return {
        "verified_return_mean": float(Gs.mean()),
        "verified_return_se": float(Gs.std(ddof=1) / np.sqrt(M)),
        "verified_success_rate": float(np.mean(succ)),
        "verified_M": M,
        "verified_master_seed": TEST_SEED,
    }
