import json
from pathlib import Path

import gymnasium as gym
import imageio.v3 as imageio
import matplotlib
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN

from src.deep.envs import make_env
from src.deep.eval import ENVS
from src.rng_factory import SeededRNG

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

VAL_SEED = 67
TEST_SEED = 420

OBS_DIMS = {
    "LunarLander-v3": {
        "names": ["x", "y", "vx", "vy", "angle", "ang_vel", "leg1", "leg2"],
        "pairs": [(0, 1), (2, 3)],
    },
    "CartPole-v1": {
        "names": ["x", "x_dot", "theta", "theta_dot"],
        "pairs": [(0, 2), (1, 3)],
    },
}


def load_coverage(exp_dir: Path) -> np.ndarray:
    files = sorted(exp_dir.glob("run_*/obs_coverage.npy"))
    return np.concatenate([np.load(f) for f in files])


def plot_obs_coverage(
    variant_dirs: dict, env_id: str, env_kwargs: dict, out_path: Path, gridsize=60
):
    spec = OBS_DIMS[env_id]
    nrows, ncols = len(variant_dirs), len(spec["pairs"])
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )

    for i, (label, d) in enumerate(variant_dirs.items()):
        obs = load_coverage(d)
        tr = champion_trace(d, env_id, env_kwargs)  # <-- once per variant

        for j, (a, b) in enumerate(spec["pairs"]):
            ax = axes[i][j]
            hb = ax.hexbin(
                obs[:, a],
                obs[:, b],
                gridsize=gridsize,
                norm=mcolors.LogNorm(),
                mincnt=1,
                cmap="viridis",
            )
            ax.plot(
                tr[:, a], tr[:, b], color="red", lw=1.5, alpha=0.9, label="champion"
            )  # exploited path
            ax.scatter(
                tr[0, a], tr[0, b], c="white", s=40, ec="k", zorder=5
            )  # start marker
            ax.set_xlabel(spec["names"][a])
            ax.set_ylabel(spec["names"][b])
            ax.set_title(f"{label}: {spec['names'][a]} x {spec['names'][b]}")
            fig.colorbar(hb, ax=ax, label="samples (log)")

    axes[0][0].legend(loc="upper right")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def find_champion(exp_dir, metric="eval_return_mean", smooth=1):
    df = pd.read_parquet(exp_dir / "all_evals.parquet").sort_values(
        ["run_no.", "env_steps"]
    )
    sm_df = df.groupby("run_no.")[metric].transform(
        lambda x: x.rolling(smooth, center=True, min_periods=1).mean()
    )
    best = df.loc[sm_df.idxmax()]
    return best["run_no."], int(best["env_steps"]), float(best[metric])


def load_champion_box(exp_dir: Path, env: gym.Env):
    run_no, champ_step, sel_value = find_champion(exp_dir)
    snaps = torch.load(exp_dir / run_no / "snapshots.pt", weights_only=False)
    idx = list(snaps["steps"]).index(champ_step)
    policy = json.loads((exp_dir / "meta.json").read_text())["algo_config"]["policy"]
    model = DQN(
        policy,
        env,
        policy_kwargs={"net_arch": snaps["net_arch"]},
        buffer_size=1,
        device="cpu",
    )
    model.q_net.load_state_dict(snaps["snapshots"][idx])
    model.policy.set_training_mode(False)
    return model, run_no, champ_step, sel_value


def rollout(model: DQN, env: gym.Env, seed, cap, render=False, trace=False):
    obs, _ = env.reset(seed=seed)
    frames = [env.render()] if render else None
    path = [obs] if trace else None
    done, L, r, term, trunc, G = False, 0, 0.0, False, False, 0.0
    while not done and L < cap:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(int(act))
        if render:
            frames.append(env.render())
        if trace:
            path.append(obs)
        G += r
        done = term or trunc
        L += 1
    return G, L, r, term, trunc, frames, path


def champion_trace(exp_dir: Path, env_id: str, env_kwargs: dict, tries=20):
    env = make_env(env_id, env_kwargs)
    model, _, _, _ = load_champion_box(exp_dir, env)
    is_success = ENVS[env_id]["is_success"]
    cap = env.spec.max_episode_steps or 1000
    best = None

    for t in range(tries):
        G, L, r, term, trunc, _, path = rollout(model, env, t, cap, trace=True)

        if best is None:
            best = path

        if is_success(term, trunc, r, G, L):
            best = path
            break

    env.close()
    return np.asarray(best)


def record_champion_box(
    exp_dir: Path, env_id: str, env_kwargs: dict, out_path: Path, fps=30, tries=20
):
    env = make_env(env_id, env_kwargs, render_mode="rgb_array")
    model, _, _, _ = load_champion_box(exp_dir, env)
    is_success = ENVS[env_id]["is_success"]
    cap = env.spec.max_episode_steps or 1000
    best = None

    # Arbitrary seeds not derived from SeedSequence.
    # This is for the video only, not for measurement.
    for t in range(tries):
        G, L, r, term, trunc, frames, _ = rollout(model, env, t, cap, render=True)
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
    env = make_env(env_id, env_kwargs)
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
        G, L, r, term, trunc, _, _ = rollout(model, env, rng.next_seed(), cap)
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
