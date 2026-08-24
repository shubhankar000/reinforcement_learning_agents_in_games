import json
from pathlib import Path

import imageio.v3 as imageio
from stable_baselines3 import DQN

from src.deep.envs import make_env
from src.deep.figures import rollout
from src.rng_factory import SeededRNG

ENV_ID = "CarRacing-v3"
ENV_KWARGS = {"continuous": False}  # matches ENVS["CarRacing-v3"]["base_kwargs"]
EXP_DIR = Path("runs/CarRacing-v3/dqn/2026-07-22-11-22-59-PM-main")
VID_DIR = Path("runs/CarRacing-v3/dqn/videos")

VIDEO_SEED = 1337  # distinct from VAL_SEED=67 / TEST_SEED=420
N_EXTRA, START_IDX, FPS = 2, 2, 30


def track_completion(env):
    """Fraction of track tiles driven over. Reads better than raw return.

    Must be called AFTER the rollout and BEFORE the next reset - both
    attributes are cleared by CarRacing.reset().
    """
    core = env.unwrapped
    n_tiles = len(core.track)
    return core.tile_visited_count / n_tiles if n_tiles else float("nan")


def record_extra(exp_dir, env_id, env_kwargs, vid_dir, n, start_idx, fps):
    """Roll out the saved champion on n fresh seeds -> champion_{start_idx+i}.mp4/gif.

    Unlike figures.record_champion_box these clips are NOT selected for success -
    they are whatever the seed produced, so they are a fair sample of the policy.
    """
    rng = SeededRNG(VIDEO_SEED)
    env = make_env(env_id, env_kwargs, render_mode="rgb_array")
    model = DQN.load(exp_dir / "champion_model.zip", device="cpu")
    cap = env.spec.max_episode_steps or 1000

    clips = []
    for i in range(n):
        seed = rng.next_seed()
        G, L, r, term, trunc, frames, _ = rollout(model, env, seed, cap, render=True)
        pct = track_completion(env)

        out = vid_dir / f"champion_{start_idx + i}"
        imageio.imwrite(
            out.with_suffix(".mp4"),
            frames,
            fps=fps,
            macro_block_size=1,
            plugin="FFMPEG",
        )
        imageio.imwrite(out.with_suffix(".gif"), frames, duration=1000 / fps, loop=0)

        clips.append(
            {
                "name": out.name,
                "seed": int(seed),
                "return": float(G),
                "steps": int(L),
                "track_completion": float(pct),
                "terminated": bool(term),
                "truncated": bool(trunc),
            }
        )
        print(f"{out.name}  seed={seed}  G={G:8.2f}  L={L:4d}  track={pct:6.1%}")

    env.close()

    # Sidecar so the captions can be quoted in the report without re-running.
    (vid_dir / "extra_clips.json").write_text(json.dumps(clips, indent=2))
    return clips


def main():
    VID_DIR.mkdir(parents=True, exist_ok=True)
    record_extra(EXP_DIR, ENV_ID, ENV_KWARGS, VID_DIR, N_EXTRA, START_IDX, FPS)


if __name__ == "__main__":
    main()
