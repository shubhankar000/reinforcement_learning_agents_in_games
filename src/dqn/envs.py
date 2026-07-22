import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_vecenv(env_id, n_envs, env_kwargs, use_subproc=False):
    def _factory():
        def _init():
            return Monitor(make_env(env_id, env_kwargs))

        return _init

    VecCls = SubprocVecEnv if use_subproc else DummyVecEnv
    return VecCls([_factory() for _ in range(n_envs)])


def wrap_image(env):
    """
    Image Wrapper for car racing pixel env.
    (4, 84, 84) uint8 channels first for NatureCNN
    """
    env = GrayscaleObservation(env, keep_dim=False)  # (96,96,3) -> (96,96)
    env = ResizeObservation(env, (84, 84))  # resize to (84,84) for natureCNN
    env = FrameStackObservation(env, 4)  # (4, 84,84)
    return env


def make_env(env_id, env_kwargs, render_mode=None):
    env = gym.make(env_id, **env_kwargs, render_mode=render_mode)
    if (
        isinstance(env.observation_space, gym.spaces.Box)
        and len(env.observation_space.shape) == 3
    ):
        env = wrap_image(env)
    return env
