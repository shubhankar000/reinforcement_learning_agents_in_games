import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_vecenv(env_id, n_envs, env_kwargs, use_subproc=False):
    def _factory():
        def _init():
            return Monitor(gym.make(env_id, **env_kwargs))

        return _init

    VecCls = SubprocVecEnv if use_subproc else DummyVecEnv
    return VecCls([_factory() for _ in range(n_envs)])
