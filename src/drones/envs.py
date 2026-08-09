"""
Create modified env for Pyflyt.
Apply the following changes to the env, that apply for every env:

1. Strip velocity observations from the obs vector. This is done so that the frame-stack is tested and arch is stressed to infer velocities from the frame-stack as memory. Note: memory does no apply to LSTM, as it is infinite horizon by construction.
2. Apply a bugfix to the randomness injected to the motor RPM by the PyFlyt env. By default, the noise applies to all 4 motors simultaneously, applying only thrust noise, while what we really want is attitude noise, which is done through applying different noise to all 4 motors.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401
from gymnasium.wrappers import FrameStackObservation
from PyFlyt.core.abstractions.motors import Motors
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from src.rng_factory import SeededRNG

# ============== PyFlyt physics fix ============== #
# Note: most of the code is copied from pyflyt except the 1 line fix


def physics_update_fix(
    self, pwm: np.ndarray, rotation: None | np.ndarray = None
) -> None:
    """Converts motor PWM values to forces, this motor allows negative thrust.

    Fix applied and highlighted

    Args:
        pwm (np.ndarray): [num_motors, ] array defining the pwm values of each motor from -1 to 1.
        rotation (np.ndarray): (num_motors, 3, 3) rotation matrices to rotate each booster's thrust axis around, this is readily obtained from the `gimbals` component.

    """
    assert np.all(pwm >= -1.0) and np.all(pwm <= 1.0), (
        f"`{pwm=} has values out of bounds of -1.0 and 1.0.`"
    )
    if rotation is not None:
        assert rotation.shape == (
            self.num_motors,
            3,
            3,
        ), f"`rotation` should be of shape (num_motors, 3, 3), got {rotation.shape}"

    # model the motor using first order ODE, y' = T/tau * (setpoint - y)
    self.throttle += (self.physics_period / self.tau) * (pwm - self.throttle)

    # noise in the motor
    self.throttle += (
        self.np_random.normal(size=self.throttle.shape)
        * self.throttle
        * self.noise_ratio
    )  # <======= FIX APPLIED

    # compute thrust and torque in jitted manner
    (thrust, torque) = self._jitted_compute_thrust_torque(
        rotation,
        self.throttle,
        self.max_rpm,
        self.thrust_unit,
        self.thrust_coef,
        self.torque_coef,
    )

    # apply the forces
    for idx, thr, tor in zip(self.motor_ids, thrust, torque):
        self.p.applyExternalForce(
            self.uav_id, idx, thr, [0.0, 0.0, 0.0], self.p.LINK_FRAME
        )
        self.p.applyExternalTorque(self.uav_id, idx, tor, self.p.LINK_FRAME)


def apply_motor_noise_fix():
    """
    Apply the motor noise fix to PyFlyt by replacing its physics method. This function is idempotent
    """
    Motors.physics_update = physics_update_fix


# ============== gym Wrappers for obs trim ============== #
# Obs dim range meanings
# ang_vel 0:3 | quat 3:7 | lin_vel 7:10 | lin_pos 10:13 | prev_action 13:17 | aux 17:21
# pole_top_pos 21:24 | pole_bot_pos 24:27 | pole_top_vel 27:30 | pole_bot_vel 30:33
KEEP = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    21,
    22,
    23,
    24,
    25,
    26,
]  # 20 dims — drops lin_vel, motor throttle state and BOTH pole vel


class StripVelocities(gym.ObservationWrapper):
    """
    Remove all velocities from the obs vector, down to 20 final obs
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = env.observation_space.low[KEEP].astype(np.float32)
        high = env.observation_space.high[KEEP].astype(np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, observation: Any) -> Any:
        return observation[KEEP].astype(np.float32)


class RandomStartingOrientation(gym.Wrapper):
    """
    Randomly tilt the drone at the start of each episode, for generalizability.

    The base env reads `start_orn` while resetting, so set it before calling reset
    """

    def __init__(self, env, spread=0.1):
        super().__init__(env)
        self.spread = spread
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        # NOTE: Using our own rng separate from the env's physics rng, so that starting tilt is decided by the seed we are giving and nothing else

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.env.unwrapped.start_orn = self._rng.uniform(
            -self.spread, self.spread, size=(1, 3)
        )
        return self.env.reset(seed=seed, options=options)


def make_env(seed: int, stripped=True, flight_mode=-1, k_frames=8, spread=0.1):
    def build():
        apply_motor_noise_fix()
        env = gym.make("PyFlyt/QuadX-Pole-Balance-v4", flight_mode=flight_mode)
        if stripped:
            env = StripVelocities(env)
        env = RandomStartingOrientation(env, spread)
        if k_frames > 1:  # else Box(20,) (for lstm)
            env = FrameStackObservation(env, k_frames)  # Box(k, 20)
        env = Monitor(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return build


def make_vecenv(
    seed: int,
    n_envs: int = 8,
    use_subproc=True,
    normalize=True,
    stripped=True,
    flight_mode=-1,
    k_frames=8,
    spread=0.1,
):
    # Seedsequenced from master seed rng for full reproducibility
    rng = SeededRNG(seed)

    env_list = [
        make_env(rng.next_seed(), stripped, flight_mode, k_frames, spread)
        for _ in range(n_envs)
    ]

    # Require start_method='fork' cos inside a joblib worker the default context is loky.
    if use_subproc:
        venv = SubprocVecEnv(env_list, start_method='fork')
    else:
        venv = DummyVecEnv(env_list)

    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    return venv
