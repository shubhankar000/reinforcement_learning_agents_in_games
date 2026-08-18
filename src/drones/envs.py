"""
Create modified env for Pyflyt.
Apply the following changes to the env, that apply for every env:

Update: This file also now contains the env for Task B, waypoiints with pixels.

Task A (pole-balancing):
    1. Strip velocity observations from the obs vector. This is done so that the frame-stack is tested and arch is stressed to infer velocities from the frame-stack as memory. Note: memory does no apply to LSTM, as it is infinite horizon by construction.
    2. Apply a bugfix to the randomness injected to the motor RPM by the PyFlyt env. By default, the noise applies to all 4 motors simultaneously, applying only thrust noise, while what we really want is attitude noise, which is done through applying different noise to all 4 motors.

Task B (waypoint navigation):
    1. Since PyFlyt's reset hook is not general, it hardcodes a begin_reset() call, so drone options cannot be injected.Only full reimplementation can fix this.
    2. WN uses a dict obs with pixels + KIN. Pixels is processed by the CNN as a frame stack, KIN is concated post-aggregator.
"""

from typing import Any
import collections

import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401
from gymnasium.wrappers import FrameStackObservation
from PyFlyt.core.abstractions.motors import Motors
from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.drones.config import KEEP_A, KEEP_B, VECTOR_DIM
from src.drones.scene import N_OBSTACLES, build_scene
from src.rng_factory import SeededRNG


CAMERA_RESOLUTION = (96, 96)
CAMERA_ANGLE_DEG = 0  # default is 20 uptilt for FPV racing
CAMERA_FOV_DEG = 90  # PyFlyt default
K_FRAMES = 8


# ============== PyFlyt physics fix ============== #
# Note: most of the code is copied from pyflyt except the 1 line fix


def physics_update_fix(
    self: Motors, pwm: np.ndarray, rotation: None | np.ndarray = None
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


class StripVelocities(gym.ObservationWrapper):
    """
    Remove all velocities from the obs vector, down to 20 final obs
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = env.observation_space.low[KEEP_A].astype(np.float32)
        high = env.observation_space.high[KEEP_A].astype(np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, observation: Any) -> Any:
        return observation[KEEP_A].astype(np.float32)


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


class PixelDictObs(gym.ObservationWrapper):
    """
    Create a gym Dict{pixels (K,96,96), vector(30,)} observation from PyFlyt's default Dict{attitude, target_deltas} for the multi-obs Task B.

    pixels gets fed to the CNN, while the vector gets concated later after the aggregator for additional information to the network

    Reimplements FlattenWaypointEnv's padding logic. The original wrapper destroys the dict structure and we lose pixels. Plus we need the separation since cnn runs on the frame-stack but vector is concated once after the aggregator
    """

    def __init__(self, env: gym.Env, k_frames=K_FRAMES):
        super().__init__(env)
        self.k = k_frames
        self.n_targets = env.unwrapped.waypoints.num_targets
        h, w = env.unwrapped.camera_resolution
        self.frame_queue = collections.deque(maxlen=k_frames)

        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Box(0, 255, (k_frames, h, w), dtype=np.uint8),
                "vector": gym.spaces.Box(
                    -np.inf, np.inf, (VECTOR_DIM,), dtype=np.float32
                ),
            }
        )

    def reset(self, *, seed=None, options=None):
        self.frame_queue.clear()
        return super().reset(seed=seed, options=options)

    def grey_frame(self):
        """
        Convert a rgba frame to greyscale, drop the alpha channel
        """
        rgba = np.asarray(self.unwrapped.env.drones[0].rgbaImg)

        # Apply Pillow's ITU-R BT.601 luminance transformation
        grey = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
        grey = np.clip(grey, 0, 255)

        return grey.astype(
            np.uint8
        )  # cast as uint8, stays as int64 otherwise, then SB3 fails to normalise

    def observation(self, obs):
        frame = self.grey_frame()
        if not self.frame_queue:
            self.frame_queue.extend([frame] * self.k)  # first obs has duplicates
        else:
            self.frame_queue.append(frame)

        vector = np.concatenate(
            [obs["attitude"][KEEP_B], self.pad(obs["target_deltas"])]
        ).astype(np.float32)

        return {"pixels": np.stack(self.frame_queue, axis=0), "vector": vector}

    def pad(self, deltas):
        """
        PyFlyt's remaining deltas x,y,z position are of type Sequence, they change length as new waypoints are reached. Thats why PyFlyt provides FlattenWaypointEnv, but it kills the dict structure and pixels/vector separation.

        change the FlattenWaypointEnv's padding structure. All 4 target delta values stay in the vector at all times (constant length).
        Slot 0 (deltas[0]) is always the active slot (reward only given to active slot).
        Use a 'present' boolean flag to indicate this.
        Slots look like:
        [dx, dy, dz, present_flag]
        """
        out = np.zeros((self.n_targets, 4))
        for i, d in enumerate(deltas):
            out[i, :3] = d
            out[i, 3] = 1.0

        return out.ravel()


class WaypointsWithObstacles(QuadXWaypointsEnv):
    def __init__(
        self,
        n_obstacles: int = N_OBSTACLES,
        camera_resolution: tuple[int, int] = CAMERA_RESOLUTION,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_obstacles = n_obstacles
        self.camera_resolution = camera_resolution
        self.obstacle_ids = []

    def reset(self, *, seed: int = None, options: dict = None):
        """
        Copy of QuadXWaypointsEnv.reset() with drone_options + scene with obstacles inserted
        """
        drone_options = {
            "use_camera": True,
            "use_gimbal": False,  # Gimbling horizon-locks the camera, strips attitude information from pixels
            "camera_angle_degrees": CAMERA_ANGLE_DEG,
            "camera_FOV_degrees": CAMERA_FOV_DEG,
            "camera_resolution": self.camera_resolution,
        }

        super().begin_reset(seed, options, drone_options)
        self.waypoints.reset(self.env, self.np_random)
        self.info["num_targets_reached"] = 0
        self.info["obstacle_collision"] = (
            False  # Inject this kv, use later for env term/trunc info in wandb
        )

        self.obstacle_ids = build_scene(
            self.env,
            self.np_random,
            self.waypoints.targets,
            self.start_pos[0],
            self.flight_dome_size,
            self.waypoints.min_height,
            self.env.planeId,
            self.n_obstacles,
        )

        super().end_reset()

        return self.state, self.info

    def compute_term_trunc_reward(self) -> None:
        """
        Same as QuadXWaypointsEnv, but treat obstacle contact as floor contact (r=-100, terminate episode)
        """
        super().compute_term_trunc_reward()

        drone_id = self.env.drones[0].Id
        if self.obstacle_ids and np.any(
            self.env.contact_array[drone_id, self.obstacle_ids]
        ):
            self.reward = -100.0
            self.info["collision"] = True
            self.info["obstacle_collision"] = True
            self.termination |= True


def make_env(
    seed: int,
    stripped=True,
    flight_mode=-1,
    k_frames=K_FRAMES,
    spread=0.1,
):
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

    if use_subproc:
        venv = SubprocVecEnv(env_list, start_method="fork")
    else:
        venv = DummyVecEnv(env_list)

    return venv


def make_env_b(
    seed: int,
    n_obstacles: int = N_OBSTACLES,
    k_frames: int = K_FRAMES,
    camera_resolution: tuple[int, int] = CAMERA_RESOLUTION,
    flight_mode: int = 0,
):
    def build():
        apply_motor_noise_fix()
        # construct env directly since not registered with gym
        env = WaypointsWithObstacles(
            n_obstacles=n_obstacles,
            camera_resolution=camera_resolution,
            flight_mode=flight_mode,
        )
        env = PixelDictObs(env, k_frames)
        env = Monitor(
            env,
            info_keywords=(
                "num_targets_reached",
                "obstacle_collision",
                "collision",
                "out_of_bounds",
                "env_complete",
            ),
        )
        env.reset(seed=seed)
        env.action_space.seed(seed)

        return env

    return build


def make_vecenv_b(
    seed: int,
    n_envs=8,
    use_subproc: bool = True,
    n_obstacles: int = N_OBSTACLES,
    k_frames: int = K_FRAMES,
    flight_mode: int = 0,
):
    rng = SeededRNG(seed)

    env_list = [
        make_env_b(rng.next_seed(), n_obstacles, k_frames, flight_mode=flight_mode)
        for _ in range(n_envs)
    ]

    if use_subproc:
        venv = SubprocVecEnv(env_list, start_method="fork")
    else:
        venv = DummyVecEnv(env_list)

    return venv
