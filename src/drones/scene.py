"""
Procedural scene generation for Task B; PyFlyt waypoints with pixels

Procedurally generate obstacles from the pybullet library and insert them in the scene. This is done to generalise the drone's learning to avoid obstacles as its navigating to the waypoints.

Generated obstacles follow this property:
    1. they have no mass, are static and do not move.
    2. collision with obstacles is considered a termination
    3. object placements must be valid. No overlapping objects. this is done through rejection sampling against other obstacles.
    4. they are visible against the background and the floor in greyscale via valid luminance values
    5. obstacle coordinates are not provided to the agent, the agent must learn to navigate around them using pixels.
    6. Obstacle placements are drawn from the same distribution as the env waypoints

Pybullet doesnt have a proper docs website. Used directly from their github: https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet_data
from PyFlyt.core import Aviary

N_OBSTACLES = 5  # number of obstacles in each scene
MIN_OBJ_SEP = 1.2  # in meters, obstacle<->obstacle separation
MIN_OBJ_WP_SEP = 0.8  # obstacle<->waypoint separation
HALF_EXTENT = (0.15, 0.34)  # pybullet convention, use to define body's radius
LUMA = (0.1, 0.85)  # visually separable luma from sky and floor
REJECTION_MAX_TRIES = 600
SHAPES = ("box", "sphere", "cylinder")
DIST_MIN = 1.0  # mirrors pyflyt
DOME_FRAC = 0.9  # mirrors pyflyt


@dataclass(frozen=True)
class Obstacle:
    shape: str  # one of the SHAPES
    half_extent: float
    position: tuple  # (x, y, z)
    rotation: float  # rotation of objects, no-op for cylinder/sphere
    luma: float  # greyscale channel value


def sample_position(rng: np.random.Generator, dome_size, min_height: float):
    """
    Sample 1 point from the envs own waypoint distribution
    """
    dist = rng.uniform(DIST_MIN, DOME_FRAC * dome_size)
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0, 2 * np.pi)

    x = dist * np.sin(phi) * np.cos(theta)
    y = dist * np.sin(phi) * np.sin(theta)
    z = max(np.abs(dist * np.cos(phi)), min_height)

    return x, y, z


def sample_obstacle(
    rng: np.random.Generator, waypoints, start_pos, dome_size, min_height, n=N_OBSTACLES
):
    """
    Rejection-sample n obstacles. Accept only if:
    - min distance to every accepted obstacle >= MIN_OBJ_SEP
    - distance to start pos >= MIN_OBJ_SEP
    - min distance to every waypoint >= MIN_OBJ_WP_SEP
    """
    obstacles: list[Obstacle] = []
    tries = 0
    start_pos = np.asarray(start_pos)
    waypoints = np.asarray(waypoints)

    while len(obstacles) < n:
        tries += 1
        if tries > REJECTION_MAX_TRIES:
            raise RuntimeError(
                "Max tries exceeded when rejection sampling obstacles into the scene"
            )

        pos = np.asarray(sample_position(rng, dome_size, min_height))

        if np.linalg.norm(pos - start_pos) < MIN_OBJ_SEP:
            continue
        if np.linalg.norm(waypoints - pos, axis=1).min() < MIN_OBJ_WP_SEP:
            continue
        if (
            obstacles
            and min(np.linalg.norm(pos - o.position) for o in obstacles) < MIN_OBJ_SEP
        ):
            continue

        sampled_shape = SHAPES[rng.integers(low=0, high=len(SHAPES))]
        half = rng.uniform(*HALF_EXTENT)
        rotation = rng.uniform(0, 2 * np.pi)
        luma = rng.uniform(*LUMA)
        obstacles.append(
            Obstacle(sampled_shape, half, tuple(pos.tolist()), rotation, luma)
        )

    return obstacles, tries


def spawn_obstacles(p: Aviary, obstacles: list[Obstacle]):
    """
    Create the obstacles, return the ids
    """
    ids = []

    for o in obstacles:
        half = o.half_extent
        if o.shape == "box":
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[half, half, half]
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[half, half, half],
                rgbaColor=[o.luma] * 3 + [1.0],
            )  # Add 1.0 alpha channel
        elif o.shape == "sphere":
            collision = p.createCollisionShape(p.GEOM_SPHERE, radius=half)
            visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=half, rgbaColor=[o.luma] * 3 + [1.0]
            )
        elif o.shape == "cylinder":
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=half, height=2 * half
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=half,
                length=2 * half,
                rgbaColor=[o.luma] * 3 + [1.0],
            )
        else:
            raise ValueError(f"unacceptable shape {o.shape}")

        quaternion = p.getQuaternionFromEuler(
            [0.0, 0.0, o.rotation]
        )  # only rotate on yaw axis
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=o.position,
            baseOrientation=quaternion,
        )
        ids.append(body_id)

    return ids


def set_floor(p: Aviary, plane_id):
    """
    Set the floor texture to the blue and white checkerboard pattern, constant throughout, not sampled (other floor textures are noisy or look bad)
    Works through side-effect
    """
    path = Path(pybullet_data.getDataPath())
    texture = p.loadTexture(str(path / "checker_blue.png"))
    p.changeVisualShape(plane_id, -1, textureUniqueId=texture)


def build_scene(
    p, rng, waypoints, start_pos, dome_size, min_height, plane_id, n=N_OBSTACLES
):
    """
    Called by the env's reset and waypoints have already been set
    """
    set_floor(p, plane_id)
    obstacles, tries = sample_obstacle(
        rng, waypoints, start_pos, dome_size, min_height, n
    )
    ids = spawn_obstacles(p, obstacles)
    return ids


def scene_params() -> dict:
    """
    Return scene parameters as a dict for meta.json and eval guard
    """
    return {
        # "n_obstacles": N_OBSTACLES, # pulled from run_one rather than here.
        "min_object_separation": MIN_OBJ_SEP,
        "min_object_wp_separation": MIN_OBJ_WP_SEP,
        "half_extent": HALF_EXTENT,
        "luma": LUMA,
        "rejection_max_tries": REJECTION_MAX_TRIES,
        "shapes": SHAPES,
        "distance_minimum": DIST_MIN,
        "dome_fraction": DOME_FRAC,
    }


if __name__ == "__main__":
    import time

    import gymnasium as gym
    import PyFlyt.gym_envs

    env = gym.make("PyFlyt/QuadX-Waypoints-v4", render_mode="human")

    while True:
        env.reset(seed=43)
        u = env.unwrapped
        p = u.env
        ids = build_scene(
            p,
            u.np_random,
            u.waypoints.targets,
            u.start_pos[0],
            u.flight_dome_size,
            0.1,
            p.planeId,
        )
        p.register_all_new_bodies()

        for _ in range(600):
            p.step()
            time.sleep(1 / 50)  # adjust denominator lower for longer time to view
