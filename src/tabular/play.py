"""
play.py — play the tabular ToyText games yourself with the keyboard.

These are TURN-BASED envs, so we step exactly once per key press using an
event-driven pygame loop. (gymnasium.utils.play.play is built for real-time
envs — it steps every frame at a fixed fps — which is wrong for grid worlds.)

We render with render_mode="rgb_array" and blit the frames into OUR OWN pygame
window. We do NOT use render_mode="human": that would make the env open and
drive its own window, leaving us no clean hook to step on key-press.

Run:
    uv run python -m src.tabular.play FrozenLake-v1
    uv run python -m src.tabular.play CliffWalking-v1 slip
    uv run python -m src.tabular.play Taxi-v4
    uv run python -m src.tabular.play Taxi-v4 rainy

Controls: arrow keys move; Taxi also uses P=pickup, D=dropoff.
          R resets the episode, Esc / window-close quits.
"""

import sys

import gymnasium as gym
import numpy as np
import pygame

# key -> action id, matching each env's action encoding
KEYMAP = {
    "FrozenLake-v1": {  # 0 L, 1 D, 2 R, 3 U
        pygame.K_LEFT: 0,
        pygame.K_DOWN: 1,
        pygame.K_RIGHT: 2,
        pygame.K_UP: 3,
    },
    "CliffWalking-v1": {  # 0 U, 1 R, 2 D, 3 L
        pygame.K_UP: 0,
        pygame.K_RIGHT: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
    },
    "Taxi-v4": {  # 0 S, 1 N, 2 E, 3 W, 4 pickup, 5 dropoff
        pygame.K_DOWN: 0,
        pygame.K_UP: 1,
        pygame.K_RIGHT: 2,
        pygame.K_LEFT: 3,
        pygame.K_p: 4,
        pygame.K_d: 5,
    },
}

DEFAULT_KWARGS = {
    "FrozenLake-v1": {"map_name": "4x4", "is_slippery": False},
    "CliffWalking-v1": {},
    "Taxi-v4": {},
}

VARIANT_KEY = {  # which kwarg turns on stochastic dynamics
    "FrozenLake-v1": "is_slippery",
    "CliffWalking-v1": "is_slippery",
    "Taxi-v4": "is_rainy",
}

SCALE = 3  # window magnification — toy-text frames are small

INSTRUCTIONS = {
    "FrozenLake-v1": (
        "FROZEN LAKE — reach the goal (G, bottom-right) from the start (top-left).\n"
        "  Controls: arrow keys move one tile.\n"
        "  Falling in a hole (H) ends the episode. Slippery ice makes moves random.\n"
    ),
    "CliffWalking-v1": (
        "CLIFF WALKING — walk from start (bottom-left) to goal (bottom-right).\n"
        "  Controls: arrow keys move one tile.\n"
        "  Stepping onto the cliff (bottom edge) costs -100 and sends you to start.\n"
    ),
    "Taxi-v4": (
        "TAXI — drive to the passenger, pick them up, drop them at the destination.\n"
        "  Controls: arrow keys drive; P = pick up; D = drop off.\n"
        "  Pickup only works while ON the passenger's tile; dropoff only at the\n"
        "  destination with the passenger aboard. Doing either elsewhere costs -10\n"
        "  and nothing visibly changes. Colours: passenger and destination are the\n"
        "  coloured letters (R/G/Y/B); the taxi turns green once the passenger is in.\n"
    ),
}


def draw(screen, frame):
    # gym renders (H, W, 3); pygame's make_surface wants (W, H, 3)
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    surf = pygame.transform.scale(
        surf, (surf.get_width() * SCALE, surf.get_height() * SCALE)
    )
    screen.blit(surf, (0, 0))
    pygame.display.flip()


def play(env_id: str, **overrides):
    kwargs = {**DEFAULT_KWARGS[env_id], **overrides}
    keymap = KEYMAP[env_id]
    print("\n" + INSTRUCTIONS[env_id] + "  R = reset episode, Esc = quit.\n")
    env = gym.make(env_id, render_mode="rgb_array", **kwargs)

    obs, _ = env.reset()
    frame = env.render()

    pygame.init()
    screen = pygame.display.set_mode((frame.shape[1] * SCALE, frame.shape[0] * SCALE))
    pygame.display.set_caption(f"{env_id} — arrows move, R reset, Esc quit")
    draw(screen, frame)

    total_r, steps, done = 0.0, 0, False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    total_r, steps, done = 0.0, 0, False
                    draw(screen, env.render())
                elif not done and event.key in keymap:
                    obs, r, term, trunc, _ = env.step(keymap[event.key])
                    total_r += r
                    steps += 1
                    done = term or trunc
                    draw(screen, env.render())
                    if done:
                        print(
                            f"episode over — return {total_r:.2f} in {steps} steps"
                            f"{' (terminated)' if term else ' (truncated)'}."
                            " Press R to play again."
                        )
        pygame.time.wait(20)  # ~50 Hz event poll; cheap idle

    env.close()
    pygame.quit()


MENU = [
    ("FrozenLake-v1", "cross the frozen lake to the goal, don't fall in a hole"),
    ("CliffWalking-v1", "walk to the goal without stepping off the cliff"),
    ("Taxi-v4", "pick up the passenger and drop them at the destination"),
]


def choose_env() -> str:
    prompt = "\nChoose an environment:\n"
    for i, (env_id, help_text) in enumerate(MENU, start=1):
        prompt += f"  {i}) {env_id:<16} — {help_text}\n"
    prompt += "> "
    while True:
        choice = input(prompt).strip()
        if choice in ("1", "2", "3"):
            return MENU[int(choice) - 1][0]
        print("Please enter 1, 2 or 3.")


if __name__ == "__main__":
    # CLI arg skips the menu (e.g. `... play Taxi-v4 rainy`); otherwise prompt.
    env_id = sys.argv[1] if len(sys.argv) > 1 else choose_env()
    stochastic = len(sys.argv) > 2 and sys.argv[2] in ("slip", "rainy", "stochastic")
    extra = {VARIANT_KEY[env_id]: True} if stochastic else {}
    play(env_id, **extra)
