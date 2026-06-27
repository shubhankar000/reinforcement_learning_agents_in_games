"""
File containing value iteration function, thats only possible on gridworlds. This function computes the optimal Q and V values from the grid world transition probability matrix. Used to compare the algo against V* and Q*.
"""

import gymnasium as gym
import numpy as np


def value_iteration(P: dict, gamma: float, theta=1e-10):
    n_states = len(P)
    n_actions = len(P[0])

    V = np.zeros((n_states,))

    while True:
        delta = 0
        V_new = np.zeros((n_states,))
        for s in range(n_states):
            Q = np.zeros((n_actions,))
            for a in range(n_actions):
                q = 0.0
                for p, s_next, r, terminal in P[s][a]:
                    q += p * (r + gamma * V[s_next] * int(not terminal))
                Q[a] = q

            V_new[s] = np.max(Q)
            delta = np.maximum(delta, np.abs(V_new[s] - V[s]))

        V = V_new.copy()

        if delta < theta:
            break

    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            q = 0.0
            for p, s_next, r, terminal in P[s][a]:
                q += p * (r + gamma * V[s_next] * int(not terminal))
            Q[s, a] = q

    return Q, V


if __name__ == "__main__":
    env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        # render_mode="human",
        reward_schedule=(1, 0, 0),  # [reward_goal,  reward_hole, reward_frozen]
        is_slippery=True,
    )
    Q, V = value_iteration(env.unwrapped.P, 0.95)

    np.savez("./src/tabular/q_v_star_frozenlake_slippery", q_star=Q, v_star=V)

