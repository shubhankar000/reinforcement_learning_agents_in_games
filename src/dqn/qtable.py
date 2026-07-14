from stable_baselines3 import DQN
import torch as th


def create_qtable(model: DQN, n_states):
    obs = th.arange(n_states, device=model.device)
    with th.no_grad():
        q = model.q_net(obs)

    return q.cpu().numpy()
