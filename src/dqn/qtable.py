import torch as th


def create_qtable(model: th.nn.Module, n_states):
    obs = th.arange(n_states, device=model.device)
    with th.no_grad():
        q = model.q_net(obs)

    return q.cpu().numpy()
