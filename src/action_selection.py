from abc import ABC, abstractmethod
import random
import numpy as np
import torch
from config import DEVICE


class Qnet_SelectActionMeta(ABC):
    def __init__(self, action_dim) -> None:
        self.action_dim = action_dim

    def random_action(self):
        return random.randint(0, self.action_dim - 1)

    def __call__(self, state, online_qnet, eps) -> int:
        if eps > 0 and random.random() < eps:
            return self.random_action()
        else:
            return self.call(state=state, online_qnet=online_qnet)

    @abstractmethod
    def call(self, state, online_qnet) -> int:
        pass


class single_dqn_eps_greedy(Qnet_SelectActionMeta):
    def __init__(self, action_dim) -> None:
        super().__init__(action_dim)

    def call(self, state, online_qnet) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = online_qnet(state_t)
        return q_values.argmax().item()


class mean_logvar_std_greedy(Qnet_SelectActionMeta):
    def __init__(self, action_dim, beta) -> None:
        super().__init__(action_dim)
        self.beta = beta

    def call(self, state, online_qnet) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mean, _ = online_qnet(state_t)
        return mean.argmax().item()


class mean_logvar_actionselection(Qnet_SelectActionMeta):
    def __init__(self, action_dim, beta) -> None:
        super().__init__(action_dim)
        self.beta = beta

    def call(self, state, online_qnet) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mean, logvar = online_qnet(state_t)
        q_values = mean + self.beta * logvar
        return q_values.argmax().item()


class mean_logvar_maxexpected(Qnet_SelectActionMeta):
    def __init__(self, action_dim) -> None:
        super().__init__(action_dim)

    def call(self, state, online_qnet) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mean, logvar = online_qnet(state_t)
        var = torch.exp(logvar)
        optimistic_q = (mean + torch.sqrt(mean**2 + 4 * var)) / 2
        return optimistic_q.argmax().item()


class ensemble_action_weighted_sum(Qnet_SelectActionMeta):
    def __init__(self, action_dim) -> None:
        super().__init__(action_dim)

    def call(self, state, online_qnet) -> int:
        with torch.no_grad():
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            )
            q_values = torch.stack(
                [qnet(state_tensor) for qnet in online_qnet.qnets], dim=0
            ).squeeze(1)

            possibilities = np.array(online_qnet.possibility)
            possibilities_normalized = possibilities / (possibilities.sum() + 1e-8)
            poss_tensor = torch.tensor(
                possibilities_normalized, dtype=torch.float32, device=q_values.device
            )

            weighted_q = (q_values * poss_tensor[:, None]).sum(dim=0)
            action = int(torch.argmax(weighted_q).item())
        return action


class ensemble_action_majority_voting(Qnet_SelectActionMeta):
    def __init__(self, action_dim) -> None:
        super().__init__(action_dim)

    def call(self, state, online_qnet):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        votes = np.zeros(self.action_dim)
        for i, qnet in enumerate(online_qnet.qnets):
            q_values = qnet(state_t)
            best_action = q_values.argmax().item()
            votes[best_action] += online_qnet.possibility[i]
        return int(np.argmax(votes))


def select_action_eps_greedy_meanvarQnet(state, q_network, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mean, logvar = q_network(state_t)
        return mean.squeeze().argmax().item()


def selection_action_logvar(state, q_network, beta=1):
    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mean, logvar = q_network(state_t)
    q_values = mean + beta * logvar
    return q_values.argmax().item()


def select_action_uncertainty_aware(state, q_network, beta=1):
    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mean, logvar = q_network(state_t)
        variance = torch.exp(logvar)
    q_values = mean + beta * torch.sqrt(variance)
    return q_values.argmax().item()


class AC_SelectAction:
    def __init__(self, action_dim) -> None:
        self.action_dim = action_dim

    def __call__(self, state, online_actor, eps=0.1):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action = online_actor(state_t).cpu().data.numpy()[0]
        action += np.random.normal(0, eps, size=self.action_dim)
        return action
