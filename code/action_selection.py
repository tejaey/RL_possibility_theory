import random
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class SelectActionMeta(ABC):
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


class single_dqn_eps_greedy(SelectActionMeta):
    def __init__(self, action_dim) -> None:
        super().__init__(action_dim)

    def call(self, state, online_qnet) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = online_qnet(state_t)
        return q_values.argmax().item()


class ensemble_action_weighted_sum(SelectActionMeta):
    def __init__(self, action_dim) -> None:
        super().__init__(action_dim)

    def call(self, state, online_qnet) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values_list = [qnet(state_t) for qnet in online_qnet.qnets]
        poss = np.array(online_qnet.possibility)
        poss_normalized = poss / poss.sum() + 1e-8
        q_values_tensor = torch.stack(q_values_list)
        weighted_q_values = (q_values_tensor.squeeze(1) * poss_normalized[:, None]).sum(
            dim=0
        )
        return int(weighted_q_values.argmax().item())


class ensemble_action_majority_voting(SelectActionMeta):
    def __init__(self, action_dim) -> None:
        super().__init__(action_dim)

    def call(self, state, online_qnet):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        votes = np.zeros(self.action_dim)
        for i, qnet in enumerate(online_qnet.qnets):
            q_values = qnet(state_t)
            best_action = q_values.argmax().item()
            votes[best_action] += online_qnet.possibility[i]
        return int(np.argmax(votes))
