import torch
from typing import Literal
import numpy as np
import torch.nn as nn

from config import DEVICE


class SimpleDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class MeanVarianceQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.logvar_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.shared(state)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar


class EnsembleDQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_ensemble=5, hidden_dim=128):
        super().__init__()
        self.num_ensemble = num_ensemble
        self.qnets = nn.ModuleList(
            [SimpleDQN(state_dim, action_dim, hidden_dim) for _ in range(num_ensemble)]
        )
        self.possibility = [1.0 for _ in range(num_ensemble)]

    def forward(self, state):
        return [qnet(state) for qnet in self.qnets]


class Actor(SimpleDQN):
    pass


class RewardModel(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + state_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, next_state):
        return self.net(torch.cat([state, next_state], dim=1))


class SimpleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(SimpleCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


class SimpleDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class EnsembleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_ensemble=5, hidden_dim=128):
        super().__init__()
        self.num_ensemble = num_ensemble
        self.critic = nn.ModuleList(
            [
                SimpleCritic(
                    state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
                )
                for _ in range(num_ensemble)
            ]
        )
        self.possibility = [1.0 for _ in range(num_ensemble)]

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return [critic(x) for critic in self.critics]

    def aggregated_q(
        self,
        state,
        action,
        method: Literal[
            "weighted_average", "min", "max", "max_possibility"
        ] = "weighted_average",
    ):
        q_list = self.forward(state, action)
        # Compute each critic's Q-value, then aggregate them. q_list = self.forward(state, action)  # each: [batch, 1]
        q_tensor = torch.stack(q_list, dim=0)  # shape: [num_ensemble, batch, 1]
        match method:
            case "min":
                agg_q, _ = torch.min(q_tensor, dim=0)
            case "max":
                agg_q, _ = torch.max(q_tensor, dim=0)
            case "weighted_average":
                poss = torch.tensor(
                    self.possibilities, device=q_tensor.device, dtype=q_tensor.dtype
                )
                poss = poss / poss.sum()
                poss = poss.view(self.num_ensemble, 1, 1)  # reshape for broadcasting
                agg_q = (poss * q_tensor).sum(dim=0)
            case "max_possibility":
                max_index, _ = max(enumerate(self.possibility), key=lambda x: x[1])
                agg_q = q_list[max_index]
        # elif method == "possibility_weighted_max":
        #     poss = torch.tensor(
        #         self.possibilities, device=q_tensor.device, dtype=q_tensor.dtype
        #     )
        #     scaled_q = q_tensor * poss.view(self.num_ensemble, 1, 1)
        #     agg_q, _ = torch.max(scaled_q, dim=0)
        return agg_q


class QuantileModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        return state + self.net(torch.cat([state, action], dim=1))


class EnsembleQuantileModels(nn.Module):
    def __init__(self, state_dim, action_dim, quantiles, hidden_dim=128):
        """
        Each quantile gets its own independent quantile model.
        """
        super().__init__()
        self.quantiles = quantiles
        self.models = nn.ModuleList(
            [QuantileModel(state_dim, action_dim, hidden_dim) for _ in quantiles]
        )

    def forward(self, state, action):
        outputs = [model(state, action) for model in self.models]
        return outputs


class StdPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Predicts the log-standard-deviation along each dimension of the state.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()
        )
        self.log_std_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        feat = self.fc(x)
        log_std = self.log_std_head(feat)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-20, max=2)
        return log_std


class MeanStdPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Predicts both the next-state mean and diagonal standard deviation
        for a Gaussian transition model given (state, action).
        """
        super().__init__()
        # shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.log_std_head = nn.Linear(hidden_dim, state_dim)

        nn.init.constant_(self.log_std_head.bias, -1.0)

    def forward(self, state, action):
        """
        :param state:   tensor of shape (B, state_dim)
        :param action:  tensor of shape (B, action_dim)
        :returns:
          - mean: tensor of shape (B, state_dim)
          - std:  tensor of shape (B, state_dim), each > 0
        """
        x = torch.cat([state, action], dim=-1)  # (B, state_dim + action_dim)
        h = self.shared(x)  # (B, hidden_dim)

        mean = self.mean_head(h)  # (B, state_dim)
        log_std = self.log_std_head(h)  # (B, state_dim)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        return mean, std
