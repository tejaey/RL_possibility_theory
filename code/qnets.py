import torch, random
from torch import optim
import numpy as np
import torch.nn as nn


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
