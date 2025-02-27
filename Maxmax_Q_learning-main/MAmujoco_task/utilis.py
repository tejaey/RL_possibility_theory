import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, if_norm = False, hidden_dim = 64):
        super(MLPNetwork, self).__init__()
        # Normalize the input data
        self.if_norm = if_norm
        self.in_fn = nn.BatchNorm1d(input_dim)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        if self.if_norm:
            h1 = F.relu(self.fc1(self.in_fn(x)))
        else:
            h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out


