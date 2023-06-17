"""Implementation of DQN model as a PyTorch nn.Module."""

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions, lr=1e-4):
        super(DQN, self).__init__()
        assert len(obs_shape) == 1, "only 1D obs spaces supported"
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.net = nn.Sequential(
            torch.nn.Linear(obs_shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )
        self.opt = torch.optim.AdamW(self.net.parameters(), lr)

    def forward(self, x):
        return self.net(x)
