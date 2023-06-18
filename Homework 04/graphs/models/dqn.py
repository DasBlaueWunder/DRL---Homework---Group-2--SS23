"""Implementation of DQN model as a PyTorch nn.Module."""

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions, config):
        super(DQN, self).__init__()
        assert len(obs_shape) == 1, "only 1D obs spaces supported"
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        h1, h2, h3 = config.layer_units
        self.net = nn.Sequential(
            torch.nn.Linear(obs_shape[0], h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Linear(h2, h3),
            torch.nn.ReLU(),
            torch.nn.Linear(h3, n_actions),
        )
        self.opt = torch.optim.AdamW(
            self.net.parameters(), config.lr, weight_decay=config.lr_decay
        )

    def forward(self, x):
        return self.net(x)
