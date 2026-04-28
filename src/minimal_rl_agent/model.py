"""Policy network: maps state vectors to one predicted value per action (Q-values)."""

from __future__ import annotations

import torch
from torch import nn


class DQN(nn.Module):
    """
    A small MLP: state → scores for each discrete action.

    The outputs are not probabilities. They are Q-values: higher means the model
    expects more *discounted* future reward if it takes that action from this state.
    """

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return shape (batch, action_size)."""
        return self.net(state)
