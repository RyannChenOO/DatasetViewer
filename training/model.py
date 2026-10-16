"""Small, explainable MLP used for next-action prediction."""

from __future__ import annotations

import torch
from torch import nn


IMAGE_SIZE = 8
IMAGE_FEATURES = IMAGE_SIZE * IMAGE_SIZE * 3
INPUT_FEATURES = IMAGE_FEATURES + 2
ACTION_NAMES = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]


class NavigationMLP(nn.Module):
    """RGB pixels + goal distance + goal angle -> four action logits."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(ACTION_NAMES)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LinearBaseline(nn.Module):
    """No-hidden-layer benchmark required for comparison."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(INPUT_FEATURES, len(ACTION_NAMES))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)
