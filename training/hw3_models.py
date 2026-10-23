"""Architectures for the HW3 controlled design experiment."""

from __future__ import annotations

import torch
from torch import nn

from model import ACTION_NAMES, IMAGE_FEATURES, INPUT_FEATURES


class HW2Baseline(nn.Module):
    """The original early-fusion MLP: concatenate every input immediately."""

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


class GoalResidualBase(nn.Module):
    """Shared two-branch encoder and residual correction used by both HW3 variants."""

    def __init__(self) -> None:
        super().__init__()
        self.visual_encoder = nn.Sequential(nn.Linear(IMAGE_FEATURES, 64), nn.ReLU())
        self.goal_encoder = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.correction = nn.Sequential(nn.Linear(80, 64), nn.Tanh())
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(ACTION_NAMES)),
        )

    def branches(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual = self.visual_encoder(inputs[:, :IMAGE_FEATURES])
        goal = self.goal_encoder(inputs[:, IMAGE_FEATURES:])
        correction = self.correction(torch.cat([visual, goal], dim=1))
        return visual, goal, correction


class GoalGatedResidualMLP(GoalResidualBase):
    """Use goal information to decide how much residual correction each feature receives."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(16, 64), nn.Sigmoid())

    def forward(
        self, inputs: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        visual, goal, correction = self.branches(inputs)
        gate = self.gate(goal)
        logits = self.classifier(visual + gate * correction)
        return (logits, gate) if return_gate else logits


class UngatedResidualMLP(GoalResidualBase):
    """Ablation: always add the complete correction instead of learning a gate."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        visual, _, correction = self.branches(inputs)
        return self.classifier(visual + correction)

