from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def activation(name: str) -> nn.Module:
    choices = {"mish": nn.Mish, "tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}
    try:
        return choices[name.lower()]()
    except KeyError as error:
        raise ValueError(f"unknown activation {name}") from error


def mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation_name: str = "mish",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(previous, width), activation(activation_name)])
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class ValueCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        activation_name: str = "mish",
    ):
        super().__init__()
        self.network = mlp(observation_dim, hidden_dims, 1, activation_name)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

