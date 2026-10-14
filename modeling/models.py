from __future__ import annotations

from typing import Final

import torch
from torch import nn

from config import DROPOUT, MODEL_A_HIDDEN, N_FEATURES, WINDOW_DAYS

MODEL_B_CHANNELS: Final[int] = 32
MODEL_B_KERNEL: Final[int] = 3
MODEL_B_DILATION: Final[int] = 2


class ModelA(nn.Module):

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden: int = MODEL_A_HIDDEN,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        layers: list[nn.Module] = [nn.Linear(n_features, hidden), nn.ReLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 18) -> (batch,) raw logits. Deliberately no sigmoid."""
        if x.dim() != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"ModelA expects (batch, {self.n_features}), got "
                f"{tuple(x.shape)}"
            )
        return self.net(x).squeeze(-1)


class ModelB(nn.Module):

    def __init__(
        self,
        n_features: int = N_FEATURES,
        days: int = WINDOW_DAYS,
        channels: int = MODEL_B_CHANNELS,
        kernel_size: int = MODEL_B_KERNEL,
        dilation: int = MODEL_B_DILATION,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd so the 'same' padding below is "
                f"exact, got {kernel_size}"
            )
        receptive_field = 1 + (kernel_size - 1) + dilation * (kernel_size - 1)
        if receptive_field > days:
            raise ValueError(
                f"receptive field of {receptive_field} days exceeds the "
                f"{days}-day window; the second layer would see padding only"
            )
        self.n_features = n_features
        self.days = days
        self.receptive_field = receptive_field

        self.conv1 = nn.Conv1d(
            n_features, channels, kernel_size,
            padding=(kernel_size - 1) // 2, dilation=1,
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=dilation * (kernel_size - 1) // 2, dilation=dilation,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() != 3 or x.shape[1] != self.n_features or x.shape[2] != self.days:
            raise ValueError(
                f"ModelB expects (batch, {self.n_features}, {self.days}) "
                f"— features as channels, days as length — got "
                f"{tuple(x.shape)}. A (batch, days, features) tensor would "
                f"convolve across the wrong axis without raising."
            )
        h = self.activation(self.conv1(x))
        h = self.activation(self.conv2(h))
        pooled = self.dropout(h.mean(dim=2))
        return self.head(pooled).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
