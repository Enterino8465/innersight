"""Module 2 — temporal pattern encoder (dilated causal CNN + attention pooling).

Encodes a 28-day window of per-user z-scored deviations, shape ``(batch, 18, T)``,
into a fixed ``(batch, 128)`` embedding. A stack of four dilated **causal**
convolutions grows the receptive field (3 → 7 → 15 → 23 days) without ever
letting day ``t`` see the future, and an attention pool collapses the time axis
into a single embedding while exposing which days the model focused on.

Design choices (council-reviewed):
    * Causal padding (left-pad only) so the encoding is strictly anti-leakage.
    * LayerNorm, not BatchNorm: with a ~0.4% positive rate most batches are
      all-negative, so batch statistics would be dominated by normal users and
      distort the rare positives. LayerNorm normalises per sample.
    * Residual connections (1×1 conv projection on channel change) for stable
      gradients through the depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConvBlock(nn.Module):
    """One dilated causal conv block: pad → conv → LayerNorm → ReLU → dropout → residual."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float) -> None:
        super().__init__()
        # Left padding only makes the conv causal: output t sees inputs ≤ t.
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        # Match channels on the residual path when they differ.
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_channels, T)
        res = self.residual(x)                  # shape: (batch, out_channels, T)
        h = F.pad(x, (self.pad, 0))             # left-pad only → shape: (batch, in_channels, T + pad)
        h = self.conv(h)                        # shape: (batch, out_channels, T)
        # LayerNorm over channels → put channels last, then restore.
        h = h.transpose(1, 2)                   # shape: (batch, T, out_channels)
        h = self.norm(h)                        # shape: (batch, T, out_channels)
        h = h.transpose(1, 2)                   # shape: (batch, out_channels, T)
        h = F.relu(h)
        h = self.dropout(h)
        return h + res                          # shape: (batch, out_channels, T)


class TemporalPatternEncoder(nn.Module):
    """Dilated causal CNN with attention pooling over the time dimension.

    Args:
        in_channels: Number of z-scored input features (channels).
        hidden: Hidden channels in layers 1-3.
        out_dim: Output embedding dimension (also layer 4's output channels).
        dropout: Dropout probability inside each conv block.
        kernel_size: Convolution kernel size (shared by all layers).
    """

    def __init__(
        self,
        in_channels: int = 18,
        hidden: int = 64,
        out_dim: int = 128,
        dropout: float = 0.3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        # Dilations 1, 2, 4, 8 → receptive field 3, 7, 15, 23 days.
        self.blocks = nn.ModuleList([
            _CausalConvBlock(in_channels, hidden, kernel_size, dilation=1, dropout=dropout),
            _CausalConvBlock(hidden, hidden, kernel_size, dilation=2, dropout=dropout),
            _CausalConvBlock(hidden, hidden, kernel_size, dilation=4, dropout=dropout),
            _CausalConvBlock(hidden, out_dim, kernel_size, dilation=8, dropout=dropout),
        ])
        # Attention scorer: one scalar score per time step.
        self.attention = nn.Linear(out_dim, 1)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conv stack and return time-major features."""
        # x shape: (batch, in_channels, T)
        h = x
        for block in self.blocks:
            h = block(h)                        # shape: (batch, out_dim, T) after the last block
        return h.transpose(1, 2)                # shape: (batch, T, out_dim)

    def _attention_weights(self, h_time: torch.Tensor) -> torch.Tensor:
        """Softmax attention weights over time from time-major features."""
        # h_time shape: (batch, T, out_dim)
        scores = self.attention(h_time).squeeze(-1)   # shape: (batch, T)
        return F.softmax(scores, dim=1)               # shape: (batch, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a deviation window into a temporal embedding.

        Args:
            x: Channels-first window, shape ``(batch, in_channels, T)``.

        Returns:
            Temporal embedding, shape ``(batch, out_dim)``.
        """
        h_time = self._backbone(x)                          # shape: (batch, T, out_dim)
        attn = self._attention_weights(h_time)              # shape: (batch, T)
        # Attention-weighted average over the time dimension.
        return torch.einsum("bt,btd->bd", attn, h_time)     # shape: (batch, out_dim)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return the per-day attention weights for visualisation.

        Args:
            x: Channels-first window, shape ``(batch, in_channels, T)``.

        Returns:
            Attention weights over time, shape ``(batch, T)``; each row sums to 1.
        """
        return self._attention_weights(self._backbone(x))   # shape: (batch, T)
