"""Module 4 — the complete InsiderThreat fusion model.

Chains all four modules into one network:

    Module 1 (per-user deviations, precomputed) →
    Module 2 (TemporalPatternEncoder: 28-day window → 128-d embedding) →
    Module 3 (GraphContextEncoder: neighbourhood context → 128-d embedding) →
    Module 4 (fuse temporal + graph + static context → classification head).

The temporal embedding is injected as the graph's user node features (the graph
builder seeds those with placeholder zeros), so the graph stage refines the
temporal representation with relational context. Static, slow-moving signals —
OCEAN personality scores, role and department — bypass the convolutions (a
constant-over-time feature would get zero gradient through a temporal conv) and
join only at the fusion step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from innersight.models.graph_encoder import GraphContextEncoder
from innersight.models.temporal_encoder import TemporalPatternEncoder

_OCEAN_DIM = 5
_ROLE_EMB_DIM = 8
_DEPT_EMB_DIM = 3


class InsiderThreatDetector(nn.Module):
    """Full temporal + graph + static-context fusion model for insider detection.

    Args:
        metadata: ``HeteroData.metadata()`` → ``(node_types, edge_types)`` for the graph encoder.
        num_roles: Size of the role embedding table.
        num_depts: Size of the department embedding table.
        temporal_config: Overrides forwarded to :class:`TemporalPatternEncoder`.
        graph_config: Overrides forwarded to :class:`GraphContextEncoder`.
        head_dropout: Dropout used in the classification head.
    """

    def __init__(
        self,
        metadata,
        num_roles: int = 50,
        num_depts: int = 15,
        temporal_config: dict | None = None,
        graph_config: dict | None = None,
        head_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        tcfg = temporal_config or {}
        gcfg = graph_config or {}

        self.temporal = TemporalPatternEncoder(
            in_channels=tcfg.get("in_channels", 18),
            hidden=tcfg.get("hidden", 64),
            out_dim=tcfg.get("out_dim", 128),
            dropout=tcfg.get("dropout", 0.3),
            kernel_size=tcfg.get("kernel_size", 3),
        )
        self.graph = GraphContextEncoder(
            metadata=metadata,
            hidden_dim=gcfg.get("hidden_dim", 128),
            num_layers=gcfg.get("num_layers", 2),
            heads=gcfg.get("heads", 4),
            dropout=gcfg.get("dropout", 0.3),
        )
        self.role_emb = nn.Embedding(num_roles, _ROLE_EMB_DIM)
        self.dept_emb = nn.Embedding(num_depts, _DEPT_EMB_DIM)

        self._temporal_dim = tcfg.get("out_dim", 128)
        self._graph_dim = gcfg.get("hidden_dim", 128)
        # Fused = temporal (128) + graph (128) + OCEAN (5) + role (8) + dept (3) = 272.
        self.fused_dim = (
            self._temporal_dim + self._graph_dim + _OCEAN_DIM + _ROLE_EMB_DIM + _DEPT_EMB_DIM
        )
        self.head = nn.Sequential(
            nn.Linear(self.fused_dim, 128), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(64, 1),
        )

    def _fuse(self, deviation_sequences, x_dict, edge_index_dict, edge_attr_dict,
              ocean, roles, depts) -> torch.Tensor:
        """Compute the fused (N, 272) representation shared by forward/get_embeddings."""
        temporal_emb = self.temporal(deviation_sequences)        # shape: (N, 128)

        # Overwrite the graph's placeholder user features with the temporal embeddings.
        graph_out = self.graph({**x_dict, "user": temporal_emb}, edge_index_dict, edge_attr_dict)
        graph_emb = graph_out.get("user")                        # shape: (N, 128)
        if graph_emb is None or graph_emb.shape[0] != temporal_emb.shape[0]:
            # Users absent from the graph fall back to temporal-only (zero graph part).
            graph_emb = torch.zeros_like(temporal_emb)

        static = torch.cat([
            ocean,                                               # shape: (N, 5)
            self.role_emb(roles),                                # shape: (N, 8)
            self.dept_emb(depts),                                # shape: (N, 3)
        ], dim=1)                                                # shape: (N, 16)

        return torch.cat([temporal_emb, graph_emb, static], dim=1)  # shape: (N, 272)

    def forward(self, deviation_sequences, x_dict, edge_index_dict, edge_attr_dict,
                ocean, roles, depts) -> torch.Tensor:
        """Score each user-window.

        Args:
            deviation_sequences: ``(N, 18, 28)`` z-scored deviation windows.
            x_dict / edge_index_dict / edge_attr_dict: windowed HeteroData parts
                (``x_dict['user']`` is overwritten with the temporal embeddings).
            ocean: ``(N, 5)`` normalized OCEAN personality scores.
            roles: ``(N,)`` integer role ids.
            depts: ``(N,)`` integer department ids.

        Returns:
            ``(N, 1)`` logits.
        """
        fused = self._fuse(deviation_sequences, x_dict, edge_index_dict, edge_attr_dict,
                           ocean, roles, depts)
        return self.head(fused)                                  # shape: (N, 1)

    def get_embeddings(self, deviation_sequences, x_dict, edge_index_dict, edge_attr_dict,
                       ocean, roles, depts) -> torch.Tensor:
        """Return the fused ``(N, 272)`` vector before the head (stored in Qdrant)."""
        return self._fuse(deviation_sequences, x_dict, edge_index_dict, edge_attr_dict,
                          ocean, roles, depts)

    def get_temporal_attention(self, deviation_sequences) -> torch.Tensor:
        """Per-day temporal attention weights ``(N, T)`` for the timeline view."""
        return self.temporal.get_attention_weights(deviation_sequences)
