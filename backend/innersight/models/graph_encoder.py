"""Module 3 — heterogeneous graph context encoder (GATv2 + edge features).

Consumes a windowed :class:`HeteroData` (see ``build_windowed_graph``) and
produces a context-aware embedding per node. User node features are the injected
temporal embeddings (Module 2); pc/url/file features are behavioural aggregates.
Edge features (counts, after-hours fractions, novelty flags) are projected to the
hidden size and fed into GATv2 attention so the model can weight *how* a user
connects to an entity, not just whether.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

from innersight.models.graph_schema import (
    ALL_EDGE_TYPES,
    ALL_REV_EDGE_TYPES,
    WINDOWED_EDGE_FEATURE_DIMS,
    WINDOWED_NODE_FEATURE_DIMS,
)


def _edge_key(edge_type: tuple[str, str, str]) -> str:
    """ModuleDict-safe string key for an edge-type tuple."""
    return "__".join(edge_type)


def _edge_dim_lookup() -> dict[tuple[str, str, str], int]:
    """Edge-type → feature dim for forward edges and their (same-dim) reverses."""
    dims = dict(WINDOWED_EDGE_FEATURE_DIMS)
    for forward, reverse in zip(ALL_EDGE_TYPES, ALL_REV_EDGE_TYPES):
        dims[reverse] = WINDOWED_EDGE_FEATURE_DIMS[forward]
    return dims


class GraphContextEncoder(nn.Module):
    """Heterogeneous GATv2 encoder that consumes node and edge features.

    Args:
        metadata: ``HeteroData.metadata()`` → ``(node_types, edge_types)``.
        hidden_dim: Hidden / output embedding size for every node type.
        num_layers: Number of HeteroConv(GATv2) message-passing layers.
        heads: Attention heads per GATv2Conv (output is concatenated back to
            ``hidden_dim``, so ``hidden_dim`` must be divisible by ``heads``).
        dropout: Dropout used inside attention and after each layer.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        node_dims = WINDOWED_NODE_FEATURE_DIMS
        edge_dims = _edge_dim_lookup()

        # Per-node-type input projection: raw features → hidden_dim.
        self.input_projections = nn.ModuleDict({
            nt: nn.Linear(node_dims.get(nt, hidden_dim), hidden_dim) for nt in node_types
        })
        # Per-edge-type projection: edge features → hidden_dim (for GATv2's edge_dim).
        self.edge_projections = nn.ModuleDict({
            _edge_key(et): nn.Linear(edge_dims.get(et, hidden_dim), hidden_dim) for et in edge_types
        })

        # num_layers × HeteroConv(GATv2Conv), with per-layer/per-node-type LayerNorm.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    et: GATv2Conv(
                        hidden_dim, hidden_dim // heads, heads=heads, concat=True,
                        edge_dim=hidden_dim, add_self_loops=False, dropout=dropout,
                    )
                    for et in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types}))

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Encode every node type, returning ``{node_type: (N, hidden_dim)}``.

        Args:
            x_dict: ``{node_type: (N, in_dim)}`` raw node features.
            edge_index_dict: ``{edge_type: (2, E)}`` connectivity.
            edge_attr_dict: ``{edge_type: (E, edge_dim)}`` raw edge features.

        Returns:
            ``{node_type: (N, hidden_dim)}`` for every node type with nodes.
        """
        # Project node features, skipping node types with zero nodes.
        h_dict = {
            nt: self.input_projections[nt](x)
            for nt, x in x_dict.items()
            if nt in self.input_projections and x.shape[0] > 0
        }
        active = set(h_dict.keys())

        # Keep only edges whose endpoints are both present this forward pass.
        edge_index_dict = {
            et: ei for et, ei in edge_index_dict.items() if et[0] in active and et[2] in active
        }
        # Project edge features to hidden_dim for the matching active edges.
        proj_edge_attr = {
            et: self.edge_projections[_edge_key(et)](edge_attr_dict[et])
            for et in edge_index_dict
            if et in edge_attr_dict and _edge_key(et) in self.edge_projections
        }

        for conv, norm in zip(self.convs, self.norms):
            h_in = h_dict
            out = conv(h_in, edge_index_dict, edge_attr_dict=proj_edge_attr)
            updated: dict[str, torch.Tensor] = {}
            for nt, val in out.items():
                residual = h_in.get(nt)
                if residual is not None and residual.shape == val.shape:
                    val = val + residual                       # residual connection
                val = norm[nt](val)                            # LayerNorm
                val = F.dropout(F.relu(val), p=self.dropout, training=self.training)
                updated[nt] = val
            # Carry forward node types that received no messages this layer.
            for nt, val in h_in.items():
                updated.setdefault(nt, val)
            h_dict = updated

        return h_dict
