"""Heterogeneous GraphSAGE encoder for InnerSight UEBA.

Architecture
------------
1. Per-node-type input projection  (nn.Linear, in_dim → hidden_dim)
2. N × HeteroConv(SAGEConv) message-passing layers
3. ReLU + dropout between every layer

The model returns a dict mapping node_type → embedding tensor so that the
caller can extract whichever node types it needs (typically 'user' for the
anomaly-scoring head).

Edge cases handled
------------------
- Node types with zero nodes (e.g. 'pc' absent in a NeighborLoader batch):
  skipped during input projection; their edge types are filtered out so
  HeteroConv never receives empty source/destination feature tensors.
- Edge types with zero edges but non-empty endpoints: HeteroConv propagates
  zero messages, which is correct — the node just keeps its own signal.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.nn import Linear

# Allow running as __main__ from backend/ or the repo root.
_FILE_DIR = os.path.abspath(os.path.dirname(__file__))
_BACKEND  = os.path.abspath(os.path.join(_FILE_DIR, '..'))
_PKG_ROOT = os.path.abspath(os.path.join(_BACKEND, '..', '..'))
for _p in (_PKG_ROOT, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from innersight.backend.models.graph_schema import NODE_FEATURE_DIMS


class HeteroGraphSAGE(nn.Module):
    """Heterogeneous GraphSAGE encoder.

    Parameters
    ----------
    metadata:
        ``HeteroData.metadata()`` → ``(node_types, edge_types)``.
    hidden_dim:
        Output dimensionality for every node type at every layer.
    num_layers:
        Number of HeteroConv message-passing layers (= neighbourhood hops).
    dropout:
        Dropout probability applied after each layer's ReLU.
    """

    def __init__(
        self,
        metadata: tuple,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout    = dropout

        node_types, edge_types = metadata

        # ── Input projections ─────────────────────────────────────────────────
        # One Linear per node type that maps raw features → hidden_dim.
        # PyG's Linear is used (supports lazy init, but we pass explicit dims).
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            in_dim = NODE_FEATURE_DIMS[node_type]
            self.input_projections[node_type] = Linear(in_dim, hidden_dim)

        # ── Message-passing layers ────────────────────────────────────────────
        # Each layer is a HeteroConv wrapping one SAGEConv per edge type.
        # (-1, -1) means SAGEConv infers input dims lazily on first call, which
        # works correctly here because we always pass hidden_dim-projected features.
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_type: SAGEConv((-1, -1), hidden_dim)
                for edge_type in edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
    ) -> dict:
        """Run the full encoder.

        Parameters
        ----------
        x_dict:
            ``{node_type: FloatTensor(N, in_dim)}`` — raw node features.
        edge_index_dict:
            ``{edge_type: LongTensor(2, E)}`` — edge connectivity.

        Returns
        -------
        dict
            ``{node_type: FloatTensor(N, hidden_dim)}`` for every node type
            that had at least one node.  Node types with zero nodes are absent.
        """
        # Step a — input projection (skip node types with no nodes in this batch)
        x_dict = {
            nt: self.input_projections[nt](x)
            for nt, x in x_dict.items()
            if nt in self.input_projections and x.shape[0] > 0
        }

        # Filter edge_index_dict to only edge types where both endpoints are present.
        # This avoids HeteroConv trying to look up features for absent node types.
        active = set(x_dict.keys())
        edge_index_dict = {
            et: ei
            for et, ei in edge_index_dict.items()
            if et[0] in active and et[2] in active
        }

        # Step b — message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v)                                         for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        return x_dict

    def get_user_embeddings(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """Convenience wrapper: returns only the 'user' embedding tensor."""
        return self.forward(x_dict, edge_index_dict)['user']


class InsiderThreatGNN(nn.Module):
    """Full insider-threat detector: GraphSAGE backbone + MLP classification head.

    The backbone encodes every node type into a shared ``hidden_dim``-dimensional
    space via heterogeneous message passing.  The head then maps each user's
    embedding to a single logit for binary threat classification.

    Parameters
    ----------
    metadata:
        ``HeteroData.metadata()`` tuple passed straight to ``HeteroGraphSAGE``.
    hidden_dim:
        Backbone output width and head input width.
    num_layers:
        Number of GraphSAGE message-passing layers.
    dropout:
        Dropout probability used in both the backbone and the head.
    head_layers:
        Hidden layer widths for the MLP head.  ``[128, 64]`` adds two hidden
        layers before the final 1-dim output; ``[]`` means a single linear map.
    """

    def __init__(
        self,
        metadata: tuple,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        head_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        if head_layers is None:
            head_layers = [128, 64]

        self.backbone = HeteroGraphSAGE(metadata, hidden_dim, num_layers, dropout)

        # Build the MLP head generically from the head_layers list.
        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for out_dim in head_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """Return raw logits for every user node.

        Returns
        -------
        torch.Tensor
            Shape ``(num_users, 1)``.  No sigmoid applied — use
            ``BCEWithLogitsLoss`` during training.
        """
        embeddings_dict = self.backbone(x_dict, edge_index_dict)
        user_embeds = embeddings_dict['user']   # (num_users, hidden_dim)
        return self.head(user_embeds)           # (num_users, 1)

    def get_embeddings(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """Return user node embeddings for downstream use (e.g. Qdrant in Phase 7).

        Returns
        -------
        torch.Tensor
            Shape ``(num_users, hidden_dim)``.
        """
        return self.backbone(x_dict, edge_index_dict)['user']


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import logging
    from torch.nn.parameter import UninitializedParameter

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    DIVIDER = '=' * 60

    model_dir  = os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints')
    graph_path = os.path.join(model_dir, 'graphs', 'train_graph.pt')

    print(f'\nLoading training graph from: {graph_path}')
    graph = torch.load(graph_path, weights_only=False)

    print('\nGraph contents:')
    for nt in graph.node_types:
        print(f'  {nt:<8}: {graph[nt].x.shape[0]:>8,} nodes  feat_dim={graph[nt].x.shape[1]}')
    for et in graph.edge_types:
        print(f'  {str(et)}: {graph[et].edge_index.shape[1]:,} edges')

    # ── InsiderThreatGNN ──────────────────────────────────────────────────────
    print(f'\n{DIVIDER}')
    print('InsiderThreatGNN — backbone + classification head')
    print(DIVIDER)

    model = InsiderThreatGNN(
        metadata=graph.metadata(),
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        head_layers=[128, 64],
    )

    # Forward before param-count: SAGEConv(-1,-1) weights are lazy.
    model.eval()
    with torch.no_grad():
        logits = model(graph.x_dict, graph.edge_index_dict)

    print('\nModel architecture:')
    print(model)

    # Count only materialized parameters (lazy SAGEConv for absent edge types
    # remain uninitialized until those node types appear in a batch).
    init_p = [p for p in model.parameters() if not isinstance(p, UninitializedParameter)]
    uninit  = sum(1 for p in model.parameters() if isinstance(p, UninitializedParameter))
    total   = sum(p.numel() for p in init_p)
    print(f'\nTotal parameters (initialized): {total:,}')
    if uninit:
        print(f'  ({uninit} lazy tensors not yet materialized'
              f' — edge types whose node type has 0 nodes in every batch)')

    # ── Output verification ───────────────────────────────────────────────────
    num_users = graph['user'].x.shape[0]
    print(f'\nLogits shape : {tuple(logits.shape)}  (expected: ({num_users}, 1))')
    assert logits.shape == (num_users, 1), f"Shape mismatch: {logits.shape}"
    print('Shape assertion passed.')

    print(f'\nLogits (first 5 users):\n  {logits[:5].squeeze().tolist()}')

    # ── Embeddings check ─────────────────────────────────────────────────────
    with torch.no_grad():
        embeds = model.get_embeddings(graph.x_dict, graph.edge_index_dict)
    print(f'\nget_embeddings() shape: {tuple(embeds.shape)}  (for Qdrant in Phase 7)')

    # ── HeteroGraphSAGE backbone alone ───────────────────────────────────────
    print(f'\n{DIVIDER}')
    print('HeteroGraphSAGE backbone only (no head)')
    print(DIVIDER)

    backbone = HeteroGraphSAGE(graph.metadata(), hidden_dim=128, num_layers=2, dropout=0.3)
    backbone.eval()
    with torch.no_grad():
        enc = backbone(graph.x_dict, graph.edge_index_dict)

    print('\nBackbone output shapes per node type:')
    for nt, emb in enc.items():
        print(f'  {nt:<8}: {tuple(emb.shape)}')
    absent = set(graph.node_types) - set(enc.keys())
    if absent:
        print(f'  Skipped (0 nodes): {sorted(absent)}')
