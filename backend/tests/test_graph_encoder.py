"""Unit tests for the GATv2 GraphContextEncoder (Phase 5.2)."""

import torch

from innersight.models.graph_encoder import GraphContextEncoder
from innersight.models.graph_schema import (
    EDGE_HTTP,
    EDGE_LOGON,
    NODE_PC,
    NODE_URL,
    NODE_USER,
    REV_EDGE_HTTP,
    REV_EDGE_LOGON,
)
from torch_geometric.data import HeteroData

# Dense bipartite connectivity (every destination node has ≥2 neighbours) so that
# GATv2 attention is non-trivial — a single-neighbour softmax is constant 1.0 and
# would yield zero gradients for the attention and edge-feature projections.
_SRC = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
_DST = [0, 1, 1, 2, 2, 0, 0, 1, 1, 2]


def _synthetic_graph(n_url=3):
    """5 users, 3 PCs, n_url URLs; dense logon (dim 5) + http (dim 4) edges + reverses."""
    torch.manual_seed(0)
    g = HeteroData()
    g[NODE_USER].x = torch.randn(5, 128)
    g[NODE_PC].x = torch.randn(3, 8)
    g[NODE_URL].x = torch.randn(n_url, 8) if n_url else torch.zeros(0, 8)

    le = torch.tensor([_SRC, _DST])
    la = torch.randn(le.shape[1], 5)
    g[EDGE_LOGON].edge_index = le
    g[EDGE_LOGON].edge_attr = la
    g[REV_EDGE_LOGON].edge_index = le.flip(0)
    g[REV_EDGE_LOGON].edge_attr = la

    if n_url:
        he = torch.tensor([_SRC, _DST])
        ha = torch.randn(he.shape[1], 4)
        g[EDGE_HTTP].edge_index = he
        g[EDGE_HTTP].edge_attr = ha
        g[REV_EDGE_HTTP].edge_index = he.flip(0)
        g[REV_EDGE_HTTP].edge_attr = ha
    return g


def test_forward_user_output_shape():
    g = _synthetic_graph()
    enc = GraphContextEncoder(g.metadata())
    enc.eval()
    out = enc(g.x_dict, g.edge_index_dict, g.edge_attr_dict)
    assert NODE_USER in out
    assert out[NODE_USER].shape == (5, 128)


def test_edge_features_are_consumed():
    g = _synthetic_graph()
    enc = GraphContextEncoder(g.metadata())
    enc.eval()
    with torch.no_grad():
        out_real = enc(g.x_dict, g.edge_index_dict, g.edge_attr_dict)[NODE_USER]
        zeroed = {et: torch.zeros_like(v) for et, v in g.edge_attr_dict.items()}
        out_zero = enc(g.x_dict, g.edge_index_dict, zeroed)[NODE_USER]
    # If edge features were ignored, these would be identical.
    assert (out_real - out_zero).abs().max().item() > 1e-4


def test_handles_empty_node_type():
    g = _synthetic_graph(n_url=0)  # zero URL nodes
    enc = GraphContextEncoder(g.metadata())
    enc.eval()
    out = enc(g.x_dict, g.edge_index_dict, g.edge_attr_dict)
    assert out[NODE_USER].shape == (5, 128)


def test_gradient_flows_to_all_parameters():
    g = _synthetic_graph()
    enc = GraphContextEncoder(g.metadata())
    enc.train()
    out = enc(g.x_dict, g.edge_index_dict, g.edge_attr_dict)
    # Sum over ALL node outputs: in a 2-layer encoder read only at 'user', the
    # final layer's updates to other node types are dead-ends — summing every
    # output gives each parameter a gradient path, so this catches genuinely
    # broken layers/residuals rather than that expected read-out asymmetry.
    sum(v.sum() for v in out.values()).backward()
    for name, param in enc.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum().item() > 0.0, f"{name} has a zero gradient"


def test_parameter_count_in_expected_range():
    # GATv2 per edge type × 2 layers (each with lin_l + lin_r + lin_edge of 128×128)
    # is parameter-heavy: ~420K for this 4-edge-type graph (more than the rough
    # ~200K estimate). The band guards against accidental architecture changes.
    g = _synthetic_graph()
    n_params = sum(p.numel() for p in GraphContextEncoder(g.metadata()).parameters())
    assert 350_000 <= n_params <= 500_000, f"unexpected parameter count: {n_params}"
