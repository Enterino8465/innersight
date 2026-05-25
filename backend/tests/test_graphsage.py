"""Tests for HeteroGraphSAGE, InsiderThreatGNN, and build_model factory."""

import pytest
import torch
from torch_geometric.data import HeteroData

from innersight.backend.models.graphsage import HeteroGraphSAGE, InsiderThreatGNN
from innersight.backend.models.factory import build_model
from innersight.backend.models.mlp import InsiderThreatMLP


# ── HeteroGraphSAGE ───────────────────────────────────────────────────────────

def test_hetero_graphsage_creation(small_hetero_graph):
    """HeteroGraphSAGE initialises without error using the fixture's metadata."""
    model = HeteroGraphSAGE(
        metadata=small_hetero_graph.metadata(),
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    )
    assert model.hidden_dim == 32
    assert len(model.convs) == 2
    # One input projection per node type in the fixture (user, pc, url)
    assert set(model.input_projections.keys()) == {'user', 'pc', 'url'}


def test_forward_pass_shapes(small_hetero_graph):
    """Forward produces 'user' key with shape (N_user, hidden_dim)."""
    hidden_dim = 32
    model = HeteroGraphSAGE(
        metadata=small_hetero_graph.metadata(),
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        out = model(small_hetero_graph.x_dict, small_hetero_graph.edge_index_dict)

    assert 'user' in out
    assert out['user'].shape == (10, hidden_dim)


def test_forward_output_all_node_types(small_hetero_graph):
    """All non-empty node types appear in the output dict."""
    model = HeteroGraphSAGE(
        metadata=small_hetero_graph.metadata(),
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        out = model(small_hetero_graph.x_dict, small_hetero_graph.edge_index_dict)

    # fixture has user, pc, url (all non-empty)
    assert set(out.keys()) == {'user', 'pc', 'url'}


def test_forward_skips_zero_node_type(small_hetero_graph):
    """A node type with 0 nodes is absent from the output dict."""
    g = small_hetero_graph
    metadata = g.metadata()

    model = HeteroGraphSAGE(metadata=metadata, hidden_dim=16, num_layers=1, dropout=0.0)
    model.eval()

    # Inject an empty 'pc' node tensor — simulates a NeighborLoader batch where
    # no PC was sampled.
    x_dict = {k: v for k, v in g.x_dict.items()}
    x_dict['pc'] = torch.zeros(0, 8)

    with torch.no_grad():
        out = model(x_dict, g.edge_index_dict)

    assert 'pc' not in out
    assert 'user' in out


# ── InsiderThreatGNN ──────────────────────────────────────────────────────────

def test_insider_threat_gnn_logit_shape(small_hetero_graph):
    """InsiderThreatGNN forward returns logits of shape (num_users, 1)."""
    model = InsiderThreatGNN(
        metadata=small_hetero_graph.metadata(),
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        head_layers=[16],
    )
    model.eval()
    with torch.no_grad():
        logits = model(small_hetero_graph.x_dict, small_hetero_graph.edge_index_dict)

    assert logits.shape == (10, 1)


def test_insider_threat_gnn_no_sigmoid(small_hetero_graph):
    """Raw logits must include negative values (no sigmoid applied in forward)."""
    torch.manual_seed(0)
    model = InsiderThreatGNN(
        metadata=small_hetero_graph.metadata(),
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        head_layers=[16],
    )
    model.eval()
    with torch.no_grad():
        logits = model(small_hetero_graph.x_dict, small_hetero_graph.edge_index_dict)

    # A randomly-initialised network should produce at least some negative logits
    assert logits.min().item() < 0


def test_get_embeddings_shape(small_hetero_graph):
    """get_embeddings() returns (num_users, hidden_dim)."""
    hidden_dim = 24
    model = InsiderThreatGNN(
        metadata=small_hetero_graph.metadata(),
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.0,
        head_layers=[],
    )
    model.eval()
    with torch.no_grad():
        emb = model.get_embeddings(
            small_hetero_graph.x_dict, small_hetero_graph.edge_index_dict
        )

    assert emb.shape == (10, hidden_dim)


def test_get_embeddings_differs_from_logits(small_hetero_graph):
    """get_embeddings returns the backbone output, not the head output."""
    hidden_dim = 32
    model = InsiderThreatGNN(
        metadata=small_hetero_graph.metadata(),
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.0,
        head_layers=[16],
    )
    model.eval()
    with torch.no_grad():
        emb    = model.get_embeddings(
            small_hetero_graph.x_dict, small_hetero_graph.edge_index_dict
        )
        logits = model(small_hetero_graph.x_dict, small_hetero_graph.edge_index_dict)

    # Embeddings are hidden_dim-wide; logits are 1-wide
    assert emb.shape[1] == hidden_dim
    assert logits.shape[1] == 1


def test_gnn_empty_head_layers(small_hetero_graph):
    """head_layers=[] reduces head to a single linear layer."""
    model = InsiderThreatGNN(
        metadata=small_hetero_graph.metadata(),
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        head_layers=[],
    )
    model.eval()
    with torch.no_grad():
        logits = model(small_hetero_graph.x_dict, small_hetero_graph.edge_index_dict)

    assert logits.shape == (10, 1)


# ── build_model factory ───────────────────────────────────────────────────────

def test_build_model_mlp(small_hetero_graph):
    """build_model returns InsiderThreatMLP for type='mlp'."""
    config = {'model': {'type': 'mlp', 'layer_sizes': [18, 8, 1]}}
    model, device = build_model(config)
    assert isinstance(model, InsiderThreatMLP)
    assert isinstance(device, torch.device)


def test_build_model_graphsage(small_hetero_graph):
    """build_model returns InsiderThreatGNN for type='graphsage'."""
    config = {
        'model': {
            'type':        'graphsage',
            'hidden_dim':  16,
            'num_layers':  1,
            'dropout':     0.0,
            'head_layers': [8],
        }
    }
    model, device = build_model(config, metadata=small_hetero_graph.metadata())
    assert isinstance(model, InsiderThreatGNN)
    assert isinstance(device, torch.device)


def test_build_model_graphsage_requires_metadata():
    """build_model raises AssertionError when metadata is omitted for graphsage."""
    config = {'model': {'type': 'graphsage', 'hidden_dim': 16}}
    with pytest.raises(AssertionError, match='metadata'):
        build_model(config, metadata=None)


def test_build_model_unknown_type_raises():
    """build_model raises ValueError for an unrecognised model type."""
    config = {'model': {'type': 'transformer', 'layer_sizes': [18, 1]}}
    with pytest.raises(ValueError, match='unknown model type'):
        build_model(config)


def test_build_model_model_on_device(small_hetero_graph):
    """Model parameters live on the device returned by build_model."""
    config = {
        'model': {'type': 'graphsage', 'hidden_dim': 8,
                  'num_layers': 1, 'dropout': 0.0, 'head_layers': []}
    }
    model, device = build_model(config, metadata=small_hetero_graph.metadata())
    # Run a forward pass first to materialise lazy SAGEConv weights
    model.eval()
    with torch.no_grad():
        _ = model(
            {k: v.to(device) for k, v in small_hetero_graph.x_dict.items()},
            {k: v.to(device) for k, v in small_hetero_graph.edge_index_dict.items()},
        )
    from torch.nn.parameter import UninitializedParameter
    init_params = [p for p in model.parameters()
                   if not isinstance(p, UninitializedParameter)]
    assert all(p.device.type == device.type for p in init_params)
