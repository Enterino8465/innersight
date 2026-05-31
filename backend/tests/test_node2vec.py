"""Tests for backend/models/node2vec_trainer.py."""

import importlib
import torch
import pytest
from torch_geometric.data import HeteroData

# Node2Vec requires pyg-lib or torch-cluster; skip the whole module if absent.
_pyg_cluster = importlib.util.find_spec('pyg_lib') or importlib.util.find_spec('torch_cluster')
pytestmark = pytest.mark.skipif(
    _pyg_cluster is None,
    reason='pyg-lib or torch-cluster required for Node2Vec (node2vec tests)',
)

from innersight.backend.models.node2vec_trainer import (
    _hetero_to_homo,
    train_node2vec,
    save_embeddings,
    load_embeddings,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _tiny_homo_graph(n_nodes: int = 10, n_edges: int = 20, seed: int = 0):
    """Return a homogeneous (2, E) edge_index for a random graph."""
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n_nodes, (n_edges,), generator=g)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=g)
    # Add self-loops to guarantee all nodes are connected
    loop = torch.arange(n_nodes)
    src  = torch.cat([src, loop])
    dst  = torch.cat([dst, loop])
    return torch.stack([src, dst], dim=0)


def _tiny_hetero_graph():
    """Three-node HeteroData: 2 users, 1 url, connected via http edges."""
    data = HeteroData()

    data['user'].x = torch.randn(2, 4)
    data['url'].x  = torch.randn(1, 4)
    # No 'pc' or 'file' nodes — they should default to 0 count

    # user → url edges
    data['user', 'http_request', 'url'].edge_index = torch.tensor([[0, 1], [0, 0]])
    # url → user reverse
    data['url', 'rev_http_request', 'user'].edge_index = torch.tensor([[0, 0], [0, 1]])

    return data


# ── test_node2vec_toy ─────────────────────────────────────────────────────────

def test_node2vec_toy():
    """Node2Vec on a tiny graph: shape correct, embeddings non-zero."""
    n_nodes  = 10
    emb_dim  = 8
    edge_idx = _tiny_homo_graph(n_nodes=n_nodes, n_edges=20)

    from torch_geometric.nn import Node2Vec
    model = Node2Vec(
        edge_index=edge_idx,
        embedding_dim=emb_dim,
        walk_length=5,
        context_size=3,
        walks_per_node=2,
        sparse=True,
        num_nodes=n_nodes,
    )
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    loader = model.loader(batch_size=4, shuffle=False, num_workers=0)

    for epoch in range(2):
        model.train()
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            model.loss(pos_rw, neg_rw).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        emb = model().detach()

    assert emb.shape == (n_nodes, emb_dim), f"Expected ({n_nodes},{emb_dim}), got {emb.shape}"
    assert not torch.all(emb == 0), "Embeddings are all zeros — training had no effect"


# ── test_node2vec_save_load ───────────────────────────────────────────────────

def test_node2vec_save_load(tmp_path):
    """Save and load round-trip preserves embedding values and user_to_idx."""
    emb      = torch.randn(5, 16)
    u2i      = {'alice': 0, 'bob': 1, 'carol': 2, 'dave': 3, 'eve': 4}
    out_path = str(tmp_path / 'emb.pt')

    save_embeddings(emb, u2i, out_path)
    loaded_emb, loaded_u2i = load_embeddings(out_path)

    torch.testing.assert_close(loaded_emb, emb)
    assert loaded_u2i == u2i


# ── test_homogeneous_conversion ───────────────────────────────────────────────

def test_homogeneous_conversion_node_count():
    """_hetero_to_homo: connected nodes ≤ sum of all node type counts."""
    graph = _tiny_hetero_graph()

    homo_edge_index, total_nodes, user_indices, offsets = _hetero_to_homo(graph)

    n_user = graph['user'].x.shape[0]   # 2
    n_url  = graph['url'].x.shape[0]    # 1
    raw_total = n_user + n_url           # 3 (pc and file absent)

    # After pruning isolated nodes, total_nodes ≤ raw_total
    assert total_nodes <= raw_total, (
        f"total_nodes={total_nodes} exceeds raw_total={raw_total}"
    )
    # All user nodes are connected (they all appear in edge_index)
    assert user_indices.shape[0] == n_user, (
        f"Expected {n_user} user indices, got {user_indices.shape[0]}"
    )


def test_homogeneous_conversion_edge_index_shape():
    """homo_edge_index has 2 rows and contains no index ≥ total_nodes."""
    graph = _tiny_hetero_graph()
    homo_edge_index, total_nodes, _, _ = _hetero_to_homo(graph)

    assert homo_edge_index.shape[0] == 2
    assert homo_edge_index.max().item() < total_nodes


def test_homogeneous_conversion_no_isolated_nodes():
    """Every node that appears in homo_edge_index has a valid compact index."""
    graph = _tiny_hetero_graph()
    homo_edge_index, total_nodes, _, _ = _hetero_to_homo(graph)

    # All indices should be in [0, total_nodes)
    assert homo_edge_index.min().item() >= 0
    assert homo_edge_index.max().item() < total_nodes


# ── test_train_node2vec_on_hetero ─────────────────────────────────────────────

def test_train_node2vec_on_hetero():
    """train_node2vec on a tiny HeteroData returns (n_users, emb_dim) tensor."""
    graph  = _tiny_hetero_graph()
    config = {
        'embedding_dim':  8,
        'walk_length':    3,
        'context_size':   2,
        'walks_per_node': 2,
        'epochs':         2,
        'p': 1.0,
        'q': 1.0,
    }
    emb = train_node2vec(graph, config)

    n_users = graph['user'].x.shape[0]
    assert emb.shape == (n_users, 8), f"Expected ({n_users},8), got {emb.shape}"
    assert emb.dtype == torch.float32
    assert torch.isfinite(emb).all()
