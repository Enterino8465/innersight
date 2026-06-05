"""End-to-end integration tests for the Temporal+Graph pipeline (Phase 5)."""

import pandas as pd
import pytest
import torch
import torch.nn as nn

from innersight.models.graph_builder import build_windowed_graph
from innersight.models.graph_encoder import GraphContextEncoder
from innersight.models.losses import FocalLoss
from innersight.models.temporal_encoder import TemporalPatternEncoder
from innersight.scripts.train_temporal_graph import (
    ChainedTemporalGraph,
    _chained_logits,
    _filter_logs,
)
from innersight.utils.reproducibility import seed_everything

_TEMPORAL_CFG = {"in_channels": 18, "hidden": 64, "out_dim": 128, "dropout": 0.3, "kernel_size": 3}
_GRAPH_CFG = {"hidden_dim": 128, "num_layers": 2, "heads": 4, "dropout": 0.3}


@pytest.fixture()
def graph_and_logs():
    """Synthetic logs for 10 users → a windowed graph with 10 user nodes."""
    dates = pd.date_range("2010-06-01", periods=20, freq="D")
    users = [f"u{i:02d}" for i in range(10)]
    pcs = ["PC0", "PC1", "PC2"]
    logon_rows, http_rows = [], []
    for i, u in enumerate(users):
        for d in range(3):
            logon_rows.append({"id": f"L{u}{d}", "date": dates[d + i % 5],
                               "user": u, "pc": pcs[i % 3], "activity": "Logon"})
        http_rows.append({"id": f"H{u}", "date": dates[i % 5], "user": u,
                          "pc": pcs[i % 3], "url": f"http://site{i % 4}.com", "content": ""})
    logs = {"logon": pd.DataFrame(logon_rows), "http": pd.DataFrame(http_rows)}
    graph = build_windowed_graph(logs, "2010-06-01", "2010-06-28")
    return logs, graph


def _ordered_users(graph):
    """User ids in node-index order (0..N-1)."""
    return sorted(graph.user_to_idx, key=lambda u: graph.user_to_idx[u])


def test_temporal_graph_chain_forward(graph_and_logs):
    _logs, graph = graph_and_logs
    users = _ordered_users(graph)
    n = len(users)
    assert n == 10

    temporal = TemporalPatternEncoder()
    temporal.eval()
    graph_enc = GraphContextEncoder(graph.metadata())
    graph_enc.eval()
    head = nn.Linear(128, 1)

    windows = torch.randn(n, 18, 28)
    with torch.no_grad():
        emb = temporal(windows)                         # shape: (10, 128)
        assert emb.shape == (n, 128)

        x_dict = dict(graph.x_dict)
        x_dict["user"] = emb                            # inject temporal embeddings
        enriched = graph_enc(x_dict, graph.edge_index_dict, graph.edge_attr_dict)["user"]
        assert enriched.shape == (n, 128)               # enriched user embeddings

        logits = head(enriched)                         # shape: (10, 1)
        assert logits.shape == (n, 1)


def test_temporal_graph_trains_loss_decreases(graph_and_logs):
    _logs, graph = graph_and_logs
    users = _ordered_users(graph)
    n = len(users)

    seed_everything(0)
    model = ChainedTemporalGraph(_TEMPORAL_CFG, _GRAPH_CFG, graph.metadata())
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Separable problem: the first 3 users are positive with spiked windows.
    targets = torch.zeros(n, 1)
    targets[:3] = 1.0
    windows = torch.randn(n, 18, 28)
    windows[:3] += 4.0

    def _loss():
        model.eval()
        with torch.no_grad():
            return criterion(_chained_logits(model, windows, users, graph), targets).item()

    initial = _loss()
    for _ in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(_chained_logits(model, windows, users, graph), targets)
        loss.backward()
        optimizer.step()
    assert _loss() < initial


def test_inductive_split_excludes_test_users(graph_and_logs):
    logs, full_graph = graph_and_logs
    test_users = {"u00", "u01"}

    # Removed from the training graph (their log rows are filtered out).
    train_graph = build_windowed_graph(_filter_logs(logs, test_users),
                                       "2010-06-01", "2010-06-28")
    for u in test_users:
        assert u not in train_graph.user_to_idx

    # Present again in the full (eval) graph when added back.
    for u in test_users:
        assert u in full_graph.user_to_idx
