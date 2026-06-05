"""End-to-end integration tests for the fusion model pipeline (Phase 6)."""

import resource
import sys

import pandas as pd
import pytest
import torch
import torch.nn as nn

from innersight.models.fusion_model import InsiderThreatDetector
from innersight.models.graph_builder import build_windowed_graph
from innersight.models.losses import FocalLoss
from innersight.models.mlp import InsiderThreatMLP
from innersight.models.temporal_encoder import TemporalPatternEncoder
from innersight.scripts.train_temporal_graph import ChainedTemporalGraph, _chained_logits
from innersight.utils.reproducibility import seed_everything

NUM_ROLES = 8
NUM_DEPTS = 4
_TEMPORAL_CFG = {"in_channels": 18, "hidden": 64, "out_dim": 128, "dropout": 0.3, "kernel_size": 3}
_GRAPH_CFG = {"hidden_dim": 128, "num_layers": 2, "heads": 4, "dropout": 0.3}


@pytest.fixture()
def fusion_data():
    """Synthetic graph (10 users, 2 insiders) + aligned windows / OCEAN / role / dept."""
    dates = pd.date_range("2010-06-01", periods=60, freq="D")
    users = [f"u{i:02d}" for i in range(10)]
    pcs = ["PC0", "PC1", "PC2"]
    rows = []
    for i, u in enumerate(users):
        for d in range(3):
            rows.append({"id": f"L{u}{d}", "date": dates[(d + i) % 30],
                         "user": u, "pc": pcs[i % 3], "activity": "Logon"})
    graph = build_windowed_graph({"logon": pd.DataFrame(rows)}, "2010-06-01", "2010-06-28")

    ordered = sorted(graph.user_to_idx, key=lambda u: graph.user_to_idx[u])
    n = len(ordered)
    assert n == 10

    seed_everything(0)
    windows = torch.randn(n, 18, 28)
    labels = torch.zeros(n, 1)
    labels[:2] = 1.0           # first 2 users are insiders
    windows[:2] += 4.0          # with spiked windows (separable)

    inputs = {
        "windows": windows,
        "ocean": torch.randn(n, 5),
        "roles": torch.randint(0, NUM_ROLES, (n,)),
        "depts": torch.randint(0, NUM_DEPTS, (n,)),
        "labels": labels,
        "user_ids": ordered,
    }
    return graph, inputs


def _make_fusion(graph):
    return InsiderThreatDetector(graph.metadata(), num_roles=NUM_ROLES, num_depts=NUM_DEPTS,
                                 temporal_config=_TEMPORAL_CFG, graph_config=_GRAPH_CFG)


def _fusion_forward(model, graph, inputs):
    return model(inputs["windows"], graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict,
                 inputs["ocean"], inputs["roles"], inputs["depts"])


# ── Forward / embeddings ─────────────────────────────────────────────────────

def test_fusion_model_forward_end_to_end(fusion_data):
    graph, inputs = fusion_data
    model = _make_fusion(graph)
    model.eval()
    with torch.no_grad():
        out = _fusion_forward(model, graph, inputs)
    assert out.shape == (10, 1)


def test_fusion_embeddings_272_dim(fusion_data):
    graph, inputs = fusion_data
    model = _make_fusion(graph)
    model.eval()
    with torch.no_grad():
        emb = model.get_embeddings(inputs["windows"], graph.x_dict, graph.edge_index_dict,
                                   graph.edge_attr_dict, inputs["ocean"], inputs["roles"], inputs["depts"])
    assert emb.shape == (10, 272)
    assert not torch.isnan(emb).any()


# ── Training ─────────────────────────────────────────────────────────────────

def test_fusion_model_trains_loss_decreases(fusion_data):
    graph, inputs = fusion_data
    seed_everything(0)
    model = _make_fusion(graph)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def _loss():
        model.eval()
        with torch.no_grad():
            return criterion(_fusion_forward(model, graph, inputs), inputs["labels"]).item()

    initial = _loss()
    for _ in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(_fusion_forward(model, graph, inputs), inputs["labels"])
        loss.backward()
        optimizer.step()
    assert _loss() < initial


# ── Full ladder ──────────────────────────────────────────────────────────────

def test_full_progressive_ladder_shapes(fusion_data):
    graph, inputs = fusion_data
    n = inputs["windows"].shape[0]
    windows = inputs["windows"]

    # Rung 1 — MLP on flattened windows (18×28 = 504).
    mlp = InsiderThreatMLP([504, 64, 32, 1])
    mlp.eval()
    with torch.no_grad():
        assert mlp(windows.reshape(n, -1)).shape == (n, 1)

    # Rung 2 — Temporal CNN + linear head.
    temporal = TemporalPatternEncoder()
    head = nn.Linear(128, 1)
    temporal.eval()
    with torch.no_grad():
        assert head(temporal(windows)).shape == (n, 1)

    # Rung 3 — Temporal + Graph (chained).
    chained = ChainedTemporalGraph(_TEMPORAL_CFG, _GRAPH_CFG, graph.metadata())
    chained.eval()
    with torch.no_grad():
        assert _chained_logits(chained, windows, inputs["user_ids"], graph).shape == (n, 1)

    # Rung 4 — Full fusion.
    fusion = _make_fusion(graph)
    fusion.eval()
    with torch.no_grad():
        assert _fusion_forward(fusion, graph, inputs).shape == (n, 1)


# ── Memory ───────────────────────────────────────────────────────────────────

def _peak_rss_mb() -> float:
    """Peak process RSS in MB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def test_cpu_memory_reasonable(fusion_data):
    graph, inputs = fusion_data
    # torch is already warm (fixture built a model), so the delta in peak RSS is
    # the fusion model's own footprint — must stay well under 500 MB on CPU.
    before = _peak_rss_mb()
    model = _make_fusion(graph)
    model.eval()
    with torch.no_grad():
        out = _fusion_forward(model, graph, inputs)
    assert out.shape == (10, 1)
    delta = _peak_rss_mb() - before
    assert delta < 500.0, f"fusion model added {delta:.1f} MB peak RSS (expected < 500)"
