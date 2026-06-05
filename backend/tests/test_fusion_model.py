"""Unit tests for the InsiderThreatDetector fusion model (Phase 6)."""

import pytest
import torch
from torch_geometric.data import HeteroData

from innersight.models.fusion_model import InsiderThreatDetector
from innersight.models.graph_schema import (
    EDGE_HTTP,
    EDGE_LOGON,
    NODE_PC,
    NODE_URL,
    NODE_USER,
    REV_EDGE_HTTP,
    REV_EDGE_LOGON,
)

N_USERS = 10
NUM_ROLES = 50
NUM_DEPTS = 15


@pytest.fixture()
def fusion_setup():
    """Synthetic windowed graph (10 users, 5 PCs, 3 URLs) + a fusion model + inputs."""
    torch.manual_seed(0)
    g = HeteroData()
    g[NODE_USER].x = torch.zeros(N_USERS, 128)  # placeholder (overwritten in forward)
    g[NODE_PC].x = torch.randn(5, 8)
    g[NODE_URL].x = torch.randn(3, 8)

    # Dense-ish edges so every node has neighbours (non-trivial attention).
    le_src = list(range(N_USERS)) + list(range(N_USERS))
    le_dst = [i % 5 for i in range(N_USERS)] + [(i + 1) % 5 for i in range(N_USERS)]
    le = torch.tensor([le_src, le_dst])
    la = torch.randn(le.shape[1], 5)
    g[EDGE_LOGON].edge_index = le
    g[EDGE_LOGON].edge_attr = la
    g[REV_EDGE_LOGON].edge_index = le.flip(0)
    g[REV_EDGE_LOGON].edge_attr = la

    he_src = list(range(N_USERS)) + list(range(N_USERS))
    he_dst = [i % 3 for i in range(N_USERS)] + [(i + 1) % 3 for i in range(N_USERS)]
    he = torch.tensor([he_src, he_dst])
    ha = torch.randn(he.shape[1], 4)
    g[EDGE_HTTP].edge_index = he
    g[EDGE_HTTP].edge_attr = ha
    g[REV_EDGE_HTTP].edge_index = he.flip(0)
    g[REV_EDGE_HTTP].edge_attr = ha

    model = InsiderThreatDetector(g.metadata(), num_roles=NUM_ROLES, num_depts=NUM_DEPTS)
    inputs = {
        "deviation_sequences": torch.randn(N_USERS, 18, 28),
        "ocean": torch.randn(N_USERS, 5),
        "roles": torch.randint(0, NUM_ROLES, (N_USERS,)),
        "depts": torch.randint(0, NUM_DEPTS, (N_USERS,)),
    }
    return model, g, inputs


def _forward(model, g, inputs):
    return model(inputs["deviation_sequences"], g.x_dict, g.edge_index_dict, g.edge_attr_dict,
                 inputs["ocean"], inputs["roles"], inputs["depts"])


# ── Shapes ───────────────────────────────────────────────────────────────────

def test_forward_output_shape(fusion_setup):
    model, g, inputs = fusion_setup
    model.eval()
    with torch.no_grad():
        out = _forward(model, g, inputs)
    assert out.shape == (N_USERS, 1)


def test_get_embeddings_shape(fusion_setup):
    model, g, inputs = fusion_setup
    model.eval()
    with torch.no_grad():
        emb = model.get_embeddings(inputs["deviation_sequences"], g.x_dict, g.edge_index_dict,
                                   g.edge_attr_dict, inputs["ocean"], inputs["roles"], inputs["depts"])
    # 272 = temporal(128) + graph(128) + static(16)
    assert emb.shape == (N_USERS, 272)


def test_get_temporal_attention_shape(fusion_setup):
    model, _g, inputs = fusion_setup
    attn = model.get_temporal_attention(inputs["deviation_sequences"])
    assert attn.shape == (N_USERS, 28)


# ── Gradient flow ────────────────────────────────────────────────────────────

def test_gradient_flow_all_modules(fusion_setup):
    model, g, inputs = fusion_setup
    model.train()
    out = _forward(model, g, inputs)
    out.sum().backward()
    # Each module must receive gradient (proves the full chain is connected).
    for name, module in (("temporal", model.temporal), ("graph", model.graph),
                         ("role_emb", model.role_emb), ("dept_emb", model.dept_emb),
                         ("head", model.head)):
        assert any(p.grad is not None and p.grad.abs().sum().item() > 0.0
                   for p in module.parameters()), f"{name} received no gradient"


# ── Static context ───────────────────────────────────────────────────────────

def test_static_context_affects_output(fusion_setup):
    model, g, inputs = fusion_setup
    model.eval()
    with torch.no_grad():
        out_a = _forward(model, g, inputs)
        flipped = dict(inputs)
        flipped["ocean"] = inputs["ocean"] + 5.0  # different OCEAN, same windows/graph
        out_b = model(flipped["deviation_sequences"], g.x_dict, g.edge_index_dict, g.edge_attr_dict,
                      flipped["ocean"], flipped["roles"], flipped["depts"])
    # If static context were ignored, the outputs would be identical.
    assert (out_a - out_b).abs().max().item() > 1e-4


# ── Parameter count ──────────────────────────────────────────────────────────

def test_parameter_count(fusion_setup):
    model, _g, _inputs = fusion_setup
    n_params = sum(p.numel() for p in model.parameters())
    # temporal ~63K + graph ~420K (4 edge types) + embeddings + head ~43K ≈ 530K.
    assert 400_000 <= n_params <= 700_000, f"unexpected parameter count: {n_params}"
