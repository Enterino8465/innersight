"""Unit tests for the windowed graph builder (Phase 5.1)."""

import pandas as pd
import torch

from innersight.models.graph_builder import build_windowed_graph
from innersight.models.graph_schema import (
    EDGE_FILE_COPY,
    EDGE_HTTP,
    EDGE_LOGON,
    EDGE_USB,
    NODE_FILE,
    NODE_PC,
    NODE_URL,
    NODE_USER,
    REV_EDGE_LOGON,
    USER_TEMPORAL_DIM,
    WINDOWED_EDGE_FEATURE_DIMS,
)


def _ts(s):
    return pd.Timestamp(s)


def _logs():
    """Synthetic raw logs: events inside June + a prior-May logon for is_new tests."""
    logon = pd.DataFrame({
        "id": [f"L{i}" for i in range(6)],
        "date": [_ts("2010-05-10 09:00"),  # prior: u1@PC1
                 _ts("2010-06-02 08:00"), _ts("2010-06-03 22:00"), _ts("2010-06-05 13:00"),
                 _ts("2010-06-06 10:00"), _ts("2010-06-07 11:00")],
        "user": ["u1", "u1", "u1", "u2", "u3", "u2"],
        "pc": ["PC1", "PC1", "PC1", "PC2", "PC1", "PC2"],
        "activity": ["Logon"] * 6,
    })
    device = pd.DataFrame({
        "id": ["D0", "D1"],
        "date": [_ts("2010-06-04 23:00"), _ts("2010-06-05 14:00")],
        "user": ["u1", "u2"], "pc": ["PC1", "PC2"], "activity": ["Connect", "Connect"],
    })
    http = pd.DataFrame({
        "id": ["H0", "H1"],
        "date": [_ts("2010-06-02 12:00"), _ts("2010-07-15 10:00")],  # 2nd is OUTSIDE window
        "user": ["u1", "u3"], "pc": ["PC1", "PC1"],
        "url": ["http://monster.com/jobs", "http://news.com"], "content": ["", ""],
    })
    file = pd.DataFrame({
        "id": ["F0"], "date": [_ts("2010-06-04 23:30")],
        "user": ["u1"], "pc": ["PC1"], "filename": ["secret.docx"], "content": [""],
        "to_removable_media": ["True"],
    })
    return {"logon": logon, "device": device, "http": http, "file": file}


def _graph():
    return build_windowed_graph(_logs(), "2010-06-01", "2010-06-28", prior_days=60)


# ── Structure ────────────────────────────────────────────────────────────────

def test_returns_hetero_data_with_node_types():
    g = _graph()
    for nt in (NODE_USER, NODE_PC, NODE_URL, NODE_FILE):
        assert nt in g.node_types
    assert g[NODE_USER].x.shape[1] == USER_TEMPORAL_DIM
    assert g[NODE_PC].x.shape[1] == 8
    assert g[NODE_URL].x.shape[1] == 8
    assert g[NODE_FILE].x.shape[1] == 6


def test_edge_feature_dimensions():
    g = _graph()
    for etype, dim in WINDOWED_EDGE_FEATURE_DIMS.items():
        assert g[etype].edge_attr.shape[1] == dim
        assert g[etype].edge_index.shape[0] == 2
        assert g[etype].edge_index.shape[1] == g[etype].edge_attr.shape[0]
    # reverse edge carries the same features as its forward counterpart
    assert torch.equal(g[REV_EDGE_LOGON].edge_attr, g[EDGE_LOGON].edge_attr)
    assert torch.equal(g[REV_EDGE_LOGON].edge_index, g[EDGE_LOGON].edge_index.flip(0))


# ── Windowing ────────────────────────────────────────────────────────────────

def test_only_events_within_window():
    # The July http visit (news.com) must be excluded; only monster.com remains.
    g = _graph()
    assert "monster.com" in g.url_to_idx
    assert "news.com" not in g.url_to_idx


# ── is_new flags ─────────────────────────────────────────────────────────────

def test_is_new_flag_against_prior_days():
    g = _graph()
    # USB edge is_new_pc is column index 3 (count, frac_after_hours, max_burst_day, is_new_pc).
    li, pi = g.user_to_idx, g.pc_to_idx
    new_flags = {}
    ei, ea = g[EDGE_USB].edge_index, g[EDGE_USB].edge_attr
    inv_u = {v: k for k, v in li.items()}
    inv_p = {v: k for k, v in pi.items()}
    for e in range(ei.shape[1]):
        new_flags[(inv_u[ei[0, e].item()], inv_p[ei[1, e].item()])] = ea[e, 3].item()
    # u1→PC1 had a prior-May logon → NOT new (0.0); u2→PC2 is new (1.0).
    assert new_flags[("u1", "PC1")] == 0.0
    assert new_flags[("u2", "PC2")] == 1.0


# ── Robustness ───────────────────────────────────────────────────────────────

def test_missing_log_types_no_crash():
    # r1-style: only logon, no email/file/http/device.
    logs = {"logon": _logs()["logon"]}
    g = build_windowed_graph(logs, "2010-06-01", "2010-06-28")
    assert g[EDGE_HTTP].edge_index.shape[1] == 0
    assert g[EDGE_FILE_COPY].edge_index.shape[1] == 0
    assert g[EDGE_USB].edge_index.shape[1] == 0
    assert g[NODE_URL].x.shape == (0, 8)
    # logon edges still present
    assert g[EDGE_LOGON].edge_index.shape[1] > 0


def test_node_index_mappings_present_and_correct():
    g = _graph()
    for attr in ("user_to_idx", "pc_to_idx", "url_to_idx", "file_to_idx"):
        assert isinstance(getattr(g, attr), dict)
    assert set(g.user_to_idx) >= {"u1", "u2", "u3"}
    assert set(g.pc_to_idx) == {"PC1", "PC2"}
    assert set(g.file_to_idx) == {"secret.docx"}
    # indices are a contiguous 0..N-1 range
    assert sorted(g.user_to_idx.values()) == list(range(len(g.user_to_idx)))
    assert g[NODE_USER].x.shape[0] == len(g.user_to_idx)
