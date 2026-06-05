"""Unit tests for innersight.training.evaluation (Phase 3)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from innersight.data.answers import InsiderRecord
from innersight.training.evaluation import (
    compute_metrics,
    detection_latency,
    format_results_table,
    per_scenario_metrics,
    run_cross_validation,
    temporal_stratified_kfold,
)

_EXPECTED_KEYS = {
    "auprc", "auroc", "f1_best", "threshold_best", "p_at_10", "p_at_20", "p_at_50",
}


def _signal_probs(labels, rng, strength=0.35):
    return np.clip(strength * labels + 0.5 * rng.random(len(labels)), 0, 1)


# ── compute_metrics ──────────────────────────────────────────────────────────

def test_compute_metrics_keys():
    rng = np.random.default_rng(0)
    labels = (rng.random(200) < 0.1).astype(float)
    m = compute_metrics(_signal_probs(labels, rng), labels)
    assert set(m.keys()) == _EXPECTED_KEYS


def test_compute_metrics_auprc_matches_sklearn():
    rng = np.random.default_rng(1)
    labels = (rng.random(200) < 0.15).astype(float)
    probs = _signal_probs(labels, rng)
    assert compute_metrics(probs, labels)["auprc"] == average_precision_score(labels, probs)


def test_compute_metrics_all_zero_labels_no_crash():
    rng = np.random.default_rng(2)
    m = compute_metrics(rng.random(50), np.zeros(50))
    assert m["auprc"] == 0.0 and m["auroc"] == 0.0 and m["p_at_10"] == 0.0


def test_compute_metrics_all_one_labels_no_crash():
    rng = np.random.default_rng(3)
    m = compute_metrics(rng.random(50), np.ones(50))
    assert m["auprc"] == 0.0 and m["p_at_10"] == 0.0


# ── per_scenario_metrics ─────────────────────────────────────────────────────

def test_per_scenario_metrics_keyed_by_scenario():
    rng = np.random.default_rng(4)
    labels = (rng.random(200) < 0.1).astype(float)
    probs = _signal_probs(labels, rng)
    scenarios = np.zeros(200)
    pos = np.where(labels == 1)[0]
    scenarios[pos[: len(pos) // 2]] = 2
    scenarios[pos[len(pos) // 2:]] = 4
    out = per_scenario_metrics(probs, labels, scenarios)
    assert set(out.keys()) == {2, 4}
    assert all(set(v.keys()) == _EXPECTED_KEYS for v in out.values())


# ── temporal_stratified_kfold ────────────────────────────────────────────────

def _windowed_users(n_users=40, n_insiders=10, windows_per_user=5):
    users = np.array([f"u{i:02d}" for i in range(n_users)])
    user_ids = np.repeat(users, windows_per_user)
    insider = {u for u in users[:n_insiders]}
    labels = np.array([1.0 if u in insider else 0.0 for u in user_ids])
    return user_ids, labels


def test_kfold_produces_n_folds():
    user_ids, labels = _windowed_users()
    folds = temporal_stratified_kfold(user_ids, labels, n_folds=5, seed=42)
    assert len(folds) == 5


def test_kfold_no_user_in_both_train_and_val():
    user_ids, labels = _windowed_users()
    for train_idx, val_idx in temporal_stratified_kfold(user_ids, labels, n_folds=5, seed=42):
        assert set(user_ids[train_idx]).isdisjoint(set(user_ids[val_idx]))


def test_kfold_insiders_distributed_across_folds():
    user_ids, labels = _windowed_users(n_users=40, n_insiders=10)
    folds = temporal_stratified_kfold(user_ids, labels, n_folds=5, seed=42)
    # Every fold's validation set should contain at least one insider window.
    for _train_idx, val_idx in folds:
        assert labels[val_idx].sum() > 0


# ── detection_latency ────────────────────────────────────────────────────────

def test_detection_latency_keys():
    aw = {"ins": InsiderRecord("ins", 2, "4.2", pd.Timestamp("2010-06-01"),
                               pd.Timestamp("2010-06-20"), "f.csv")}
    metas = [{"user_id": "ins", "window_start": pd.Timestamp("2010-06-03")}]
    dl = detection_latency(np.array([0.9]), metas, 0.5, aw)
    for key in ("median_days", "p90_days", "detected_count", "total_insiders", "per_insider"):
        assert key in dl
    assert dl["detected_count"] == 1 and dl["total_insiders"] == 1


# ── format_results_table ─────────────────────────────────────────────────────

def test_format_results_table_non_empty_string():
    rng = np.random.default_rng(5)
    user_ids, labels = _windowed_users()
    X = rng.standard_normal((len(labels), 8))
    metas = [{"user_id": u, "window_start": pd.Timestamp("2010-06-01"), "scenario": 0} for u in user_ids]
    aw = {u: InsiderRecord(u, 2, "4.2", pd.Timestamp("2010-06-01"), pd.Timestamp("2010-06-20"), "f.csv")
          for u in set(user_ids[labels == 1])}

    def model_fn(Xtr, ytr, Xva, yva, seed):
        r = np.random.default_rng(seed)
        return np.clip(0.5 * yva + 0.3 * r.random(len(yva)), 0, 1)

    res = run_cross_validation(model_fn, X, labels, metas, aw, n_folds=5, seeds=[42])
    table = format_results_table({"ModelA": res})
    assert isinstance(table, str) and len(table) > 0
    assert "ModelA" in table and "AUPRC" in table
