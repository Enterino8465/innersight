"""End-to-end integration tests for the Phase 3 baseline pipeline.

Exercises windows → handcrafted features → classifier → evaluation harness on
synthetic data. The XGBoost case trains a real model, so it is marked
``@pytest.mark.slow`` and excluded from the default run (also avoids the macOS
torch+xgboost OpenMP conflict in the default suite).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from innersight.data.answers import InsiderRecord
from innersight.features.temporal_features import (
    N_WINDOW_FEATURES,
    extract_all_window_features,
    get_feature_names,
)
from innersight.models.dataset import DeviationWindowDataset
from innersight.models.losses import FocalLoss
from innersight.training.evaluation import (
    compute_metrics,
    per_scenario_metrics,
    temporal_stratified_kfold,
)
from innersight.schema import FEATURE_NAMES

N_FEAT = len(FEATURE_NAMES)
N_DAYS = 90
_EXPECTED_KEYS = {
    "auprc", "auroc", "f1_best", "threshold_best", "p_at_10", "p_at_20", "p_at_50",
}


@pytest.fixture()
def phase3_dataset():
    """20 users (18 normal, 2 insiders w/ different scenarios), 90 days, spiking insiders."""
    dates = pd.date_range("2010-06-01", periods=N_DAYS, freq="D")
    rng = np.random.default_rng(11)
    # (user_id, scenario, attack_start_idx, attack_end_idx)
    insiders = [("ins_a", 2, 40, 60), ("ins_b", 3, 30, 52)]
    insider_attacks = {u: (s, a0, a1) for u, s, a0, a1 in insiders}

    frames = []
    users = [f"norm{i:02d}" for i in range(18)] + [u for u, *_ in insiders]
    for user in users:
        mat = rng.normal(0.0, 1.0, size=(N_DAYS, N_FEAT))
        if user in insider_attacks:
            _, a0, a1 = insider_attacks[user]
            mat[a0:a1 + 1] += rng.normal(6.0, 1.0, size=(a1 - a0 + 1, N_FEAT))
        f = pd.DataFrame(mat, columns=FEATURE_NAMES)
        f.insert(0, "date", dates)
        f.insert(0, "user", user)
        frames.append(f)
    deviations = pd.concat(frames, ignore_index=True)

    attack_windows = {
        u: InsiderRecord(u, s, "4.2", dates[a0], dates[a1], f"r4.2-{u}.csv")
        for u, s, a0, a1 in insiders
    }
    dataset = DeviationWindowDataset(deviations, attack_windows, window_size=28, stride=7)
    return dataset, attack_windows


def _features(phase3_dataset):
    dataset, attack_windows = phase3_dataset
    X, y, metas = extract_all_window_features(dataset)
    return X, y, metas, attack_windows


@pytest.mark.slow
def test_xgboost_end_to_end(phase3_dataset):
    from xgboost import XGBClassifier

    X, y, metas, _ = _features(phase3_dataset)
    assert y.sum() > 0, "fixture must yield positive windows"
    user_ids = np.array([m["user_id"] for m in metas])

    train_idx, val_idx = temporal_stratified_kfold(user_ids, y, n_folds=2, seed=42)[0]
    n_pos = float(np.count_nonzero(y[train_idx] == 1))
    n_neg = float(np.count_nonzero(y[train_idx] == 0))
    clf = XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, tree_method="hist",
        scale_pos_weight=(n_neg / n_pos) if n_pos > 0 else 1.0,
        eval_metric="aucpr", n_jobs=1, random_state=42,
    )
    clf.fit(X[train_idx], y[train_idx])
    val_probs = clf.predict_proba(X[val_idx])[:, 1]

    metrics = compute_metrics(val_probs, y[val_idx])
    auprc = metrics["auprc"]
    assert isinstance(auprc, float) and 0.0 <= auprc <= 1.0


def test_evaluation_harness_keys(phase3_dataset):
    X, y, metas, _ = _features(phase3_dataset)
    rng = np.random.default_rng(0)
    probs = np.clip(0.4 * y + 0.4 * rng.random(len(y)), 0, 1)
    assert set(compute_metrics(probs, y).keys()) == _EXPECTED_KEYS


def test_temporal_features_shape(phase3_dataset):
    X, _, _, _ = _features(phase3_dataset)
    assert X.shape[1] == len(get_feature_names()) == N_WINDOW_FEATURES


def test_per_scenario_metrics(phase3_dataset):
    X, y, metas, _ = _features(phase3_dataset)
    scenarios = np.array([m["scenario"] for m in metas])
    rng = np.random.default_rng(1)
    probs = np.clip(0.4 * y + 0.4 * rng.random(len(y)), 0, 1)
    out = per_scenario_metrics(probs, y, scenarios)
    present = {int(s) for s in scenarios if s != 0}
    assert set(out.keys()) == present
    assert present == {2, 3}


def test_focal_loss_trains_mlp():
    # A tiny separable problem: FocalLoss + a small MLP should reduce the loss.
    torch.manual_seed(0)
    n, d = 256, 16
    X = torch.randn(n, d)
    w = torch.randn(d, 1)
    y = (X @ w > 0).float()  # shape: (n, 1)

    model = torch.nn.Sequential(
        torch.nn.Linear(d, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
    )
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    with torch.no_grad():
        initial_loss = criterion(model(X), y).item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    final_loss = criterion(model(X), y).item()

    assert final_loss < initial_loss
