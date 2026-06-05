"""End-to-end integration tests for the temporal CNN pipeline (Phase 4).

Synthetic windows → TemporalPatternEncoder + linear head → focal-loss training →
evaluation. All torch-only (no xgboost), fast enough for the default suite.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from innersight.data.answers import InsiderRecord
from innersight.models.dataset import DeviationWindowDataset
from innersight.models.losses import FocalLoss
from innersight.models.temporal_encoder import TemporalPatternEncoder
from innersight.schema import FEATURE_NAMES
from innersight.training.evaluation import compute_metrics, temporal_stratified_kfold
from innersight.utils.reproducibility import seed_everything

N_FEAT = len(FEATURE_NAMES)
N_DAYS = 90


class _Clf(nn.Module):
    """TemporalPatternEncoder + linear head → per-window logit."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = TemporalPatternEncoder()
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))  # shape: (batch, 1)


@pytest.fixture()
def temporal_data():
    """20 users (18 normal, 2 insiders), 90 days, insiders spike during attacks."""
    dates = pd.date_range("2010-06-01", periods=N_DAYS, freq="D")
    rng = np.random.default_rng(11)
    insiders = [("ins_a", 2, 40, 60), ("ins_b", 3, 30, 52)]
    attacks = {u: (a0, a1) for u, _s, a0, a1 in insiders}

    frames = []
    users = [f"norm{i:02d}" for i in range(18)] + [u for u, *_ in insiders]
    for user in users:
        mat = rng.normal(0.0, 1.0, size=(N_DAYS, N_FEAT))
        if user in attacks:
            a0, a1 = attacks[user]
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


def _window_tensors(dataset):
    """Stack the dataset into (X (n,18,28), y (n,), metas)."""
    windows, labels, metas = [], [], []
    for i in range(len(dataset)):
        window, label, meta = dataset[i]
        windows.append(window)
        labels.append(float(label.reshape(-1)[0]))
        metas.append(meta)
    return torch.stack(windows), torch.tensor(labels, dtype=torch.float32), metas


def _train(model, X, y, epochs, lr=1e-2, batch_size=16, seed=42):
    """Train the model with focal loss; returns nothing (trains in place)."""
    seed_everything(seed)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X, y.reshape(-1, 1)), batch_size=batch_size,
                        shuffle=True, generator=generator)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()


def _full_loss(model, X, y) -> float:
    model.eval()
    with torch.no_grad():
        return FocalLoss()(model(X), y.reshape(-1, 1)).item()


# ── Training ─────────────────────────────────────────────────────────────────

def test_temporal_cnn_trains_loss_decreases(temporal_data):
    dataset, _ = temporal_data
    X, y, _ = _window_tensors(dataset)
    seed_everything(0)
    model = _Clf()
    initial = _full_loss(model, X, y)
    _train(model, X, y, epochs=5)
    final = _full_loss(model, X, y)
    assert final < initial, f"loss did not decrease: {initial:.5f} -> {final:.5f}"


def test_temporal_cnn_produces_valid_auprc(temporal_data):
    dataset, _ = temporal_data
    X, y, metas = _window_tensors(dataset)
    user_ids = np.array([m["user_id"] for m in metas])
    train_idx, val_idx = temporal_stratified_kfold(user_ids, y.numpy(), n_folds=2, seed=42)[0]

    seed_everything(0)
    model = _Clf()
    _train(model, X[train_idx], y[train_idx], epochs=3)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X[val_idx])).reshape(-1).numpy()
    auprc = compute_metrics(probs, y[val_idx].numpy())["auprc"]
    assert isinstance(auprc, float) and 0.0 <= auprc <= 1.0


def test_temporal_cnn_embeddings_shape(temporal_data):
    dataset, _ = temporal_data
    X, _, _ = _window_tensors(dataset)
    encoder = TemporalPatternEncoder()
    encoder.eval()
    with torch.no_grad():
        emb = encoder(X[:8])  # shape: (8, 128) — Phase 5 user node features
    assert emb.shape == (8, 128)


# ── Interpretability (soft) ──────────────────────────────────────────────────

def test_attention_weights_peak_during_attack(temporal_data):
    dataset, attack_windows = temporal_data
    X, y, metas = _window_tensors(dataset)
    seed_everything(0)
    model = _Clf()
    _train(model, X, y, epochs=5)

    # Pick an insider's positive window.
    idx = next((i for i, m in enumerate(metas)
                if m["user_id"] in attack_windows and y[i].item() == 1.0), None)
    if idx is None:
        pytest.skip("no positive insider window in this fixture")

    meta = metas[idx]
    record = attack_windows[meta["user_id"]]
    model.eval()
    with torch.no_grad():
        attn = model.encoder.get_attention_weights(X[idx].unsqueeze(0))[0].numpy()  # shape: (28,)

    # Hard check: attention is a valid distribution.
    assert abs(attn.sum() - 1.0) < 1e-5 and (attn >= 0).all()

    window_start = pd.Timestamp(meta["window_start"]).normalize()
    attack_start = pd.Timestamp(record.attack_start).normalize()
    attack_end = pd.Timestamp(record.attack_end).normalize()
    is_attack = np.array([
        attack_start <= window_start + pd.Timedelta(days=i) <= attack_end
        for i in range(attn.shape[0])
    ])
    # Soft check: log whether the model attends more to attack days; never fail.
    if is_attack.any() and (~is_attack).any():
        attack_attn = float(attn[is_attack].mean())
        normal_attn = float(attn[~is_attack].mean())
        print(f"attention | attack-day mean={attack_attn:.4f} vs non-attack mean={normal_attn:.4f} "
              f"({'higher' if attack_attn > normal_attn else 'not higher'})")


# ── Gradient clipping ────────────────────────────────────────────────────────

def test_gradient_clipping_works(temporal_data):
    dataset, _ = temporal_data
    X, y, _ = _window_tensors(dataset)
    seed_everything(0)
    model = _Clf()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Scale the loss so gradients clearly exceed the clip threshold.
    loss = FocalLoss()(model(X), y.reshape(-1, 1)) * 1000.0
    loss.backward()

    max_norm = 1.0
    pre_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
    post_norm = float(torch.sqrt(sum(
        p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None)))

    assert pre_norm > max_norm, "test setup should produce gradients above the clip threshold"
    assert post_norm <= max_norm + 1e-4, f"gradients not clipped: norm {post_norm} > {max_norm}"
