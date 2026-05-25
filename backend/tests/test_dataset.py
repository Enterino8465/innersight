import pytest
import pandas as pd
import torch
from innersight.backend.models.dataset import Standardizer, build_features_tensor, build_dataloaders
from innersight.backend.config import FEATURE_COLS

N_FEAT = len(FEATURE_COLS)


# ── Standardizer.fit / transform ──────────────────────────────────────────────

def test_standardizer_fit_sets_mean_and_std():
    scaler = Standardizer()
    X      = torch.randn(200, N_FEAT)
    scaler.fit(X)
    assert scaler.mean is not None
    assert scaler.std  is not None
    assert scaler.mean.shape == (N_FEAT,)
    assert scaler.std.shape  == (N_FEAT,)


def test_standardizer_fit_transform_near_zero_mean():
    torch.manual_seed(0)
    X      = torch.randn(1000, N_FEAT) * 5 + 3
    scaler = Standardizer()
    X_std  = scaler.fit_transform(X)
    assert X_std.mean(dim=0).abs().max().item() < 1e-5


def test_standardizer_fit_transform_near_unit_std():
    torch.manual_seed(0)
    X      = torch.randn(1000, N_FEAT) * 5 + 3
    scaler = Standardizer()
    X_std  = scaler.fit_transform(X)
    # std = (X.std + eps) / (X.std + eps) ≈ 1; allow small tolerance from eps
    assert (X_std.std(dim=0) - 1).abs().max().item() < 0.01


def test_standardizer_transform_preserves_shape():
    scaler = Standardizer()
    scaler.fit(torch.randn(100, N_FEAT))
    X   = torch.randn(7, N_FEAT)
    out = scaler.transform(X)
    assert out.shape == X.shape


def test_standardizer_unfitted_raises():
    with pytest.raises(RuntimeError, match="fitted"):
        Standardizer().transform(torch.randn(4, N_FEAT))


def test_standardizer_eps_prevents_division_by_zero():
    """Constant feature column should not produce NaN or Inf."""
    X          = torch.randn(100, N_FEAT)
    X[:, 0]    = 5.0  # constant column → std = 0
    scaler     = Standardizer()
    X_std      = scaler.fit_transform(X)
    assert torch.isfinite(X_std).all()


# ── Standardizer save / load ──────────────────────────────────────────────────

def test_standardizer_save_load_round_trip(tmp_path):
    torch.manual_seed(1)
    X      = torch.randn(200, N_FEAT)
    scaler = Standardizer()
    scaler.fit(X)

    path   = str(tmp_path / 'std.pt')
    scaler.save(path)
    loaded = Standardizer.load(path)

    torch.testing.assert_close(loaded.mean, scaler.mean)
    torch.testing.assert_close(loaded.std,  scaler.std)


def test_standardizer_load_produces_identical_transform(tmp_path):
    torch.manual_seed(2)
    X      = torch.randn(200, N_FEAT)
    scaler = Standardizer()
    scaler.fit(X)

    path   = str(tmp_path / 'std.pt')
    scaler.save(path)
    loaded = Standardizer.load(path)

    X_new  = torch.randn(50, N_FEAT)
    torch.testing.assert_close(scaler.transform(X_new), loaded.transform(X_new))


def test_standardizer_save_unfitted_raises(tmp_path):
    with pytest.raises(RuntimeError):
        Standardizer().save(str(tmp_path / 'std.pt'))


# ── build_features_tensor ─────────────────────────────────────────────────────

def _logon_df(n_rows=3):
    return pd.DataFrame({
        'id':       [f'L{i}' for i in range(n_rows)],
        'date':     pd.to_datetime(['2010-06-01 09:00', '2010-06-01 17:00',
                                    '2010-06-02 10:00'])[:n_rows],
        'user':     ['u1', 'u1', 'u2'][:n_rows],
        'pc':       ['PC1', 'PC1', 'PC2'][:n_rows],
        'activity': ['Logon', 'Logoff', 'Logon'][:n_rows],
    })


def test_build_features_tensor_shapes():
    """Minimal logon data → correct tensor shapes, dtypes, and user_ids list."""
    X, y, user_ids = build_features_tensor({'logon': _logon_df()}, labels=set())

    assert X.dtype == torch.float32
    assert y.dtype == torch.float32
    assert X.shape[1] == N_FEAT
    assert y.shape[1] == 1
    assert X.shape[0] == y.shape[0]
    assert len(user_ids) == X.shape[0]


def test_build_features_tensor_label_dtype():
    X, y, _ = build_features_tensor({'logon': _logon_df(1)}, labels=set())
    assert y.dtype == torch.float32


def test_build_features_tensor_malicious_label():
    """A known-malicious (user, date) tuple should produce y=1."""
    import datetime
    labels = {('u1', datetime.date(2010, 6, 1))}
    X, y, _ = build_features_tensor({'logon': _logon_df()}, labels=labels)
    assert y[0, 0].item() == 1.0


# ── build_dataloaders with EmbeddingManager ───────────────────────────────────

def test_build_dataloaders_with_embeddings():
    """Passing an active EmbeddingManager doubles the feature width per batch."""
    from torch.utils.data import TensorDataset, DataLoader
    from innersight.backend.models.embeddings import EmbeddingManager

    EMB_DIM = 32
    USERS   = ['u1', 'u2', 'u3']

    # Build a tiny synthetic EmbeddingManager without touching disk
    mgr = EmbeddingManager.__new__(EmbeddingManager)
    mgr.embeddings    = torch.randn(len(USERS), EMB_DIM)
    mgr.user_to_idx   = {u: i for i, u in enumerate(USERS)}
    mgr.embedding_dim = EMB_DIM
    mgr.available     = True

    # Fake data dict matching build_dataloaders' contract
    logon_df = pd.DataFrame({
        'id':       ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'],
        'date':     pd.to_datetime(['2010-06-01'] * 3 + ['2010-06-02'] * 3),
        'user':     ['u1', 'u2', 'u3', 'u1', 'u2', 'u3'],
        'pc':       ['PC1'] * 6,
        'activity': ['Logon'] * 6,
    })
    data = {
        'labels': set(),
        'splits': {
            'train': {'logon': logon_df},
            'val':   {'logon': logon_df},
            'test':  {'logon': logon_df},
        },
    }

    loaders = build_dataloaders(data, batch_size=4, embedding_manager=mgr)
    X_batch, _ = next(iter(loaders['train_loader']))

    assert X_batch.shape[1] == N_FEAT + EMB_DIM, (
        f"Expected {N_FEAT + EMB_DIM} features, got {X_batch.shape[1]}"
    )
