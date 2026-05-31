import os
import pytest
import torch
from innersight.backend.training.trainer import train


# ── smoke test ────────────────────────────────────────────────────────────────

def test_train_smoke(tmp_path, monkeypatch, synthetic_loaders):
    """train() completes for 2 epochs on synthetic data, emits events, saves checkpoints."""
    import innersight.backend.training.trainer as trainer_mod

    # Redirect all file I/O to tmp_path
    monkeypatch.setattr(trainer_mod, '_BEST_MODEL_PT_PATH', str(tmp_path / 'model.pt'))
    monkeypatch.setattr(trainer_mod, '_BEST_MODEL_PATH',    str(tmp_path / 'model.npz'))
    monkeypatch.setattr(trainer_mod, '_PREPROCESSOR_PATH',  str(tmp_path / 'prep.npz'))
    monkeypatch.setattr(trainer_mod, '_STANDARDIZER_PATH',  str(tmp_path / 'std.pt'))

    # Bypass data loading and feature engineering
    monkeypatch.setattr(trainer_mod, 'load_data',        lambda: {})
    monkeypatch.setattr(trainer_mod, 'build_dataloaders',
                        lambda data, batch_size, embedding_manager=None: synthetic_loaders)

    config = {
        'epochs':      2,
        'batch_size':  8,
        'lr':          0.001,
        'layer_sizes': [18, 8, 1],   # tiny — fast
        'pos_weight':  50.0,
        'patience':    5,
    }

    events = []
    result = train(config, event_callback=events.append)

    # Checkpoints written
    assert os.path.exists(str(tmp_path / 'model.pt')),   "PyTorch checkpoint missing"
    assert os.path.exists(str(tmp_path / 'model.npz')),  "Numpy compat checkpoint missing"
    assert os.path.exists(str(tmp_path / 'prep.npz')),   "Numpy preprocessor missing"
    assert os.path.exists(str(tmp_path / 'std.pt')),     "PyTorch standardizer missing"

    # Event stream was emitted
    assert any('epoch' in e for e in events),            "No epoch events emitted"
    assert any(e.get('status') == 'done' for e in events), "No 'done' event emitted"

    # Result dict has the expected keys
    for key in ('best_val_f1', 'test_loss', 'test_precision', 'test_recall', 'test_f1'):
        assert key in result, f"Missing key: {key}"


def test_train_class_imbalance_event(tmp_path, monkeypatch, synthetic_loaders):
    """The first event emitted must contain class_imbalance info."""
    import innersight.backend.training.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, '_BEST_MODEL_PT_PATH', str(tmp_path / 'model.pt'))
    monkeypatch.setattr(trainer_mod, '_BEST_MODEL_PATH',    str(tmp_path / 'model.npz'))
    monkeypatch.setattr(trainer_mod, '_PREPROCESSOR_PATH',  str(tmp_path / 'prep.npz'))
    monkeypatch.setattr(trainer_mod, '_STANDARDIZER_PATH',  str(tmp_path / 'std.pt'))
    monkeypatch.setattr(trainer_mod, 'load_data',        lambda: {})
    monkeypatch.setattr(trainer_mod, 'build_dataloaders',
                        lambda data, batch_size, embedding_manager=None: synthetic_loaders)

    events = []
    train({'epochs': 1, 'batch_size': 8, 'lr': 1e-3,
           'layer_sizes': [18, 4, 1], 'pos_weight': 50.0, 'patience': 5},
          event_callback=events.append)

    assert events[0].get('class_imbalance') is not None
    ratio = events[0]['class_imbalance']['ratio']
    assert 0.0 <= ratio <= 1.0


def test_train_epoch_events_match_config(tmp_path, monkeypatch, synthetic_loaders):
    """Epoch field in val events should not exceed the configured epochs."""
    import innersight.backend.training.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, '_BEST_MODEL_PT_PATH', str(tmp_path / 'model.pt'))
    monkeypatch.setattr(trainer_mod, '_BEST_MODEL_PATH',    str(tmp_path / 'model.npz'))
    monkeypatch.setattr(trainer_mod, '_PREPROCESSOR_PATH',  str(tmp_path / 'prep.npz'))
    monkeypatch.setattr(trainer_mod, '_STANDARDIZER_PATH',  str(tmp_path / 'std.pt'))
    monkeypatch.setattr(trainer_mod, 'load_data',        lambda: {})
    monkeypatch.setattr(trainer_mod, 'build_dataloaders',
                        lambda data, batch_size, embedding_manager=None: synthetic_loaders)

    n_epochs = 3
    events   = []
    train({'epochs': n_epochs, 'batch_size': 8, 'lr': 1e-3,
           'layer_sizes': [18, 4, 1], 'pos_weight': 50.0, 'patience': 10},
          event_callback=events.append)

    val_events = [e for e in events if 'val_f1' in e]
    assert len(val_events) == n_epochs
    assert all(e['epoch'] <= n_epochs for e in val_events)


def test_train_checkpoint_is_loadable(tmp_path, monkeypatch, synthetic_loaders):
    """Saved .pt checkpoint must be loadable as a valid state dict."""
    import innersight.backend.training.trainer as trainer_mod
    from innersight.backend.models.mlp import InsiderThreatMLP

    pt_path = str(tmp_path / 'model.pt')
    monkeypatch.setattr(trainer_mod, '_BEST_MODEL_PT_PATH', pt_path)
    monkeypatch.setattr(trainer_mod, '_BEST_MODEL_PATH',    str(tmp_path / 'model.npz'))
    monkeypatch.setattr(trainer_mod, '_PREPROCESSOR_PATH',  str(tmp_path / 'prep.npz'))
    monkeypatch.setattr(trainer_mod, '_STANDARDIZER_PATH',  str(tmp_path / 'std.pt'))
    monkeypatch.setattr(trainer_mod, 'load_data',        lambda: {})
    monkeypatch.setattr(trainer_mod, 'build_dataloaders',
                        lambda data, batch_size, embedding_manager=None: synthetic_loaders)

    train({'epochs': 1, 'batch_size': 8, 'lr': 1e-3,
           'layer_sizes': [18, 8, 1], 'pos_weight': 50.0, 'patience': 5})

    ckpt  = torch.load(pt_path, map_location='cpu', weights_only=True)
    model = InsiderThreatMLP(ckpt['layer_sizes'])
    model.load_state_dict(ckpt['state_dict'])       # should not raise
    out   = model(torch.randn(4, 18))
    assert out.shape == (4, 1)


def test_train_early_stopping(tmp_path, monkeypatch, synthetic_loaders):
    """With patience=1 and many epochs, training stops before reaching max epochs."""
    import innersight.backend.training.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, '_BEST_MODEL_PT_PATH', str(tmp_path / 'model.pt'))
    monkeypatch.setattr(trainer_mod, '_BEST_MODEL_PATH',    str(tmp_path / 'model.npz'))
    monkeypatch.setattr(trainer_mod, '_PREPROCESSOR_PATH',  str(tmp_path / 'prep.npz'))
    monkeypatch.setattr(trainer_mod, '_STANDARDIZER_PATH',  str(tmp_path / 'std.pt'))
    monkeypatch.setattr(trainer_mod, 'load_data',        lambda: {})
    monkeypatch.setattr(trainer_mod, 'build_dataloaders',
                        lambda data, batch_size, embedding_manager=None: synthetic_loaders)

    events = []
    train({'epochs': 50, 'batch_size': 8, 'lr': 1e-3,
           'layer_sizes': [18, 4, 1], 'pos_weight': 50.0, 'patience': 1},
          event_callback=events.append)

    val_events = [e for e in events if 'val_f1' in e]
    assert len(val_events) < 50, "Early stopping should have fired before 50 epochs"
