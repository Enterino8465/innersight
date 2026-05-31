"""Full pipeline integration test: synthetic CSVs → train → score.

Marked @pytest.mark.slow — excluded from the default pytest run.
Run explicitly with:  pytest -m slow  (or  pytest tests/test_integration.py)
"""

import json
import os
import datetime
import pytest
import pandas as pd
import torch

from innersight.backend.data.pipeline import load_data


# ── Synthetic dataset fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synth_data_dir(tmp_path_factory):
    """Create a minimal CERT-like CSV dataset in a temp directory.

    Layout mirrors what pipeline.load_data() expects:
      <root>/logon.csv
      <root>/answers/answers.csv   (one malicious event)

    Returns (data_dir: str, test_date: str)
    """
    root = tmp_path_factory.mktemp("cert_synth")
    answers_dir = root / "answers"
    answers_dir.mkdir()

    users = [f"u{i:03d}" for i in range(8)]

    rows = []
    # train: 2010-06-01 → 2010-09-30  (weekly, ~23 dates × 8 users = 184 rows)
    # val:   2010-10-01 → 2010-11-30  (weekly, ~9 dates × 8 users = 72 rows)
    # test:  2010-12-01 → 2011-01-31  (weekly, ~8 dates × 8 users = 64 rows)
    for date in pd.date_range("2010-06-01", "2011-01-31", freq="7D"):
        for u in users:
            rows.append({
                "id":       f"L{len(rows)}",
                "date":     date.strftime("%Y-%m-%d %H:%M:%S"),
                "user":     u,
                "pc":       f"PC{users.index(u) % 3}",
                "activity": "Logon",
            })
            # Add a logoff for variety
            rows.append({
                "id":       f"L{len(rows)}",
                "date":     (date + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                "user":     u,
                "pc":       f"PC{users.index(u) % 3}",
                "activity": "Logoff",
            })

    pd.DataFrame(rows).to_csv(root / "logon.csv", index=False)

    # One malicious event in the training window
    pd.DataFrame([{"user": "u000", "date": "2010-07-05"}]).to_csv(
        answers_dir / "answers.csv", index=False
    )

    # First Tuesday of the test split (dates are weekly Tuesdays from 2010-06-01)
    test_date = "2010-12-07"
    return str(root), test_date


# ── Helpers ───────────────────────────────────────────────────────────────────

def _redirect_trainer(monkeypatch, tmp_path):
    """Patch trainer module paths to write into tmp_path."""
    import innersight.backend.training.trainer as t
    monkeypatch.setattr(t, "_BEST_MODEL_PT_PATH", str(tmp_path / "model.pt"))
    monkeypatch.setattr(t, "_STANDARDIZER_PATH",  str(tmp_path / "std.pt"))


def _redirect_scoring(monkeypatch, tmp_path):
    """Patch scoring module paths to read from tmp_path / write alerts there."""
    import innersight.backend.scoring.scoring as s
    monkeypatch.setattr(s, "_BEST_MODEL_PT_PATH", str(tmp_path / "model.pt"))
    monkeypatch.setattr(s, "_STANDARDIZER_PATH",  str(tmp_path / "std.pt"))
    monkeypatch.setattr(s, "_ALERTS_PATH",        str(tmp_path / "alerts.json"))


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_load_data_with_synthetic_csvs(synth_data_dir):
    """load_data() must succeed with only logon.csv present."""
    data_dir, _ = synth_data_dir
    data = load_data(data_dir)

    assert "splits" in data and "labels" in data
    assert set(data["splits"].keys()) == {"train", "val", "test"}
    assert isinstance(data["labels"], set)

    train_logon = data["splits"]["train"]["logon"]
    assert len(train_logon) > 0, "Training split should have rows"
    assert "u000" in train_logon["user"].values

    print(f"\n  [load_data] train={len(train_logon)} rows  "
          f"labels={len(data['labels'])}")


@pytest.mark.slow
def test_train_full_pipeline(tmp_path, monkeypatch, synth_data_dir):
    """train() completes on synthetic data and writes all expected checkpoints."""
    import innersight.backend.training.trainer as trainer_mod

    data_dir, _ = synth_data_dir

    _redirect_trainer(monkeypatch, tmp_path)
    monkeypatch.setattr(trainer_mod, "load_data", lambda: load_data(data_dir))

    config = {
        "epochs":      3,
        "batch_size":  32,
        "lr":          0.001,
        "layer_sizes": [18, 32, 1],   # small for speed
        "pos_weight":  50.0,
        "patience":    2,
    }

    events = []
    result = trainer_mod.train(config, event_callback=events.append)

    # Checkpoint files must exist
    for fname in ("model.pt", "std.pt"):
        path = str(tmp_path / fname)
        assert os.path.exists(path), f"Missing checkpoint: {fname}"
        assert os.path.getsize(path) > 0, f"Empty checkpoint: {fname}"

    # Event stream correctness
    assert any("epoch" in e for e in events), "No epoch events emitted"
    done_events = [e for e in events if e.get("status") == "done"]
    assert done_events, "No 'done' event emitted"
    cm = done_events[0]["confusion_matrix"]
    assert isinstance(cm, list) and len(cm) == 2, "confusion_matrix must be 2×2"

    # Result keys
    for key in ("best_val_f1", "test_loss", "test_precision", "test_recall", "test_f1"):
        assert key in result, f"Missing result key: {key}"
        assert isinstance(result[key], float)

    print(f"\n  [train] epochs_run={len([e for e in events if 'val_f1' in e])}  "
          f"best_val_f1={result['best_val_f1']}  test_f1={result['test_f1']}")


@pytest.mark.slow
def test_checkpoint_is_valid_pytorch(tmp_path, monkeypatch, synth_data_dir):
    """Saved .pt checkpoint loads cleanly as InsiderThreatMLP state dict."""
    import innersight.backend.training.trainer as trainer_mod
    from innersight.backend.models.mlp import InsiderThreatMLP

    data_dir, _ = synth_data_dir
    _redirect_trainer(monkeypatch, tmp_path)
    monkeypatch.setattr(trainer_mod, "load_data", lambda: load_data(data_dir))

    trainer_mod.train(
        {"epochs": 1, "batch_size": 32, "lr": 1e-3,
         "layer_sizes": [18, 16, 1], "pos_weight": 50.0, "patience": 5}
    )

    ckpt = torch.load(str(tmp_path / "model.pt"), map_location="cpu", weights_only=True)
    m    = InsiderThreatMLP(ckpt["layer_sizes"])
    m.load_state_dict(ckpt["state_dict"])
    out = m(torch.randn(4, 18))
    assert out.shape == (4, 1)


@pytest.mark.slow
def test_score_employees_after_training(tmp_path, monkeypatch, synth_data_dir):
    """score_employees() runs after training and returns valid alert dicts."""
    import innersight.backend.training.trainer as trainer_mod
    import innersight.backend.scoring.scoring as scoring_mod

    data_dir, test_date = synth_data_dir

    _redirect_trainer(monkeypatch, tmp_path)
    _redirect_scoring(monkeypatch, tmp_path)
    monkeypatch.setattr(trainer_mod, "load_data", lambda: load_data(data_dir))
    monkeypatch.setattr(scoring_mod,  "load_data", lambda: load_data(data_dir))

    # Train first so the checkpoint exists
    trainer_mod.train(
        {"epochs": 2, "batch_size": 32, "lr": 1e-3,
         "layer_sizes": [18, 16, 1], "pos_weight": 50.0, "patience": 5}
    )

    # Score with threshold=0 to guarantee alerts for any active user
    alerts = scoring_mod.score_employees(test_date, threshold=0.0)

    assert isinstance(alerts, list), "score_employees must return a list"
    assert len(alerts) > 0, (
        f"Expected at least one alert for active users on {test_date} with threshold=0"
    )

    # Verify alert schema
    required_keys = {"id", "user", "date", "score", "status", "created_at", "top_features"}
    for alert in alerts:
        missing = required_keys - set(alert.keys())
        assert not missing, f"Alert missing keys: {missing}"
        assert isinstance(alert["score"], float)
        assert 0.0 <= alert["score"] <= 1.0
        assert alert["status"] == "open"
        assert isinstance(alert["top_features"], list)

    # Alerts must have been persisted
    assert os.path.exists(str(tmp_path / "alerts.json"))
    with open(str(tmp_path / "alerts.json")) as f:
        persisted = json.load(f)
    assert len(persisted) == len(alerts)

    print(f"\n  [score_employees] date={test_date}  alerts={len(alerts)}")
    for a in alerts[:3]:
        print(f"    user={a['user']}  score={a['score']:.4f}  "
              f"top={a['top_features']}")


@pytest.mark.slow
def test_load_alerts_after_scoring(tmp_path, monkeypatch, synth_data_dir):
    """load_alerts() and update_alert_status() work on the persisted results."""
    import innersight.backend.training.trainer as trainer_mod
    import innersight.backend.scoring.scoring as scoring_mod

    data_dir, test_date = synth_data_dir

    _redirect_trainer(monkeypatch, tmp_path)
    _redirect_scoring(monkeypatch, tmp_path)
    monkeypatch.setattr(trainer_mod, "load_data", lambda: load_data(data_dir))
    monkeypatch.setattr(scoring_mod,  "load_data", lambda: load_data(data_dir))

    trainer_mod.train(
        {"epochs": 1, "batch_size": 32, "lr": 1e-3,
         "layer_sizes": [18, 16, 1], "pos_weight": 50.0, "patience": 5}
    )
    alerts = scoring_mod.score_employees(test_date, threshold=0.0)
    assert alerts

    loaded = scoring_mod.load_alerts()
    assert loaded == sorted(loaded, key=lambda a: a["score"], reverse=True)

    # Status update
    first_id = loaded[0]["id"]
    updated  = scoring_mod.update_alert_status(first_id, "muted")
    assert updated["status"] == "muted"
    assert scoring_mod.load_alerts(status_filter="muted")[0]["id"] == first_id
