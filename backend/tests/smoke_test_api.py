#!/usr/bin/env python3
"""
API smoke test — uses Flask test_client(), no live server needed.

Run standalone:  python tests/smoke_test_api.py
Run via pytest:  pytest tests/smoke_test_api.py -v
"""
from __future__ import annotations

import json
import os
import queue as queue_module
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_fake_checkpoints(tmp_dir: str, layer_sizes: list[int] | None = None) -> None:
    from innersight.backend.models.mlp import InsiderThreatMLP
    from innersight.backend.models.dataset import Standardizer
    if layer_sizes is None:
        from innersight.backend.config import DEFAULT_TRAINING_CONFIG
        layer_sizes = DEFAULT_TRAINING_CONFIG["layer_sizes"]
    model = InsiderThreatMLP(layer_sizes)
    torch.save(
        {"state_dict": model.state_dict(), "layer_sizes": layer_sizes},
        os.path.join(tmp_dir, "model.pt"),
    )
    std = Standardizer()
    std.fit(torch.randn(64, layer_sizes[0]))
    std.save(os.path.join(tmp_dir, "std.pt"))


def _write_fake_alerts(tmp_dir: str) -> None:
    import datetime
    alerts = [
        {
            "id": "smoke-alert-001",
            "user": "u000",
            "date": "2010-12-06",
            "score": 0.85,
            "status": "open",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "top_features": ["logon_count", "after_hours_rate"],
        }
    ]
    with open(os.path.join(tmp_dir, "alerts.json"), "w") as f:
        json.dump(alerts, f)


def _patch_api(tmp_dir: str) -> None:
    """Redirect all file-path constants in api and scoring modules to tmp_dir."""
    import innersight.backend.api as api_mod
    import innersight.backend.scoring.scoring as scoring_mod
    api_mod._BEST_MODEL_PT_PATH       = os.path.join(tmp_dir, "model.pt")
    api_mod._STANDARDIZER_PATH        = os.path.join(tmp_dir, "std.pt")
    api_mod._model_cache              = None
    api_mod._score_history_cache      = {}
    scoring_mod._BEST_MODEL_PT_PATH   = os.path.join(tmp_dir, "model.pt")
    scoring_mod._STANDARDIZER_PATH    = os.path.join(tmp_dir, "std.pt")
    scoring_mod._ALERTS_PATH          = os.path.join(tmp_dir, "alerts.json")


def _make_fast_train(tmp_dir: str):
    """Return a mock train() that writes valid checkpoints and emits events instantly."""
    def _fast(config, event_callback=None):
        layer_sizes = config.get("layer_sizes")
        _write_fake_checkpoints(tmp_dir, layer_sizes)
        if event_callback:
            event_callback({
                "epoch": 1, "val_f1": 0.5, "val_loss": 0.3, "train_loss": 0.4,
            })
            event_callback({
                "status": "done",
                "confusion_matrix": [[10, 0], [0, 1]],
            })
        return {
            "best_val_f1": 0.5, "test_loss": 0.3,
            "test_precision": 0.5, "test_recall": 0.5, "test_f1": 0.5,
        }
    return _fast


def _drain_events(api_mod, timeout: float = 30.0) -> dict | None:
    """Drain api_mod._event_queue until a 'done' or 'error' event arrives."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            event = api_mod._event_queue.get(timeout=0.5)
            if event.get("status") in ("done", "error"):
                return event
        except queue_module.Empty:
            continue
    return None


def _clear_event_queue(api_mod) -> None:
    while not api_mod._event_queue.empty():
        try:
            api_mod._event_queue.get_nowait()
        except queue_module.Empty:
            break


# ── pytest fixture ────────────────────────────────────────────────────────────

@pytest.fixture()
def smoke_setup(tmp_path):
    """Yields (client, tmp_dir) with paths patched and fake data written."""
    import innersight.backend.api as api_mod

    tmp_dir = str(tmp_path)
    _write_fake_checkpoints(tmp_dir)
    _write_fake_alerts(tmp_dir)
    _patch_api(tmp_dir)

    yield api_mod.app.test_client(), tmp_dir


# ── individual tests (usable by both pytest and __main__) ─────────────────────

def test_status(smoke_setup):
    client, _ = smoke_setup
    rv = client.get("/api/status")
    assert rv.status_code == 200, f"Expected 200, got {rv.status_code}"
    assert rv.get_json().get("status") == "idle"


def test_config(smoke_setup):
    client, _ = smoke_setup
    rv = client.get("/api/config")
    assert rv.status_code == 200, f"Expected 200, got {rv.status_code}"
    body = rv.get_json()
    assert "layer_sizes" in body
    assert isinstance(body["layer_sizes"], list)
    assert body["layer_sizes"][-1] == 1


def test_alerts(smoke_setup):
    client, _ = smoke_setup
    rv = client.get("/api/alerts")
    assert rv.status_code == 200, f"Expected 200, got {rv.status_code}"
    data = rv.get_json()
    assert isinstance(data, list), f"Expected list, got {type(data)}"


def test_employees(smoke_setup):
    client, _ = smoke_setup
    rv = client.get("/api/employees")
    assert rv.status_code == 200, f"Expected 200, got {rv.status_code}"
    data = rv.get_json()
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    assert len(data) > 0, "Expected at least one employee (alerts were seeded)"
    required = {"user", "department", "latest_score", "alert_count", "status"}
    missing = required - set(data[0].keys())
    assert not missing, f"Employee missing keys: {missing}"


def test_train_invalid_config(smoke_setup):
    client, _ = smoke_setup
    rv = client.post("/api/train", json={"epochs": -1})
    assert rv.status_code == 400, f"Expected 400, got {rv.status_code}"
    assert "error" in rv.get_json()


def test_train_starts_and_completes(tmp_path):
    """POST /api/train → async → drain SSE → confirm done event received."""
    import innersight.backend.api as api_mod

    tmp_dir = str(tmp_path)
    _write_fake_checkpoints(tmp_dir)
    _write_fake_alerts(tmp_dir)
    _patch_api(tmp_dir)
    _clear_event_queue(api_mod)

    # Patch run_train (the name api.py imports train as)
    original_run_train = api_mod.run_train
    api_mod.run_train   = _make_fast_train(tmp_dir)

    client = api_mod.app.test_client()
    try:
        config = {
            "epochs": 1, "batch_size": 8, "lr": 0.001,
            "layer_sizes": [18, 8, 1], "pos_weight": 50.0, "patience": 5,
        }
        rv = client.post("/api/train", json=config)
        assert rv.status_code == 200, f"Expected 200, got {rv.status_code}: {rv.data}"
        assert rv.get_json().get("status") == "started"

        done = _drain_events(api_mod, timeout=30)
        assert done is not None, "Training did not emit a 'done' event within 30 s"
        assert done.get("status") == "done", f"Unexpected terminal event: {done}"
    finally:
        api_mod.run_train = original_run_train


# ── standalone __main__ ───────────────────────────────────────────────────────

def _run_standalone() -> bool:
    results: list[tuple[str, bool, str]] = []

    def _check(label: str, fn) -> None:
        try:
            fn()
            results.append((label, True, ""))
            print(f"  [PASS] {label}")
        except Exception as exc:
            results.append((label, False, str(exc)))
            print(f"  [FAIL] {label} — {exc}")

    print("InnerSight API smoke test")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as d:
        import innersight.backend.api as api_mod

        _write_fake_checkpoints(d)
        _write_fake_alerts(d)
        _patch_api(d)
        client = api_mod.app.test_client()
        setup = (client, d)

        _check("GET /api/status",              lambda: test_status(setup))
        _check("GET /api/config",              lambda: test_config(setup))
        _check("GET /api/alerts",              lambda: test_alerts(setup))
        _check("GET /api/employees",           lambda: test_employees(setup))
        _check("POST /api/train (bad config)", lambda: test_train_invalid_config(setup))

    with tempfile.TemporaryDirectory() as d2:
        _check("POST /api/train (async done)",
               lambda: test_train_starts_and_completes(Path(d2)))

    print()
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    if failed:
        print(f"RESULT: {passed}/{total} passed, {failed} FAILED")
    else:
        print(f"RESULT: {passed}/{total} passed — all green")
    return failed == 0


if __name__ == "__main__":
    ok = _run_standalone()
    sys.exit(0 if ok else 1)
