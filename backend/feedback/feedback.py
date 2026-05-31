"""Feedback actions: online model correction, muting, and blocking alerts."""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from innersight.backend.b2_data.pipeline import load_data
from innersight.backend.b2_features.features import build_user_day_features
from innersight.backend.b8_scoring.scoring import update_alert_status, _read_alerts_file
from innersight.backend.config import (
    ALERTS_FILE       as _ALERTS_PATH,
    CORRECTIONS_FILE  as _CORRECTIONS_PATH,
    BLOCK_LOG_FILE    as _BLOCK_LOG_PATH,
    BEST_MODEL_PT_FILE  as _BEST_MODEL_PT_PATH,
    BEST_MODEL_FILE     as _BEST_MODEL_PATH,    # .npz — kept for load_best_model() compat
    STANDARDIZER_FILE   as _STANDARDIZER_PATH,
    FEATURE_COLS      as _FEATURE_COLS,
    CORRECTION_LR,
    DEFAULT_TRAINING_CONFIG,
)
from innersight.backend.models.mlp import InsiderThreatMLP, get_device
from innersight.backend.models.dataset import Standardizer
from innersight.backend.utils import safe_json_read, safe_json_write

# Number of gradient steps taken per online correction
_CORRECTION_STEPS = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_learn(alert_id: str) -> dict:
    """Mark an alert as a false positive and fine-tune the model accordingly.

    Appends a correction record (label=0) to the corrections log, loads the
    current PyTorch checkpoint, runs a few gradient steps to push the model
    toward predicting "normal" for this user-day, saves both the ``.pt`` and
    legacy ``.npz`` checkpoints, then sets the alert status to ``'learned'``.

    Args:
        alert_id: UUID of the alert to correct.

    Returns:
        The updated alert dict.
    """
    # ── 1. Find alert ─────────────────────────────────────────────────────────
    alert = _get_alert(alert_id)

    # ── 2. Persist correction (label=0 → model should predict "normal") ───────
    corrections = _read_json(_CORRECTIONS_PATH)
    correction  = {
        'alert_id':   alert_id,
        'user':       alert['user'],
        'date':       alert['date'],
        'label':      0,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    corrections.append(correction)
    _write_json(_CORRECTIONS_PATH, corrections)

    # ── 3. Extract feature tensor for this correction ─────────────────────────
    X_raw = _features_for_correction(correction)

    if X_raw is not None:
        # ── 4. Load model and standardizer ────────────────────────────────────
        device = get_device()
        try:
            checkpoint  = torch.load(
                _BEST_MODEL_PT_PATH, map_location=device, weights_only=True
            )
            layer_sizes = checkpoint.get("layer_sizes", DEFAULT_TRAINING_CONFIG["layer_sizes"])
            model       = InsiderThreatMLP(layer_sizes)
            model.load_state_dict(checkpoint["state_dict"])
        except FileNotFoundError:
            logger.warning(
                'apply_learn | no checkpoint at %s — skipping weight update',
                _BEST_MODEL_PT_PATH,
            )
            return update_alert_status(alert_id, 'learned')

        model.to(device)

        standardizer = Standardizer.load(_STANDARDIZER_PATH)

        # ── 5. Standardize and build label tensor ──────────────────────────────
        X_std = standardizer.transform(X_raw).to(device)
        y_t   = torch.tensor(
            [[float(correction['label'])]], dtype=torch.float32, device=device
        )

        # ── 6. Fine-tune: a few gradient steps on this single sample ──────────
        # pos_weight=1.0 — no class weighting for corrections; we just want to
        # nudge the model toward normal for this specific user-day.
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0], dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=CORRECTION_LR)
        model.train()
        final_loss = 0.0
        for _ in range(_CORRECTION_STEPS):
            optimizer.zero_grad()
            loss = criterion(model(X_std), y_t)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        # ── 7. Persist updated checkpoint ─────────────────────────────────────
        _save_model(model)
        logger.info(
            'apply_learn | alert=%s user=%s date=%s | steps=%d loss=%.4f',
            alert_id, alert['user'], alert['date'], _CORRECTION_STEPS, final_loss,
        )
    else:
        logger.warning(
            'apply_learn | alert=%s — no feature data found for user=%s date=%s; '
            'skipping weight update',
            alert_id, alert['user'], alert['date'],
        )

    # ── 8. Update alert status ────────────────────────────────────────────────
    return update_alert_status(alert_id, 'learned')


def apply_mute(alert_id: str) -> dict:
    """Silence an alert without changing the model.

    Args:
        alert_id: UUID of the alert to mute.

    Returns:
        The updated alert dict.
    """
    return update_alert_status(alert_id, 'muted')


def apply_block(alert_id: str) -> tuple[dict, str]:
    """Flag an alert for access revocation and log the action.

    Args:
        alert_id: UUID of the alert to block.

    Returns:
        ``(updated_alert, notification_message)`` tuple.
    """
    updated = update_alert_status(alert_id, 'blocked')
    user    = updated['user']
    message = (
        f'Access revocation request sent for user {user}. '
        '(Simulation only — no real action taken.)'
    )
    log = _read_json(_BLOCK_LOG_PATH)
    log.append({
        'alert_id':  alert_id,
        'user':      user,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'message':   message,
    })
    _write_json(_BLOCK_LOG_PATH, log)
    return updated, message


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_alert(alert_id: str) -> dict:
    """Return the alert dict for *alert_id*, or raise ``KeyError``."""
    for alert in _read_alerts_file():
        if alert['id'] == alert_id:
            return alert
    raise KeyError(f'Alert {alert_id!r} not found')


def _features_for_correction(correction: dict) -> Optional[torch.Tensor]:
    """Build a raw (1, n_features) float32 tensor for *correction*, or ``None``.

    Searches all dataset splits for logs matching the correction's user and date.
    Returns ``None`` if no log data is found for that user-day.
    """
    user        = correction['user']
    target_date = pd.Timestamp(correction['date']).normalize()

    data     = load_data()
    filtered: dict[str, list] = {}
    for logs_dict in data['splits'].values():
        for log_name, df in logs_dict.items():
            if df.empty or 'date' not in df.columns:
                continue
            day_df = df[df['date'].dt.normalize() == target_date]
            if not day_df.empty:
                filtered.setdefault(log_name, []).append(day_df)

    if not filtered:
        return None

    day_logs = {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in filtered.items()
    }

    feat_df = build_user_day_features(day_logs, malicious_tuples=data['labels'])
    row     = feat_df[feat_df['user'] == user]
    if row.empty:
        return None

    X_full = np.zeros((1, len(_FEATURE_COLS)), dtype=np.float32)
    for j, col in enumerate(_FEATURE_COLS):
        if col in row.columns:
            X_full[0, j] = float(row.iloc[0][col])
    return torch.tensor(X_full, dtype=torch.float32)


def _save_model(model: InsiderThreatMLP) -> None:
    """Save *model* in both PyTorch (.pt) and legacy numpy (.npz) formats.

    The .npz file keeps ``load_best_model()`` and ``api.py``'s ``_get_model()``
    working without any changes.
    """
    os.makedirs(os.path.dirname(_BEST_MODEL_PT_PATH), exist_ok=True)

    # PyTorch checkpoint — includes layer_sizes so loaders don't need to guess
    torch.save(
        {"state_dict": model.state_dict(), "layer_sizes": model.layer_sizes},
        _BEST_MODEL_PT_PATH,
    )

    # Legacy numpy checkpoint — nn.Linear weight is (out, in); Network expects (in, out)
    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    arrays: dict  = {}
    for i, layer in enumerate(linear_layers):
        arrays[f'W_{i}'] = layer.weight.detach().cpu().numpy().T           # (in, out)
        arrays[f'b_{i}'] = layer.bias.detach().cpu().numpy().reshape(1, -1)  # (1, out)
    arrays['n_layers'] = np.array(len(linear_layers))
    np.savez(_BEST_MODEL_PATH, **arrays)


def _read_json(path: str) -> list:
    return safe_json_read(path, default=[])


def _write_json(path: str, data) -> None:
    safe_json_write(path, data)
