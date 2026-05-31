"""Employee risk scoring — supports both InsiderThreatMLP and InsiderThreatGNN.

Public API (unchanged):
  score_employees(date, threshold=0.7)  → list[dict]
  load_alerts(status_filter)            → list[dict]
  update_alert_status(alert_id, status) → dict

Checkpoint format (Option A — model_type tag):
  MLP:      {'state_dict', 'layer_sizes', 'model_type': 'mlp'}
  GraphSAGE:{'state_dict', 'metadata', 'config', 'graphs_dir', 'model_type': 'graphsage'}

Backward-compat: checkpoints without 'model_type' are assumed to be MLP.
"""

import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Allow running as __main__ from backend/ or the repo root.
_FILE_DIR = os.path.abspath(os.path.dirname(__file__))
_BACKEND  = os.path.abspath(os.path.join(_FILE_DIR, '..'))
_PKG_ROOT = os.path.abspath(os.path.join(_BACKEND, '..', '..'))
for _p in (_PKG_ROOT, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

from innersight.backend.data.pipeline import load_data
from innersight.backend.features.features import build_user_day_features
from innersight.backend.config import (
    ALERTS_FILE, BEST_MODEL_PT_FILE, STANDARDIZER_FILE,
    FEATURE_COLS, DEFAULT_TRAINING_CONFIG,
)
from innersight.backend.models.mlp import InsiderThreatMLP, get_device
from innersight.backend.models.dataset import Standardizer
from innersight.backend.utils import safe_json_read, safe_json_write

_ALERTS_PATH        = ALERTS_FILE
_BEST_MODEL_PT_PATH = BEST_MODEL_PT_FILE
_STANDARDIZER_PATH  = STANDARDIZER_FILE
_FEATURE_COLS       = FEATURE_COLS


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str, device: torch.device):
    """Load a checkpoint file, returning ``(ckpt_dict, model_type)``."""
    # weights_only=False required for GNN checkpoints (contain dicts/tuples).
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_type = ckpt.get('model_type', 'mlp')   # backward compat default
    return ckpt, model_type


def _load_model_and_standardizer() -> Optional[tuple]:
    """Load either an MLP or GNN model from the canonical checkpoint path.

    Returns
    -------
    tuple
        ``(model, standardizer_or_None, device, model_type)`` on success,
        ``None`` on failure (error logged).
    """
    try:
        device = get_device()
        ckpt, model_type = _load_checkpoint(_BEST_MODEL_PT_PATH, device)
    except FileNotFoundError as exc:
        logger.error('Checkpoint not found — run training first: %s', exc)
        return None
    except Exception as exc:
        logger.error('Failed to load checkpoint: %s', exc, exc_info=True)
        return None

    if model_type == 'mlp':
        layer_sizes = ckpt.get('layer_sizes', DEFAULT_TRAINING_CONFIG['layer_sizes'])
        model = InsiderThreatMLP(layer_sizes)
        model.load_state_dict(ckpt['state_dict'])
        model.to(device).eval()
        try:
            standardizer = Standardizer.load(_STANDARDIZER_PATH)
        except Exception as exc:
            logger.error('Failed to load standardizer: %s', exc)
            return None
        return model, standardizer, device, 'mlp'

    if model_type == 'graphsage':
        from innersight.backend.models.graphsage import InsiderThreatGNN
        model_cfg = ckpt.get('config', {})
        model = InsiderThreatGNN(
            metadata=ckpt['metadata'],
            hidden_dim=model_cfg.get('hidden_dim', 128),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.3),
            head_layers=model_cfg.get('head_layers', [128, 64]),
        )
        model.load_state_dict(ckpt['state_dict'])
        model.to(device).eval()
        return model, None, device, 'graphsage'

    logger.error("Unknown model_type %r in checkpoint", model_type)
    return None


# ---------------------------------------------------------------------------
# Top-features helper
# ---------------------------------------------------------------------------

def get_top_features(
    features_row: pd.Series,
    mean: np.ndarray,
    std: np.ndarray,
    n: int = 3,
) -> list[str]:
    """Return the names of the *n* features with the highest z-scores."""
    values   = np.array([features_row.get(c, 0.0) for c in _FEATURE_COLS], dtype=np.float64)
    z_scores = (values - mean) / (std + 1e-8)
    top_idx  = np.argsort(z_scores)[::-1][:n]
    return [_FEATURE_COLS[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_employees(date: str, threshold: float = 0.7) -> list[dict]:
    """Score all employees and persist alerts above *threshold*.

    Works with both MLP and GNN checkpoints; which model is active is
    detected automatically from the checkpoint's ``model_type`` field.

    Args:
        date:      ISO date string, e.g. ``'2010-09-15'``.
        threshold: Risk-score threshold; users at or below it are ignored.

    Returns:
        List of newly created alert dicts (may be empty).
    """
    loaded = _load_model_and_standardizer()
    if loaded is None:
        return []
    model, standardizer, device, model_type = loaded

    if model_type == 'mlp':
        return _score_mlp(model, standardizer, device, date, threshold)
    else:
        return _score_gnn(model, device, date, threshold)


# ---------------------------------------------------------------------------
# MLP scoring path (unchanged logic from original implementation)
# ---------------------------------------------------------------------------

def _score_mlp(
    model: InsiderThreatMLP,
    standardizer: Standardizer,
    device: torch.device,
    date: str,
    threshold: float,
) -> list[dict]:
    mean_np = standardizer.mean.cpu().numpy()   # type: ignore[union-attr]
    std_np  = standardizer.std.cpu().numpy()    # type: ignore[union-attr]

    data        = load_data()
    target_date = pd.Timestamp(date).normalize()

    filtered_logs: dict[str, list] = {}
    for logs_dict in data['splits'].values():
        for log_name, df in logs_dict.items():
            if df.empty or 'date' not in df.columns:
                continue
            day_df = df[df['date'].dt.normalize() == target_date]
            if not day_df.empty:
                filtered_logs.setdefault(log_name, []).append(day_df)

    day_logs = {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in filtered_logs.items()
    }

    feat_df = build_user_day_features(day_logs, malicious_tuples=data['labels'])
    if feat_df.empty:
        return []

    n_rows = len(feat_df)
    X_np   = np.zeros((n_rows, len(_FEATURE_COLS)), dtype=np.float32)
    for j, col in enumerate(_FEATURE_COLS):
        if col in feat_df.columns:
            X_np[:, j] = feat_df[col].values.astype(np.float32)

    X_t   = torch.tensor(X_np, dtype=torch.float32)
    X_std = standardizer.transform(X_t).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(X_std)).cpu().numpy().flatten()

    now_iso    = datetime.now(timezone.utc).isoformat()
    new_alerts = []
    for i, row in feat_df.reset_index(drop=True).iterrows():
        score = float(probs[i])
        if score <= threshold:
            continue
        top_feats = get_top_features(row, mean_np, std_np)
        new_alerts.append({
            'id':           str(uuid.uuid4()),
            'user':         str(row['user']),
            'date':         str(row['date']),
            'score':        round(score, 4),
            'status':       'open',
            'created_at':   now_iso,
            'top_features': top_feats,
        })

    _persist_alerts(new_alerts)
    logger.info(
        'score_employees (mlp) | date=%s | users_scored=%d | alerts=%d',
        date, n_rows, len(new_alerts),
    )
    return new_alerts


# ---------------------------------------------------------------------------
# GNN scoring path
# ---------------------------------------------------------------------------

def _score_gnn(
    model,
    device: torch.device,
    date: str,
    threshold: float,
) -> list[dict]:
    """Score users using the GNN model on the full training graph.

    The GNN score for each user is derived from the trained graph
    representation (2-hop neighbourhood context).  Feature-level
    explanation falls back to the same 18 hand-crafted flat features
    used by the MLP, computed from the raw logs for *date*.  If raw
    logs are unavailable, top_features is left empty.
    """
    # ── Load the graph to get user IDs ────────────────────────────────────────
    try:
        ckpt, _ = _load_checkpoint(_BEST_MODEL_PT_PATH, device)
        graphs_dir = ckpt.get('graphs_dir', 'checkpoints')
        graph_path = os.path.join(graphs_dir, 'graphs', 'train_graph.pt')
        graph = torch.load(graph_path, weights_only=False)
    except Exception as exc:
        logger.error('GNN scoring: failed to load graph: %s', exc)
        return []

    idx_to_user: dict[int, str] = graph.idx_to_user  # {row_idx: user_id}

    # ── Full-graph inference ──────────────────────────────────────────────────
    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

    with torch.no_grad():
        logits = model(x_dict, ei_dict)             # (num_users, 1)
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()   # (num_users,)

    # ── Flat features for explanation (best-effort) ───────────────────────────
    user_flat_features: dict[str, pd.Series] = {}
    mean_np = std_np = None
    try:
        data        = load_data()
        target_date = pd.Timestamp(date).normalize()
        filtered_logs: dict[str, list] = {}
        for logs_dict in data['splits'].values():
            for log_name, df in logs_dict.items():
                if df.empty or 'date' not in df.columns:
                    continue
                day_df = df[df['date'].dt.normalize() == target_date]
                if not day_df.empty:
                    filtered_logs.setdefault(log_name, []).append(day_df)
        day_logs = {
            name: pd.concat(parts, ignore_index=True)
            for name, parts in filtered_logs.items()
        }
        feat_df = build_user_day_features(day_logs, malicious_tuples=data['labels'])
        if not feat_df.empty:
            # Build population mean/std over available users for z-scoring
            X_np = np.zeros((len(feat_df), len(_FEATURE_COLS)), dtype=np.float64)
            for j, col in enumerate(_FEATURE_COLS):
                if col in feat_df.columns:
                    X_np[:, j] = feat_df[col].values.astype(np.float64)
            mean_np = X_np.mean(axis=0)
            std_np  = X_np.std(axis=0)
            for _, row in feat_df.reset_index(drop=True).iterrows():
                user_flat_features[str(row['user'])] = row
    except Exception as exc:
        logger.debug('GNN scoring: flat features unavailable (%s) — top_features will be empty', exc)

    # ── Build alerts ──────────────────────────────────────────────────────────
    now_iso    = datetime.now(timezone.utc).isoformat()
    new_alerts = []
    for idx, score in enumerate(probs):
        score = float(score)
        if score <= threshold:
            continue
        user_id = idx_to_user.get(idx, f'user_{idx}')

        if mean_np is not None and user_id in user_flat_features:
            top_feats = get_top_features(user_flat_features[user_id], mean_np, std_np)
        else:
            top_feats = []   # GNNExplainer can fill this in Phase 8

        new_alerts.append({
            'id':           str(uuid.uuid4()),
            'user':         user_id,
            'date':         date,
            'score':        round(score, 4),
            'status':       'open',
            'created_at':   now_iso,
            'top_features': top_feats,
        })

    _persist_alerts(new_alerts)
    logger.info(
        'score_employees (gnn) | date=%s | users_scored=%d | alerts=%d',
        date, len(probs), len(new_alerts),
    )
    return new_alerts


# ---------------------------------------------------------------------------
# Alert CRUD
# ---------------------------------------------------------------------------

def load_alerts(status_filter: Optional[str] = None) -> list[dict]:
    """Return all persisted alerts, optionally filtered by status."""
    alerts = _read_alerts_file()
    if status_filter is not None:
        alerts = [a for a in alerts if a.get('status') == status_filter]
    return sorted(alerts, key=lambda a: a['score'], reverse=True)


def update_alert_status(alert_id: str, new_status: str) -> dict:
    """Set the status field of *alert_id* and persist."""
    alerts  = _read_alerts_file()
    updated = None
    for alert in alerts:
        if alert['id'] == alert_id:
            alert['status'] = new_status
            updated         = alert
            break
    if updated is None:
        raise KeyError(f'Alert {alert_id!r} not found')
    _write_alerts_file(alerts)
    return updated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _persist_alerts(new_alerts: list[dict]) -> None:
    existing = _read_alerts_file()
    existing.extend(new_alerts)
    _write_alerts_file(existing)


def _read_alerts_file() -> list[dict]:
    return safe_json_read(_ALERTS_PATH, default=[])


def _write_alerts_file(alerts: list[dict]) -> None:
    safe_json_write(_ALERTS_PATH, alerts)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    # innersight/ lives at Developer/innersight; Developer/ is the pkg root.
    _pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    print('=' * 60)
    print('scoring.py smoke test')
    print('=' * 60)

    loaded = _load_model_and_standardizer()
    if loaded is None:
        print(f'\nNo trained model found at {_BEST_MODEL_PT_PATH}')
        print('Run training first.')
        sys.exit(0)

    model, standardizer, device, model_type = loaded
    from torch.nn.parameter import UninitializedParameter
    total_params = sum(
        p.numel() for p in model.parameters()
        if not isinstance(p, UninitializedParameter)
    )
    print(f'\nModel type : {model_type}')
    print(f'Params     : {total_params:,}  (initialized)')
    print(f'Device     : {device}')
    if standardizer is not None:
        print(f'Standardizer mean[:3]: {standardizer.mean[:3].tolist()}')

    test_date = '2010-09-15'
    print(f'\nScoring date: {test_date}  (threshold=0.3 for demo) …')
    try:
        alerts = score_employees(test_date, threshold=0.3)
        print(f'Alerts generated: {len(alerts)}')
        for a in alerts[:5]:
            print(f"  user={a['user']}  score={a['score']}  top={a['top_features']}")
    except Exception as exc:
        print(f'Scoring error: {exc}')
        if model_type == 'mlp' and standardizer is not None:
            print('\nFalling back to synthetic inference check …')
            from innersight.backend.models.dataset import Standardizer as _S
            X_syn = torch.randn(4, len(_FEATURE_COLS))
            X_std = standardizer.transform(X_syn).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(X_std))
            print(f'  Synthetic probs: {probs.cpu().numpy().flatten().round(4).tolist()}')
            print('  Inference OK')
