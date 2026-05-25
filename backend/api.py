import json
import logging
import os
import queue as queue_module
import threading
import traceback
import contextlib
import io
from datetime import datetime, timedelta, timezone

import pandas as pd
import torch

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from innersight.backend.b2_data.pipeline import load_data as _load_raw_data
from innersight.backend.b2_features.features import build_user_day_features
from innersight.backend.b7_training.trainer import train as run_train
from innersight.backend.b8_scoring.scoring import load_alerts
from innersight.backend.b9_feedback.feedback import apply_learn, apply_mute, apply_block
from innersight.backend.config import (
    DATA_DIR, LDAP_FILE, BEST_MODEL_PT_FILE, STANDARDIZER_FILE,
    FEATURE_COLS, DEFAULT_TRAINING_CONFIG, setup_logging,
)
from innersight.backend.models.mlp import InsiderThreatMLP, get_device
from innersight.backend.models.dataset import Standardizer

setup_logging()
logger = logging.getLogger(__name__)

_BEST_MODEL_PT_PATH = BEST_MODEL_PT_FILE
_STANDARDIZER_PATH  = STANDARDIZER_FILE
_FEATURE_COLS       = FEATURE_COLS

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:5173'])

_event_queue: queue_module.Queue = queue_module.Queue()

# ── module-level caches (populated lazily) ───────────────────────────────────
_data_cache = None           # full CERT dataset from load_data()
_ldap_cache = None           # LDAP DataFrame
_model_cache = None          # {"model": InsiderThreatMLP, "standardizer": Standardizer, "device": torch.device} or None
_score_history_cache: dict   = {}   # (user_id, days) -> [{date, score}]

_LDAP_PATH = LDAP_FILE
_DATA_DIR  = DATA_DIR


def _get_data():
    global _data_cache
    if _data_cache is None:
        try:
            _data_cache = _load_raw_data(_DATA_DIR)
        except Exception as exc:
            logger.error('Failed to load data from %s: %s', _DATA_DIR, exc)
            raise
    return _data_cache


def _get_ldap() -> pd.DataFrame:
    global _ldap_cache
    if _ldap_cache is None:
        if os.path.exists(_LDAP_PATH):
            _ldap_cache = pd.read_csv(_LDAP_PATH)
        else:
            _ldap_cache = pd.DataFrame(columns=['user_id', 'department'])
    return _ldap_cache


def _get_model():
    global _model_cache
    if _model_cache is None:
        try:
            device     = get_device()
            # weights_only=False: GNN checkpoints contain metadata dicts/tuples.
            checkpoint = torch.load(
                _BEST_MODEL_PT_PATH, map_location=device, weights_only=False
            )
            model_type = checkpoint.get('model_type', 'mlp')

            if model_type == 'graphsage':
                from innersight.backend.models.graphsage import InsiderThreatGNN
                model_cfg = checkpoint.get('config', {})
                model = InsiderThreatGNN(
                    metadata=checkpoint['metadata'],
                    hidden_dim=model_cfg.get('hidden_dim', 128),
                    num_layers=model_cfg.get('num_layers', 2),
                    dropout=model_cfg.get('dropout', 0.3),
                    head_layers=model_cfg.get('head_layers', [128, 64]),
                )
                model.load_state_dict(checkpoint['state_dict'])
                model.to(device).eval()
                # GNN has no standardizer; score-history endpoint falls back gracefully.
                _model_cache = {
                    'model_type': 'graphsage',
                    'model': model,
                    'standardizer': None,
                    'device': device,
                }
            else:
                layer_sizes  = checkpoint.get('layer_sizes', DEFAULT_TRAINING_CONFIG['layer_sizes'])
                model        = InsiderThreatMLP(layer_sizes)
                model.load_state_dict(checkpoint['state_dict'])
                model.to(device).eval()
                standardizer = Standardizer.load(_STANDARDIZER_PATH)
                _model_cache = {
                    'model_type': 'mlp',
                    'model': model,
                    'standardizer': standardizer,
                    'device': device,
                }
        except FileNotFoundError:
            logger.warning('No trained model found at %s', _BEST_MODEL_PT_PATH)
            _model_cache = None
        except Exception as exc:
            logger.error('Failed to load model: %s', exc)
            _model_cache = None
    return _model_cache


def _score_user_day(
    user_logs: dict,
    model: InsiderThreatMLP,
    standardizer: Standardizer,
    device: torch.device,
) -> float:
    """Return a risk score in (0, 1) for a single user-day log dict."""
    if not user_logs:
        return 0.0
    feat_df = build_user_day_features(user_logs, malicious_tuples=set())
    if feat_df.empty:
        return 0.0
    X_full = torch.zeros(1, len(_FEATURE_COLS))
    for j, col in enumerate(_FEATURE_COLS):
        if col in feat_df.columns:
            X_full[0, j] = float(feat_df.iloc[0].get(col, 0))
    X_std = standardizer.transform(X_full).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(X_std))
    return float(prob.cpu().item())


def _user_dept_map() -> dict[str, str]:
    ldap = _get_ldap()
    user_col = next((c for c in ldap.columns if 'user' in c.lower()), None)
    dept_col  = next(
        (c for c in ldap.columns if c.lower() in ('department', 'functional_unit', 'business_unit')),
        None,
    )
    if user_col is None or dept_col is None:
        return {}
    return dict(zip(ldap[user_col].astype(str), ldap[dept_col].astype(str)))


# ── input validation ─────────────────────────────────────────────────────────

def _validate_train_config(config: dict) -> str | None:
    """Return an error string if config is invalid, else None."""
    for key in ('epochs', 'batch_size', 'patience'):
        val = config.get(key)
        if not isinstance(val, int) or val <= 0:
            return f"'{key}' must be a positive integer, got {val!r}"

    for key in ('lr', 'pos_weight'):
        val = config.get(key)
        if not isinstance(val, (int, float)) or val <= 0:
            return f"'{key}' must be a positive number, got {val!r}"

    sizes = config.get('layer_sizes')
    if not isinstance(sizes, list) or len(sizes) == 0:
        return "'layer_sizes' must be a non-empty list"
    if not all(isinstance(s, int) and s > 0 for s in sizes):
        return "'layer_sizes' must contain only positive integers"
    if sizes[-1] != 1:
        return f"'layer_sizes' last element must be 1 (output neuron), got {sizes[-1]}"

    return None


def _alert_exists(alert_id: str) -> bool:
    from innersight.backend.b8_scoring.scoring import _read_alerts_file
    return any(a['id'] == alert_id for a in _read_alerts_file())


# ── generic error handler ─────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    logger.error('Unhandled exception: %s\n%s', exc, traceback.format_exc())
    return jsonify({'error': 'Internal server error'}), 500


# ── legacy training worker (kept for /api/train backward compat) ─────────────

def _train_worker(config):
    global _model_cache, _score_history_cache
    try:
        run_train(config, event_callback=_event_queue.put)
        _model_cache         = None  # force reload with newly trained weights
        _score_history_cache = {}    # stale against old model
    except Exception as exc:
        _event_queue.put({'status': 'error', 'message': str(exc)})


# ── existing endpoints ────────────────────────────────────────────────────────

@app.get('/api/config')
def get_config():
    return jsonify({'layer_sizes': DEFAULT_TRAINING_CONFIG['layer_sizes']})


@app.post('/api/train')
def post_train():
    config = request.get_json() or {}
    for k, v in DEFAULT_TRAINING_CONFIG.items():
        config.setdefault(k, v)

    error = _validate_train_config(config)
    if error:
        logger.warning('POST /api/train rejected: %s', error)
        return jsonify({'error': error}), 400

    threading.Thread(target=_train_worker, args=(config,), daemon=True).start()
    return jsonify({'status': 'started'})


@app.get('/api/events')
def get_events():
    def stream():
        while True:
            try:
                event = _event_queue.get(timeout=60)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get('status') in ('done', 'error'):
                    break
            except queue_module.Empty:
                yield ": keepalive\n\n"
    return Response(
        stream(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.get('/api/status')
def get_status():
    return jsonify({'status': 'idle'})


# ── alert action endpoints ────────────────────────────────────────────────────

@app.post('/api/alert/<alert_id>/learn')
def post_alert_learn(alert_id):
    if not _alert_exists(alert_id):
        return jsonify({'error': 'Alert not found'}), 404
    alert = apply_learn(alert_id)
    global _model_cache
    _model_cache = None          # invalidate so next request reloads updated weights
    return jsonify({'alert': alert, 'message': 'Model updated'})


@app.post('/api/alert/<alert_id>/mute')
def post_alert_mute(alert_id):
    if not _alert_exists(alert_id):
        return jsonify({'error': 'Alert not found'}), 404
    return jsonify({'alert': apply_mute(alert_id)})


@app.post('/api/alert/<alert_id>/block')
def post_alert_block(alert_id):
    if not _alert_exists(alert_id):
        return jsonify({'error': 'Alert not found'}), 404
    alert, notification = apply_block(alert_id)
    return jsonify({'alert': alert, 'notification': notification})


# ── new query endpoints ───────────────────────────────────────────────────────

@app.get('/api/alerts')
def get_alerts():
    status_filter = request.args.get('status') or None
    alerts = load_alerts(status_filter=status_filter)
    return jsonify(alerts)


@app.get('/api/employees')
def get_employees():
    alerts = load_alerts()
    dept_map = _user_dept_map()

    # Aggregate per user: latest score, alert count, latest status
    latest: dict[str, dict] = {}
    counts: dict[str, int]  = {}
    for alert in alerts:          # already sorted score desc
        user = alert['user']
        counts[user] = counts.get(user, 0) + 1
        if user not in latest:
            latest[user] = alert  # highest-score alert per user

    employees = []
    for user, al in latest.items():
        employees.append({
            'user':         user,
            'department':   dept_map.get(user, 'Unknown'),
            'latest_score': al['score'],
            'alert_count':  counts[user],
            'status':       al['status'],
        })

    employees.sort(key=lambda e: e['latest_score'], reverse=True)
    return jsonify(employees)


@app.get('/api/employee/<user_id>/activity')
def get_employee_activity(user_id):
    from_str = request.args.get('from')
    to_str   = request.args.get('to')

    try:
        data = _get_data()
    except Exception:
        return jsonify({'error': 'Data not loaded'}), 503

    malicious = data['labels']

    # Collect all rows for this user across all splits and log types
    events = []

    for logs_dict in data['splits'].values():
        for log_type, df in logs_dict.items():
            if df.empty or 'user' not in df.columns:
                continue
            udf = df[df['user'] == user_id].copy()
            if from_str:
                udf = udf[udf['date'] >= pd.Timestamp(from_str)]
            if to_str:
                udf = udf[udf['date'] <= pd.Timestamp(to_str)]
            if udf.empty:
                continue

            for _, row in udf.iterrows():
                ts    = row['date']
                day   = ts.date() if hasattr(ts, 'date') else ts
                susp  = (user_id, day) in malicious
                desc  = _describe_event(log_type, row)
                events.append({
                    'timestamp':   str(ts),
                    'type':        _LOG_TYPE_MAP.get(log_type, log_type),
                    'description': desc,
                    'suspicious':  susp,
                })

    events.sort(key=lambda e: e['timestamp'])
    return jsonify({'user': user_id, 'events': events})


_LOG_TYPE_MAP = {
    'logon':  'logon',
    'device': 'usb',
    'file':   'file',
    'email':  'email',
    'http':   'http',
}


def _describe_event(log_type: str, row: pd.Series) -> str:
    try:
        if log_type == 'logon':
            action = 'on' if str(row.get('activity', '')).lower() == 'logon' else 'off'
            return f"Logged {action} at {row.get('pc', '?')}"
        if log_type == 'device':
            action = str(row.get('activity', 'Connect')).lower()
            return f"USB {action} at {row.get('pc', '?')}"
        if log_type == 'file':
            fname = os.path.basename(str(row.get('filename', '?')))
            removable = str(row.get('to_removable_media', '')).lower() == 'true'
            suffix = ' → removable drive' if removable else ''
            return f"File access: {fname}{suffix}"
        if log_type == 'email':
            to_addr = str(row.get('to', '?'))[:60]
            size_kb  = int(row.get('size', 0)) // 1024
            return f"Email to {to_addr} ({size_kb} KB)"
        if log_type == 'http':
            url = str(row.get('url', '?'))[:80]
            return f"Visited {url}"
    except Exception:
        pass
    return log_type


@app.get('/api/employee/<user_id>/score-history')
def get_employee_score_history(user_id):
    days = int(request.args.get('days', 30))
    cache_key = (user_id, days)

    if cache_key in _score_history_cache:
        return jsonify(_score_history_cache[cache_key])

    model_tuple = _get_model()
    if model_tuple is None:
        return jsonify({'error': 'No trained model available'}), 503

    if model_tuple.get('model_type') == 'graphsage':
        # Score-history requires per-day flat-feature inference; not yet supported
        # for GNN models (planned for Phase 8 with GNNExplainer).
        return jsonify({'error': 'Score history not yet supported for GNN models'}), 503

    model        = model_tuple['model']
    standardizer = model_tuple['standardizer']
    device       = model_tuple['device']

    try:
        data = _get_data()
    except Exception:
        return jsonify({'error': 'Data not loaded'}), 503

    # Find latest date in dataset
    max_date = None
    for logs_dict in data['splits'].values():
        for df in logs_dict.values():
            if df.empty or 'date' not in df.columns:
                continue
            cand = df['date'].max()
            if max_date is None or cand > max_date:
                max_date = cand

    if max_date is None:
        return jsonify([])

    end_date   = max_date.normalize()
    start_date = end_date - timedelta(days=days - 1)

    history = []
    current = start_date
    while current <= end_date:
        day_logs: dict[str, pd.DataFrame] = {}
        for logs_dict in data['splits'].values():
            for log_name, df in logs_dict.items():
                if df.empty or 'user' not in df.columns or 'date' not in df.columns:
                    continue
                day_df = df[
                    (df['user'] == user_id) &
                    (df['date'].dt.normalize() == current)
                ]
                if not day_df.empty:
                    day_logs.setdefault(log_name, []).append(day_df)

        merged_logs = {
            name: pd.concat(parts, ignore_index=True)
            for name, parts in day_logs.items()
        }

        score = _score_user_day(merged_logs, model, standardizer, device) if merged_logs else 0.0
        history.append({'date': current.strftime('%Y-%m-%d'), 'score': round(score, 4)})
        current += timedelta(days=1)

    _score_history_cache[cache_key] = history
    return jsonify(history)


if __name__ == '__main__':
    app.run(port=5001, debug=True)
