#!/usr/bin/env python
"""MLP baseline for insider-threat detection (Phase 3).

Trains a small feed-forward MLP on **flattened** 28-day deviation windows
(18 features × 28 days = 504 dims, optionally + 5 OCEAN psychometric dims) with
focal loss, under the same leakage-safe user-level cross-validation harness used
everywhere else. This is the neural counterpart to the XGBoost baseline, which
instead used the 129 handcrafted temporal features.

Hyperparameter defaults mirror configs/train_baseline_mlp.yaml.

Usage:
    python -m innersight.scripts.train_mlp_baseline \
        --version r4.2 --store-dir feature_store --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from innersight.config import setup_logging
from innersight.data.answers import load_insiders
from innersight.data.feature_store import FeatureStore
from innersight.models.dataset import DeviationWindowDataset
from innersight.models.losses import FocalLoss
from innersight.scripts import compute_baselines
from innersight.training.evaluation import (
    compute_metrics,
    detection_latency,
    format_results_table,
    per_scenario_metrics,
    run_cross_validation,
    temporal_stratified_kfold,
)
from innersight.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)

# Training hyperparameters (mirror configs/train_baseline_mlp.yaml).
HIDDEN_SIZES = (64, 32)
DROPOUT = 0.4
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 64

_DEVICE = torch.device("cpu")  # CPU keeps cross-validation reproducible


class _BaselineMLP(nn.Module):
    """input_dim → 64 → 32 → 1 with ReLU + dropout; emits raw logits."""

    def __init__(self, input_dim: int, hidden=HIDDEN_SIZES, dropout: float = DROPOUT) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # shape: (batch, 1)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train the MLP baseline (flattened windows + focal loss) with user-level CV.',
    )
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', metavar='PATH',
                   help='Dataset directory (answers/ labels, OCEAN scores, and to compute deviations)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--output', default='mlp_results.json', metavar='PATH',
                   help='Where to write the results JSON (default: mlp_results.json)')
    p.add_argument('--n-folds', type=int, default=5, help='CV folds per seed (default: 5)')
    p.add_argument('--seeds', default='42,123,456',
                   help='Comma-separated CV seeds (default: "42,123,456")')
    return p.parse_args(argv)


def _train_one_mlp(X_train, y_train, X_val, y_val, seed: int) -> np.ndarray:
    """Train an MLP on one fold and return validation positive-class probabilities."""
    seed_everything(seed)

    # Standardise on train statistics only (fit on train, apply to val).
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    Xtr = torch.tensor((X_train - mu) / sd, dtype=torch.float32, device=_DEVICE)  # shape: (n_train, d)
    Xva = torch.tensor((X_val - mu) / sd, dtype=torch.float32, device=_DEVICE)    # shape: (n_val, d)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=_DEVICE).reshape(-1, 1)  # shape: (n_train, 1)

    model = _BaselineMLP(X_train.shape[1]).to(_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    n = Xtr.shape[0]
    generator = torch.Generator().manual_seed(seed)

    best_auprc = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_since_improvement = 0

    for _epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xtr[idx])               # shape: (batch, 1)
            loss = criterion(logits, ytr[idx])
            if torch.isnan(loss):
                logger.warning("train_mlp_baseline | NaN loss; skipping batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # Early stopping on validation AUPRC.
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(Xva)).reshape(-1).cpu().numpy()  # shape: (n_val,)
        auprc = compute_metrics(val_probs, y_val)["auprc"]
        if auprc > best_auprc:
            best_auprc = auprc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xva)).reshape(-1).cpu().numpy()  # shape: (n_val,)


def _oof_predictions(X, y, user_ids, n_folds: int, seed: int) -> np.ndarray:
    """Out-of-fold predictions for per-scenario metrics and detection latency."""
    oof = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in temporal_stratified_kfold(user_ids, y, n_folds=n_folds, seed=seed):
        oof[val_idx] = _train_one_mlp(X[train_idx], y[train_idx], X[val_idx], y[val_idx], seed)
    return oof


def _load_ocean(version: str, data_dir: str | None) -> dict[str, np.ndarray] | None:
    """Return {user_id: OCEAN(5,)} from psychometric data, or None if unavailable."""
    if not data_dir:
        return None
    try:
        from innersight.data.adapters import get_adapter

        psych = get_adapter(version).load_psychometric(Path(data_dir))
        cols = ["O", "C", "E", "A", "N"]
        if psych.empty or not all(c in psych.columns for c in cols):
            return None
        return {str(row["user_id"]): row[cols].to_numpy(dtype=float) for _, row in psych.iterrows()}
    except Exception as exc:  # psychometric is optional — never fail the run over it
        logger.warning("train_mlp_baseline | could not load OCEAN scores: %s", exc)
        return None


def _build_features(dataset, ocean_map):
    """Flatten each (18, 28) window to 504 dims, optionally appending 5 OCEAN dims."""
    rows: list[np.ndarray] = []
    labels: list[float] = []
    metas: list[dict] = []
    for i in range(len(dataset)):
        window, label, meta = dataset[i]
        flat = np.asarray(window, dtype=float).reshape(-1)  # shape: (504,)
        if ocean_map is not None:
            flat = np.concatenate([flat, ocean_map.get(str(meta["user_id"]), np.zeros(5))])
        rows.append(flat)
        labels.append(float(np.asarray(label).reshape(-1)[0]))
        metas.append(meta)
    if not rows:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float), metas
    return np.vstack(rows), np.asarray(labels, dtype=float), metas


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str | None):
    """Return cached deviations, computing them from data_dir if absent."""
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    if not data_dir:
        logger.error("train_mlp_baseline | no deviations cached for %s and no --data-dir. "
                     "Run compute_baselines first.", version)
        return None
    logger.info("train_mlp_baseline | deviations not cached; computing from %s …", data_dir)
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        logger.error("train_mlp_baseline | compute_baselines failed (rc=%d).", rc)
        return None
    return store.load_deviations(version)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    seed_everything(seeds[0] if seeds else 42)

    store = FeatureStore(args.store_dir)
    deviations = _load_deviations(store, args.version, args.store_dir, args.data_dir)
    if deviations is None:
        return 1

    if not args.data_dir:
        logger.error("train_mlp_baseline | --data-dir is required to load attack windows (answers/).")
        return 1
    answers_dir = Path(args.data_dir) / 'answers'
    if not answers_dir.exists():
        logger.error("train_mlp_baseline | no answers/ directory at %s.", answers_dir)
        return 1
    attack_windows = {r.user_id: r for r in load_insiders(answers_dir, args.version)}
    logger.info("train_mlp_baseline | %d insiders for %s.", len(attack_windows), args.version)

    ocean_map = _load_ocean(args.version, args.data_dir)
    dataset = DeviationWindowDataset(deviations, attack_windows)
    X, y, metas = _build_features(dataset, ocean_map)
    n_pos = int(y.sum())
    logger.info("train_mlp_baseline | %d windows × %d features (%d positive)%s.",
                X.shape[0], X.shape[1], n_pos, " (+OCEAN)" if ocean_map is not None else "")
    if X.shape[0] == 0 or n_pos == 0:
        logger.error("train_mlp_baseline | need labelled positive windows to train; aborting.")
        return 1

    user_ids = np.array([m['user_id'] for m in metas])
    scenarios = np.array([m.get('scenario', 0) for m in metas])

    cv = run_cross_validation(_train_one_mlp, X, y, metas, attack_windows, n_folds=args.n_folds, seeds=seeds)
    logger.info("train_mlp_baseline | CV AUPRC=%.4f ± %.4f | P@10=%.3f | F1=%.3f",
                cv['mean']['auprc'], cv['std']['auprc'], cv['mean']['p_at_10'], cv['mean']['f1_best'])

    oof = _oof_predictions(X, y, user_ids, args.n_folds, seeds[0])
    scenario_metrics = per_scenario_metrics(oof, y, scenarios)
    threshold = compute_metrics(oof, y)['threshold_best']
    latency = detection_latency(oof, metas, threshold, attack_windows)
    logger.info("train_mlp_baseline | detection: %d/%d insiders flagged, median latency=%s days.",
                latency['detected_count'], latency['total_insiders'], latency['median_days'])
    logger.info("\n%s", format_results_table({f"MLP ({args.version})": cv}))

    results = {
        "version": args.version,
        "feature": "flattened_window_504" + ("+ocean5" if ocean_map is not None else ""),
        "n_windows": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "n_positive": n_pos,
        "n_folds": args.n_folds,
        "seeds": seeds,
        "cross_validation": cv,
        "per_scenario": scenario_metrics,
        "detection_latency": latency,
        "oof_threshold": threshold,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("train_mlp_baseline | results written to %s.", output_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
