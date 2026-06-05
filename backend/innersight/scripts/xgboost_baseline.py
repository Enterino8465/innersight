#!/usr/bin/env python
"""XGBoost baseline for insider-threat detection (Phase 3).

End-to-end classical baseline: it turns each 28-day deviation window into a
handcrafted feature vector and trains an XGBoost classifier under leakage-safe,
user-level cross-validation, reporting the same metrics every later model is
judged on (AUPRC, P@k, per-scenario, detection latency).

Usage:
    python -m innersight.scripts.xgboost_baseline \
        --version r4.2 --store-dir feature_store --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from innersight.config import setup_logging
from innersight.data.answers import load_insiders
from innersight.data.feature_store import FeatureStore
from innersight.features.temporal_features import extract_all_window_features, get_feature_names
from innersight.models.dataset import DeviationWindowDataset
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Run the XGBoost baseline with user-level cross-validation.',
    )
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', metavar='PATH',
                   help='Dataset directory (for answers/ labels and to compute deviations if uncached)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--output', default='xgboost_results.json', metavar='PATH',
                   help='Where to write the results JSON (default: xgboost_results.json)')
    p.add_argument('--n-folds', type=int, default=5, help='CV folds per seed (default: 5)')
    p.add_argument('--seeds', default='42,123,456',
                   help='Comma-separated CV seeds (default: "42,123,456")')
    return p.parse_args(argv)


def _make_xgb_model_fn():
    """Return a model_fn(X_train, y_train, X_val, y_val, seed) -> val positive-class probs."""
    from xgboost import XGBClassifier

    def model_fn(X_train, y_train, X_val, y_val, seed):
        n_pos = float(np.count_nonzero(y_train == 1))
        n_neg = float(np.count_nonzero(y_train == 0))
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            tree_method='hist',
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_val)[:, 1]  # shape: (n_val,)

    return model_fn


def _oof_predictions(model_fn, X, y, user_ids, n_folds: int, seed: int) -> np.ndarray:
    """Out-of-fold predictions: every sample scored by a model that didn't train on its user."""
    oof = np.zeros(len(y), dtype=float)  # shape: (n,)
    for train_idx, val_idx in temporal_stratified_kfold(user_ids, y, n_folds=n_folds, seed=seed):
        oof[val_idx] = model_fn(X[train_idx], y[train_idx], X[val_idx], y[val_idx], seed)
    return oof


def _feature_importance(model_fn, X, y, top_k: int = 20) -> list[dict]:
    """Train one model on all data and return the top-k features by importance."""
    from xgboost import XGBClassifier

    n_pos = float(np.count_nonzero(y == 1))
    n_neg = float(np.count_nonzero(y == 0))
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=(n_neg / n_pos) if n_pos > 0 else 1.0,
        eval_metric='aucpr', tree_method='hist', n_jobs=-1, random_state=42,
    )
    clf.fit(X, y)
    names = get_feature_names()
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:top_k]
    return [{"feature": names[i], "importance": float(importances[i])} for i in order]


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str | None):
    """Return the cached deviations, computing them from data_dir if absent."""
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    if not data_dir:
        logger.error(
            "xgboost_baseline | no deviations cached for %s and no --data-dir to compute them. "
            "Run compute_baselines first.", version,
        )
        return None
    logger.info("xgboost_baseline | deviations not cached; computing from %s …", data_dir)
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        logger.error("xgboost_baseline | compute_baselines failed (rc=%d).", rc)
        return None
    return store.load_deviations(version)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    seed_everything(seeds[0] if seeds else 42)

    store = FeatureStore(args.store_dir)

    # 1. Deviations (from cache, or computed from --data-dir).
    deviations = _load_deviations(store, args.version, args.store_dir, args.data_dir)
    if deviations is None:
        return 1

    # 2. Attack windows come from answers/ — required for labels and latency.
    if not args.data_dir:
        logger.error("xgboost_baseline | --data-dir is required to load attack windows (answers/).")
        return 1
    answers_dir = Path(args.data_dir) / 'answers'
    if not answers_dir.exists():
        logger.error("xgboost_baseline | no answers/ directory at %s.", answers_dir)
        return 1
    insiders = load_insiders(answers_dir, args.version)
    attack_windows = {r.user_id: r for r in insiders}
    logger.info("xgboost_baseline | %d insiders for %s.", len(attack_windows), args.version)

    # 3-4. Build windows and extract handcrafted temporal features.
    dataset = DeviationWindowDataset(deviations, attack_windows)
    X, y, metas = extract_all_window_features(dataset)
    n_pos = int(y.sum())
    logger.info("xgboost_baseline | %d windows × %d features (%d positive).", X.shape[0], X.shape[1], n_pos)
    if X.shape[0] == 0 or n_pos == 0:
        logger.error("xgboost_baseline | need labelled positive windows to train; aborting.")
        return 1

    user_ids = np.array([m['user_id'] for m in metas])
    scenarios = np.array([m.get('scenario', 0) for m in metas])

    # 5. Cross-validated metrics (3 seeds for variance).
    model_fn = _make_xgb_model_fn()
    cv = run_cross_validation(model_fn, X, y, metas, attack_windows, n_folds=args.n_folds, seeds=seeds)
    logger.info("xgboost_baseline | CV AUPRC=%.4f ± %.4f | P@10=%.3f | F1=%.3f",
                cv['mean']['auprc'], cv['std']['auprc'], cv['mean']['p_at_10'], cv['mean']['f1_best'])

    # 6. Per-scenario metrics + detection latency on out-of-fold predictions.
    oof = _oof_predictions(model_fn, X, y, user_ids, args.n_folds, seeds[0])
    scenario_metrics = per_scenario_metrics(oof, y, scenarios)
    threshold = compute_metrics(oof, y)['threshold_best']
    latency = detection_latency(oof, metas, threshold, attack_windows)
    logger.info("xgboost_baseline | detection: %d/%d insiders flagged, median latency=%s days.",
                latency['detected_count'], latency['total_insiders'], latency['median_days'])

    # 7. Feature importance (top 20).
    top_features = _feature_importance(model_fn, X, y, top_k=20)
    logger.info("xgboost_baseline | top features: %s",
                ", ".join(f"{f['feature']}({f['importance']:.3f})" for f in top_features[:5]))

    logger.info("\n%s", format_results_table({f"XGBoost ({args.version})": cv}))

    # 8. Persist results.
    results = {
        "version": args.version,
        "n_windows": int(X.shape[0]),
        "n_positive": n_pos,
        "n_folds": args.n_folds,
        "seeds": seeds,
        "cross_validation": cv,
        "per_scenario": scenario_metrics,
        "detection_latency": latency,
        "oof_threshold": threshold,
        "feature_importance_top20": top_features,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("xgboost_baseline | results written to %s.", output_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
