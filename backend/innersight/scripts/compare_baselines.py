#!/usr/bin/env python
"""Compare the Phase 3 baselines (XGBoost vs. MLP) under one harness.

Runs both baselines in-process on the same windows — XGBoost on the 129
handcrafted temporal features, the MLP on the flattened 504-dim window — scores
them with the shared cross-validation harness, and prints a side-by-side table.
The Phase 3 gate is AUPRC > 0.20 for the best baseline.

Usage:
    python -m innersight.scripts.compare_baselines \
        --version r4.2 --store-dir feature_store --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from innersight.config import setup_logging
from innersight.data.answers import load_insiders
from innersight.data.feature_store import FeatureStore
from innersight.features.temporal_features import extract_all_window_features
from innersight.models.dataset import DeviationWindowDataset
from innersight.scripts import compute_baselines
from innersight.scripts.train_mlp_baseline import _build_features, _load_ocean, _train_one_mlp
from innersight.scripts.xgboost_baseline import _make_xgb_model_fn
from innersight.training.evaluation import format_results_table, run_cross_validation
from innersight.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)

# Phase 3 validation gate: the best baseline must exceed this mean AUPRC.
GATE_AUPRC_THRESHOLD = 0.20


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Compare the XGBoost and MLP baselines under shared cross-validation.',
    )
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', metavar='PATH',
                   help='Dataset directory (answers/ labels, OCEAN scores, and to compute deviations)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--output', default='compare_results.json', metavar='PATH',
                   help='Where to write the comparison JSON (default: compare_results.json)')
    p.add_argument('--n-folds', type=int, default=5, help='CV folds per seed (default: 5)')
    p.add_argument('--seeds', default='42,123,456',
                   help='Comma-separated CV seeds (default: "42,123,456")')
    return p.parse_args(argv)


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str | None):
    """Return cached deviations, computing them from data_dir if absent."""
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    if not data_dir:
        logger.error("compare_baselines | no deviations cached for %s and no --data-dir. "
                     "Run compute_baselines first.", version)
        return None
    logger.info("compare_baselines | deviations not cached; computing from %s …", data_dir)
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        logger.error("compare_baselines | compute_baselines failed (rc=%d).", rc)
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
        logger.error("compare_baselines | --data-dir is required to load attack windows (answers/).")
        return 1
    answers_dir = Path(args.data_dir) / 'answers'
    if not answers_dir.exists():
        logger.error("compare_baselines | no answers/ directory at %s.", answers_dir)
        return 1
    attack_windows = {r.user_id: r for r in load_insiders(answers_dir, args.version)}
    logger.info("compare_baselines | %d insiders for %s.", len(attack_windows), args.version)

    dataset = DeviationWindowDataset(deviations, attack_windows)

    # XGBoost: 129 handcrafted temporal features.
    X_xgb, y, metas = extract_all_window_features(dataset)
    # MLP: flattened 504-dim window (+5 OCEAN if available); same window order/labels.
    ocean_map = _load_ocean(args.version, args.data_dir)
    X_mlp, _, _ = _build_features(dataset, ocean_map)

    n_pos = int(y.sum())
    logger.info("compare_baselines | %d windows (%d positive) | XGB dim=%d, MLP dim=%d.",
                len(y), n_pos, X_xgb.shape[1] if X_xgb.size else 0, X_mlp.shape[1] if X_mlp.size else 0)
    if len(y) == 0 or n_pos == 0:
        logger.error("compare_baselines | need labelled positive windows; aborting.")
        return 1

    logger.info("compare_baselines | running XGBoost baseline …")
    xgb_cv = run_cross_validation(
        _make_xgb_model_fn(), X_xgb, y, metas, attack_windows, n_folds=args.n_folds, seeds=seeds)
    logger.info("compare_baselines | running MLP baseline …")
    mlp_cv = run_cross_validation(
        _train_one_mlp, X_mlp, y, metas, attack_windows, n_folds=args.n_folds, seeds=seeds)

    table = format_results_table({"XGBoost": xgb_cv, "MLP": mlp_cv})
    logger.info("compare_baselines | comparison:\n%s", table)

    best_auprc = max(xgb_cv["mean"]["auprc"], mlp_cv["mean"]["auprc"])
    gate_passed = best_auprc > GATE_AUPRC_THRESHOLD
    if gate_passed:
        logger.info("GATE PASSED — best baseline AUPRC=%.4f > %.2f.", best_auprc, GATE_AUPRC_THRESHOLD)
    else:
        logger.error("GATE FAILED — best baseline AUPRC=%.4f ≤ %.2f.", best_auprc, GATE_AUPRC_THRESHOLD)

    results = {
        "version": args.version,
        "n_windows": int(len(y)),
        "n_positive": n_pos,
        "n_folds": args.n_folds,
        "seeds": seeds,
        "gate_threshold": GATE_AUPRC_THRESHOLD,
        "gate_passed": gate_passed,
        "best_auprc": best_auprc,
        "xgboost": xgb_cv,
        "mlp": mlp_cv,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("compare_baselines | results written to %s.", output_path)
    return 0 if gate_passed else 1


if __name__ == '__main__':
    sys.exit(main())
