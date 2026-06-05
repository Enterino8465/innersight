#!/usr/bin/env python
"""Evaluate the Temporal+Graph model against the progressive ladder (Phase 5).

Trains the chained Temporal+Graph model under the shared user-level CV protocol,
builds the four-rung ladder table (XGBoost → MLP → Temporal CNN → Temporal+Graph)
from the saved baseline results, reports per-scenario metrics and detection
latency, and logs whether the graph stage adds at least 0.02 AUPRC over the
temporal-only model.

Usage:
    python -m innersight.scripts.evaluate_graph \
        --version r4.2 --data-dir /path/to/data --baseline-results-dir .
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
from innersight.data.pipeline import load_version
from innersight.models.dataset import DeviationWindowDataset
from innersight.scripts import compute_baselines
from innersight.scripts.train_temporal_graph import (
    _build_period_graphs,
    _build_window_registry,
    _fit_fold,
    _load_config,
)
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

# Minimum AUPRC lift the graph stage must add over the temporal-only model.
_GATE_MARGIN = 0.02


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str | None):
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    if not data_dir:
        logger.error("evaluate_graph | no deviations cached for %s and no --data-dir.", version)
        return None
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        return None
    return store.load_deviations(version)


def _load_baseline(results_dir: str, fname: str) -> dict | None:
    """Load a prior phase's cross-validation result, or None if unavailable."""
    path = Path(results_dir) / fname
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["cross_validation"]
    except (KeyError, ValueError) as exc:
        logger.warning("evaluate_graph | could not read %s: %s", path, exc)
        return None


def _evaluate_gate(graph_auprc: float, temporal_cv: dict | None) -> dict:
    """Log and return the Temporal+Graph vs. Temporal-only validation gate."""
    if temporal_cv is None:
        logger.warning("evaluate_graph | no Temporal CNN results (temporal_results.json) — "
                       "cannot compute the graph-uplift gate.")
        return {"status": "no_baseline", "graph_auprc": graph_auprc}
    temporal_auprc = temporal_cv["mean"]["auprc"]
    diff = graph_auprc - temporal_auprc
    if diff >= _GATE_MARGIN:
        logger.info("GATE PASS: Temporal+Graph AUPRC %.3f > Temporal-only %.3f by %.3f (≥ %.2f)",
                    graph_auprc, temporal_auprc, diff, _GATE_MARGIN)
        status = "pass"
    else:
        logger.info(
            "GATE NOTE: Graph adds < %.2f AUPRC. CERT's synthetic graph may lack relational signal.\n"
            "  Keeping Qdrant for k-NN discovery. Document finding.\n"
            "  (Temporal+Graph %.3f vs Temporal-only %.3f, difference %.3f.)",
            _GATE_MARGIN, graph_auprc, temporal_auprc, diff)
        status = "note"
    return {"status": status, "graph_auprc": graph_auprc,
            "temporal_auprc": temporal_auprc, "difference": diff}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Evaluate the Temporal+Graph model and compare the full progressive ladder.',
    )
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', metavar='PATH',
                   help='Dataset directory (raw logs for graphs, answers/ labels, deviations)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--config', metavar='PATH', help='Path to a train_temporal_graph.yaml config')
    p.add_argument('--output', default='graph_eval_results.json', metavar='PATH',
                   help='Where to write the results JSON (default: graph_eval_results.json)')
    p.add_argument('--baseline-results-dir', default='.', metavar='PATH',
                   help="Directory holding prior-phase result JSONs (default: '.')")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    temporal_cfg, graph_cfg, train_cfg, eval_cfg = _load_config(args.config)
    seeds = [int(s) for s in eval_cfg["seeds"]]
    n_folds = int(eval_cfg["n_folds"])
    seed_everything(seeds[0] if seeds else 42)

    if not args.data_dir:
        logger.error("evaluate_graph | --data-dir is required (raw logs build the graphs).")
        return 1

    store = FeatureStore(args.store_dir)
    deviations = _load_deviations(store, args.version, args.store_dir, args.data_dir)
    if deviations is None:
        return 1

    answers_dir = Path(args.data_dir) / 'answers'
    if not answers_dir.exists():
        logger.error("evaluate_graph | no answers/ directory at %s.", answers_dir)
        return 1
    attack_windows = {r.user_id: r for r in load_insiders(answers_dir, args.version)}

    logger.info("evaluate_graph | loading raw logs for graph construction …")
    logs = load_version(args.data_dir, args.version).logs

    win_dataset = DeviationWindowDataset(deviations, attack_windows)
    windows_t, user_ids, period_keys, y, metas = _build_window_registry(win_dataset)
    n_pos = int(y.sum())
    logger.info("evaluate_graph | %d windows over %d periods (%d positive).",
                len(y), len(set(period_keys)), n_pos)
    if len(y) == 0 or n_pos == 0:
        logger.error("evaluate_graph | need labelled positive windows; aborting.")
        return 1

    full_graphs = _build_period_graphs(logs, period_keys, exclude_users=None)
    metadata = next(iter(full_graphs.values())).metadata()
    registry = {"windows": windows_t, "user_ids": user_ids, "periods": period_keys, "y": y}

    def fit(train_pos, val_pos, seed):
        return _fit_fold(train_pos, val_pos, seed, registry=registry, logs=logs,
                         full_graphs=full_graphs, temporal_cfg=temporal_cfg,
                         graph_cfg=graph_cfg, train_cfg=train_cfg, metadata=metadata)

    # Index-registry trick: X carries original positions so the CV harness's
    # slicing hands model_fn the indices it needs for graph lookup.
    positions = np.arange(len(y)).reshape(-1, 1)

    def model_fn(X_train, _y_train, X_val, _y_val, seed):
        return fit(X_train.ravel().astype(int), X_val.ravel().astype(int), seed)

    graph_cv = run_cross_validation(model_fn, positions, y, metas, attack_windows,
                                    n_folds=n_folds, seeds=seeds)
    logger.info("evaluate_graph | Temporal+Graph AUPRC=%.4f ± %.4f.",
                graph_cv['mean']['auprc'], graph_cv['std']['auprc'])

    # Out-of-fold predictions for per-scenario metrics + detection latency.
    scenarios = np.array([m.get('scenario', 0) for m in metas])
    oof = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in temporal_stratified_kfold(user_ids, y, n_folds=n_folds, seed=seeds[0]):
        oof[val_idx] = fit(train_idx, val_idx, seeds[0])
    scenario_metrics = per_scenario_metrics(oof, y, scenarios)
    threshold = compute_metrics(oof, y)['threshold_best']
    latency = detection_latency(oof, metas, threshold, attack_windows)
    logger.info("evaluate_graph | detection: %d/%d insiders flagged, median latency=%s days.",
                latency['detected_count'], latency['total_insiders'], latency['median_days'])

    # ── Progressive ladder table ────────────────────────────────────────────
    xgb_cv = _load_baseline(args.baseline_results_dir, "xgboost_results.json")
    mlp_cv = _load_baseline(args.baseline_results_dir, "mlp_results.json")
    temporal_cv = _load_baseline(args.baseline_results_dir, "temporal_results.json")
    ladder = {}
    if xgb_cv is not None:
        ladder["XGBoost"] = xgb_cv
    if mlp_cv is not None:
        ladder["MLP"] = mlp_cv
    if temporal_cv is not None:
        ladder["TemporalCNN"] = temporal_cv
    ladder["TemporalGraph"] = graph_cv
    table = format_results_table(ladder)
    logger.info("evaluate_graph | progressive ladder:\n%s", table)

    gate = _evaluate_gate(graph_cv["mean"]["auprc"], temporal_cv)

    results = {
        "version": args.version,
        "model": "temporal_graph",
        "n_windows": int(len(y)),
        "n_positive": n_pos,
        "n_folds": n_folds,
        "seeds": seeds,
        "cross_validation": graph_cv,
        "ladder_table": table,
        "gate": gate,
        "per_scenario": scenario_metrics,
        "detection_latency": latency,
        "oof_threshold": threshold,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("evaluate_graph | results written to %s.", output_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
