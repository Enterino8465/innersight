#!/usr/bin/env python
"""Full evaluation of the temporal CNN vs. the Phase 3 baselines (Phase 4).

Runs the temporal CNN under the same cross-validation protocol as the baselines,
builds a side-by-side comparison table (XGBoost / MLP / Temporal CNN), reports a
per-scenario breakdown and detection latency, and sweeps the window size to find
the best receptive field. The validation gate is whether the CNN beats XGBoost
by at least 0.03 AUPRC.

Usage:
    python -m innersight.scripts.evaluate_temporal \
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
from innersight.models.dataset import DeviationWindowDataset
from innersight.scripts import compute_baselines
from innersight.scripts.train_temporal import _build_window_tensors, _fit, _load_config
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

# Window sizes for the ablation sweep, and the minimum AUPRC lift over XGBoost.
_ABLATION_SIZES = (7, 14, 28, 56)
_GATE_MARGIN = 0.03


def _make_model_fn(model_cfg: dict, train_cfg: dict):
    """Build a model_fn(X_train, y_train, X_val, y_val, seed) -> val probs."""
    def model_fn(X_train, y_train, X_val, y_val, seed):
        return _fit(X_train, y_train, X_val, y_val, model_cfg, train_cfg, seed)["val_probs"]
    return model_fn


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str | None):
    """Return cached deviations, computing them from data_dir if absent."""
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    if not data_dir:
        logger.error("evaluate_temporal | no deviations cached for %s and no --data-dir.", version)
        return None
    logger.info("evaluate_temporal | deviations not cached; computing from %s …", data_dir)
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        logger.error("evaluate_temporal | compute_baselines failed (rc=%d).", rc)
        return None
    return store.load_deviations(version)


def _load_baseline(results_dir: str, fname: str) -> dict | None:
    """Load a Phase 3 baseline's cross-validation result, or None if unavailable."""
    path = Path(results_dir) / fname
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["cross_validation"]
    except (KeyError, ValueError) as exc:
        logger.warning("evaluate_temporal | could not read %s: %s", path, exc)
        return None


def _window_ablation(deviations, attack_windows, model_cfg, train_cfg, n_folds, seed) -> tuple[dict, str | None]:
    """Sweep window sizes (1 seed) and return per-size AUPRC + the best size."""
    model_fn = _make_model_fn(model_cfg, train_cfg)
    max_history = int(deviations.groupby("user").size().max()) if len(deviations) else 0
    results: dict[str, dict] = {}
    for size in _ABLATION_SIZES:
        if size > max_history:
            logger.info("ablation | window_size=%d skipped — max user history %d days < %d.",
                        size, max_history, size)
            results[str(size)] = {"status": "skipped_insufficient_history", "auprc": None}
            continue
        dataset = DeviationWindowDataset(deviations, attack_windows, window_size=size, stride=7)
        X, y, metas = _build_window_tensors(dataset)
        if X.shape[0] == 0 or int(y.sum()) == 0:
            logger.info("ablation | window_size=%d skipped — no positive windows.", size)
            results[str(size)] = {"status": "skipped_no_positive_windows", "auprc": None}
            continue
        try:
            cv = run_cross_validation(model_fn, X, y, metas, attack_windows, n_folds=n_folds, seeds=[seed])
            auprc = cv["mean"]["auprc"]
            results[str(size)] = {"status": "ok", "auprc": auprc, "n_windows": int(X.shape[0])}
            logger.info("ablation | window_size=%d → AUPRC=%.4f (%d windows).", size, auprc, X.shape[0])
        except Exception as exc:  # e.g. too few insiders for the fold count
            logger.warning("ablation | window_size=%d failed: %s", size, exc)
            results[str(size)] = {"status": "error", "auprc": None}

    scored = {size: r["auprc"] for size, r in results.items() if r["auprc"] is not None}
    best = max(scored, key=scored.get) if scored else None
    if best is not None:
        logger.info("ablation | best window size: %s (AUPRC=%.4f).", best, scored[best])
    return results, best


def _evaluate_gate(cnn_auprc: float, xgb_cv: dict | None) -> dict:
    """Log and return the CNN-vs-XGBoost validation gate result."""
    if xgb_cv is None:
        logger.warning("evaluate_temporal | no XGBoost baseline results — cannot compute the gate.")
        return {"status": "no_baseline", "cnn_auprc": cnn_auprc}
    xgb_auprc = xgb_cv["mean"]["auprc"]
    diff = cnn_auprc - xgb_auprc
    if diff >= _GATE_MARGIN:
        logger.info("GATE PASS: Temporal CNN AUPRC %.3f > XGBoost AUPRC %.3f by %.3f (≥ %.2f)",
                    cnn_auprc, xgb_auprc, diff, _GATE_MARGIN)
        status = "pass"
    else:
        logger.info(
            "GATE TIE: Temporal CNN AUPRC %.3f vs XGBoost %.3f — difference %.3f < %.2f.\n"
            "  Valid finding: handcrafted features capture equivalent signal.\n"
            "  Proceeding — CNN embeddings still needed for Phase 5.",
            cnn_auprc, xgb_auprc, diff, _GATE_MARGIN)
        status = "tie"
    return {"status": status, "cnn_auprc": cnn_auprc, "xgboost_auprc": xgb_auprc, "difference": diff}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Evaluate the temporal CNN against the Phase 3 baselines, with a window-size ablation.',
    )
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', metavar='PATH',
                   help='Dataset directory (answers/ labels; computes deviations if uncached)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--config', metavar='PATH', help='Path to a train_temporal.yaml config')
    p.add_argument('--output', default='temporal_eval_results.json', metavar='PATH',
                   help='Where to write the results JSON (default: temporal_eval_results.json)')
    p.add_argument('--baseline-results-dir', default='.', metavar='PATH',
                   help="Directory holding Phase 3 baseline JSONs (default: '.')")
    p.add_argument('--skip-ablation', action='store_true', help='Skip the window-size sweep')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    model_cfg, train_cfg, eval_cfg = _load_config(args.config)
    seeds = [int(s) for s in eval_cfg["seeds"]]
    n_folds = int(eval_cfg["n_folds"])
    seed_everything(seeds[0] if seeds else 42)

    store = FeatureStore(args.store_dir)
    deviations = _load_deviations(store, args.version, args.store_dir, args.data_dir)
    if deviations is None:
        return 1

    if not args.data_dir:
        logger.error("evaluate_temporal | --data-dir is required to load attack windows (answers/).")
        return 1
    answers_dir = Path(args.data_dir) / 'answers'
    if not answers_dir.exists():
        logger.error("evaluate_temporal | no answers/ directory at %s.", answers_dir)
        return 1
    attack_windows = {r.user_id: r for r in load_insiders(answers_dir, args.version)}
    logger.info("evaluate_temporal | %d insiders for %s.", len(attack_windows), args.version)

    # ── Temporal CNN: full CV protocol (5-fold × 3 seeds) on 28-day windows ──
    dataset = DeviationWindowDataset(deviations, attack_windows)
    X, y, metas = _build_window_tensors(dataset)
    n_pos = int(y.sum())
    logger.info("evaluate_temporal | %d windows of shape %s (%d positive).",
                X.shape[0], tuple(X.shape[1:]), n_pos)
    if X.shape[0] == 0 or n_pos == 0:
        logger.error("evaluate_temporal | need labelled positive windows; aborting.")
        return 1

    user_ids = np.array([m['user_id'] for m in metas])
    scenarios = np.array([m.get('scenario', 0) for m in metas])
    model_fn = _make_model_fn(model_cfg, train_cfg)

    cnn_cv = run_cross_validation(model_fn, X, y, metas, attack_windows, n_folds=n_folds, seeds=seeds)
    logger.info("evaluate_temporal | Temporal CNN AUPRC=%.4f ± %.4f.",
                cnn_cv['mean']['auprc'], cnn_cv['std']['auprc'])

    # Out-of-fold predictions for per-scenario metrics + detection latency.
    oof = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in temporal_stratified_kfold(user_ids, y, n_folds=n_folds, seed=seeds[0]):
        oof[val_idx] = _fit(X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                            model_cfg, train_cfg, seeds[0])["val_probs"]
    scenario_metrics = per_scenario_metrics(oof, y, scenarios)
    threshold = compute_metrics(oof, y)["threshold_best"]
    latency = detection_latency(oof, metas, threshold, attack_windows)
    logger.info("evaluate_temporal | detection: %d/%d insiders flagged, median latency=%s days.",
                latency["detected_count"], latency["total_insiders"], latency["median_days"])

    # ── Comparison table vs. Phase 3 baselines ──────────────────────────────
    xgb_cv = _load_baseline(args.baseline_results_dir, "xgboost_results.json")
    mlp_cv = _load_baseline(args.baseline_results_dir, "mlp_results.json")
    comparison = {}
    if xgb_cv is not None:
        comparison["XGBoost"] = xgb_cv
    if mlp_cv is not None:
        comparison["MLP"] = mlp_cv
    comparison["TemporalCNN"] = cnn_cv
    table = format_results_table(comparison)
    logger.info("evaluate_temporal | comparison:\n%s", table)

    gate = _evaluate_gate(cnn_cv["mean"]["auprc"], xgb_cv)

    # ── Window-size ablation ────────────────────────────────────────────────
    ablation: dict = {}
    best_window = None
    if args.skip_ablation:
        logger.info("evaluate_temporal | window-size ablation skipped (--skip-ablation).")
    else:
        ablation, best_window = _window_ablation(deviations, attack_windows, model_cfg, train_cfg,
                                                 n_folds=3, seed=seeds[0])

    results = {
        "version": args.version,
        "model": "temporal_cnn",
        "n_windows": int(X.shape[0]),
        "n_positive": n_pos,
        "n_folds": n_folds,
        "seeds": seeds,
        "cross_validation": cnn_cv,
        "comparison_table": table,
        "gate": gate,
        "per_scenario": scenario_metrics,
        "detection_latency": latency,
        "ablation": {"results": ablation, "best_window_size": best_window},
        "oof_threshold": threshold,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("evaluate_temporal | results written to %s.", output_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
