#!/usr/bin/env python
"""CLI to compute Module 1 per-user baselines and z-scored deviations.

Orchestrates the full Module 1 pipeline for a single CERT version:

    load data → daily features → global std reference → role cohorts →
    per-user EMA baseline → z-scored deviations → feature store + metadata

Results are written to the feature store (Parquet + JSON) so downstream
modules never recompute. A fresh cache is skipped unless ``--force`` is given.

Usage:
    python -m innersight.scripts.compute_baselines --version r4.2 --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from innersight.config import DEFAULT_BASELINE_CONFIG, setup_logging
from innersight.data.answers import get_malicious_dates
from innersight.data.feature_store import FeatureStore
from innersight.data.pipeline import load_version
from innersight.features.features import build_user_day_features
from innersight.models.baseline import (
    PerUserBaseline,
    compute_global_median_stds,
    compute_role_cohorts,
)
from innersight.schema import FEATURE_NAMES
from innersight.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Compute Module 1 per-user baselines and z-scored deviations for a CERT version.',
    )
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', required=True, metavar='PATH', help='Path to the dataset directory')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store output directory (default: 'feature_store')")
    p.add_argument('--force', action='store_true', help='Recompute even if the cache is fresh')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    store = FeatureStore(args.store_dir)

    # 3. Skip if the cache is already up to date with the source CSVs.
    if not args.force and not store.is_stale(args.version, data_dir):
        logger.info("compute_baselines | %s cache is fresh; nothing to do (use --force to recompute).",
                    args.version)
        return 0

    # 4. Load all logs + LDAP + answer labels for this version.
    try:
        dataset = load_version(data_dir, args.version)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("compute_baselines | failed to load %s from %s: %s", args.version, data_dir, exc)
        return 1

    # 5. Malicious (user, date) labels — answers/ is optional (e.g. unlabelled data).
    answers_dir = data_dir / 'answers'
    if answers_dir.exists():
        malicious_dates = get_malicious_dates(answers_dir, args.version)
    else:
        logger.warning("compute_baselines | no answers/ directory at %s; proceeding with no attack labels.",
                       answers_dir)
        malicious_dates = set()

    features_df = build_user_day_features(dataset.logs, malicious_dates)
    if features_df.empty:
        logger.error("compute_baselines | no features produced for %s (empty logs?). Aborting.", args.version)
        return 1

    # 6. Persist the raw daily features.
    store.save_features(args.version, features_df)

    # 7. Per-feature std reference used to floor the z-score denominator.
    global_median_stds = compute_global_median_stds(features_df)  # shape: (18,)

    # 8. Cohort priors for cold-starting users (role → department → global).
    cohort_stats = compute_role_cohorts(features_df, dataset.ldap)

    # 9. Per-user EMA baseline configured from the project defaults.
    baseline = PerUserBaseline.from_config(DEFAULT_BASELINE_CONFIG, global_median_stds)

    # 10. Z-score every user-day against their evolving baseline.
    deviations_df = baseline.compute_deviations_df(features_df, cohort_stats)

    # 11. Persist the deviation matrices.
    store.save_deviations(args.version, deviations_df)

    # 12. Record provenance + summary statistics for reproducibility.
    dev_values = deviations_df[FEATURE_NAMES].to_numpy(dtype=float)  # shape: (n_rows, 18)
    mean_abs_deviation = float(np.abs(dev_values).mean()) if dev_values.size else 0.0
    user_count = int(features_df['user'].nunique())
    insider_count = len(dataset.insiders)
    metadata = {
        'version': args.version,
        'seed': args.seed,
        'baseline_config': dict(DEFAULT_BASELINE_CONFIG),
        'user_count': user_count,
        'insider_count': insider_count,
        'malicious_day_count': len(malicious_dates),
        'feature_row_count': int(len(features_df)),
        'deviation_row_count': int(len(deviations_df)),
        'mean_abs_deviation': mean_abs_deviation,
        'provenance': dataset.provenance,
    }
    store.save_metadata(args.version, metadata)

    logger.info(
        "compute_baselines | %s done: %d users, %d insiders, %d rows, mean|dev|=%.4f → %s",
        args.version, user_count, insider_count, len(deviations_df), mean_abs_deviation, args.store_dir,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
