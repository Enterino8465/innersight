#!/usr/bin/env python
"""CLI to validate Module 1 baselines against known insiders (the Phase 2 gate).

Loads the z-scored deviations computed by ``compute_baselines`` and checks that
every known insider's deviations spike during their attack window relative to
their pre-attack behaviour. Prints GATE PASSED / GATE FAILED.

Usage:
    python -m innersight.scripts.validate_baselines \
        --version r4.2 --store-dir feature_store --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from innersight.config import setup_logging
from innersight.data.answers import load_insiders
from innersight.data.feature_store import FeatureStore
from innersight.schema import FEATURE_NAMES
from innersight.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)

# Default insider attack-window mean |z| an insider must exceed to pass the gate.
DEFAULT_GATE_THRESHOLD = 2.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Validate Module 1 baselines against known insiders (Phase 2 gate).',
    )
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', required=True, metavar='PATH', help='Path to the dataset directory')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--threshold', type=float, default=DEFAULT_GATE_THRESHOLD,
                   help=f'Min attack-window mean |z| per insider (default: {DEFAULT_GATE_THRESHOLD})')
    return p.parse_args(argv)


def _mean_abs_z(df, feature_cols: list[str]) -> float:
    """Mean absolute z-score over all rows and feature columns (0.0 if empty)."""
    if df.empty:
        return 0.0
    return float(np.abs(df[feature_cols].to_numpy(dtype=float)).mean())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    seed_everything(42)

    store = FeatureStore(args.store_dir)
    data_dir = Path(args.data_dir)

    # 1. Deviations must already be computed.
    deviations = store.load_deviations(args.version)
    if deviations is None:
        logger.error(
            "validate_baselines | no deviations cached for %s in %s. "
            "Run compute_baselines first.", args.version, args.store_dir,
        )
        return 1

    feature_cols = [c for c in FEATURE_NAMES if c in deviations.columns]
    deviations = deviations.copy()
    deviations["date"] = pd.to_datetime(deviations["date"])

    # 2. Insider ground truth.
    answers_dir = data_dir / "answers"
    if not answers_dir.exists():
        logger.error("validate_baselines | no answers/ directory at %s; cannot validate.", answers_dir)
        return 1
    insiders = load_insiders(answers_dir, args.version)
    if not insiders:
        logger.error("validate_baselines | no insiders found for %s; nothing to validate.", args.version)
        return 1

    insider_ids = {r.user_id for r in insiders}

    # 3-4. Per-insider attack vs pre-attack deviations.
    results = []
    for record in insiders:
        rows = deviations[deviations["user"] == record.user_id].sort_values("date")
        if rows.empty:
            logger.warning("validate_baselines | insider %s has no deviation rows; marking FAIL.",
                           record.user_id)
            results.append((record, 0.0, 0.0, 0.0, False))
            continue

        start = record.attack_start.normalize()
        end = record.attack_end.normalize()
        days = rows["date"].dt.normalize()
        attack_rows = rows[(days >= start) & (days <= end)]
        pre_rows = rows[days < start]

        attack_z = _mean_abs_z(attack_rows, feature_cols)
        pre_z = _mean_abs_z(pre_rows, feature_cols)
        max_z = (
            float(np.abs(attack_rows[feature_cols].to_numpy(dtype=float)).max())
            if not attack_rows.empty else 0.0
        )
        lift = attack_z / pre_z if pre_z > 0 else float("inf")
        passed = attack_z > args.threshold
        results.append((record, attack_z, pre_z, max_z, passed))

        logger.info(
            "insider %-10s scenario=%d | pre-attack |z|=%.3f  attack |z|=%.3f  max |z|=%.3f  "
            "lift=%.2fx  %s",
            record.user_id, record.scenario, pre_z, attack_z, max_z, lift,
            "PASS" if passed else "FAIL",
        )

    # 5. Normal-user baseline for context.
    normal_rows = deviations[~deviations["user"].isin(insider_ids)]
    normal_z = _mean_abs_z(normal_rows, feature_cols)
    logger.info(
        "validate_baselines | normal users (%d) mean |z|=%.3f over %d rows",
        normal_rows["user"].nunique(), normal_z, len(normal_rows),
    )

    # 6. Gate verdict.
    n_pass = sum(1 for *_, passed in results if passed)
    all_passed = n_pass == len(results)
    logger.info(
        "validate_baselines | %s: %d/%d insiders exceed |z| > %.2f",
        args.version, n_pass, len(results), args.threshold,
    )
    if all_passed:
        logger.info("GATE PASSED — all insiders are detectable during their attack windows.")
        return 0
    logger.error("GATE FAILED — %d insider(s) below threshold; baseline may be broken.",
                 len(results) - n_pass)
    return 1


if __name__ == '__main__':
    sys.exit(main())
