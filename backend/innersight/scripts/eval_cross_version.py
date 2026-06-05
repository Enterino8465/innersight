#!/usr/bin/env python
"""CLI to evaluate cross-version generalization (Phase 6).

Loads a model trained on one CERT version and scores it against a different
version's data, reporting how well the learned temporal/graph representations
transfer to completely different users and graph instances.

Usage:
    python -m innersight.scripts.eval_cross_version \
        --checkpoint checkpoints/fusion_r5.2.pt \
        --source-version r5.2 --target-version r4.2 \
        --target-data-dir /data/cert_r4.2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from innersight.config import setup_logging
from innersight.training.cross_version import evaluate_cross_version
from innersight.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Evaluate a trained model on a different CERT version (cross-version generalization).',
    )
    p.add_argument('--checkpoint', required=True, metavar='PATH', help='Trained model checkpoint (.pt)')
    p.add_argument('--source-version', required=True, help="Version the model was trained on, e.g. 'r5.2'")
    p.add_argument('--target-version', required=True, help="Version to evaluate against, e.g. 'r4.2'")
    p.add_argument('--target-data-dir', required=True, metavar='PATH', help='Target version dataset directory')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--output', default='cross_version_results.json', metavar='PATH',
                   help='Where to write the results JSON (default: cross_version_results.json)')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    seed_everything(42)

    if not Path(args.checkpoint).exists():
        logger.error("eval_cross_version | checkpoint not found: %s", args.checkpoint)
        return 1
    if not Path(args.target_data_dir).exists():
        logger.error("eval_cross_version | target data dir not found: %s", args.target_data_dir)
        return 1

    try:
        results = evaluate_cross_version(
            checkpoint_path=args.checkpoint,
            source_version=args.source_version,
            target_data_dir=args.target_data_dir,
            target_version=args.target_version,
            store_dir=args.store_dir,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        logger.error("eval_cross_version | evaluation failed: %s", exc)
        return 1

    logger.info("eval_cross_version | %s→%s AUPRC=%.4f (n=%d, pos=%d).",
                args.source_version, args.target_version,
                results.get("metrics", {}).get("auprc", 0.0),
                results.get("n_windows", 0), results.get("n_positive", 0))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("eval_cross_version | results written to %s.", output_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
