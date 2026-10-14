"""Stage 7's last step: turn the measured artifacts into the deliverables.

::

    python -m evaluation.run_stage7            # report, JSON and both figures
    python -m evaluation.run_stage7 --no-figures

This reads and never measures. Both phase artifacts must already exist; the
B@7 control, the ``pos_weight`` ablation and the figures are optional, and a
missing one produces a visible *"not run"* in the report rather than a gap a
reader would not notice. That is §11's cut order made operational: cut from
the bottom, and the report says what was cut.

Deliverables, in the brief's order:

1. ``STAGE7_REPORT.md``
2. ``outputs/final_results.json``
3. ``outputs/figures/stage7_comparison.png``
4. ``outputs/figures/stage7_detection_latency.png``
5. ``outputs/insider_detection.csv`` — written by Phase 1
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import ensure_directories
from evaluation import figures, report

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_stage7",
        description="Write STAGE7_REPORT.md, final_results.json and the figures.",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="build from the sample artifacts. The report is stamped with a "
             "warning saying its numbers are meaningless.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="deliverables 1 and 2 only — the two that must survive if time "
             "runs out.",
    )
    parser.add_argument("--quiet", action="store_true", help="warnings and errors only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    ensure_directories()

    try:
        sources = report.Sources.load(sample=args.sample)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 4

    missing = [
        name
        for name, value in (
            ("B@7 capacity control (§6)", sources.control_b7),
            ("pos_weight ablation (§7)", sources.ablation_a),
            ("Model A validation metrics", sources.val_a),
            ("Model B validation metrics", sources.val_b),
        )
        if value is None
    ]
    for name in missing:
        logger.warning("missing: %s — the report will say 'not run'", name)

    report.write(sources)

    if not args.no_figures:
        figures.comparison(sources.phase2, sources.val_a, sources.val_b)
        figures.latency(sources.phase2)

    failed = [
        guard
        for block in (sources.phase1, sources.phase2)
        for guard in block["guards"]
        if not guard["passed"]
    ]
    if failed:
        logger.warning(
            "%d guard(s) failed in the runs this report is built from; they "
            "are listed in Appendix A and must be addressed before "
            "submission.", len(failed),
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
