"""Stage 7, Phase 2 — the test split. One pass, both models, then stop.

::

    python -m evaluation.run_phase2            # the real thing, ~2 min
    python -m evaluation.run_phase2 --sample   # smoke test on the 30-user data

Before this runs, three things must already be true, and two of them are
checked rather than trusted:

1. **Phase 1 has finished** — its artifact must exist on disk (§4). Once test
   is opened, no model may change; any adjustment made afterwards is fitting
   to the test set.
2. **The B@7 control and the ``pos_weight`` ablation have been run.** Both are
   validation-only and both belong to Phase 1. Not checked here, because
   neither writes a file this module can require without coupling to a
   training run — but the report will have a hole where their tables go.
3. **Nothing about either model has changed since Stage 6.** The checkpoints
   are read, not rewritten.

The run appends to ``outputs/TEST_SPLIT_OPENED.json``. A second opening is
refused unless ``--force --reason '...'`` is given, and the reason is recorded
so the report can disclose it. That is not a lock — it is a note to the person
about to do it that the result is no longer a single-pass test number.

Exit codes: ``0`` clean, ``2`` bad device, ``3`` a guard failed, ``4`` the
ordering or the audit trail refused the run.
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import DEVICE_CHOICES, DEVICE_PREFERENCE, SEEDS, resolve_device
from evaluation import testpass
from evaluation.corpus import EvaluationError

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_phase2",
        description=(
            "Stage 7 Phase 2 — score both models on the test split, once, "
            "using each checkpoint's own stored standardiser and threshold."
        ),
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="use the 30-user sample. Its numbers are meaningless and it does "
             "not count as opening the real test split.",
    )
    parser.add_argument(
        "--device", choices=DEVICE_CHOICES, default=DEVICE_PREFERENCE,
        help="use the device the models were trained on (default: %(default)s).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS), metavar="SEED",
        help="default: %(default)s.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="open the test split again after it has already been opened. "
             "Requires --reason. Only legitimate for a bug in the evaluation "
             "code — never for a disappointing result.",
    )
    parser.add_argument(
        "--reason", default="first pass", metavar="TEXT",
        help="recorded in outputs/TEST_SPLIT_OPENED.json alongside the "
             "timestamp (default: %(default)r).",
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

    try:
        device = resolve_device(args.device)
    except RuntimeError as error:
        logger.error("%s", error)
        return 2

    if args.force and args.reason == "first pass":
        logger.error(
            "--force requires an explicit --reason. It goes in the audit "
            "trail and, if the numbers are reported, in the write-up."
        )
        return 4

    try:
        artifact = testpass.require_phase1(sample=args.sample)
        logger.info("phase 1 artifact found: %s", artifact)
        opening = testpass.record_opening(
            reason=args.reason, seeds=args.seeds, sample=args.sample,
            force=args.force,
        )
    except (testpass.PhaseOrderError, EvaluationError) as error:
        logger.error("%s", error)
        return 4

    logger.warning(
        "opening the TEST split (%s). From this point no model, threshold or "
        "standardiser may change. Recorded at %s.",
        "sample" if args.sample else "REAL", opening["when"],
    )

    result = testpass.run(seeds=args.seeds, sample=args.sample, device=device)
    testpass.write(result)

    if not result.passed:
        logger.error(
            "a fatal guard failed on the test pass. Report the failure — do "
            "not work around it (brief §9)."
        )
        return 3

    logger.info(
        "Test pass complete. Build the report with: python -m "
        "evaluation.run_stage7%s", " --sample" if args.sample else "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
