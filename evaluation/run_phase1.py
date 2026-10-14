"""Stage 7, Phase 1 — the validation analyses, from checkpoints that exist.

Nothing here trains and nothing here opens the test split::

    python -m evaluation.run_phase1            # 3 seeds, full data, ~2 min
    python -m evaluation.run_phase1 --sample   # smoke test, seconds
    python -m evaluation.run_phase1 --device mps

**Read the detection-latency block before anything else.** It is the analysis
that can change what the project is allowed to claim: if recall climbs steeply
with position inside the attack window, "detects insiders" has to become
"identifies ongoing exfiltration campaigns" and the median first-detection day
has to be stated beside it. That decision belongs in the report, and this run
is what settles it.

Exit codes: ``0`` all guards passed, ``3`` a guard failed. A failure here is
worth stopping for — every Phase 2 number depends on the same loading path
these guards check.
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import DEVICE_CHOICES, DEVICE_PREFERENCE, SEEDS, resolve_device
from evaluation import phase1

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_phase1",
        description=(
            "Stage 7 Phase 1 — detection latency, the P@10 inspection and the "
            "guard block, all on validation. Does not open the test split."
        ),
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="use the 30-user sample checkpoints and data. Seconds, not "
             "minutes. Its numbers are meaningless and must never be "
             "reported; it only proves the code runs end to end.",
    )
    parser.add_argument(
        "--device", choices=DEVICE_CHOICES, default=DEVICE_PREFERENCE,
        help="use the device the models were TRAINED on (MPS) — inference on "
             "another backend moves the last decimal and the "
             "reproduces-training guard will say so (default: %(default)s).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS), metavar="SEED",
        help="default: %(default)s. Latency and bucket recall are pooled "
             "across all of them; one seed is a draw of the split lottery.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, metavar="K",
        help="how many top-scoring benign rows to inspect per model and seed "
             "(brief §8; default: %(default)s).",
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

    if args.sample:
        logger.warning(
            "--sample: 30 users, ~30%% insiders. A smoke test. Its latency "
            "medians and precision figures are not results."
        )

    logger.info(
        "Stage 7 Phase 1 — validation only, seeds %s, device %r. No training, "
        "no threshold selection, no test rows.", args.seeds, device,
    )

    result = phase1.run(
        seeds=args.seeds, sample=args.sample, device=device, top_k=args.top_k
    )
    phase1.write(result)

    if not result.passed:
        logger.error(
            "a fatal guard failed — do not proceed to Phase 2 until it is "
            "understood. Brief §9: stop and report it rather than working "
            "around it."
        )
        return 3
    logger.info(
        "Phase 1 complete. Finish the B@7 control (evaluation.run_b7) and the "
        "pos_weight ablation (modeling.run_model_a --no-pos-weight) before "
        "opening test."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
