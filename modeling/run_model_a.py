"""Stage 5 entry point: Model A, three seeds, validation only.

Model A is the control arm of the experiment. It sees one day's eighteen
z-scores and nothing else, so whatever Model B gains over it in Stage 7 is the
measured value of temporal context — the only quantity this project set out to
find. Everything except the architecture is shared machinery living in
:mod:`modeling.data`, :mod:`modeling.train` and :mod:`modeling.runner`; this
file is the wiring::

    python -m modeling.run_model_a                    # 3 seeds, full data
    python -m modeling.run_model_a --sample           # fast smoke test
    python -m modeling.run_model_a --no-pos-weight    # the ablation
    python -m modeling.run_model_a --device cpu       # force the backend

**The ablation is worth the five minutes.** ``--no-pos-weight`` trains the
identical model on the identical rows with plain BCE, and writes to its own
output paths so it cannot overwrite the real result. Whatever the two tables
say, the claim that class weighting matters at 0.4% positives has then been
*measured* rather than asserted. If it turns out to matter less than expected,
that is a finding: a model can rank correctly while predicting tiny
probabilities everywhere, and AUPRC only cares about the ranking.

**This stage does not open the test split.** Not once, not to peek. Train on
train, select on validation, report validation; Stage 7 opens test a single
time, for both models together.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from config import (
    BATCH_SIZE,
    DEVICE_CHOICES,
    DEVICE_PREFERENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_A_HIDDEN,
    MODEL_A_NAME,
    SEEDS,
    resolve_device,
)
from modeling.data import SplitData
from modeling.models import ModelA
from modeling.runner import run_experiment

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modeling.run_model_a",
        description="Stage 5 — train Model A (one day, MLP) across three seeds.",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="use the 30-user sample. Seconds, not minutes. The numbers it "
             "produces are meaningless (30 users, ~30%% insider rate) and "
             "must never be reported as results — it only proves the code "
             "runs end to end. Writes to .sample output paths.",
    )
    parser.add_argument(
        "--device", choices=DEVICE_CHOICES, default=DEVICE_PREFERENCE,
        help="auto prefers MPS and falls back to CPU. Both models must use "
             "the SAME device, or a second variable has entered a two-model "
             "comparison (default: %(default)s).",
    )
    parser.add_argument(
        "--no-pos-weight", action="store_true",
        help="the ablation: train with plain BCE. Writes to separate output "
             "paths tagged 'nopw' so the real result is never overwritten.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS), metavar="SEED",
        help="default: %(default)s. All three are reported as mean +/- sd; "
             "one seed is one draw of the split lottery, not a result.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help=f"override config.MAX_EPOCHS (currently {MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="warnings and errors only.",
    )
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

    logger.info(
        "Model A — %d -> %d -> 1 MLP, raw logit out. Device requested %r, "
        "resolved %r. batch %d, lr %g, max epochs %d.",
        18, MODEL_A_HIDDEN, args.device, device, BATCH_SIZE, LEARNING_RATE,
        args.max_epochs or MAX_EPOCHS,
    )
    if args.sample:
        logger.warning(
            "--sample: 30 users, ~30%% insiders. This is a smoke test. Its "
            "AUPRC is not a result and must not appear in the report."
        )

    summary: dict[str, Any] = run_experiment(
        model_name=MODEL_A_NAME,
        model_factory=ModelA,
        view_selector=lambda split: SplitData.point_view(split),
        input_kind="point",
        seeds=args.seeds,
        sample=args.sample,
        device=device,
        use_pos_weight=not args.no_pos_weight,
        tag="nopw" if args.no_pos_weight else "",
        max_epochs=args.max_epochs,
    )

    if summary["leakage_suspected_any_seed"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
