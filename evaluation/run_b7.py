"""Phase 1, task 2 — the capacity control: Model B on a 7-day window.

Model B has 7.6x Model A's parameters, so *"B wins because it is bigger"* is a
live alternative explanation for the entire result. Answering it does not need
a new architecture — it needs the *same* architecture with less context::

    python -m evaluation.run_b7            # 3 seeds, validation only, ~4 min
    python -m evaluation.run_b7 --sample   # smoke test, seconds

Convolutions plus global pooling do not care about sequence length, so
``ModelB(days=7)`` has **exactly** the parameter count of ``ModelB(days=28)``:
4,897 either way. Nothing about the model's capacity changes. The only thing
that changes is how far back it can see. Seven is the floor — the two dilated
k=3 layers have a receptive field of 7 days, and a shorter window would leave
the second layer convolving over padding.

| outcome | conclusion |
|---|---|
| B@7 lands between A and B@28 | context is the driver — demonstrated, not asserted |
| B@7 ~ B@28 | capacity was doing the work, and the report must say so |
| B@7 ~ A | the 28-day span specifically is what matters |

**This is a control, not a competing model.** It is never evaluated on test,
never appears in the headline table, and its checkpoints are written under a
``w7`` tag so :func:`evaluation.checkpoints.load_checkpoint` cannot pick them
up by accident.

Nothing in ``modeling/`` is touched. This file calls
:func:`modeling.runner.run_experiment` with a different view selector, exactly
as ``run_model_b.py`` does — which is also what makes the comparison fair, as
every other knob comes from the same shared code path.

One wart to know about, and it is deliberately not fixed here:
``runner.save_checkpoint`` stamps every window checkpoint with
``config.WINDOW_DAYS`` rather than the view's actual length, so these files
claim 28 days on disk. Stage 7 may not edit ``runner.py`` (brief §14), and the
control is validation-only and never reloaded, so the field is simply not
trusted — :func:`evaluation.checkpoints.load_variant_checkpoint` requires the
caller to state the window length and verifies the weights accept it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from config import (
    DEVICE_CHOICES,
    DEVICE_PREFERENCE,
    MODEL_B_NAME,
    SEEDS,
    metrics_path,
    resolve_device,
)
from modeling.data import SplitData
from modeling.models import ModelB, count_parameters
from modeling.runner import run_experiment

logger = logging.getLogger(__name__)

MIN_WINDOW_DAYS = 7
CONTROL_TAG = "w7"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_b7",
        description=(
            "Phase 1 capacity control — Model B on a shorter window. "
            "Identical architecture, identical parameter count, less context."
        ),
    )
    parser.add_argument(
        "--window-days", type=int, default=MIN_WINDOW_DAYS, metavar="N",
        help=f"days of context (default: %(default)s, the receptive-field "
             f"floor). Anything below {MIN_WINDOW_DAYS} is refused.",
    )
    parser.add_argument("--sample", action="store_true", help="30-user smoke test.")
    parser.add_argument(
        "--device", choices=DEVICE_CHOICES, default=DEVICE_PREFERENCE,
        help="must match the device Model A and Model B@28 trained on "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS), metavar="SEED",
        help="default: %(default)s.",
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

    days = int(args.window_days)
    if days < MIN_WINDOW_DAYS:
        logger.error(
            "a %d-day window is below the %d-day receptive field of ModelB's "
            "two dilated layers; the second convolution would see padding "
            "only and the control would measure the padding, not the context.",
            days, MIN_WINDOW_DAYS,
        )
        return 2

    try:
        device = resolve_device(args.device)
    except RuntimeError as error:
        logger.error("%s", error)
        return 2

    reference = count_parameters(ModelB())
    control = count_parameters(ModelB(days=days))
    logger.info(
        "capacity control — ModelB on %d days. Parameters: %s at 28 days, %s "
        "at %d days.", days, f"{reference:,}", f"{control:,}", days,
    )
    if control != reference:
        logger.error(
            "the two window lengths do not have the same parameter count "
            "(%d vs %d). The control only separates context from capacity if "
            "capacity is held exactly fixed — stop.", control, reference,
        )
        return 2
    logger.info(
        "identical parameter counts confirmed: the only variable between this "
        "run and Model B@28 is how far back the model can see."
    )
    logger.info(
        "validation only — this control is never scored on test, and its "
        "checkpoints are tagged %r so the headline loader refuses them.",
        CONTROL_TAG,
    )

    summary: dict[str, Any] = run_experiment(
        model_name=MODEL_B_NAME,
        model_factory=lambda: ModelB(days=days),
        view_selector=lambda split: SplitData.window_view(split, days),
        input_kind="window",
        seeds=args.seeds,
        sample=args.sample,
        device=device,
        use_pos_weight=True,
        tag=f"{CONTROL_TAG}" if days == MIN_WINDOW_DAYS else f"w{days}",
    )

    logger.info(
        "control metrics -> %s",
        metrics_path(
            MODEL_B_NAME, sample=args.sample,
            tag=CONTROL_TAG if days == MIN_WINDOW_DAYS else f"w{days}",
        ),
    )
    logger.info(
        "Compare this AUPRC against Model A's and Model B@28's. Between them "
        "means context is the driver; level with B@28 means capacity was "
        "doing the work and the report must say so."
    )
    return 3 if summary["leakage_suspected_any_seed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
