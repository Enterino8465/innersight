"""Stage 6 entry point: Model B, three seeds, validation only.

Model B is the experimental arm. It reads the 28 days ending on the day being
judged and predicts the same label from the same rows Model A predicted, under
the same standardiser, loss, optimiser, learning rate, batch size,
early-stopping rule, threshold sweep, seeds and device. **One variable
changes: temporal context.** Everything that makes that true lives in
:mod:`modeling.data`, :mod:`modeling.train` and :mod:`modeling.runner`, and
Stage 6 did not edit any of them — this file differs from ``run_model_a.py``
in exactly two expressions, the model factory and the view selector::

    python -m modeling.run_model_b                    # 3 seeds, full data
    python -m modeling.run_model_b --sample           # fast smoke test
    python -m modeling.run_model_b --no-pos-weight    # the ablation
    python -m modeling.run_model_b --device cpu       # force the backend

**Use the same device Model A used.** Model A trained on MPS and it is
recorded in every one of its checkpoints. Floating-point results differ
subtly between backends, so training B elsewhere puts a second variable into a
two-model comparison and a reader can fairly ask whether the difference is
architecture or arithmetic.

**Read the pre-flight block before believing any result.** ``pos_weight``
must print 229.7 / 225.3 / 232.1 for seeds 42 / 123 / 456. If it prints ~244,
the 27-day row mask is not being applied before the weight is computed, and
the two models are not on the same rows — 244 is the whole-table figure. That
one number is the cheapest tripwire for the most likely way this stage breaks
silently.

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
    MODEL_B_NAME,
    N_FEATURES,
    SEEDS,
    WINDOW_DAYS,
    resolve_device,
)
from modeling.data import SplitData
from modeling.models import ModelB
from modeling.runner import run_experiment

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modeling.run_model_b",
        description=(
            "Stage 6 — train Model B (28 days, 1D CNN) across three seeds."
        ),
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
        help=f"override config.MAX_EPOCHS (currently {MAX_EPOCHS}). Raising "
             f"it for Model B alone would bias the comparison — if patience "
             f"is clipping a still-improving run, raise it for BOTH models "
             f"and re-run Model A too (90 seconds).",
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
        "Model B — 1D CNN over (%d features, %d days), raw logit out. Device "
        "requested %r, resolved %r. batch %d, lr %g, max epochs %d.",
        N_FEATURES, WINDOW_DAYS, args.device, device, BATCH_SIZE,
        LEARNING_RATE, args.max_epochs or MAX_EPOCHS,
    )
    logger.info(
        "identical-data tripwire: pos_weight must print ~229.7 / 225.3 / "
        "232.1 for seeds 42 / 123 / 456. ~244 means the row mask is not "
        "applied before the weight and the two models are not on the same "
        "rows."
    )

    if args.sample:
        logger.warning(
            "--sample: 30 users, ~30%% insiders. This is a smoke test. Its "
            "AUPRC is not a result and must not appear in the report."
        )

    summary: dict[str, Any] = run_experiment(
        model_name=MODEL_B_NAME,
        model_factory=ModelB,
        view_selector=lambda split: SplitData.window_view(split),
        input_kind="window",
        seeds=args.seeds,
        sample=args.sample,
        device=device,
        use_pos_weight=not args.no_pos_weight,
        tag="nopw" if args.no_pos_weight else "",
        max_epochs=args.max_epochs,
    )

    _log_patience_watch(summary)

    if summary["leakage_suspected_any_seed"]:
        return 3
    return 0


def _log_patience_watch(summary: dict[str, Any]) -> None:
    """Warn if early stopping plausibly clipped a still-improving Model B run.

    All three of Model A's seeds stopped on patience with room to spare (best
    epochs 16 / 40 / 29, stopped at 23 / 47 / 36), so patience-7 was never
    binding against the epoch cap. A 4,897-parameter CNN can converge more
    slowly than a 641-parameter MLP, and a Model B whose best epoch is its
    *last* epoch was still improving when the cap cut it off — which would
    bias the comparison against B, in the direction of the hypothesis being
    tested. That is the one bias worth being loud about.

    The fix, if it fires, is to raise ``EARLY_STOP_PATIENCE`` (or
    ``MAX_EPOCHS``) and **re-run both models**. Raising it for B alone turns
    the training budget into a second variable.
    """
    for seed_run in summary["per_seed"]:
        epochs_run, best = seed_run["epochs_run"], seed_run["best_epoch"]
        if not seed_run["stopped_early"] and best >= epochs_run - 1:
            logger.warning(
                "seed %d: best epoch %d of %d, and the run hit the epoch cap "
                "rather than stopping on patience — validation AUPRC was "
                "still improving when training ended. Raise max epochs for "
                "BOTH models and re-run Model A too.",
                seed_run["seed"], best, epochs_run,
            )
        elif best >= epochs_run - 1:
            logger.warning(
                "seed %d: best epoch %d of %d run — the best epoch is the "
                "last one, which patience should have made impossible. "
                "Inspect the history before quoting this seed.",
                seed_run["seed"], best, epochs_run,
            )


if __name__ == "__main__":
    raise SystemExit(main())
