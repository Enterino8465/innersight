"""Turning a checkpoint and a split into probabilities, and nothing else.

Two rules govern this module, and both are §9 of the brief:

**The threshold is applied, never chosen.** :func:`describe_at_threshold`
computes precision, recall, F1 and the flagged-row count *at a cutoff handed
in*. There is no sweep here and :func:`modeling.metrics.sweep_threshold` is
not imported. Re-selecting a threshold on the rows being reported is the
easiest leak in the project to commit by accident and the hardest to see
afterwards, because the resulting number looks better and is not obviously
wrong.

**The scoring path is the training path.** :func:`modeling.train.predict` and
:func:`modeling.metrics.probabilities` do the work — the same batching, the
same single sigmoid, the same float64 conversion the training loop used every
epoch. A Stage 7 that reimplements inference is a Stage 7 that can disagree
with its own training curves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import BATCH_SIZE
from evaluation.checkpoints import Checkpoint
from evaluation.corpus import EvalSplit, EvaluationError
from modeling.metrics import (
    Evaluation,
    ThresholdChoice,
    evaluate,
    probabilities,
)
from modeling.train import predict

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Scored:
    """One model's probabilities over one split, with the metrics they imply.

    ``probs`` is kept alongside the aggregate because every Stage 7 analysis
    that is not a headline number — detection latency, the position buckets,
    the false-positive inspection — is a question about individual rows.
    """

    checkpoint: Checkpoint
    split: EvalSplit
    probs: np.ndarray
    evaluation: Evaluation
    threshold: float

    @property
    def model_name(self) -> str:
        return self.checkpoint.model_name

    @property
    def seed(self) -> int:
        return self.checkpoint.seed

    @property
    def flagged(self) -> np.ndarray:
        """Boolean mask of rows at or above the checkpoint's threshold."""
        return self.probs >= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "seed": self.seed,
            "split": self.split.name,
            "checkpoint": str(self.checkpoint.path),
            "threshold_applied": self.threshold,
            "threshold_source": "checkpoint (chosen on validation)",
            "n_flagged": int(self.flagged.sum()),
            "context_days": self.checkpoint.context_days,
            "n_parameters": self.checkpoint.n_parameters,
            **self.evaluation.as_dict(),
        }


def score(
    checkpoint: Checkpoint,
    split: EvalSplit,
    *,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
) -> Scored:
    """Score every target row of ``split`` with ``checkpoint``'s weights.

    The threshold applied is the one inside the checkpoint, full stop. It is
    passed to :func:`modeling.metrics.evaluate` as an already-made choice so
    that ``choose_threshold`` never has a chance to fire.
    """
    model = checkpoint.build().to(device)
    view = checkpoint.view(split.data).to(device)

    if len(view) != split.n_rows:
        raise EvaluationError(
            f"{checkpoint.path.name} produced a view of {len(view)} rows for a "
            f"split of {split.n_rows} target rows"
        )

    probs = probabilities(predict(model, view, batch_size, device))
    if not np.isfinite(probs).all():
        raise EvaluationError(
            f"{checkpoint.path.name} produced {int((~np.isfinite(probs)).sum())} "
            f"non-finite prediction(s) on split {split.name!r}"
        )

    applied = describe_at_threshold(split.labels, probs, checkpoint.threshold)
    evaluation = evaluate(
        split.labels,
        probs,
        split.scenarios,
        choose_threshold=False,
        threshold=applied,
    )
    return Scored(
        checkpoint=checkpoint,
        split=split,
        probs=probs,
        evaluation=evaluation,
        threshold=checkpoint.threshold,
    )


def describe_at_threshold(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> ThresholdChoice:
    """Precision, recall and F1 **at a fixed cutoff** — a description, not a
    choice.

    The distinction is the whole of §9's warning. ``sweep_threshold`` searches
    for the cutoff that flatters these numbers most; this function is handed
    one and reports what it does. The returned object is a
    :class:`~modeling.metrics.ThresholdChoice` only because that is the type
    :func:`modeling.metrics.evaluate` accepts — the values in it are measured
    on the rows given, not selected over them.
    """
    labels = np.asarray(labels).astype(np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    predicted = probs >= threshold
    n_predicted = int(predicted.sum())
    positives = int(labels.sum())

    hits = float(labels[predicted].sum()) if n_predicted else 0.0
    precision = hits / n_predicted if n_predicted else 0.0
    recall = hits / positives if positives else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return ThresholdChoice(
        threshold=float(threshold),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        predicted_positives=n_predicted,
    )


def log_scored(scored: Scored, *, label: str | None = None) -> None:
    """The block a reader of the run log needs for one model on one split."""
    where = label or f"{scored.split.name} | {scored.model_name} seed {scored.seed}"
    ev = scored.evaluation
    logger.info(
        "%s | AUPRC %.4f (random %.5f, %.1fx)  AUROC %s  P@10 %.3f  P@50 %.3f",
        where, ev.auprc, ev.random_baseline, ev.lift,
        f"{ev.auroc:.4f}" if ev.auroc is not None else "n/a",
        ev.precision_at_k.get(10, float("nan")),
        ev.precision_at_k.get(50, float("nan")),
    )
    logger.info(
        "%s | threshold %.6f from checkpoint -> %s rows flagged, precision "
        "%.4f, recall %.4f",
        where, scored.threshold, f"{int(scored.flagged.sum()):,}",
        ev.threshold.precision if ev.threshold else float("nan"),
        ev.threshold.recall if ev.threshold else float("nan"),
    )
    for scenario in ev.per_scenario:
        logger.info("%s | %s", where, scenario.phrase())
