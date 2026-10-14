from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from config import (
    BENIGN_SCENARIO,
    LEAKAGE_AUPRC,
    MIN_DISTINCT_PREDICTIONS,
    PRECISION_AT_K,
    SCENARIOS,
    THRESHOLD_SWEEP_POINTS,
)

logger = logging.getLogger(__name__)


def probabilities(logits: torch.Tensor) -> np.ndarray:
    """Logits -> probabilities, as float64 numpy on the CPU.

    The single place the sigmoid is applied. The models emit raw logits and
    ``BCEWithLogitsLoss`` applies its own internally; putting one in
    ``forward`` as well would squash the output twice. float64 here is safe —
    this is numpy, off the accelerator, where MPS's float32-only rule does not
    apply — and it keeps ``distinct`` from undercounting through rounding.
    """
    return torch.sigmoid(logits.detach().float().cpu()).numpy().astype(np.float64)


@dataclass(frozen=True, slots=True)
class PredictionStats:
    """The spread of predicted probabilities — the collapse detector.

    If a weighted loss has failed to bite, the cheapest way to reduce loss is
    to output the same near-zero number for every row. The model converges to
    "nobody is an insider", which is 99.6% accurate and worthless, and **the
    loss curve looks perfectly healthy while it happens**. There is no
    exception and no warning. Counting distinct predicted values is what
    catches it: a model that is genuinely ranking produces tens of thousands
    of different scores, a collapsed one produces a handful.
    """

    minimum: float
    maximum: float
    mean: float
    std: float
    distinct: int
    n_rows: int

    @classmethod
    def of(cls, probs: np.ndarray) -> PredictionStats:
        return cls(
            minimum=float(probs.min()),
            maximum=float(probs.max()),
            mean=float(probs.mean()),
            std=float(probs.std()),
            distinct=int(np.unique(np.round(probs, 9)).size),
            n_rows=int(probs.size),
        )

    @property
    def collapse_floor(self) -> int:
        """How many distinct values a healthy model must produce here.

        ``MIN_DISTINCT_PREDICTIONS`` is the bar on a real validation split of
        ~90,000 rows. It cannot be the bar on 372 rows — no model can produce
        1,000 distinct scores for 372 inputs, so a fixed threshold would fire
        on every small run and be ignored by the time it mattered. A guard
        that cries wolf is a guard that has been turned off.
        """
        return min(MIN_DISTINCT_PREDICTIONS, max(2, self.n_rows // 10))

    @property
    def collapsed(self) -> bool:
        return self.distinct < self.collapse_floor

    def as_dict(self) -> dict[str, float | int]:
        return {
            "min": self.minimum, "max": self.maximum, "mean": self.mean,
            "std": self.std, "distinct": self.distinct,
            "n_rows": self.n_rows, "collapse_floor": self.collapse_floor,
        }


@dataclass(frozen=True, slots=True)
class ThresholdChoice:
    """A decision threshold, and the validation numbers that chose it."""

    threshold: float
    f1: float
    precision: float
    recall: float
    predicted_positives: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold, "f1": self.f1,
            "precision": self.precision, "recall": self.recall,
            "predicted_positives": self.predicted_positives,
        }


@dataclass(frozen=True, slots=True)
class ScenarioRecall:
    """Recall for one attack type, as a count first and a rate second."""

    scenario: int
    hits: int
    total: int

    @property
    def recall(self) -> float | None:
        return self.hits / self.total if self.total else None

    def phrase(self) -> str:
        if not self.total:
            return f"scenario {self.scenario}: no positive rows in this split"
        return (
            f"scenario {self.scenario}: {self.hits} of {self.total} attack "
            f"days ({self.hits / self.total:.1%})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario, "hits": self.hits,
            "attack_days": self.total, "recall": self.recall,
        }


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Everything Stage 5 reports about one model on one split."""

    auprc: float
    random_baseline: float
    auroc: float | None
    precision_at_k: dict[int, float]
    stats: PredictionStats
    n_rows: int
    n_positives: int
    threshold: ThresholdChoice | None = None
    per_scenario: tuple[ScenarioRecall, ...] = field(default_factory=tuple)

    @property
    def lift(self) -> float:
        """How many times better than guessing. The one-number summary."""
        return self.auprc / self.random_baseline if self.random_baseline else float("nan")

    @property
    def suspiciously_good(self) -> bool:
        """Above ~0.95 AUPRC here means leakage, not a triumph."""
        return self.auprc >= LEAKAGE_AUPRC

    def as_dict(self) -> dict[str, Any]:
        return {
            "auprc": self.auprc,
            "random_baseline": self.random_baseline,
            "lift_over_random": self.lift,
            "auroc": self.auroc,
            "precision_at_k": {str(k): v for k, v in self.precision_at_k.items()},
            "prediction_stats": self.stats.as_dict(),
            "n_rows": self.n_rows,
            "n_positives": self.n_positives,
            "threshold": self.threshold.as_dict() if self.threshold else None,
            "per_scenario": [s.as_dict() for s in self.per_scenario],
        }


def auprc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Average precision. Returns the baseline itself if one class is absent."""
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float(labels.mean())
    return float(average_precision_score(labels, probs))


def auroc(labels: np.ndarray, probs: np.ndarray) -> float | None:
    """AUROC, or None when it is undefined (a split with one class)."""
    if labels.sum() == 0 or labels.sum() == labels.size:
        return None
    return float(roc_auc_score(labels, probs))


def precision_at_k(labels: np.ndarray, probs: np.ndarray, k: int) -> float:
    """Of the k highest-scoring days, the fraction that were attack days.

    Ties are broken by taking the k rows a stable descending sort puts first,
    which is the pessimistic-neutral choice: it does not reorder tied rows to
    flatter the model.
    """
    k = min(k, labels.size)
    if k == 0:
        return float("nan")
    order = np.argsort(-probs, kind="stable")[:k]
    return float(labels[order].mean())


def sweep_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    points: int = THRESHOLD_SWEEP_POINTS,
) -> ThresholdChoice:
    """Pick the probability cutoff that maximises F1 **on validation**.

    Never assume 0.5. With ``pos_weight`` in the loss the outputs are
    deliberately not calibrated: the model is trained to over-predict the rare
    class, so its "yes" region can sit anywhere. A model without its threshold
    is not a working system, and recomputing the threshold on test data at
    report time is leakage of exactly the kind that is easy to do by accident
    — so the chosen value is saved inside the checkpoint.

    Candidates are quantiles of the observed scores rather than a linear grid
    over [0, 1]: the outputs of a weighted model cluster, and a linear grid
    spends most of its budget on empty stretches of the axis.
    """
    if labels.sum() == 0:
        return ThresholdChoice(0.5, 0.0, 0.0, 0.0, int((probs >= 0.5).sum()))

    quantiles = np.quantile(probs, np.linspace(0.0, 1.0, points))
    candidates = np.unique(np.concatenate([quantiles, probs[labels == 1]]))
    positives = labels.sum()

    best = ThresholdChoice(0.5, -1.0, 0.0, 0.0, 0)
    for threshold in candidates:
        predicted = probs >= threshold
        n_predicted = int(predicted.sum())
        if n_predicted == 0:
            continue
        hits = float(labels[predicted].sum())
        precision = hits / n_predicted
        recall = hits / positives
        if precision + recall == 0.0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best.f1:
            best = ThresholdChoice(
                threshold=float(threshold), f1=float(f1),
                precision=float(precision), recall=float(recall),
                predicted_positives=n_predicted,
            )
    return best


def per_scenario_recall(
    labels: np.ndarray,
    scenarios: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> tuple[ScenarioRecall, ...]:
    """Recall for scenarios 1, 2 and 3 separately, at the chosen threshold.

    Separately, because an unweighted average across the three is a
    scenario-2 result wearing a hat: scenario 2 supplies ~88% of every split's
    positive rows. A model that finds every scenario-2 day and no others
    scores ~0.88 "overall recall" while missing two of the three attack types
    entirely.
    """
    flagged = probs >= threshold
    out: list[ScenarioRecall] = []
    for scenario in SCENARIOS:
        rows = (scenarios == scenario) & (labels == 1)
        out.append(
            ScenarioRecall(
                scenario=scenario,
                hits=int(flagged[rows].sum()),
                total=int(rows.sum()),
            )
        )
    return tuple(out)


def evaluate(
    labels: np.ndarray,
    probs: np.ndarray,
    scenarios: np.ndarray | None = None,
    *,
    choose_threshold: bool = True,
    threshold: ThresholdChoice | None = None,
) -> Evaluation:
    """The full validation picture for one set of predictions.

    ``threshold`` may be passed in to *apply* an already-chosen cutoff (Stage
    7 will do exactly that with the one stored in the checkpoint) instead of
    choosing a new one.
    """
    labels = np.asarray(labels).astype(np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.shape != probs.shape:
        raise ValueError(
            f"labels {labels.shape} and probabilities {probs.shape} disagree"
        )

    chosen = threshold
    if chosen is None and choose_threshold:
        chosen = sweep_threshold(labels, probs)

    scenario_recalls: tuple[ScenarioRecall, ...] = ()
    if scenarios is not None and chosen is not None:
        scenario_recalls = per_scenario_recall(
            labels, np.asarray(scenarios), probs, chosen.threshold
        )

    return Evaluation(
        auprc=auprc(labels, probs),
        random_baseline=float(labels.mean()),
        auroc=auroc(labels, probs),
        precision_at_k={k: precision_at_k(labels, probs, k) for k in PRECISION_AT_K},
        stats=PredictionStats.of(probs),
        n_rows=int(labels.size),
        n_positives=int(labels.sum()),
        threshold=chosen,
        per_scenario=scenario_recalls,
    )


def log_evaluation(evaluation: Evaluation, label: str = "validation") -> None:
    """Print the block a reader of the run log needs, warnings included."""
    logger.info(
        "%s AUPRC %.4f  (random baseline %.5f, %.1fx)  AUROC %s",
        label, evaluation.auprc, evaluation.random_baseline, evaluation.lift,
        f"{evaluation.auroc:.4f}" if evaluation.auroc is not None else "n/a",
    )
    for k, value in evaluation.precision_at_k.items():
        logger.info("%s P@%d %.3f", label, k, value)
    stats = evaluation.stats
    logger.info(
        "%s predicted prob  min %.6f  max %.6f  mean %.6f  std %.6f  "
        "distinct %s", label, stats.minimum, stats.maximum, stats.mean,
        stats.std, f"{stats.distinct:,}",
    )
    if stats.collapsed:
        logger.warning(
            "COLLAPSE GUARD: only %d distinct predicted values across %s rows "
            "(floor %d). The model is scoring nearly every row the same — "
            "check that pos_weight was applied.",
            stats.distinct, f"{stats.n_rows:,}", stats.collapse_floor,
        )
    if evaluation.auprc <= evaluation.random_baseline:
        logger.warning(
            "AUPRC %.4f is at or below the random baseline %.5f — the model "
            "is not ranking.", evaluation.auprc, evaluation.random_baseline,
        )
    if evaluation.suspiciously_good:
        logger.warning(
            "AUPRC %.4f is above %.2f. At %.3f%% positives that means "
            "LEAKAGE, not success — most likely a split applied by row "
            "instead of by user, or a standardiser fitted on everything. "
            "Stop and find it.",
            evaluation.auprc, LEAKAGE_AUPRC, 100 * evaluation.random_baseline,
        )
    if evaluation.threshold is not None:
        choice = evaluation.threshold
        logger.info(
            "threshold %.6f chosen on %s: F1 %.4f, precision %.4f, recall "
            "%.4f, %s rows flagged",
            choice.threshold, label, choice.f1, choice.precision,
            choice.recall, f"{choice.predicted_positives:,}",
        )
    for scenario in evaluation.per_scenario:
        logger.info("%s %s", label, scenario.phrase())


def aggregate(values: list[float]) -> dict[str, float]:
    """mean +/- sd across seeds, with the population sd of one run defined as 0.

    Stage 4 measured that scenario-1 positives swing 32 -> 56 across these
    three seeds — a 75% range from the split alone. A single run's number is
    noise dressed as a result, so every headline figure in this project is
    reported as mean +/- sd over ``config.SEEDS``.
    """
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "n": int(array.size),
    }
