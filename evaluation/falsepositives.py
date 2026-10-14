"""Reading the highest-scoring benign rows, because the label may be wrong.

§8 of the brief: Model B seed 123 scored P@10 = 0.100 against P@50 = 0.520 —
nine of its top ten validation rows are labelled benign yet outscore nearly
every true positive. That is either an honest failure mode or the most
quotable sentence in the report, and the only way to tell is to look at the
rows.

The reason to suspect the labels rather than the model is measured, not
hopeful. Stage 2 established that the published windows mark the *signature
event*, not the full behavioural anomaly: BIH0745's heaviest after-hours USB
copying sits outside his 65-minute labelled window, and KPC0073's loudest day
of all — z = 107 on after-hours USB — is formally labelled 0. A model that
ranks those days at the top is not wrong about the behaviour; it disagrees
with the annotation.

So this module prints, for each top-scoring benign row: who, when, the
eighteen z-scores in Stage 3's units, and three pieces of context the eye
cannot supply —

* whether the user is an insider at all,
* how many days separate the row from that insider's labelled window,
* which features are actually loud on that day.

and then classifies each row into one of four buckets so the report's sentence
can be written from a count rather than an impression. If they turn out to be
spurious, that is equally reportable and the classification says so.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from config import BENIGN_SCENARIO, FEATURE_NAMES
from evaluation.latency import attack_windows
from evaluation.scoring import Scored

logger = logging.getLogger(__name__)

_FEATURES_NAMED: Final[int] = 4
_LOUD_Z: Final[float] = 2.0
_NEAR_WINDOW_DAYS: Final[int] = 28

INSIDE_WINDOW_GAP: Final[str] = "insider, inside the window's date range but unlabelled"
NEAR_WINDOW: Final[str] = f"insider, within {_NEAR_WINDOW_DAYS} days of their window"
INSIDER_FAR: Final[str] = "insider, far from their labelled window"
BENIGN_USER: Final[str] = "user with no labelled attack at all"


@dataclass(frozen=True, slots=True)
class FalsePositive:
    """One high-scoring row that the ground truth calls benign."""

    model: str
    seed: int
    rank: int
    user: str
    day: str
    probability: float
    user_scenario: int
    reading: str
    days_from_window: int | None
    z_scores: dict[str, float]

    @property
    def loud_features(self) -> list[tuple[str, float]]:
        """Features at or above two sigma, loudest first."""
        loud = [(n, z) for n, z in self.z_scores.items() if abs(z) >= _LOUD_Z]
        return sorted(loud, key=lambda pair: -abs(pair[1]))

    def line(self) -> str:
        named = ", ".join(
            f"{name} z={value:+.1f}" for name, value in self.loud_features[:_FEATURES_NAMED]
        )
        distance = (
            "" if self.days_from_window is None
            else f", {self.days_from_window}d from window"
        )
        return (
            f"#{self.rank:<2} p={self.probability:.6f}  {self.user}  {self.day}  "
            f"[{self.reading}{distance}]  {named or 'nothing above 2 sigma'}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "seed": self.seed,
            "rank": self.rank,
            "user": self.user,
            "day": self.day,
            "probability": self.probability,
            "user_scenario": self.user_scenario,
            "reading": self.reading,
            "days_from_labelled_window": self.days_from_window,
            "z_scores": self.z_scores,
        }


def top_false_positives(scored: Scored, k: int = 10) -> list[FalsePositive]:
    """The ``k`` highest-scoring rows the labels call benign, in rank order.

    Rank is over *all* rows, not just benign ones, so "rank 3" means third
    highest overall — which is the number §8 is about. A benign row at rank 3
    is a different claim from the third-highest benign row.
    """
    split = scored.split
    windows = attack_windows(split)
    order = np.argsort(-scored.probs, kind="stable")

    out: list[FalsePositive] = []
    for rank, index in enumerate(order, start=1):
        if split.labels[index] == 1:
            continue
        user = str(split.users[index])
        day = split.days[index].astype("datetime64[D]")
        window = windows.get(user)

        distance: int | None = None
        if window is not None:
            if window.first_day <= day <= window.last_day:
                distance = 0
                reading = INSIDE_WINDOW_GAP
            else:
                before = int((window.first_day - day).astype(int))
                after = int((day - window.last_day).astype(int))
                distance = max(before, after)
                reading = (
                    NEAR_WINDOW if distance <= _NEAR_WINDOW_DAYS else INSIDER_FAR
                )
        else:
            reading = BENIGN_USER

        out.append(
            FalsePositive(
                model=scored.model_name,
                seed=scored.seed,
                rank=rank,
                user=user,
                day=str(day),
                probability=float(scored.probs[index]),
                user_scenario=int(split.scenarios[index]),
                reading=reading,
                days_from_window=distance,
                z_scores={
                    name: float(split.raw_features[index, i])
                    for i, name in enumerate(FEATURE_NAMES)
                },
            )
        )
        if len(out) >= k:
            break
    return out


@dataclass(frozen=True, slots=True)
class FalsePositiveReading:
    """What the inspected rows collectively say, as counts.

    The report's sentence depends on which bucket dominates, so the sentence
    is derived here from counts rather than written first and supported later.
    """

    model: str
    seed: int
    n_inspected: int
    counts: dict[str, int]
    max_z_seen: float
    sentence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "seed": self.seed,
            "inspected": self.n_inspected,
            "counts_by_reading": self.counts,
            "largest_absolute_z_seen": self.max_z_seen,
            "sentence_the_evidence_supports": self.sentence,
        }


def read(
    rows: Sequence[FalsePositive],
    *,
    model: str | None = None,
    seed: int | None = None,
) -> FalsePositiveReading:
    """Turn the inspected rows into the one sentence they justify.

    ``model`` and ``seed`` override the labels taken from the first row, which
    is how the same function produces both a per-(model, seed) reading and the
    pooled one the report quotes. Pooling matters: a sentence drawn from one
    seed's ten rows is a sentence about that seed.
    """
    if not rows:
        raise ValueError("no false positives to read")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.reading] = counts.get(row.reading, 0) + 1
    max_z = max(
        (abs(z) for row in rows for z in row.z_scores.values()), default=0.0
    )
    unlabelled_attack = counts.get(INSIDE_WINDOW_GAP, 0) + counts.get(NEAR_WINDOW, 0)
    insider_rows = unlabelled_attack + counts.get(INSIDER_FAR, 0)

    if unlabelled_attack > len(rows) / 2:
        sentence = (
            f"{unlabelled_attack} of the top {len(rows)} 'false positives' are "
            f"days belonging to known insiders that fall inside or within "
            f"{_NEAR_WINDOW_DAYS} days of their labelled attack window. Our "
            f"highest-scoring false positives are, on inspection, genuine "
            f"anomalous behaviour that the ground-truth windows fail to "
            f"label; the reported precision is therefore a lower bound."
        )
    elif insider_rows > len(rows) / 2:
        sentence = (
            f"{insider_rows} of the top {len(rows)} 'false positives' belong "
            f"to known insiders but fall well outside their labelled window. "
            f"That matches the Stage 2 finding that the published windows "
            f"mark the signature event rather than the full behavioural "
            f"anomaly, so these are best described as unlabelled anomalous "
            f"behaviour by a known insider rather than as errors."
        )
    else:
        sentence = (
            f"{counts.get(BENIGN_USER, 0)} of the top {len(rows)} false "
            f"positives belong to users with no labelled attack at all "
            f"(largest |z| seen: {max_z:.1f}). These are genuine false "
            f"positives — an honest failure mode, and the precision figures "
            f"should be read at face value rather than as a lower bound."
        )
    return FalsePositiveReading(
        model=model if model is not None else rows[0].model,
        seed=seed if seed is not None else rows[0].seed,
        n_inspected=len(rows),
        counts=counts,
        max_z_seen=max_z,
        sentence=sentence,
    )


def log_inspection(rows: Sequence[FalsePositive], reading: FalsePositiveReading) -> None:
    """Print the rows and the reading. §8 asks for them to be looked at."""
    logger.info("=" * 72)
    logger.info(
        "TOP-%d FALSE POSITIVES — %s seed %d, validation",
        reading.n_inspected, reading.model, reading.seed,
    )
    logger.info("=" * 72)
    for row in rows:
        logger.info("  %s", row.line())
    logger.info("")
    for label, count in sorted(reading.counts.items(), key=lambda kv: -kv[1]):
        logger.info("  %2d x %s", count, label)
    logger.info("")
    logger.info("  reading: %s", reading.sentence)


def is_insider_row(scenario: int) -> bool:
    """Whether this row's user is one of the release's insiders.

    70 of them on r4.2, 99 on r5.2 — hence the question asked of the scenario
    column rather than of a count.
    """
    return scenario != BENIGN_SCENARIO
