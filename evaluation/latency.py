"""Detection latency: does the model catch the insider, or notice the mess?

This is the analysis §5 calls the most important one remaining, and the reason
is not statistical. Scenario-2 attacks run 45-59 days on r4.2 and 44-60 on
r5.2, and r5.2's scenario 4 runs 33-161 — see ``config.SCENARIO_PROFILES``,
which is where the ranges live so that this module does not have to know them.
For a day at position 30 of such an attack, Model B's entire 28-day window sits
*inside* the attack — it is looking at 28 consecutive anomalous days and saying
"attack", which is easy, and roughly half of all scenario-2 positive days are
like that.

Nothing about that is leakage. The window reaches strictly backwards, contains
only that user's rows, and no label is ever visible to the model. But
*"detects the insider"* and *"notices that an attack it already missed is still
running"* are different products, and only the first is worth claiming. Which
one this project built is a measurement, and this module is the measurement.

Two views, both from predictions that already exist:

**First detection.** For each insider, the earliest day that is both labelled
an attack day and scored at or above the checkpoint's threshold, expressed as
day *N* of that insider's attack window. Insiders never detected at all are
counted separately — they are a result, not missing data.

**Recall by position.** The same positives, bucketed by how far into the
attack they fall: days 1-7, 8-14, 15-28, 29+. Flat recall with a low median
latency is genuine onset detection and may be claimed as such. Recall that
climbs steeply with position is a persistence detector, and the claim has to
become "identifies ongoing exfiltration campaigns", with the median latency
stated beside it.

Stage 3 measured that scenario-2 z-scores decay across an attack window — to
0.40x on r4.2 and 0.43x on r5.2 — so the answer is not a formality: the signal
is *weakest* late, which argues against the persistence reading, while Model
B's window fills up with attack days, which argues for it. Only the measurement
settles it. r5.2's scenario 4 is the interesting case, decaying only to 0.80x:
a campaign that stays loud for its whole 33-161 day window.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from evaluation.corpus import EvalSplit, positions_within
from evaluation.scoring import Scored

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Bucket:
    """A range of positions within an attack window, inclusive at both ends."""

    label: str
    low: int
    high: int | None

    def contains(self, position: np.ndarray) -> np.ndarray:
        inside = position >= self.low
        if self.high is not None:
            inside &= position <= self.high
        return inside


BUCKETS: Final[tuple[Bucket, ...]] = (
    Bucket("days 1-7", 1, 7),
    Bucket("days 8-14", 8, 14),
    Bucket("days 15-28", 15, 28),
    Bucket("days 29+", 29, None),
)


@dataclass(frozen=True, slots=True)
class AttackWindow:
    """One insider's labelled attack window, as it appears in the rows."""

    user: str
    scenario: int
    first_day: np.datetime64
    last_day: np.datetime64
    n_labelled_days: int
    contiguous: bool

    @property
    def span_days(self) -> int:
        return (
            int((self.last_day - self.first_day).astype("timedelta64[D]").astype(int))
            + 1
        )


def attack_windows(split: EvalSplit) -> dict[str, AttackWindow]:
    """Every insider in this split, keyed by user, with their window.

    The window is derived from the labels rather than re-read from
    ``insiders.csv``: the answer key lives on an external drive that is not
    always mounted, and more importantly the labels are what the models were
    scored against. Deriving the window from anything else would let "day 1 of
    the attack" mean one thing here and another in Stage 2.

    Non-contiguous windows are flagged rather than silently accepted. On every
    release measured so far — r4.2 and r5.2 — each window is a single date
    range and every calendar day inside it is a row, so a gap means an upstream
    assumption has moved.
    """
    windows: dict[str, AttackWindow] = {}
    positives = split.labels == 1
    if not positives.any():
        return windows

    for user in np.unique(split.users[positives]):
        rows = positives & (split.users == user)
        days = np.sort(split.days[rows].astype("datetime64[D]"))
        scenarios = np.unique(split.scenarios[rows])
        if scenarios.size != 1:
            raise ValueError(
                f"user {user} carries more than one scenario label "
                f"({scenarios.tolist()}); user_scenario is meant to be "
                f"constant per user"
            )
        span = int((days[-1] - days[0]).astype(int)) + 1
        window = AttackWindow(
            user=str(user),
            scenario=int(scenarios[0]),
            first_day=days[0],
            last_day=days[-1],
            n_labelled_days=int(days.size),
            contiguous=span == days.size,
        )
        if not window.contiguous:
            logger.warning(
                "%s's labelled attack days are not contiguous: %d days across "
                "a %d-day span. Position-within-window is still measured from "
                "the first labelled day.",
                window.user, window.n_labelled_days, span,
            )
        windows[window.user] = window
    return windows


@dataclass(frozen=True, slots=True)
class InsiderDetection:
    """When one model first caught one insider, if it ever did."""

    model: str
    seed: int
    user: str
    scenario: int
    n_attack_days: int
    window_span_days: int
    n_attack_days_flagged: int
    first_detection_day: int | None
    first_detection_date: str | None

    @property
    def detected(self) -> bool:
        return self.first_detection_day is not None

    @property
    def attack_day_recall(self) -> float:
        return self.n_attack_days_flagged / self.n_attack_days

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "seed": self.seed,
            "user": self.user,
            "scenario": self.scenario,
            "attack_days": self.n_attack_days,
            "window_span_days": self.window_span_days,
            "attack_days_flagged": self.n_attack_days_flagged,
            "attack_day_recall": self.attack_day_recall,
            "detected": self.detected,
            "first_detection_day": self.first_detection_day,
            "first_detection_date": self.first_detection_date,
        }


def detections(scored: Scored) -> list[InsiderDetection]:
    """First-detection day for every insider in this model/seed's split.

    "Detected" means a row that is both ``label == 1`` and scored at or above
    the checkpoint's threshold — the brief's definition. A high-scoring day
    *outside* the labelled window does not count here, deliberately: it may
    well be real anomalous behaviour the ground truth misses (§8 chases exactly
    that), but crediting it as a detection would be scoring the model against
    a story rather than against the labels.
    """
    split = scored.split
    flagged = scored.flagged
    windows = attack_windows(split)
    out: list[InsiderDetection] = []

    for user, window in sorted(windows.items()):
        rows = (split.labels == 1) & (split.users == user)
        days = split.days[rows].astype("datetime64[D]")
        hit = flagged[rows]

        first_day: int | None = None
        first_date: str | None = None
        if hit.any():
            order = np.argsort(days, kind="stable")
            earliest = days[order][hit[order]][0]
            first_day = int(positions_within(np.array([earliest]), window.first_day)[0])
            first_date = str(earliest)

        out.append(
            InsiderDetection(
                model=scored.model_name,
                seed=scored.seed,
                user=user,
                scenario=window.scenario,
                n_attack_days=window.n_labelled_days,
                window_span_days=window.span_days,
                n_attack_days_flagged=int(hit.sum()),
                first_detection_day=first_day,
                first_detection_date=first_date,
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Median first-detection day and the never-caught count, per scenario."""

    model: str
    scenario: int | None
    n_insiders: int
    n_detected: int
    median_first_day: float | None
    q1_first_day: float | None
    q3_first_day: float | None
    earliest: int | None
    latest: int | None

    @property
    def n_never_detected(self) -> int:
        return self.n_insiders - self.n_detected

    def phrase(self) -> str:
        scope = "all scenarios" if self.scenario is None else f"scenario {self.scenario}"
        if self.n_detected == 0:
            return (
                f"{self.model} | {scope}: none of {self.n_insiders} insider(s) "
                f"detected at all"
            )
        return (
            f"{self.model} | {scope}: median first detection on day "
            f"{self.median_first_day:.1f} of the attack "
            f"(IQR {self.q1_first_day:.1f}-{self.q3_first_day:.1f}, range "
            f"{self.earliest}-{self.latest}), "
            f"{self.n_detected} of {self.n_insiders} insiders detected, "
            f"{self.n_never_detected} never caught"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scenario": self.scenario,
            "insiders": self.n_insiders,
            "detected": self.n_detected,
            "never_detected": self.n_never_detected,
            "median_first_detection_day": self.median_first_day,
            "iqr_first_detection_day": [self.q1_first_day, self.q3_first_day],
            "earliest_first_detection_day": self.earliest,
            "latest_first_detection_day": self.latest,
        }


def summarise(
    records: Iterable[InsiderDetection], model: str, scenario: int | None = None
) -> LatencySummary:
    """Median, IQR and never-caught count over a pool of insiders.

    Pooled across seeds on purpose. A single seed's validation split holds
    roughly a dozen insiders, and Stage 4 measured that per-scenario counts
    swing 75% across ``config.SEEDS`` from the split alone — a median over one
    seed's handful of insiders is a statement about the split lottery.
    """
    chosen = [
        record
        for record in records
        if record.model == model
        and (scenario is None or record.scenario == scenario)
    ]
    days = np.array(
        [r.first_detection_day for r in chosen if r.detected], dtype=np.float64
    )
    if days.size == 0:
        return LatencySummary(model, scenario, len(chosen), 0, None, None, None, None, None)
    return LatencySummary(
        model=model,
        scenario=scenario,
        n_insiders=len(chosen),
        n_detected=int(days.size),
        median_first_day=float(np.median(days)),
        q1_first_day=float(np.percentile(days, 25)),
        q3_first_day=float(np.percentile(days, 75)),
        earliest=int(days.min()),
        latest=int(days.max()),
    )


@dataclass(frozen=True, slots=True)
class BucketRecall:
    """Recall over the attack days that fall in one position bucket."""

    model: str
    scenario: int | None
    bucket: str
    hits: int
    total: int

    @property
    def recall(self) -> float | None:
        return self.hits / self.total if self.total else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scenario": self.scenario,
            "bucket": self.bucket,
            "hits": self.hits,
            "attack_days": self.total,
            "recall": self.recall,
        }


def recall_by_position(
    scored: Sequence[Scored], scenario: int | None = None
) -> list[BucketRecall]:
    """Pooled recall per position bucket, as counts first and a rate second.

    Counts, because Stage 5 measured that recall-at-threshold swings wildly
    across seeds (Model A flagged 1,168 / 481 / 107 rows) while AUPRC barely
    moves. A mean of three bucket percentages computed on different row counts
    is not a number worth printing; summing hits and totals across seeds is.
    """
    by_model: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for item in scored:
        split = item.split
        windows = attack_windows(split)
        if not windows:
            continue
        positive = split.labels == 1
        if scenario is not None:
            positive &= split.scenarios == scenario
        if not positive.any():
            continue

        first_days = np.array(
            [
                windows[user].first_day if user in windows else np.datetime64("NaT")
                for user in split.users[positive]
            ],
            dtype="datetime64[D]",
        )
        positions = (
            split.days[positive].astype("datetime64[D]") - first_days
        ).astype(np.int64) + 1
        by_model.setdefault(item.model_name, []).append(
            (positions, item.flagged[positive])
        )

    out: list[BucketRecall] = []
    for model, chunks in by_model.items():
        positions = np.concatenate([c[0] for c in chunks])
        flagged = np.concatenate([c[1] for c in chunks])
        for bucket in BUCKETS:
            inside = bucket.contains(positions)
            out.append(
                BucketRecall(
                    model=model,
                    scenario=scenario,
                    bucket=bucket.label,
                    hits=int(flagged[inside].sum()),
                    total=int(inside.sum()),
                )
            )
    return out


PERSISTENCE_RATIO: Final[float] = 1.5
FLAT_RATIO: Final[float] = 1.25
ONSET_MEDIAN_DAYS: Final[float] = 7.0


@dataclass(frozen=True, slots=True)
class LatencyVerdict:
    """Which of §5's two readings the measurement supports, and why.

    The thresholds are declared above, before the numbers are in, so the
    conclusion is a consequence of the measurement rather than a description
    of it.
    """

    model: str
    reading: str
    claim: str
    early_recall: float | None
    late_recall: float | None
    ratio: float | None
    median_first_day: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "reading": self.reading,
            "claim_licensed": self.claim,
            "early_recall_days_1_7": self.early_recall,
            "late_recall_days_29_plus": self.late_recall,
            "late_over_early": self.ratio,
            "median_first_detection_day": self.median_first_day,
        }


def read_verdict(
    buckets: Sequence[BucketRecall], summary: LatencySummary
) -> LatencyVerdict:
    """Apply §5's decision rule to one model's measured profile."""
    model = summary.model
    early = next(
        (b.recall for b in buckets if b.model == model and b.bucket == "days 1-7"),
        None,
    )
    late = next(
        (b.recall for b in buckets if b.model == model and b.bucket == "days 29+"),
        None,
    )
    median = summary.median_first_day

    if early is None or late is None or not early:
        return LatencyVerdict(
            model, "insufficient",
            "not enough attack days in one of the buckets to read a profile; "
            "report the raw counts and draw no conclusion",
            early, late, None, median,
        )

    ratio = late / early
    if ratio >= PERSISTENCE_RATIO or (median is not None and median > ONSET_MEDIAN_DAYS):
        return LatencyVerdict(
            model, "persistence",
            "identifies ongoing exfiltration campaigns; the median first "
            "detection must be stated beside any detection claim",
            early, late, ratio, median,
        )
    if ratio <= FLAT_RATIO and median is not None and median <= ONSET_MEDIAN_DAYS:
        return LatencyVerdict(
            model, "onset",
            "detects insider activity at onset — recall is flat across "
            "position within the attack and the median first detection falls "
            "in the first week",
            early, late, ratio, median,
        )
    return LatencyVerdict(
        model, "mixed",
        "between the two readings: recall rises with position but not "
        "steeply. Claim onset detection only with the median latency and the "
        "bucket table shown.",
        early, late, ratio, median,
    )


def log_latency(
    summaries: Sequence[LatencySummary],
    buckets: Sequence[BucketRecall],
    verdicts: Sequence[LatencyVerdict],
) -> None:
    """The block the brief asks to be reported first."""
    logger.info("=" * 72)
    logger.info("DETECTION LATENCY — validation, pooled across seeds")
    logger.info("=" * 72)
    for summary in summaries:
        logger.info("  %s", summary.phrase())
    logger.info("")
    _log_bucket_table([b for b in buckets if b.scenario is None], "all scenarios")
    for scenario in sorted({b.scenario for b in buckets if b.scenario is not None}):
        _log_bucket_table(
            [b for b in buckets if b.scenario == scenario], f"scenario {scenario}"
        )
    for verdict in verdicts:
        logger.info(
            "  %s: %s — %s", verdict.model, verdict.reading.upper(), verdict.claim
        )


def _log_bucket_table(buckets: Sequence[BucketRecall], scope: str) -> None:
    """One position-bucket table, for one scope.

    Called once per scenario as well as for the pool. It used to be called only
    for the pool — the per-scenario buckets were computed, written to the JSON
    and never printed — which left a reader scoring a per-scenario prediction
    out of a JSON file. r5.2's scenario 4 is exactly such a prediction.
    """
    logger.info(
        "recall by position within the attack window (counts, pooled) — %s", scope
    )
    for model in sorted({b.model for b in buckets}):
        for bucket in BUCKETS:
            entry = next(
                (
                    b
                    for b in buckets
                    if b.model == model and b.bucket == bucket.label
                ),
                None,
            )
            if entry is None or entry.total == 0:
                logger.info("  %-10s %-12s no attack days", model, bucket.label)
                continue
            logger.info(
                "  %-10s %-12s %5d of %5d  (%.1f%%)",
                model, entry.bucket, entry.hits, entry.total,
                100.0 * (entry.recall or 0.0),
            )
    logger.info("")
