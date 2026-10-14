"""Phase 1: everything that can still be asked, asked on validation.

The order in §4 is not negotiable and this module is the reason it can be
enforced. Once the test split is opened no model may change, because any
adjustment made after seeing test numbers is fitting to the test set and the
reported result stops meaning anything. So every remaining question is
answered here first, from predictions that already exist, and Phase 2 refuses
to run until this module has written its artifact.

Three of Phase 1's four tasks live here — they need no training:

* **detection latency** (§5), the analysis that may change what the project
  claims and therefore comes first;
* **the P@10 inspection** (§8), which decides whether the reported precision
  is a lower bound or a face value;
* **the guard block** (§13) run against validation, including the one that
  matters most before test is ever touched: that re-scoring validation from
  the checkpoints reproduces the AUPRC Stages 5 and 6 recorded.

The fourth and fifth — the B@7 capacity control (§6) and the ``pos_weight``
ablation (§7) — are training runs, so they have their own entry points and
their results are read back from the metrics JSON they write.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from config import MODEL_A_NAME, MODEL_B_NAME, OUTPUT_DIR, SCENARIOS, SEEDS
from evaluation import guards
from evaluation.checkpoints import Checkpoint, headline_checkpoints
from evaluation.corpus import Corpus, EvalSplit
from evaluation.falsepositives import (
    FalsePositive,
    FalsePositiveReading,
    log_inspection,
    read,
    top_false_positives,
)
from evaluation.latency import (
    BucketRecall,
    InsiderDetection,
    LatencySummary,
    LatencyVerdict,
    detections,
    log_latency,
    read_verdict,
    recall_by_position,
    summarise,
)
from evaluation.scoring import Scored, log_scored, score
from evaluation.testpass import expected_shape, shape_guard

logger = logging.getLogger(__name__)

PHASE1_ARTIFACT = OUTPUT_DIR / "phase1_validation.json"
DETECTION_CSV = OUTPUT_DIR / "insider_detection.csv"

MODELS = (MODEL_A_NAME, MODEL_B_NAME)


@dataclass(slots=True)
class Phase1Result:
    """Everything Phase 1 measured, in one object the report can read."""

    seeds: list[int]
    sample: bool
    device: str
    scored: list[Scored] = field(default_factory=list)
    detections: list[InsiderDetection] = field(default_factory=list)
    latency: list[LatencySummary] = field(default_factory=list)
    buckets: list[BucketRecall] = field(default_factory=list)
    verdicts: list[LatencyVerdict] = field(default_factory=list)
    false_positives: list[FalsePositive] = field(default_factory=list)
    fp_readings: list[FalsePositiveReading] = field(default_factory=list)
    guards: list[guards.GuardResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.guards if g.fatal)

    def pooled_fp_reading(self) -> dict[str, Any] | None:
        """One reading over every inspected row, across models and seeds.

        The report quotes a single sentence about the top-scoring false
        positives, and it should be drawn from all of them rather than from
        whichever (model, seed) happened to be looked at first.
        """
        if not self.false_positives:
            return None
        return read(self.false_positives, model="all models", seed=-1).as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": 1,
            "split": "validation",
            "sample": self.sample,
            "device": self.device,
            "seeds": self.seeds,
            "validation_metrics": [item.as_dict() for item in self.scored],
            "detection_latency": {
                "summaries": [s.as_dict() for s in self.latency],
                "position_buckets": [b.as_dict() for b in self.buckets],
                "verdicts": [v.as_dict() for v in self.verdicts],
                "per_insider": [d.as_dict() for d in self.detections],
            },
            "false_positive_inspection": {
                "pooled": self.pooled_fp_reading(),
                "readings": [r.as_dict() for r in self.fp_readings],
                "rows": [f.as_dict() for f in self.false_positives],
            },
            "guards": [g.as_dict() for g in self.guards],
            "all_guards_passed": self.passed,
        }


def run(
    *,
    seeds: Sequence[int] = SEEDS,
    sample: bool = False,
    device: str = "cpu",
    top_k: int = 10,
    corpus: Corpus | None = None,
) -> Phase1Result:
    """Score validation with both models on every seed, then analyse it.

    Nothing is trained and nothing is chosen. Every number here comes from
    weights and a threshold that were fixed before this function was written.
    """
    corpus = corpus or Corpus.load(sample=sample)
    result = Phase1Result(seeds=list(seeds), sample=sample, device=device)

    result.guards.append(guards.split_disjointness(corpus, seeds))
    logger.info("expected shape read from %s", corpus.splits_file)

    for seed in seeds:
        logger.info("=" * 72)
        logger.info("validation | seed %d", seed)
        logger.info("=" * 72)

        checkpoints = headline_checkpoints(seed, sample=sample)
        for checkpoint in checkpoints.values():
            logger.info("  %s", checkpoint.describe())

        result.guards.append(
            guards.standardiser_from_checkpoint(corpus, seed, checkpoints.values())
        )

        splits, seed_scored = _score_seed(corpus, seed, checkpoints, device)
        result.scored.extend(seed_scored)
        result.guards.append(guards.identical_rows(seed, splits))
        result.guards.append(
            shape_guard(
                seed,
                splits[0],
                expected_shape(corpus.splits_file, seed, splits[0].name),
            )
        )

        for item in seed_scored:
            log_scored(item)
            result.detections.extend(detections(item))
            rows = top_false_positives(item, k=top_k)
            if rows:
                reading = read(rows)
                result.false_positives.extend(rows)
                result.fp_readings.append(reading)
                log_inspection(rows, reading)

    result.guards.extend(
        [
            guards.threshold_from_checkpoint(result.scored),
            guards.finite_predictions(result.scored),
            guards.beats_random(result.scored),
            guards.not_suspiciously_good(result.scored),
            guards.reproduces_training(result.scored),
        ]
    )

    _analyse_latency(result)
    log_latency(result.latency, result.buckets, result.verdicts)
    guards.report(result.guards)
    return result


def _score_seed(
    corpus: Corpus,
    seed: int,
    checkpoints: dict[str, Checkpoint],
    device: str,
) -> tuple[list[EvalSplit], list[Scored]]:
    """Build each model's validation rows and score them.

    The split is built once per model rather than once per seed even though
    both models must see identical rows — because building it *separately*
    from each checkpoint's own standardiser, and then proving the two are the
    same rows, is a stronger statement than building it once and asserting the
    standardisers matched.
    """
    splits: list[EvalSplit] = []
    scored: list[Scored] = []
    for checkpoint in checkpoints.values():
        split = corpus.build("val", seed, checkpoint.standardiser)
        splits.append(split)
        scored.append(score(checkpoint, split, device=device))
    return splits, scored


def _analyse_latency(result: Phase1Result) -> None:
    """Pooled latency summaries, position buckets and the §5 reading."""
    for model in MODELS:
        model_scored = [s for s in result.scored if s.model_name == model]
        if not model_scored:
            continue
        result.latency.append(summarise(result.detections, model))
        for scenario in SCENARIOS:
            summary = summarise(result.detections, model, scenario)
            if summary.n_insiders:
                result.latency.append(summary)

    result.buckets.extend(recall_by_position(result.scored))
    for scenario in SCENARIOS:
        result.buckets.extend(recall_by_position(result.scored, scenario))

    overall = [b for b in result.buckets if b.scenario is None]
    for model in MODELS:
        summary = next(
            (s for s in result.latency if s.model == model and s.scenario is None),
            None,
        )
        if summary is not None:
            result.verdicts.append(read_verdict(overall, summary))


def write(result: Phase1Result, *, directory: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    """Write the Phase 1 artifact and the per-insider CSV."""
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".sample" if result.sample else ""

    artifact = directory / f"phase1_validation{suffix}.json"
    artifact.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n")
    logger.info("phase 1 -> %s", artifact)

    csv_path = directory / f"insider_detection{suffix}.csv"
    _write_detection_csv(result.detections, csv_path)
    logger.info("per-insider table -> %s", csv_path)
    return artifact, csv_path


def _write_detection_csv(
    records: Sequence[InsiderDetection], path: Path
) -> None:
    """Deliverable 5: one row per (model, seed, insider), for the appendix."""
    import csv

    columns = [
        "model", "seed", "user", "scenario", "attack_days", "window_span_days",
        "attack_days_flagged", "attack_day_recall", "detected",
        "first_detection_day", "first_detection_date",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in sorted(
            records, key=lambda r: (r.model, r.scenario, r.seed, r.user)
        ):
            writer.writerow(record.as_dict())
