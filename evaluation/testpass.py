"""Phase 2: the test split, opened once, for both models, and then never again.

Everything about this module is shaped by one fact: it can only be run
honestly a single time. After the test numbers are known, no model may change
and no threshold may be re-picked, because an adjustment made with the test
result in view is fitting to the test set and the reported figure stops
meaning anything.

Code cannot stop a determined person from running this twice. What it can do
is make a second run *visible*:

* :data:`OPENED_MARKER` is written the first time and refused thereafter
  unless the caller passes an explicit reason, which is recorded in the file
  alongside every previous opening. A results table can then be checked
  against the number of times test was read.
* Phase 1's artifact must already exist. The brief's ordering (§4) becomes a
  missing-file error rather than a discipline.
* The standardiser and the threshold come out of the checkpoint and are
  compared against a train-only refit and against the stored value. Neither
  is recomputed here; there is no code path in :mod:`evaluation` that could.

The expected shape of the split is checked too, and it is read from
``splits.json`` rather than remembered: Stage 4 recorded how many rows and
positives every seed's split holds, the row mask then removes exactly
``WINDOW_DAYS - 1`` rows per user, and the difference is what evaluation should
see. Comparing the two states the guard's own question — *do the splits file
and the dataset still agree about who exists* — instead of comparing against a
snapshot of one release's answer to it. A departure would be invisible in an
AUPRC.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from config import (
    ACTIVE_RELEASE,
    MODEL_A_NAME,
    MODEL_B_NAME,
    OUTPUT_DIR,
    SCENARIOS,
    SCENARIO_COLUMN,
    SEEDS,
    TEST_SPLIT_OPENED_PATH,
    WINDOW_DAYS,
)
from evaluation import guards
from evaluation.checkpoints import headline_checkpoints
from evaluation.corpus import Corpus, EvalSplit, EvaluationError
from evaluation.latency import (
    BucketRecall,
    InsiderDetection,
    LatencySummary,
    detections,
    recall_by_position,
    summarise,
)
from evaluation.scoring import Scored, log_scored, score
from modeling.metrics import aggregate
from splitting.split import POSITIVES_COLUMN, smallest_scenario

logger = logging.getLogger(__name__)

OPENED_MARKER = TEST_SPLIT_OPENED_PATH

MODELS = (MODEL_A_NAME, MODEL_B_NAME)


class PhaseOrderError(RuntimeError):
    """Phase 2 was reached before Phase 1 finished. See brief §4."""


@dataclass(slots=True)
class TestResult:
    """The final numbers, plus every guard that says they can be believed."""

    seeds: list[int]
    sample: bool
    device: str
    scored: list[Scored] = field(default_factory=list)
    detections: list[InsiderDetection] = field(default_factory=list)
    latency: list[LatencySummary] = field(default_factory=list)
    buckets: list[BucketRecall] = field(default_factory=list)
    guards: list[guards.GuardResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.guards if g.fatal)

    def for_model(self, model: str) -> list[Scored]:
        return [s for s in self.scored if s.model_name == model]

    def headline(self, model: str) -> dict[str, Any]:
        """AUPRC, AUROC and P@k as mean +/- sd across seeds, with the baseline.

        AUPRC may be quoted this way — Stage 5 measured it stable across seeds.
        Recall at threshold may not, which is why it is absent from this
        method and pooled as counts in :meth:`per_scenario` instead.
        """
        items = self.for_model(model)
        if not items:
            return {}
        evaluations = [item.evaluation for item in items]
        return {
            "model": model,
            "context_days": items[0].checkpoint.context_days,
            "n_parameters": items[0].checkpoint.n_parameters,
            "seeds": [item.seed for item in items],
            "auprc": aggregate([e.auprc for e in evaluations]),
            "random_baseline": aggregate([e.random_baseline for e in evaluations]),
            "lift_over_random": aggregate([e.lift for e in evaluations]),
            "auroc": aggregate([e.auroc for e in evaluations if e.auroc is not None]),
            "precision_at_10": aggregate(
                [e.precision_at_k[10] for e in evaluations]
            ),
            "precision_at_50": aggregate(
                [e.precision_at_k[50] for e in evaluations]
            ),
            "rows_flagged": aggregate(
                [float(int(item.flagged.sum())) for item in items]
            ),
            "per_seed_auprc": {
                str(item.seed): item.evaluation.auprc for item in items
            },
        }

    def per_scenario(self, model: str) -> dict[str, Any]:
        """Pooled hits and attack days per scenario, summed across seeds.

        Counts, not a mean of rates. Scenario 3 contributes exactly four
        positive rows per split; "2 of 4 attack days" is honest and "50%" is a
        percentage doing an impression of a measurement.
        """
        out: dict[str, Any] = {}
        for scenario in SCENARIOS:
            hits = total = 0
            for item in self.for_model(model):
                for entry in item.evaluation.per_scenario:
                    if entry.scenario == scenario:
                        hits += entry.hits
                        total += entry.total
            out[str(scenario)] = {
                "hits_summed": hits,
                "attack_days_summed": total,
                "pooled_recall": hits / total if total else None,
            }
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": 2,
            "split": "test",
            "sample": self.sample,
            "device": self.device,
            "seeds": self.seeds,
            "headline": {model: self.headline(model) for model in MODELS},
            "per_scenario_pooled": {
                model: self.per_scenario(model) for model in MODELS
            },
            "detection_latency": {
                "summaries": [s.as_dict() for s in self.latency],
                "position_buckets": [b.as_dict() for b in self.buckets],
                "per_insider": [d.as_dict() for d in self.detections],
            },
            "per_seed": [item.as_dict() for item in self.scored],
            "guards": [g.as_dict() for g in self.guards],
            "all_guards_passed": self.passed,
        }


def require_phase1(*, sample: bool = False, directory: Path = OUTPUT_DIR) -> Path:
    """Refuse to open test until Phase 1 has written its artifact.

    §4: once the test split is opened, no model may change. Every remaining
    question therefore gets answered on validation *first*. This turns that
    rule into a precondition instead of a resolution.
    """
    suffix = ".sample" if sample else ""
    artifact = directory / f"phase1_validation{suffix}.json"
    if not artifact.is_file():
        raise PhaseOrderError(
            f"Phase 1 has not been run: {artifact} does not exist.\n"
            f"Brief §4 — once the test split is opened no model may change, "
            f"so every validation analysis must finish first.\n"
            f"Run:  python -m evaluation.run_phase1"
            f"{' --sample' if sample else ''}"
        )
    return artifact


def record_opening(
    *, reason: str, seeds: Sequence[int], sample: bool, force: bool
) -> dict[str, Any]:
    """Append to the audit trail, refusing a second opening without a reason.

    The refusal is not security. It is a note to the person doing it, at the
    moment they are doing it, that the number they are about to produce is no
    longer a single-pass test result and the report has to say so.
    """
    marker = OPENED_MARKER
    history: list[dict[str, Any]] = []
    if marker.is_file():
        previous = json.loads(marker.read_text())
        history = previous.get("openings", [])

    real_openings = [entry for entry in history if not entry.get("sample")]
    if real_openings and not sample and not force:
        when = real_openings[-1]["when"]
        raise EvaluationError(
            f"the test split was already opened at {when} "
            f"({len(real_openings)} time(s) so far).\n"
            f"A second pass is not a second experiment: any change made "
            f"between the two was informed by test numbers.\n"
            f"If you must re-run — a bug in the evaluation code, not a "
            f"disappointing result — pass --force --reason '...' and the "
            f"reason is recorded in {marker.name} for the report to disclose."
        )

    entry = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seeds": list(seeds),
        "sample": sample,
        "forced": force,
        "reason": reason,
    }
    history.append(entry)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"openings": history}, indent=2) + "\n")
    return entry


def run(
    *,
    seeds: Sequence[int] = SEEDS,
    sample: bool = False,
    device: str = "cpu",
    corpus: Corpus | None = None,
) -> TestResult:
    """One pass. Both models. Three seeds each. Then stop.

    Per seed: load both checkpoints, prove they carry the same standardiser as
    a train-only refit, build the test rows under it, apply each checkpoint's
    own stored threshold, and score. Nothing is fitted, chosen or swept.
    """
    corpus = corpus or Corpus.load(sample=sample)
    result = TestResult(seeds=list(seeds), sample=sample, device=device)

    result.guards.append(guards.split_disjointness(corpus, seeds))
    logger.info("expected shape read from %s", corpus.splits_file)

    for seed in seeds:
        logger.info("=" * 72)
        logger.info("TEST | seed %d", seed)
        logger.info("=" * 72)

        checkpoints = headline_checkpoints(seed, sample=sample)
        for checkpoint in checkpoints.values():
            logger.info("  %s", checkpoint.describe())
            logger.info(
                "    standardiser from checkpoint: mean[0..2]=%s std[0..2]=%s",
                " ".join(f"{v:+.6f}" for v in checkpoint.standardiser.mean[:3]),
                " ".join(f"{v:.6f}" for v in checkpoint.standardiser.std[:3]),
            )

        result.guards.append(
            guards.standardiser_from_checkpoint(corpus, seed, checkpoints.values())
        )

        splits: list[EvalSplit] = []
        for checkpoint in checkpoints.values():
            split = corpus.build("test", seed, checkpoint.standardiser)
            splits.append(split)
            logger.info("  %s", split.describe())
            item = score(checkpoint, split, device=device)
            log_scored(item)
            result.scored.append(item)
            result.detections.extend(detections(item))

        result.guards.append(guards.identical_rows(seed, splits))
        result.guards.append(
            shape_guard(
                seed,
                splits[0],
                expected_shape(corpus.splits_file, seed, splits[0].name),
            )
        )

    result.guards.extend(
        [
            guards.threshold_from_checkpoint(result.scored),
            guards.finite_predictions(result.scored),
            guards.beats_random(result.scored),
            guards.not_suspiciously_good(result.scored),
        ]
    )

    for model in MODELS:
        if not result.for_model(model):
            continue
        result.latency.append(summarise(result.detections, model))
        for scenario in SCENARIOS:
            summary = summarise(result.detections, model, scenario)
            if summary.n_insiders:
                result.latency.append(summary)
    result.buckets.extend(recall_by_position(result.scored))
    for scenario in SCENARIOS:
        result.buckets.extend(recall_by_position(result.scored, scenario))

    log_headline(result)
    guards.report(result.guards)
    return result


@dataclass(frozen=True, slots=True)
class ExpectedShape:
    """What ``splits.json`` says one seed's split holds, before the row mask.

    Stage 4 wrote these counts over the *whole* record of each user it
    assigned. Evaluation scores a subset of that — every user's first
    ``WINDOW_DAYS - 1`` days are warm-up for Model B's window and are predicted
    by neither model — so the two numbers are related by arithmetic rather than
    equal, and :attr:`evaluated_rows` is where that arithmetic lives.
    """

    split: str
    seed: int
    source: Path
    users: int
    rows: int
    positives: int
    scarcest_scenario: int | None
    scarcest_positives: int

    @property
    def warm_up_rows(self) -> int:
        """Rows the mask removes when every user has a full-length record."""
        return (WINDOW_DAYS - 1) * self.users

    @property
    def evaluated_rows(self) -> int:
        """The row count evaluation should see, in the healthy case.

        Checked on 2026-08-22 against the frozen r4.2 test pass recorded in
        ``outputs/phase2_test.json`` — 93,287 - 27 x 200 = 87,887, and exact on
        seeds 123 and 456 too — and against both releases' ``dataset.parquet``
        for all twelve (seed, split) pairs. It is not an approximation: the
        mask drops each user's first 27 rows, no user in either release holds
        fewer than 30 days, so the arithmetic *is* the measurement.
        """
        return self.rows - self.warm_up_rows


def expected_shape(
    splits_file: Path, seed: int, split: str
) -> ExpectedShape | None:
    """Read one split's expected shape out of Stage 4's ``splits.json``.

    Returns ``None`` — rather than raising — when the file cannot be read or
    carries no summary for this seed and split. The only caller is a non-fatal
    guard, and a guard that crashes the run it is watching is worse than the
    drift it was looking for. The tests' hand-built splits files, which hold
    ``{"splits": ...}`` and nothing else, take this path.
    """
    try:
        payload = json.loads(splits_file.read_text())
    except (OSError, ValueError) as error:
        logger.warning("cannot read the expected shape from %s: %s", splits_file, error)
        return None

    block = payload.get("summary", {}).get(str(seed), {}).get(split)
    if not isinstance(block, Mapping):
        return None

    scarcest = _scarcest_scenario(block)
    positives_by_scenario = block.get("positives_by_scenario", {})
    return ExpectedShape(
        split=split,
        seed=seed,
        source=splits_file,
        users=int(block["users"]),
        rows=int(block["rows"]),
        positives=int(block["positive_rows"]),
        scarcest_scenario=scarcest,
        scarcest_positives=int(positives_by_scenario.get(str(scarcest), 0)),
    )


def _scarcest_scenario(block: Mapping[str, Any]) -> int | None:
    """Which attack scenario rests on the fewest positive rows in this split.

    Delegated to :func:`splitting.split.smallest_scenario` rather than decided
    here, because "smallest" already has a definition — fewest positive rows,
    ties broken towards fewer insiders and then the lower id — and two
    definitions of it would eventually disagree. That function reads a frame of
    one row per user, so the summary's per-scenario counts are expanded back
    into one: the scenario's positives on its first user's row and zero on the
    rest, which reproduces both aggregations it performs (a ``groupby`` sum and
    a ``value_counts``) exactly.

    Note the scope. ``split.py`` asks this of the *whole* user table, so its
    answer describes the release; here it is asked of one seed's one split,
    whose denominators are the ones about to be reported. On r4.2 and r5.2 both
    questions answer 3, in all six test splits — which is precisely why neither
    may be hardcoded to it.
    """
    users_by_scenario = block.get("by_scenario", {})
    positives_by_scenario = block.get("positives_by_scenario", {})
    rows = [
        {
            SCENARIO_COLUMN: int(scenario),
            POSITIVES_COLUMN: (
                int(positives_by_scenario.get(scenario, 0)) if index == 0 else 0
            ),
        }
        for scenario, n_users in users_by_scenario.items()
        for index in range(int(n_users))
    ]
    table = pd.DataFrame(rows, columns=[SCENARIO_COLUMN, POSITIVES_COLUMN])
    return smallest_scenario(table)


def shape_guard(
    seed: int, split: EvalSplit, expected: ExpectedShape | None
) -> guards.GuardResult:
    """This split holds the rows ``splits.json`` says it should.

    Three comparisons, and only the first has any slack in it:

    * **Rows** must land in ``[rows - 27 x users, rows]``. Both ends are
      provable rather than chosen. A user with ``d`` days loses ``min(27, d)``
      rows to the mask, so the count cannot fall below the lower end; the mask
      never adds a row, so it cannot pass the upper. A healthy split sits
      exactly on the lower bound and the message says which.
    * **Positives** must match exactly. The mask costs zero positives on both
      releases — the earliest attack opens months into every record — so there
      is nothing here for a tolerance to absorb, and losing a user would show
      up in this line first.
    * **The scarcest scenario's positives** must match exactly, for whichever
      scenario that is. Not "scenario 3": that is the answer on both releases
      and it is the answer by coincidence.

    Non-fatal. On ``--sample`` the shape is genuinely different and the sample
    splits file says so itself, so this now checks the sample too instead of
    being noise there.
    """
    name = f"seed {seed}: {split.name} split matches splits.json"
    if expected is None:
        return guards.GuardResult(
            name,
            True,
            f"no summary for seed {seed} {split.name!r} in the splits file; "
            f"shape not checked",
            fatal=False,
        )

    detail = (
        f"{split.n_rows:,} rows (expected {expected.evaluated_rows:,} = "
        f"{expected.rows:,} - {WINDOW_DAYS - 1} warm-up days x "
        f"{expected.users:,} users), "
        f"{split.n_positives} positives (expected {expected.positives}, "
        f"{split.random_baseline:.4%})"
    )
    scarcest = expected.scarcest_scenario
    found = (
        int(((split.labels == 1) & (split.scenarios == scarcest)).sum())
        if scarcest is not None
        else 0
    )
    if scarcest is not None:
        detail += (
            f", scenario-{scarcest} positives {found} (expected exactly "
            f"{expected.scarcest_positives}; scenario {scarcest} is the "
            f"scarcest in this split)"
        )
    if split.n_rows > expected.evaluated_rows:
        detail += (
            f"; {split.n_rows - expected.evaluated_rows:,} rows above the "
            f"exact expectation, so some users hold fewer than {WINDOW_DAYS} "
            f"days of record"
        )

    off = (
        not expected.evaluated_rows <= split.n_rows <= expected.rows
        or split.n_positives != expected.positives
        or (scarcest is not None and found != expected.scarcest_positives)
    )
    return guards.GuardResult(name, not off, detail, fatal=False)


def log_headline(result: TestResult) -> None:
    """The table §12 opens with, printed at the end of the run."""
    logger.info("=" * 72)
    logger.info(
        "TEST RESULTS — %s",
        "SAMPLE (meaningless)" if result.sample else f"full r{ACTIVE_RELEASE}",
    )
    logger.info("=" * 72)
    for model in MODELS:
        head = result.headline(model)
        if not head:
            continue
        logger.info(
            "%-8s (%2d-day context, %5s params)  AUPRC %.4f +/- %.4f   "
            "random %.5f  =>  %.1fx",
            model, head["context_days"], f"{head['n_parameters']:,}",
            head["auprc"]["mean"], head["auprc"]["std"],
            head["random_baseline"]["mean"], head["lift_over_random"]["mean"],
        )
        logger.info(
            "%-8s  AUROC %.4f +/- %.4f   P@10 %.3f +/- %.3f   P@50 %.3f +/- %.3f",
            "", head["auroc"]["mean"], head["auroc"]["std"],
            head["precision_at_10"]["mean"], head["precision_at_10"]["std"],
            head["precision_at_50"]["mean"], head["precision_at_50"]["std"],
        )
        for scenario, entry in sorted(result.per_scenario(model).items()):
            pooled = entry["pooled_recall"]
            logger.info(
                "%-8s  scenario %s: %d of %d attack days across all seeds%s",
                "", scenario, entry["hits_summed"], entry["attack_days_summed"],
                f"  ({pooled:.1%})" if pooled is not None else "",
            )


def write(result: TestResult, *, directory: Path = OUTPUT_DIR) -> Path:
    """Write the test artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".sample" if result.sample else ""
    path = directory / f"phase2_test{suffix}.json"
    path.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n")
    logger.info("phase 2 -> %s", path)
    return path
