"""Scoring rows with a stored model, where nothing is ever chosen.

This module does real work — it loads a checkpoint, builds the rows and runs a
real forward pass — and it makes no decisions at all. That distinction is the
whole of the design, so it is worth being exact about:

- The **decision threshold** comes off the checkpoint. It is never swept,
  never re-selected, never recomputed. ``sweep_threshold`` is not imported
  here and its name appears nowhere in ``ui/`` or ``pipeline/``.
- The **standardiser** comes off the checkpoint.
  :meth:`evaluation.corpus.Corpus.build` takes one as a required argument, so
  there is no code path that could fit a new one on the rows about to be
  scored.
- The **weights** come off the checkpoint, and
  :meth:`evaluation.checkpoints.Checkpoint.build` puts the model in
  ``eval()``.

Because all three are fixed, the run is deterministic, which turns into the
feature this module exists for:

**reproduces_frozen_results.** On an r4.2 test run with the headline
checkpoints, the recomputed AUPRC is compared against the value already in
``outputs/final_results.json``, per model and per seed, and must agree to four
decimal places. If it does not, the run fails loudly and says by how much.

That converts the riskiest question an examiner can ask — *did your live demo
touch the test set?* — into one crisp answer: nothing was chosen, and the run
asserts that it reproduces a number that was published before the demo
started. It is a reproduction, not a second pass.

**What "not a second pass" means, precisely.** Test rows are read. They are
read to be re-scored, not to be learned from: no threshold, model, variant or
cutoff is selected in this module, no audit marker is written, and the answer
is required to equal one that already exists. A pass that cannot change any
decision is not a decision.

**Measured, not assumed.** The four-decimal bound was checked before it was
chosen: re-scoring all six frozen checkpoints on CPU against numbers computed
on MPS gave a largest deviation of 1.3e-08. Device does not move the fourth
decimal here, contrary to what one might reasonably expect of two float
backends.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from config import (
    CHECKPOINT_DIR,
    DEGENERATE_FEATURES,
    FINAL_RESULTS_PATH,
    MODEL_A_NAME,
    MODEL_B_NAME,
    PRECISION_AT_K,
    SCENARIOS,
    SEEDS,
    UI_REPRODUCTION_TOLERANCE,
    resolve_device,
)
from evaluation import latency
from evaluation.checkpoints import Checkpoint, headline_checkpoints
from evaluation.corpus import Corpus, EvalSplit
from evaluation.guards import (
    GuardResult,
    beats_random,
    finite_predictions,
    identical_rows,
    not_suspiciously_good,
    report,
    split_disjointness,
    standardiser_from_checkpoint,
    threshold_from_checkpoint,
)
from evaluation.scoring import Scored, score
from modeling.data import load_dataset
from modeling.metrics import ScenarioRecall, aggregate

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """An inference run that must stop rather than report a number."""


TEST_SPLIT: Final[str] = "test"
VALIDATION_SPLIT: Final[str] = "val"

_MODELS: Final[tuple[str, ...]] = (MODEL_A_NAME, MODEL_B_NAME)


def load_frozen_test_auprc(
    path: Path = FINAL_RESULTS_PATH,
) -> dict[tuple[str, int], float]:
    """The published per-seed test AUPRC, keyed by (model, seed).

    Read from the file rather than hardcoded, so that a report regenerated
    tomorrow is what the guard checks against and the two cannot silently
    disagree. An absent file is not an error here — a sample run has nothing
    to reproduce — but :func:`reproduces_frozen_results` will say so.
    """
    if not path.is_file():
        logger.warning("no frozen results at %s", path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    headline = payload.get("test", {}).get("headline", {})
    frozen: dict[tuple[str, int], float] = {}
    for model, block in headline.items():
        for seed, value in block.get("per_seed_auprc", {}).items():
            frozen[(model, int(seed))] = float(value)
    logger.info("read %d frozen test AUPRC value(s) from %s", len(frozen), path)
    return frozen


def reproduces_frozen_results(
    scored: Sequence[Scored],
    frozen: dict[tuple[str, int], float],
    *,
    tolerance: float = UI_REPRODUCTION_TOLERANCE,
) -> GuardResult:
    """The ninth guard: this run reproduces the published number exactly.

    Fatal. A disagreement means one of three things — the checkpoint is not
    the one the report was written from, the dataset has changed since, or the
    scoring path has drifted — and every one of them makes the number on the
    screen a different number from the one in the thesis. There is no reading
    of a failure here that is safe to continue past.
    """
    name = "re-scoring test reproduces the published AUPRC to 4 dp"

    comparable = [item for item in scored if (item.model_name, item.seed) in frozen]
    if not comparable:
        return GuardResult(
            name, True,
            "not applicable — no published value exists for these "
            "checkpoints (a sample run, a stored model, or a split other "
            "than test). Nothing was compared.",
            fatal=False,
        )

    problems: list[str] = []
    printed: list[str] = []
    for item in comparable:
        published = frozen[(item.model_name, item.seed)]
        recomputed = item.evaluation.auprc
        delta = recomputed - published
        printed.append(
            f"{item.model_name} seed {item.seed}: {recomputed:.6f} vs "
            f"{published:.6f} published (delta {delta:+.1e})"
        )
        if abs(delta) > tolerance:
            problems.append(
                f"{item.model_name} seed {item.seed}: recomputed "
                f"{recomputed:.6f} but final_results.json publishes "
                f"{published:.6f} (delta {delta:+.2e}, tolerance {tolerance:.0e})"
            )

    if problems:
        return GuardResult(
            name, False,
            f"{'; '.join(problems)}. This run does not reproduce the "
            f"published result, so the numbers on this page are NOT the "
            f"numbers in the report. Stop and find out why before quoting "
            f"either.",
        )
    return GuardResult(
        name, True,
        f"all {len(comparable)} model/seed pair(s) agree; " + "; ".join(printed),
    )


@dataclass(frozen=True, slots=True)
class SeedResult:
    """One model on one seed's split: the numbers, and where they came from."""

    model: str
    seed: int
    auprc: float
    auroc: float | None
    precision_at_k: dict[int, float]
    random_baseline: float
    lift: float
    n_rows: int
    n_positives: int
    n_flagged: int
    threshold: float
    context_days: int
    n_parameters: int
    published_auprc: float | None
    per_scenario: tuple[ScenarioRecall, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "seed": self.seed,
            "auprc": self.auprc,
            "auroc": self.auroc,
            "precision_at_k": {str(k): v for k, v in self.precision_at_k.items()},
            "random_baseline": self.random_baseline,
            "lift_over_random": self.lift,
            "n_rows": self.n_rows,
            "n_positives": self.n_positives,
            "n_flagged": self.n_flagged,
            "threshold": self.threshold,
            "threshold_source": "checkpoint (chosen on validation, not here)",
            "context_days": self.context_days,
            "n_parameters": self.n_parameters,
            "published_auprc": self.published_auprc,
            "per_scenario": [s.as_dict() for s in self.per_scenario],
        }


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """One model across the seeds: AUPRC aggregated, recall deliberately not.

    ``rows_flagged`` is a list rather than a mean because Model A's flagged
    count swung 1,168 / 481 / 107 across the three seeds on validation.
    Averaging that would print a number no seed produced.
    """

    model: str
    seeds: tuple[int, ...]
    context_days: int
    n_parameters: int
    auprc_mean: float
    auprc_std: float
    auroc_mean: float | None
    random_baseline: float
    lift_mean: float
    rows_flagged: tuple[int, ...]
    pooled_scenarios: tuple[ScenarioRecall, ...]
    median_first_detection_day: float | None
    insiders_detected: int
    insiders_total: int

    def headline(self) -> str:
        return (
            f"{self.model}: AUPRC {self.auprc_mean:.4f} +/- "
            f"{self.auprc_std:.4f} over {len(self.seeds)} seeds "
            f"(random {self.random_baseline:.5f}, {self.lift_mean:.0f}x)"
        )

    def latency_phrase(self) -> str:
        """Every detection claim is stated beside its first-detection day.

        Model B is a persistence detector rather than an onset detector, so a
        recall figure shown without latency overstates what it does. The two
        travel together or neither is shown.
        """
        if self.median_first_detection_day is None:
            return (
                f"{self.model}: none of {self.insiders_total} insiders "
                f"flagged at all, so there is no first-detection day"
            )
        return (
            f"{self.model}: {self.insiders_detected} of "
            f"{self.insiders_total} insiders flagged, median first detection "
            f"on day {self.median_first_detection_day:.0f} of the attack"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "seeds": list(self.seeds),
            "context_days": self.context_days,
            "n_parameters": self.n_parameters,
            "auprc": {"mean": self.auprc_mean, "std": self.auprc_std},
            "auroc_mean": self.auroc_mean,
            "random_baseline": self.random_baseline,
            "lift_mean": self.lift_mean,
            "rows_flagged_per_seed": list(self.rows_flagged),
            "per_scenario_pooled": [s.as_dict() for s in self.pooled_scenarios],
            "median_first_detection_day": self.median_first_detection_day,
            "insiders_detected": self.insiders_detected,
            "insiders_total": self.insiders_total,
        }


@dataclass(frozen=True, slots=True)
class Transfer:
    """A run whose model and whose rows come from different CERT releases.

    Present on an :class:`InferenceRun` only when the two disagree; ``None``
    is a matched run and is the ordinary case. It carries both release names
    rather than a bare flag because "this is a transfer" is not the useful
    sentence — "an r5.2 model scored r4.2's data" is — and because the
    degenerate-feature caveat below cannot be written without knowing which
    way round it went.

    **Every word the page shows about a transfer comes from here**, through
    :meth:`banner`, so the streamed log and the browser cannot end up saying
    two different things about the same run.
    """

    model_release: str
    data_release: str

    def headline(self) -> str:
        return (
            f"TRANSFER RUN — the model was trained on CERT "
            f"r{self.model_release} and is scoring CERT "
            f"r{self.data_release}'s data. These numbers "
            f"are a transfer measurement, not a certified result for either "
            f"release, and nothing here reproduces a published figure."
        )

    def degenerate_lines(self) -> tuple[str, ...]:
        """Which of the eighteen features are dead on each side, both ways.

        §4 of brief R11, and the reason this class exists rather than a bool.
        A cross-release run can pass every guard and still be misleading: the
        eighteen slots keep their names across releases, but three of them
        carry no information on r4.2 and do on r5.2. A reader who is not told
        that will read a transfer AUPRC as a statement about temporal context
        when part of it is a statement about dead inputs.

        Both directions are named on every transfer run, including the
        direction whose answer is "none" — an empty list stated out loud is a
        measurement, and leaving it off would make the sentence look like it
        only ever applies to r4.2.
        """
        return (
            self._side(
                self.data_release, self.model_release,
                role="the data being scored",
                consequence=(
                    f"so this model holds learned weights for inputs that are "
                    f"dead in these rows"
                ),
            ),
            self._side(
                self.model_release, self.data_release,
                role="the model's own release",
                consequence=(
                    f"so the model never learned anything from those inputs, "
                    f"while these rows do vary in them"
                ),
            ),
        )

    @staticmethod
    def _side(release: str, other: str, *, role: str, consequence: str) -> str:
        if release not in DEGENERATE_FEATURES:
            return (
                f"CERT r{release} ({role}) has no entry in "
                f"config.DEGENERATE_FEATURES, so which of its eighteen "
                f"features are structurally degenerate is NOT KNOWN here. "
                f"Treat every feature-level reading of this run as unchecked."
            )
        degenerate = DEGENERATE_FEATURES[release]
        if not degenerate:
            return (
                f"None of the eighteen features is structurally degenerate on "
                f"CERT r{release} ({role}); all eighteen separate there."
            )
        listed = "; ".join(
            f"{name} ({why})" for name, why in degenerate.items()
        )
        return (
            f"{len(degenerate)} of the eighteen features are structurally "
            f"degenerate on CERT r{release} ({role}) and informative on CERT "
            f"r{other} — {consequence}. They are: {listed}."
        )

    def banner(self) -> tuple[str, ...]:
        """The whole caveat, in the order it should be read."""
        return (self.headline(), *self.degenerate_lines())

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_release": self.model_release,
            "data_release": self.data_release,
            "headline": self.headline(),
            "degenerate_features": list(self.degenerate_lines()),
            "banner": list(self.banner()),
        }


@dataclass(frozen=True, slots=True)
class InferenceRun:
    """Everything one press of the Run button produced."""

    split: str
    sample: bool
    device: str
    seeds: tuple[int, ...]
    checkpoint_dir: Path
    per_seed: tuple[SeedResult, ...]
    models: tuple[ModelSummary, ...]
    guards: tuple[GuardResult, ...] = field(default_factory=tuple)
    transfer: Transfer | None = None

    @property
    def passed(self) -> bool:
        return not [g for g in self.guards if not g.passed and g.fatal]

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "sample": self.sample,
            "device": self.device,
            "seeds": list(self.seeds),
            "checkpoint_dir": str(self.checkpoint_dir),
            "meaningless": self.sample,
            "transfer": self.transfer.as_dict() if self.transfer else None,
            "per_seed": [r.as_dict() for r in self.per_seed],
            "models": [m.as_dict() for m in self.models],
            "guards": [g.as_dict() for g in self.guards],
            "all_guards_passed": self.passed,
        }


def run_inference(
    *,
    seeds: Sequence[int] = SEEDS,
    sample: bool = False,
    device: str = "auto",
    split: str = TEST_SPLIT,
    dataset_file: Path | None = None,
    splits_file: Path | None = None,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    checkpoints_for_seed: Callable[[int], dict[str, Checkpoint]] | None = None,
    frozen_results: Path | None = FINAL_RESULTS_PATH,
    transfer: Transfer | None = None,
    on_line: Callable[[str], None] | None = None,
) -> InferenceRun:
    """Score ``split`` with the stored checkpoints and check the guards.

    ``dataset_file`` and ``splits_file`` let an inference run read tables it
    derived itself from a chosen folder, in ``runs/<run_id>/``, rather than the
    frozen ones in ``data/``. Nothing else about the run changes: the
    checkpoints, the standardiser and the threshold still come off disk.

    ``checkpoints_for_seed`` is how a model the UI trained gets scored — it
    returns both models for one seed, exactly as
    :func:`~evaluation.checkpoints.headline_checkpoints` does, but loads them
    from the model store through the variant loader.

    ``transfer`` says that the checkpoints and the rows come from different
    CERT releases. It is **declared by the caller, never inferred here**: the
    caller is the only layer that knows what the user was asked and what they
    answered, and a mode this module worked out for itself would be a mode
    nobody opted into. It does two things and no third. The standardiser guard
    is put into its inverted mode, where a run that *matches* a refit on these
    rows fails; and the caveat is printed before the first number and carried
    on the result. Nothing about how the rows are built or scored changes.
    """
    say = on_line or (lambda line: None)
    resolved = resolve_device(device)
    seeds = tuple(int(s) for s in seeds)

    say(f"device: {resolved} (requested {device})")
    if transfer is not None:
        for line in transfer.banner():
            say(line)
    corpus = _load_corpus(dataset_file, splits_file, sample=sample, say=say)

    reproducible = (
        split == TEST_SPLIT
        and not sample
        and checkpoints_for_seed is None
        and frozen_results is not None
    )
    frozen = load_frozen_test_auprc(frozen_results) if reproducible else {}

    scored: list[Scored] = []
    guards: list[GuardResult] = [split_disjointness(corpus, seeds)]
    detections: list[latency.InsiderDetection] = []

    for seed in seeds:
        say(f"--- seed {seed}")
        checkpoints = (
            checkpoints_for_seed(seed)
            if checkpoints_for_seed is not None
            else headline_checkpoints(seed, sample=sample, directory=checkpoint_dir)
        )
        for checkpoint in checkpoints.values():
            say(f"    {checkpoint.describe()}")

        splits: list[EvalSplit] = []
        for model in _MODELS:
            checkpoint = checkpoints[model]
            eval_split = corpus.build(split, seed, checkpoint.standardiser)
            splits.append(eval_split)
            item = score(checkpoint, eval_split, device=resolved)
            scored.append(item)
            detections.extend(latency.detections(item))
            say(
                f"    {model}: AUPRC {item.evaluation.auprc:.4f} "
                f"(random {item.evaluation.random_baseline:.5f}, "
                f"{item.evaluation.lift:.0f}x), "
                f"{int(item.flagged.sum()):,} rows flagged at threshold "
                f"{item.threshold:.6f}"
            )

        guards.append(
            standardiser_from_checkpoint(
                corpus, seed, checkpoints.values(),
                transfer=transfer is not None,
            )
        )
        guards.append(identical_rows(seed, splits))

    guards.extend([
        threshold_from_checkpoint(scored),
        finite_predictions(scored),
        beats_random(scored),
        not_suspiciously_good(scored),
        reproduces_frozen_results(scored, frozen),
    ])

    run = InferenceRun(
        split=split,
        sample=sample,
        device=resolved,
        seeds=seeds,
        checkpoint_dir=checkpoint_dir,
        per_seed=tuple(_seed_results(scored, frozen)),
        models=tuple(_model_summaries(scored, detections, seeds)),
        guards=tuple(guards),
        transfer=transfer,
    )
    report(run.guards)
    for line in format_report(run).splitlines():
        say(line)
    return run


def _load_corpus(
    dataset_file: Path | None,
    splits_file: Path | None,
    *,
    sample: bool,
    say: Callable[[str], None],
) -> Corpus:
    """The rows to score: the frozen tables, or ones a run derived itself."""
    if dataset_file is None:
        say(f"reading the frozen dataset ({'sample' if sample else 'full'})")
        return Corpus.load(sample=sample)
    if splits_file is None:
        raise InferenceError(
            "a dataset built by this run needs the splits file that run also "
            "built; scoring new rows against the frozen splits.json would put "
            "users in splits they were not assigned to"
        )
    say(f"reading {dataset_file}")
    say(f"reading {splits_file}")
    return Corpus.load(frame=load_dataset(dataset_file), splits_file=splits_file)


def _seed_results(
    scored: Sequence[Scored], frozen: dict[tuple[str, int], float]
) -> list[SeedResult]:
    return [
        SeedResult(
            model=item.model_name,
            seed=item.seed,
            auprc=item.evaluation.auprc,
            auroc=item.evaluation.auroc,
            precision_at_k=dict(item.evaluation.precision_at_k),
            random_baseline=item.evaluation.random_baseline,
            lift=item.evaluation.lift,
            n_rows=item.evaluation.n_rows,
            n_positives=item.evaluation.n_positives,
            n_flagged=int(item.flagged.sum()),
            threshold=item.threshold,
            context_days=item.checkpoint.context_days,
            n_parameters=item.checkpoint.n_parameters,
            published_auprc=frozen.get((item.model_name, item.seed)),
            per_scenario=item.evaluation.per_scenario,
        )
        for item in scored
    ]


def _model_summaries(
    scored: Sequence[Scored],
    detections: Sequence[latency.InsiderDetection],
    seeds: Sequence[int],
) -> list[ModelSummary]:
    summaries: list[ModelSummary] = []
    for model in _MODELS:
        items = [item for item in scored if item.model_name == model]
        if not items:
            continue
        auprc = aggregate([item.evaluation.auprc for item in items])
        auroc = aggregate(
            [
                item.evaluation.auroc
                for item in items
                if item.evaluation.auroc is not None
            ]
        )
        pooled = _pool_scenarios(items)
        summary = latency.summarise(detections, model)
        summaries.append(
            ModelSummary(
                model=model,
                seeds=tuple(seeds),
                context_days=items[0].checkpoint.context_days,
                n_parameters=items[0].checkpoint.n_parameters,
                auprc_mean=auprc["mean"],
                auprc_std=auprc["std"],
                auroc_mean=auroc["mean"] if auroc["n"] else None,
                random_baseline=aggregate(
                    [item.evaluation.random_baseline for item in items]
                )["mean"],
                lift_mean=aggregate(
                    [item.evaluation.lift for item in items]
                )["mean"],
                rows_flagged=tuple(int(item.flagged.sum()) for item in items),
                pooled_scenarios=pooled,
                median_first_detection_day=summary.median_first_day,
                insiders_detected=summary.n_detected,
                insiders_total=summary.n_insiders,
            )
        )
    return summaries


def _pool_scenarios(items: Sequence[Scored]) -> tuple[ScenarioRecall, ...]:
    """Sum hits and attack days across seeds. Never average the recalls.

    Scenario 3 has four positive rows per split. A mean of three recalls over
    four rows each is a number with no denominator a reader can hold on to;
    "1 of 12 attack days" is one they can.
    """
    pooled: list[ScenarioRecall] = []
    for scenario in SCENARIOS:
        hits = 0
        total = 0
        for item in items:
            for recall in item.evaluation.per_scenario:
                if recall.scenario == scenario:
                    hits += recall.hits
                    total += recall.total
        pooled.append(ScenarioRecall(scenario=scenario, hits=hits, total=total))
    return tuple(pooled)


def format_report(run: InferenceRun) -> str:
    """The block the page prints when a run finishes."""
    lines: list[str] = ["", "=" * 72]
    if run.sample:
        lines.append(
            "SAMPLE RUN — 30 users at roughly a 30% insider rate. These "
            "numbers are meaningless as results; they prove the code runs."
        )
        lines.append("=" * 72)
    if run.transfer is not None:
        lines.extend(run.transfer.banner())
        lines.append("=" * 72)
    lines.append(f"{run.split} split, {len(run.seeds)} seeds, device {run.device}")
    lines.append("")

    header = f"{'model':9} {'seed':>5} {'AUPRC':>8} {'AUROC':>7}"
    header += "".join(f"{f'P@{k}':>7}" for k in PRECISION_AT_K)
    header += f" {'random':>9} {'xrandom':>8} {'flagged':>9}"
    lines.append(header)
    for result in run.per_seed:
        row = f"{result.model:9} {result.seed:>5} {result.auprc:>8.4f} "
        row += f"{result.auroc:>7.4f}" if result.auroc is not None else f"{'n/a':>7}"
        row += "".join(
            f"{result.precision_at_k.get(k, float('nan')):>7.3f}"
            for k in PRECISION_AT_K
        )
        row += (
            f" {result.random_baseline:>9.5f} {result.lift:>8.1f}"
            f" {result.n_flagged:>9,}"
        )
        lines.append(row)

    lines.append("")
    for model in run.models:
        lines.append(model.headline())
        lines.append("  " + model.latency_phrase())
        lines.append(
            f"  rows flagged per seed: "
            f"{', '.join(f'{n:,}' for n in model.rows_flagged)} "
            f"(not averaged — the spread is the finding)"
        )
        for recall in model.pooled_scenarios:
            lines.append(f"  pooled {recall.phrase()}")
    lines.append("=" * 72)
    return "\n".join(lines)
