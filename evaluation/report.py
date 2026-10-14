"""Assembling ``STAGE7_REPORT.md`` and ``outputs/final_results.json``.

Deliverables 1 and 2 of §11, and the only two that must survive if time runs
out. Everything this module writes is read back from JSON that a previous run
measured — no number in the report is computed here for the first time, which
means the report and the artifacts cannot disagree.

Two properties are worth stating because they are easy to lose:

**Optional inputs are optional.** §11 gives a cut order, so the B@7 control,
the ``pos_weight`` ablation and the figures may legitimately be missing. A
missing section says *"not run"* in the report rather than silently vanishing
— a hole a reader can see is worth more than a report that looks complete.

**The reporting rules from Stages 4-6 are applied here, not remembered.**
AUPRC is quoted mean +/- sd; recall at threshold never is, and appears only as
counts pooled across seeds. The random baseline is printed beside every AUPRC.
Scenario 3 is written as "n of 4 attack days". Those rules exist because each
one was learned by measuring something that swung, and this module is where
they stop depending on anybody's memory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    ACTIVE_RELEASE,
    EXPECTED_EMPLOYEE_COUNT,
    EXPECTED_INSIDER_COUNT,
    FINAL_RESULTS_PATH,
    MAX_EPOCHS,
    MODEL_A_NAME,
    MODEL_B_NAME,
    OUTPUT_DIR,
    RELEASE_R42,
    SCENARIO_PROFILES,
    STAGE7_REPORT_PATH,
    metrics_path,
)

logger = logging.getLogger(__name__)


_MISSING = "_not run — see §11's cut order._"

_DISPLAY = {MODEL_A_NAME: "Model A", MODEL_B_NAME: "Model B"}


def _name(model: str) -> str:
    return _DISPLAY.get(model, model)


def _only_on_r42(text: str, *, instead: str) -> str:
    """``text`` on r4.2; ``instead`` on any other release.

    A function rather than a conditional at each site, so that "is this an
    r4.2 measurement" is asked in exactly one place and reads the same way
    everywhere it is answered.
    """
    return text if ACTIVE_RELEASE == RELEASE_R42 else instead


def _positive_rate(summary: dict[str, Any]) -> float:
    """The per-day positive rate an aggregate block was measured over.

    ``random_baseline`` *is* the positive rate: the average precision of a
    model that ranks at random equals the fraction of rows that are positive.
    So the rate never needs typing, and until 2026-08-22 it was typed as
    "0.4%" in five places -- r4.2's rate, which would have appeared unchanged
    in an r5.2 report describing a different population.
    """
    return float(summary["random_baseline"]["mean"])


def _test_positive_rate(sources: Sources) -> float | None:
    """The test split's positive rate, or None if no headline block survived.

    Either model's block answers it -- the rate is a property of the rows, not
    of the model scoring them -- so this takes whichever is present rather
    than insisting on Model A.
    """
    head = sources.phase2["headline"]
    for model in (MODEL_A_NAME, MODEL_B_NAME):
        block = head.get(model)
        if block:
            return _positive_rate(block)
    return None


def _as_rate(value: float | None) -> str:
    """A positive rate as prose, or a word that admits it is missing."""
    return "an unmeasured" if value is None else f"a {value:.1%}"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        logger.warning("optional input missing: %s", path.name)
        return None
    return json.loads(path.read_text())


@dataclass(frozen=True, slots=True)
class Sources:
    """Every artifact the report is built from. Optional ones may be None."""

    phase1: dict[str, Any]
    phase2: dict[str, Any]
    val_a: dict[str, Any] | None
    val_b: dict[str, Any] | None
    control_b7: dict[str, Any] | None
    ablation_a: dict[str, Any] | None
    sample: bool

    @classmethod
    def load(cls, *, sample: bool = False, directory: Path = OUTPUT_DIR) -> Sources:
        suffix = ".sample" if sample else ""
        phase1 = _load(directory / f"phase1_validation{suffix}.json")
        phase2 = _load(directory / f"phase2_test{suffix}.json")
        if phase1 is None or phase2 is None:
            raise FileNotFoundError(
                "the report needs both phase artifacts. Run "
                "`python -m evaluation.run_phase1` then "
                "`python -m evaluation.run_phase2`."
            )
        return cls(
            phase1=phase1,
            phase2=phase2,
            val_a=_load(metrics_path(MODEL_A_NAME, sample=sample)),
            val_b=_load(metrics_path(MODEL_B_NAME, sample=sample)),
            control_b7=_load(metrics_path(MODEL_B_NAME, sample=sample, tag="w7")),
            ablation_a=_load(metrics_path(MODEL_A_NAME, sample=sample, tag="nopw")),
            sample=sample,
        )


def _pm(entry: dict[str, Any] | None, places: int = 4) -> str:
    """``mean +/- sd``, or an em dash when the figure was never measured."""
    if not entry or not entry.get("n"):
        return "—"
    return f"{entry['mean']:.{places}f} ± {entry['std']:.{places}f}"


def _counts(entry: dict[str, Any]) -> str:
    """``hits of total (rate)`` — counts first, always."""
    total = entry["attack_days_summed"]
    if not total:
        return "no attack days in these splits"
    return f"{entry['hits_summed']} of {total} ({entry['hits_summed'] / total:.1%})"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _preamble(sources: Sources) -> str:
    warning = (
        "\n> ⚠️ **These numbers come from the 30-user `--sample` data and are "
        "meaningless as results.** Regenerate from the full release before "
        "submitting.\n"
        if sources.sample
        else ""
    )
    seeds = sources.phase2["seeds"]
    return f"""# InnerSight — Stage 7: Evaluate & Compare
{warning}
An insider does nothing forbidden. They do normal things at abnormal times and
in abnormal amounts, with credentials that are genuinely theirs. So this
project never asks *"is this action bad?"* but **"is this unlike *them*?"** —
every raw daily count is a z-score against that person's own history.

Two small networks were trained on **identical rows with identical settings**.
Model A sees **one day**. Model B sees the **28 days ending on that day** and
predicts the same label. Everything else — the clipping, the standardiser, the
loss, the optimiser, the learning rate, the batch size, the early-stopping
rule, the threshold sweep, the seeds and the device — is shared code, verified
byte-identical by diff. **One variable changes: temporal context.** What
follows is the measurement of what that variable is worth.

Seeds {seeds}, reported as mean ± sd. The test split was opened once, for both
models together, using each checkpoint's own stored standardiser and decision
threshold; neither was recomputed on test.
"""


def _headline(sources: Sources) -> str:
    head = sources.phase2["headline"]
    a, b = head.get(MODEL_A_NAME, {}), head.get(MODEL_B_NAME, {})
    if not a or not b:
        return "## 1. Headline — test split\n\n" + _MISSING

    baseline = a["random_baseline"]["mean"]
    rows = [
        [
            "**AUPRC**",
            _pm(a["auprc"]),
            _pm(b["auprc"]),
            f"{baseline:.5f}",
        ],
        [
            "vs random",
            f"{a['lift_over_random']['mean']:.0f}×",
            f"{b['lift_over_random']['mean']:.0f}×",
            "1×",
        ],
        ["AUROC", _pm(a["auroc"]), _pm(b["auroc"]), "0.5"],
        ["P@10", _pm(a["precision_at_10"], 3), _pm(b["precision_at_10"], 3), "—"],
        ["P@50", _pm(a["precision_at_50"], 3), _pm(b["precision_at_50"], 3), "—"],
        [
            "parameters",
            f"{a['n_parameters']:,}",
            f"{b['n_parameters']:,}",
            "—",
        ],
        [
            "context",
            f"{a['context_days']} day",
            f"{b['context_days']} days",
            "—",
        ],
    ]
    per_seed = " · ".join(
        f"{seed}: A {a['per_seed_auprc'][seed]:.4f} / B {b['per_seed_auprc'][seed]:.4f}"
        for seed in sorted(a["per_seed_auprc"])
    )
    ratio = b["auprc"]["mean"] / a["auprc"]["mean"] if a["auprc"]["mean"] else float("nan")

    if ratio >= 1.1:
        verdict = (
            f"Model B's average precision is **{ratio:.1f}× Model A's** on "
            f"held-out users neither model has seen, at "
            f"**{b['lift_over_random']['mean']:.0f}× the random baseline of "
            f"{baseline:.5f}**."
        )
    elif ratio <= 0.9:
        verdict = (
            f"**Model B scores *below* Model A on test** — {ratio:.2f}× its "
            f"average precision, against a random baseline of "
            f"{baseline:.5f}. That contradicts the validation result and is "
            f"a finding to investigate and report, not to smooth over."
        )
    else:
        verdict = (
            f"The two models are level on test ({ratio:.2f}×, random baseline "
            f"{baseline:.5f}) — temporal context bought nothing measurable "
            f"here, which is itself the answer to the question asked."
        )

    return f"""## 1. Headline — test split

{_table(["", "Model A (1 day)", "Model B (28 days)", "random"], rows)}

Per seed — {per_seed}

{verdict} The two models differ in one respect: how many days of that person's
own recent history each one is allowed to look at.
"""


def _day_span(profile: Any) -> str:
    """A scenario profile's window length, as it is written in prose.

    A one-day-wide range is written as one number: "2 days", not "2-2 days".
    """
    if profile.min_days == profile.max_days:
        return f"{profile.min_days}"
    return f"{profile.min_days}–{profile.max_days}"


def _scenario_label(scenario: str) -> str:
    """One scenario's table label: its id, its shape, its measured day range.

    Built from ``config.SCENARIO_PROFILES``, which is measured per release.
    This was a hand-typed three-key dict, and it was right -- about r4.2, the
    only release that existed when it was written. Rendering r5.2 with it
    called a 44-to-60-day drift a 45-to-59-day one, and, because the table was
    built by iterating the *names* rather than the *results*, left scenario 4
    off the report entirely.

    Same rule as ``evaluation/figures.py``'s ``scenario_label``, and
    deliberately not shared with it: that one returns a two-line label, with a
    newline between the shape and the range, which is right for an x-axis tick
    and would break a markdown table row. What is worth sharing is the source
    of the words, and both read it from the same place.

    A scenario the release profile does not describe still gets its id and its
    counts; only the prose is missing. A missing label is cosmetic, a missing
    row is a lost finding.
    """
    profile = SCENARIO_PROFILES.get(int(scenario))
    if profile is None:
        return f"**{scenario}**"
    return f"**{scenario}** — {profile.kind} ({_day_span(profile)} days)"


def _model_counts(pooled: dict[str, Any], scenario: str) -> str:
    """One model's counts for one scenario, or a cell admitting it has none.

    The two models are scored by the same aggregation and normally carry the
    same scenario keys. The row is driven by the union of them anyway: a
    scenario one model scored and the other did not is a finding, and a
    ``KeyError`` in the middle of writing the deliverable is not how it should
    arrive.
    """
    entry = pooled.get(scenario)
    return _counts(entry) if entry else "not scored for this model"


def _per_scenario(sources: Sources) -> str:
    pooled = sources.phase2["per_scenario_pooled"]
    a, b = pooled.get(MODEL_A_NAME, {}), pooled.get(MODEL_B_NAME, {})
    if not a or not b:
        return "## 2. Per-scenario recall\n\n" + _MISSING

    rows = [
        [_scenario_label(key), _model_counts(a, key), _model_counts(b, key)]
        for key in sorted(set(a) | set(b), key=int)
    ]
    n_seeds = len(sources.phase2["seeds"]) or 1
    per_split_s3 = a["3"]["attack_days_summed"] // n_seeds
    return f"""## 2. Per-scenario recall — pooled across seeds, as counts

{_table(["Scenario", "Model A", "Model B"], rows)}

Counts rather than a mean of per-seed rates, and deliberately so. Stage 5
measured **on r4.2** Model A's flagged-row count swinging **1,168 / 481 / 107**
across the three seeds while its AUPRC moved only 0.124 → 0.169: F1
maximisation at {_as_rate(_test_positive_rate(sources))} positive rate sits on a
flat, noisy surface, so a single seed's recall is a statement about the
threshold lottery. Scenario 3 contributes
**{per_split_s3}** positive rows per split — {a['3']['attack_days_summed']}
pooled across the {n_seeds} seeds. "{a['3']['hits_summed']} of
{a['3']['attack_days_summed']} attack days" is what that supports; a
percentage on {per_split_s3} rows has {per_split_s3 + 1} possible values and
is not a measurement.
"""


def _verdict_line(verdict: dict[str, Any]) -> str:
    """One bullet stating a model's reading and the numbers behind it."""
    head = (
        f"- **{_name(verdict['model'])}** — {verdict['reading']}: "
        f"{verdict['claim_licensed']}"
    )
    early = verdict.get("early_recall_days_1_7")
    late = verdict.get("late_recall_days_29_plus")
    if early is None or late is None:
        return head
    parts = [f"early {early:.1%} → late {late:.1%}"]
    if verdict.get("late_over_early"):
        parts.append(f"{verdict['late_over_early']:.2f}×")
    median = verdict.get("median_first_detection_day")
    if median is not None:
        parts.append(f"median day {median:.0f}")
    return f"{head} ({', '.join(parts)})"


def _latency(sources: Sources) -> str:
    block = sources.phase2["detection_latency"]
    val_block = sources.phase1["detection_latency"]
    summaries = block["summaries"]
    buckets = [b for b in block["position_buckets"] if b["scenario"] is None]
    verdicts = val_block.get("verdicts", [])

    if not summaries:
        return "## 3. Detection latency\n\n" + _MISSING

    rows = []
    for entry in summaries:
        scope = "all" if entry["scenario"] is None else f"scenario {entry['scenario']}"
        median = entry["median_first_detection_day"]
        iqr = entry["iqr_first_detection_day"]
        rows.append(
            [
                _name(entry["model"]),
                scope,
                f"{median:.0f}" if median is not None else "never",
                f"{iqr[0]:.0f}–{iqr[1]:.0f}" if iqr[0] is not None else "—",
                f"{entry['detected']} / {entry['insiders']}",
                str(entry["never_detected"]),
            ]
        )

    bucket_rows = []
    for label in ("days 1-7", "days 8-14", "days 15-28", "days 29+"):
        row = [label]
        for model in (MODEL_A_NAME, MODEL_B_NAME):
            entry = next(
                (b for b in buckets if b["model"] == model and b["bucket"] == label),
                None,
            )
            if entry is None or not entry["attack_days"]:
                row.append("—")
            else:
                row.append(
                    f"{entry['hits']} / {entry['attack_days']} "
                    f"({entry['recall']:.1%})"
                )
        bucket_rows.append(row)

    reading = "\n".join(_verdict_line(v) for v in verdicts)

    drift = SCENARIO_PROFILES.get(2)
    drift_span = (
        f"Scenario-2 attacks run {_day_span(drift)} days."
        if drift is not None
        else "How long scenario-2 attacks run is not stated in this "
             "release's profile."
    )

    return f"""## 3. Detection latency — *what the project may claim*

{drift_span} For a day at position 30 of such an attack,
Model B's whole 28-day window sits *inside* the attack: it is reading 28
consecutive anomalous days. Nothing there is leakage — the window reaches
strictly backwards, holds only that user's rows, and no label is ever visible
— but *"detects the insider"* and *"notices an attack it already missed is
still running"* are different products, and only the first is worth claiming.

**First detection, on test** — day *N* of that insider's attack window:

{_table(["Model", "Scope", "Median day", "IQR", "Detected", "Never caught"], rows)}

**Recall by position within the attack window** (pooled counts, test):

{_table(["Position", "Model A", "Model B"], bucket_rows)}

**The reading** — measured on **validation**, where §5 places this analysis,
with the decision rule and its thresholds fixed in `evaluation/latency.py`
before any of these numbers existed:

{reading or "_no verdict computed_"}

Read this beside Stage 3's measurement that scenario-2 z-scores decay to
**0.40×** across an attack window. The signal is *weakest* late, so recall
that holds up late is not the window merely filling with loud days.
"""


def _control(sources: Sources) -> str:
    if sources.control_b7 is None:
        return (
            "## 4. The capacity control — Model B on a 7-day window\n\n"
            "Model B has 7.6× Model A's parameters, so *\"B wins because it is "
            "bigger\"* is a live alternative explanation for the entire "
            "result. The clean separation is the same architecture with the "
            "same parameter count and less context.\n\n" + _MISSING + "\n\n"
            "`python -m evaluation.run_b7` — three seeds, validation only, "
            "~4 minutes.\n"
        )

    b7 = sources.control_b7["aggregate"]["auprc"]
    a = (
        sources.val_a["aggregate"]["auprc"]
        if sources.val_a
        else {"mean": float("nan"), "std": 0.0, "n": 1}
    )
    b28 = (
        sources.val_b["aggregate"]["auprc"]
        if sources.val_b
        else {"mean": float("nan"), "std": 0.0, "n": 1}
    )

    span = b28["mean"] - a["mean"]
    position = (b7["mean"] - a["mean"]) / span if span else float("nan")
    if position < 0.15:
        conclusion = (
            "B@7 lands level with Model A. **The 28-day span specifically is "
            "what matters** — seven days of context buys essentially nothing, "
            "and the gain is not capacity, since B@7 has exactly B@28's "
            "parameter count."
        )
    elif position > 0.85:
        conclusion = (
            "B@7 lands level with B@28. **Capacity, not context, was doing "
            "the work** — a seven-day window recovers the whole gain, so the "
            "result should be reported as an architecture effect rather than "
            "a temporal-context one."
        )
    else:
        conclusion = (
            f"B@7 lands **between** Model A and Model B@28, at "
            f"{position:.0%} of the way from one to the other. Since B@7 and "
            f"B@28 have **identical parameter counts**, the remaining "
            f"{1 - position:.0%} cannot be capacity. **Temporal context is "
            f"the driver, and the claim is demonstrated rather than "
            f"asserted.**"
        )

    rows = [
        ["Model A", "1", f"{sources.phase2['headline'][MODEL_A_NAME]['n_parameters']:,}", _pm(a)],
        ["Model B @ 7 days", "7", f"{sources.phase2['headline'][MODEL_B_NAME]['n_parameters']:,}", _pm(b7)],
        ["Model B @ 28 days", "28", f"{sources.phase2['headline'][MODEL_B_NAME]['n_parameters']:,}", _pm(b28)],
    ]

    return f"""## 4. The capacity control — Model B on a 7-day window

Model B has 7.6× Model A's parameters, so *"B wins because it is bigger"* is a
live alternative explanation for the entire result. Convolutions plus global
pooling do not care about sequence length, so `ModelB(days=7)` has **exactly**
`ModelB(days=28)`'s parameter count. Same architecture, same capacity, less
context. Seven days is the floor: the two dilated k=3 layers have a 7-day
receptive field.

{_table(["", "Context", "Parameters", "Validation AUPRC"], rows)}

{conclusion}

This is a control and not a competing model: it was never scored on test.
"""


_NUMBER_WORDS = {0: "none", 1: "one", 2: "two", 3: "three"}


def _still_improving(seed_run: dict[str, Any]) -> bool:
    """``modeling/run_model_b.py``'s patience watch, applied by hand.

    A run that did *not* stop on patience and whose best epoch is its last was
    still improving when ``MAX_EPOCHS`` cut it off, so its score is a floor
    rather than a converged value. The watch is defined only for Model B, so it
    never executed on this ablation — but the rule predates the result, which is
    what makes it worth citing instead of eyeballing the epoch counts.
    """
    return (
        not seed_run["stopped_early"]
        and seed_run["best_epoch"] >= seed_run["epochs_run"] - 1
    )


def _by_seed(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """A run summary's per-seed records, in ascending seed order.

    Every ``a / b / c`` list in the report is written in this order, so that two
    such lists printed in the same sentence line up seed for seed.
    """
    return sorted(summary["per_seed"], key=lambda run: run["seed"])


def _per_seed_metric(summary: dict[str, Any], field: str, places: int = 4) -> str:
    """One validation metric across seeds, as ``a / b / c``."""
    return " / ".join(
        f"{run['validation'][field]:.{places}f}" for run in _by_seed(summary)
    )


def _per_seed_count(summary: dict[str, Any], field: str) -> str:
    """One integer field from each seed's run record, as ``a / b / c``."""
    return " / ".join(str(run[field]) for run in _by_seed(summary))


def _ablation_verdict(
    summary: dict[str, Any],
    weighted_summary: dict[str, Any] | None,
) -> str:
    """The paragraph under the ablation table, keyed on what was measured.

    Four endings, in priority order: the collapse guard fired; there is no
    weighted run to compare against; the unweighted model *won*; the weighted
    model won. The third exists because the run took it — an earlier version of
    this function had only the first and last, so on a release where plain BCE
    wins it asserted "average precision falls" directly beneath a table showing
    average precision doubling.
    """
    rate = _positive_rate(summary["aggregate"])

    if summary.get("collapse_guard_fired_any_seed", False):
        return (
            f"The collapse guard fired: without the weight the model scores "
            f"nearly every row the same, which is {1 - rate:.1%} accurate and "
            f"detects nothing. The loss curve looks healthy while it happens."
        )

    if weighted_summary is None:
        return (
            "The collapse guard did not fire. The weighted run is missing from "
            "this release's outputs, so the table stands alone and no "
            "comparison between the two losses is made here."
        )

    plain, weighted = summary["aggregate"], weighted_summary["aggregate"]

    if plain["auprc"]["mean"] <= weighted["auprc"]["mean"]:
        return (
            "The collapse guard did not fire, but average precision falls and "
            "the predicted probabilities compress toward zero — the weighted "
            "loss is what keeps the rare class visible in the ranking."
        )

    auroc_gap = weighted["auroc"]["mean"] - plain["auroc"]["mean"]
    ratio = plain["auprc"]["mean"] / weighted["auprc"]["mean"]
    baseline = plain["random_baseline"]["mean"]
    improving = sum(_still_improving(run) for run in summary["per_seed"])
    floor = (
        f" By that rule at least {_NUMBER_WORDS.get(improving, improving)} of "
        f"the three unweighted runs was still improving when training ended, "
        f"which makes {plain['auprc']['mean']:.4f} a lower bound."
        if improving
        else " By that rule no unweighted run was cut off while still improving."
    )

    return (
        f"**The prediction in the paragraph above was tested, and it was "
        f"wrong.** Plain BCE multiplied Model A's ranking quality by "
        f"**{ratio:.1f}** — **{_pm(plain['auprc'])} against "
        f"{_pm(weighted['auprc'])}** — and it also won P@10 "
        f"({_pm(plain['precision_at_10'], 3)} against "
        f"{_pm(weighted['precision_at_10'], 3)}) and P@50 "
        f"({_pm(plain['precision_at_50'], 3)} against "
        f"{_pm(weighted['precision_at_50'], 3)}). No seed overlaps: "
        f"{_per_seed_metric(summary, 'auprc')} without the weight against "
        f"{_per_seed_metric(weighted_summary, 'auprc')} with it. The only metric the "
        f"weighting wins is **AUROC, by {auroc_gap:.3f}** — and §6 is this "
        f"project's own argument that AUROC is the less informative of the two "
        f"at {_as_rate(rate)} positive rate.\n\n"
        f"**Nothing collapsed.** The unweighted model's mean predicted "
        f"probability is {plain['mean_predicted_prob']['mean']:.6f} against a "
        f"random baseline of {baseline:.4f} — a *calibrated* model rather than "
        f"a collapsed one — and it emitted "
        f"{plain['distinct_predicted_values']['mean']:,.0f} distinct scores.\n\n"
        f"**Reported, not adopted.** The ablation was registered in advance and "
        f"run on validation before the test split was opened. Substituting the "
        f"better model now would be choosing a model after seeing test numbers "
        f"— the practice the Phase 1 / Phase 2 ordering exists to prevent — so "
        f"the weighting stays in the results table and the measurement is "
        f"published beside it.\n\n"
        f"Two things keep this from overclaiming in the other direction. "
        f"**`pos_weight` was never ablated on Model B**, so nothing here says "
        f"anything about the CNN. And the unweighted runs went "
        f"{_per_seed_count(summary, 'epochs_run')} epochs against a ceiling of "
        f"{MAX_EPOCHS}, with best epochs at "
        f"{_per_seed_count(summary, 'best_epoch')}; `modeling/run_model_b.py` "
        f"carries a rule, written before this ablation existed, that flags a "
        f"run which hit the cap without stopping on patience while its best "
        f"epoch was its last.{floor}"
    )


def _ablation(sources: Sources) -> str:
    if sources.ablation_a is None:
        return (
            "## 5. The `pos_weight` ablation\n\n" + _MISSING + "\n\n"
            "`python -m modeling.run_model_a --no-pos-weight` — ~90 seconds, "
            "validation only, writes to `*_nopw_*` paths. The report should "
            "not assert the class-weighting claim without this table.\n"
        )

    weighted = sources.val_a["aggregate"] if sources.val_a else None
    plain = sources.ablation_a["aggregate"]

    def row(name: str, agg: dict[str, Any] | None) -> list[str]:
        if agg is None:
            return [name, *["—"] * 6]
        return [
            name,
            _pm(agg["auprc"]),
            _pm(agg["auroc"]),
            _pm(agg["precision_at_10"], 3),
            _pm(agg["precision_at_50"], 3),
            f"{agg['mean_predicted_prob']['mean']:.6f}",
            f"{agg['distinct_predicted_values']['mean']:,.0f}",
        ]

    verdict = _ablation_verdict(sources.ablation_a, sources.val_a)
    rows_per_batch = round(_positive_rate(plain) * 512)
    header = [
        "Model A",
        "AUPRC",
        "AUROC",
        "P@10",
        "P@50",
        "mean predicted prob",
        "distinct predictions",
    ]

    return f"""## 5. The `pos_weight` ablation — Model A, validation

At {_as_rate(_positive_rate(plain))} positive rate a batch of 512 holds about
{_NUMBER_WORDS.get(rows_per_batch, rows_per_batch)} positive rows, so the
cheapest way to reduce an *unweighted* loss is to push every score toward zero.
That mechanism is real, and Stages 5 and 6 both took one step further and
assumed the unweighted model would therefore converge on "nobody is an insider".
Nobody had measured it. This section is that measurement.

{_table(header, [row("weighted BCE", weighted), row("plain BCE", plain)])}

{verdict}

Not repeated on Model B: thirteen minutes for a second data point, and it was
cut. Whether the weighting helps or hurts the 28-day CNN is unmeasured.
"""


def _auroc_gap(sources: Sources) -> str:
    head = sources.phase2["headline"]
    a = head.get(MODEL_A_NAME, {})
    if not a:
        return "## 6. The AUROC/AUPRC gap\n\n" + _MISSING
    auroc = a["auroc"]["mean"]
    auprc = a["auprc"]["mean"]
    baseline = a["random_baseline"]["mean"]

    anchor = _only_on_r42(
        f"""AUROC {auroc:.3f} sits beside recent published work on r4.2 — LAN (Cai et al.,
2024) reports 0.9369 — while the average precision on those same predictions
is {auprc:.3f} against a random baseline of {baseline:.5f}. The
comparison is an anchor rather than like-for-like: LAN uses a temporal split
and a different labelling scheme. The point survives the caveat.""",
        instead=f"""AUROC {auroc:.3f} stands beside an average precision of {auprc:.3f} on those
same predictions, against a random baseline of {baseline:.5f}. **No published
AUROC is quoted here as an anchor.** The one this project used on r4.2 was
measured on r4.2, and a published number from a different population is a
comparison rather than a caveat, however carefully it is worded. Nobody has
put an r{ACTIVE_RELEASE} anchor beside this, and the honest output is to say
so rather than to borrow one.""",
    )

    return f"""## 6. The AUROC/AUPRC gap, stated as a finding

Model A scores **AUROC {auroc:.4f}** and **AUPRC {auprc:.4f}** on the *same
predictions*. Both numbers are correct and they describe different things.

{anchor} At {_as_rate(baseline)}
positive rate AUROC is dominated by the ease of ranking hundreds of thousands
of obvious negatives, and a metric that rewards a model for correctly ignoring
the {1 - baseline:.1%} of days nobody was ever going to look at does not
describe an analyst's queue. **Average precision does, which is why it is the
headline here and why the random baseline is printed beside it every time.**
"""


def _scenario_one_note(gap: float) -> str:
    """The explanation for scenario 1, only when scenario 1 actually moved."""
    if np.isnan(gap):
        return (
            "Scenario 1 had no attack days in these splits, so the prediction "
            "about it is unscored rather than missed."
        )
    if gap > 5:
        return (
            "The scenario-1 miss is the interesting one, and it has an "
            "explanation worth stating: a spike judged against **27 quiet "
            "days inside the same window** is a stronger signal than the same "
            "spike judged alone. Temporal context helps short bursts too — "
            "not only slow drifts — which is not what the hypothesis said."
        )
    if gap < -5:
        return (
            "Scenario 1 moved the other way: the 28-day model is *worse* on "
            "short bursts than the one-day model. Averaging over the window "
            "dilutes a one-day spike, which is the cost the mean-pooling "
            "choice was always going to carry — see the pooling paragraph in "
            "§8."
        )
    return (
        "Scenario 1 came out close to a tie, as predicted, which is the one "
        "part of the registered prediction the test split confirms directly."
    )


def _prediction_scorecard(sources: Sources) -> str:
    pooled = sources.phase2["per_scenario_pooled"]
    a, b = pooled.get(MODEL_A_NAME, {}), pooled.get(MODEL_B_NAME, {})
    head = sources.phase2["headline"]
    if not a or not b:
        return "## 7. The registered prediction, scored honestly\n\n" + _MISSING

    def rate(entry: dict[str, Any]) -> str:
        total = entry["attack_days_summed"]
        return f"{entry['hits_summed'] / total:.1%}" if total else "n/a"

    ratio = (
        head[MODEL_B_NAME]["auprc"]["mean"] / head[MODEL_A_NAME]["auprc"]["mean"]
        if head[MODEL_A_NAME]["auprc"]["mean"]
        else float("nan")
    )
    s1_gap = (
        (b["1"]["hits_summed"] / b["1"]["attack_days_summed"]
         - a["1"]["hits_summed"] / a["1"]["attack_days_summed"]) * 100
        if a["1"]["attack_days_summed"] and b["1"]["attack_days_summed"]
        else float("nan")
    )

    def improved(key: str) -> float:
        """Percentage-point change in pooled recall from Model A to Model B."""
        if not a[key]["attack_days_summed"] or not b[key]["attack_days_summed"]:
            return float("nan")
        return (
            b[key]["hits_summed"] / b[key]["attack_days_summed"]
            - a[key]["hits_summed"] / a[key]["attack_days_summed"]
        ) * 100

    s2_gap = improved("2")
    tie_points = 5.0
    unscored = "— not scored: no attack days in these splits"
    rows = [
        [
            "scenario 2 would improve",
            unscored if np.isnan(s2_gap)
            else "✅ held" if s2_gap > tie_points else "❌ missed",
            "n/a" if np.isnan(s2_gap) else f"{rate(a['2'])} → {rate(b['2'])}",
        ],
        [
            "scenario 1 would roughly tie",
            unscored if np.isnan(s1_gap)
            else "✅ held" if abs(s1_gap) <= tie_points else "❌ missed",
            "n/a" if np.isnan(s1_gap)
            else f"{rate(a['1'])} → {rate(b['1'])}, B "
                 f"{'ahead' if s1_gap >= 0 else 'behind'} by "
                 f"{abs(s1_gap):.1f} points",
        ],
        [
            "scenario 3 would tie",
            "✅ held" if a["3"]["hits_summed"] == b["3"]["hits_summed"] else "❌ missed",
            f"{a['3']['hits_summed']} of {a['3']['attack_days_summed']} → "
            f"{b['3']['hits_summed']} of {b['3']['attack_days_summed']} attack days",
        ],
        [
            "overall gain would be modest",
            "✅ held" if 0.9 <= ratio <= 1.5
            else "⚠️ direction right, magnitude wrong" if ratio > 1.5
            else "❌ missed — B did not gain",
            f"B is {ratio:.1f}× A, against the ~1.2× predicted",
        ],
    ]

    return f"""## 7. The registered prediction, scored honestly

Written into `STAGE6_CONTEXT.md` **before Model B existed**, and scored here
against the test result rather than quietly dropped.

{_table(["Prediction", "Outcome", "Measured"], rows)}

{_scenario_one_note(s1_gap)}

The hoped-for surgical shape — *B ties overall but wins scenario 2, thereby
identifying which attack profile needs history* — {'did not happen: B won across the board' if ratio > 1.5 and s1_gap > 5 else 'is not what the test split produced either; the measured shape is in the table above'}.
"""


def _limitations(sources: Sources) -> str:
    inspection = sources.phase1["false_positive_inspection"]
    pooled = inspection.get("pooled")
    fp_sentence = (
        pooled["sentence_the_evidence_supports"]
        if pooled
        else "_the top-scoring false positives were not inspected._"
    )

    rate = _test_positive_rate(sources)
    per_day = "not measured" if rate is None else f"{rate:.1%}"

    pooled_s3 = sources.phase2["per_scenario_pooled"].get(MODEL_A_NAME, {})
    n_seeds = len(sources.phase2["seeds"]) or 1
    s3_rows = (
        pooled_s3["3"]["attack_days_summed"] // n_seeds if "3" in pooled_s3 else "—"
    )

    attack_days = sum(e["attack_days_summed"] for e in pooled_s3.values())
    s2_share = (
        f"{pooled_s3['2']['attack_days_summed'] / attack_days:.0%}"
        if attack_days and "2" in pooled_s3
        else "an unmeasured share"
    )

    features = _only_on_r42(
        """**Only 15 of the 18 features are informative on r4.2.**
  `large_attachment_count` is identically zero across the release, and two
  others barely move.""",
        instead=f"""**How many of the 18 features are informative on r{ACTIVE_RELEASE} has not
  been measured.** On r4.2 it was 15 of 18, with `large_attachment_count`
  identically zero across the release; the equivalent pass has not been run
  here, so no count is claimed.""",
    )
    windows = _only_on_r42(
        """Stage 2 measured that the windows mark the *signature event*: BIH0745's
  heaviest after-hours USB copying sits outside his 65-minute labelled
  window, and KPC0073's loudest day of all (z = 107 on after-hours USB) is
  formally labelled 0.""",
        instead=f"""Stage 2 measured this on r4.2, where the windows marked the *signature
  event* rather than the whole of the anomalous behaviour — the two users it
  named do not exist in r{ACTIVE_RELEASE}'s population, and the equivalent
  inspection has not been done here.""",
    )

    return f"""## 8. Limitations

Every one of these was measured, not guessed.

- **The dataset is synthetic and deliberately dense.** CERT r{ACTIVE_RELEASE}
  carries {EXPECTED_INSIDER_COUNT} insiders among {EXPECTED_EMPLOYEE_COUNT:,}
  users — a {EXPECTED_INSIDER_COUNT / EXPECTED_EMPLOYEE_COUNT:.2%} rate that no
  real organisation has. The per-day positive rate ({per_day}) is realistic;
  the per-*person* rate is not.
- {features}
- **The published labels are narrower than the actual anomalous behaviour.**
  {windows} On inspection of the highest-scoring "false
  positives": {fp_sentence}
- **Scenario 3 rests on {s3_rows} test rows per split.** Its standard deviation
  across seeds measures noise on {s3_rows} rows and nothing else.
- **The Stage 3 standard-deviation floor is a population statistic computed
  over all users, test users included.** It is a median of per-user standard
  deviations, so any one held-out user contributes roughly
  {1 / EXPECTED_EMPLOYEE_COUNT:.2%} of it —
  bounded, disclosed, and not worth re-running the pipeline over.
- **A tuning attempt on Model B lost, and why it lost is the finding.**
  Measured on r4.2 validation: dropout 0.2 with mean+max pooling and a cosine
  schedule took AUPRC from 0.6767 to 0.6022 with double the spread. Mean+max
  pooling raised scenario-1 recall by 6.6 points and dropped scenario-2 by
  11.0. Scenario 2 is {s2_share} of r{ACTIVE_RELEASE}'s positives, so it
  decides the AUPRC. A max channel offers the shortcut *"find the single worst
  day"* — the right instinct for a burst, the wrong one for a drift whose
  individual days are unremarkable. **The pooling operator selects which
  attack profile the model can detect.** That variant is excluded from every
  table above by design.
"""


def _future_work() -> str:
    generalisation = _only_on_r42(
        """**r5.2 as a
generalisation test** — 2,000 users, all eighteen features exercised, and a
fourth attack scenario, which would separate "the method works" from "the
method works on r4.2".""",
        instead="""**A third release, or a
non-synthetic one** — this report is r5.2, which was itself the generalisation
test r4.2's report asked for: 2,000 users and a fourth attack scenario,
scored with the same code and the same settings. What it does not answer is
whether the method survives data that was not generated by the same
simulator.""",
    )
    return f"""## 9. Future work

Three things this project measured itself into wanting. {generalisation}
**Three weight initialisations within each split**,
which is the ensemble this project cannot currently claim: the three seeds
here are three different *splits*, so averaging them measures the split
lottery rather than model variance. **Attention pooling in place of the mean**,
which would keep the drift sensitivity the mean gives while producing a
per-day weight map — turning "this person is anomalous" into "these are the
days that made them anomalous", which is what an analyst actually opens the
alert to see.
"""


def _guard_appendix(sources: Sources) -> str:
    rows = []
    for block, phase in ((sources.phase1, "1 (validation)"), (sources.phase2, "2 (test)")):
        for guard in block["guards"]:
            rows.append(
                [
                    phase,
                    "ok" if guard["passed"] else "**FAILED**",
                    guard["guard"],
                    guard["detail"],
                ]
            )
    return f"""## Appendix A — guards

Every check from the brief's §13 list, as it ran.

{_table(["Phase", "", "Guard", "Detail"], rows)}
"""


def build(sources: Sources) -> str:
    """Assemble the whole report."""
    sections = [
        _preamble(sources),
        _headline(sources),
        _per_scenario(sources),
        _latency(sources),
        _control(sources),
        _ablation(sources),
        _auroc_gap(sources),
        _prediction_scorecard(sources),
        _limitations(sources),
        _future_work(),
        _guard_appendix(sources),
    ]
    return "\n\n---\n\n".join(section.strip() for section in sections) + "\n"


def final_results(sources: Sources) -> dict[str, Any]:
    """Deliverable 2 — every number in one machine-readable file."""
    return {
        "project": f"InnerSight — insider threat detection on CERT r{ACTIVE_RELEASE}",
        "stage": 7,
        "sample": sources.sample,
        "test": sources.phase2,
        "validation": sources.phase1,
        "validation_training_metrics": {
            MODEL_A_NAME: sources.val_a,
            MODEL_B_NAME: sources.val_b,
        },
        "capacity_control_b7": sources.control_b7,
        "pos_weight_ablation_model_a": sources.ablation_a,
        "excluded_from_all_results": {
            "model_b_v2": (
                "tuning attempt (dropout 0.2, mean+max pooling, cosine LR). "
                "Measured on r4.2 validation: AUPRC 0.6767 -> 0.6022 with "
                "double the spread. "
                "Excluded by brief §3; appears only as a discussion paragraph "
                "about the pooling operator."
            )
        },
    }


def write(
    sources: Sources,
    *,
    report_path: Path = STAGE7_REPORT_PATH,
    results_path: Path = FINAL_RESULTS_PATH,
) -> tuple[Path, Path]:
    """Write both deliverables and return where they landed."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build(sources))
    results_path.write_text(
        json.dumps(final_results(sources), indent=2, sort_keys=True) + "\n"
    )
    logger.info("report  -> %s", report_path)
    logger.info("results -> %s", results_path)
    return report_path, results_path
