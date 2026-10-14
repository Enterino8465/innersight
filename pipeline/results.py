"""The models that already exist, and what they already measured.

Nothing here computes anything. Every number this module returns was read from
a file that was written by a run that has already happened:
``outputs/final_results.json`` for the frozen r4.2 experiment, the checkpoint
files themselves for what each model *is*, and ``models/<run_id>/manifest.json``
for anything the UI trained.

**That is the whole point of the module, and it is worth being blunt about.**
The other two buttons run something. This one opens the report. It exists
because at a defence the useful thing is often not a live run but an
immediate answer — the six checkpoints, their parameter counts, their stored
thresholds, and the published numbers beside them, with no waiting and no
opportunity for anything to go wrong on stage.

It also means this view is safe by construction in a way the others have to
argue for: there is no scoring path here at all. It cannot open a test split
because it cannot open anything — it reads JSON and checkpoint headers.

Two sections come back:

``frozen``
    The six headline checkpoints. Their identity comes from the ``.pt`` files
    (parameters, context days, stored threshold, training device); their
    results come from ``final_results.json``, per seed and aggregated, with the
    per-scenario pooled counts, the first-detection days, and the guard block
    from the run that produced them.

``trained``
    Whatever the Train button has stored, from each run's manifest — the
    validation metrics, the feature config those weights assume, and the git
    commit of the code that made them.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Final

from config import (
    CHECKPOINT_DIR,
    FINAL_RESULTS_PATH,
    MODEL_A_NAME,
    MODEL_B_NAME,
    PROFILES,
    SEEDS,
    UI_REPRODUCTION_TOLERANCE,
)
from evaluation.checkpoints import CheckpointError, headline_checkpoints
from pipeline import store

logger = logging.getLogger(__name__)

_MODELS: Final[tuple[str, ...]] = (MODEL_A_NAME, MODEL_B_NAME)

_SPLITS: Final[tuple[tuple[str, str], tuple[str, str]]] = (
    ("test", "test"),
    ("validation", "validation"),
)


class ResultsError(RuntimeError):
    """The stored record could not be read. Never raised for 'nothing yet'."""


def frozen_models(
    *,
    results_path: Path = FINAL_RESULTS_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """The six headline checkpoints and everything already measured about them.

    Missing files are reported rather than raised on: a fresh checkout has no
    ``final_results.json``, and a page that 500s in that state is less useful
    than one that says which command writes it.
    """
    report = _read_results(results_path)
    problems: list[str] = []
    if report is None:
        problems.append(
            f"no {results_path.name} — run `python -m evaluation.run_stage7` "
            f"to write it. Checkpoint details below are read from the .pt "
            f"files and do not depend on it."
        )
        report = {}

    identity, checkpoint_problems = _checkpoint_identity(checkpoint_dir, seeds)
    problems.extend(checkpoint_problems)

    return {
        "source": str(results_path),
        "checkpoints": identity,
        "test": _split_block(report, "test"),
        "validation": _split_block(report, "validation"),
        "problems": problems,
    }


def _read_results(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        logger.warning("no frozen results at %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ResultsError(f"{path} is not readable JSON: {error}") from error


def _checkpoint_identity(
    directory: Path, seeds: tuple[int, ...]
) -> tuple[list[dict[str, Any]], list[str]]:
    """What each stored model *is*, read from the checkpoint itself.

    Read from the ``.pt`` files rather than from the report on purpose. The
    parameter count and the stored threshold are properties of the weights; a
    page that showed the report's copy of them could not notice the two
    disagreeing.
    """
    found: list[dict[str, Any]] = []
    problems: list[str] = []
    for seed in seeds:
        try:
            loaded = headline_checkpoints(seed, directory=directory)
        except CheckpointError as error:
            problems.append(f"seed {seed}: {error}")
            continue
        for model in _MODELS:
            checkpoint = loaded[model]
            found.append({
                "model": model,
                "seed": seed,
                "file": checkpoint.path.name,
                "architecture": checkpoint.model_class,
                "context_days": checkpoint.context_days,
                "n_parameters": checkpoint.n_parameters,
                "threshold": checkpoint.threshold,
                "trained_on": checkpoint.device_trained_on,
                "best_val_auprc": checkpoint.best_val_auprc,
                "val_random_baseline": checkpoint.val_random_baseline,
            })
    return found, problems


def _split_block(report: dict[str, Any], split: str) -> dict[str, Any] | None:
    """One split's published numbers, reshaped for the page.

    ``test`` and ``validation`` are stored under different key names by
    ``evaluation.report`` — ``per_seed`` against ``validation_metrics`` — which
    is a wart, not a distinction. Normalised here so the page has one shape.
    """
    block = report.get(split)
    if not block:
        return None

    rows = block.get("per_seed") or block.get("validation_metrics") or []
    return {
        "split": split,
        "device": block.get("device"),
        "seeds": block.get("seeds", []),
        "all_guards_passed": block.get("all_guards_passed"),
        "per_seed": [_seed_row(row) for row in rows],
        "aggregate": _aggregate(block, rows),
        "per_scenario_pooled": _pooled(block, rows),
        "latency": block.get("detection_latency", {}).get("summaries", []),
        "guards": block.get("guards", []),
    }


def _seed_row(row: dict[str, Any]) -> dict[str, Any]:
    at_k = row.get("precision_at_k", {})
    return {
        "model": row.get("model"),
        "seed": row.get("seed"),
        "auprc": row.get("auprc"),
        "auroc": row.get("auroc"),
        "precision_at_10": at_k.get("10"),
        "precision_at_50": at_k.get("50"),
        "random_baseline": row.get("random_baseline"),
        "lift_over_random": row.get("lift_over_random"),
        "n_rows": row.get("n_rows"),
        "n_positives": row.get("n_positives"),
        "n_flagged": row.get("n_flagged"),
        "threshold": row.get("threshold_applied") or row.get("threshold"),
        "context_days": row.get("context_days"),
        "n_parameters": row.get("n_parameters"),
    }


def _aggregate(block: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean +/- sd per model, from the report's own headline block if it has one.

    The test block carries a ``headline`` with the aggregates already computed
    by Stage 7. Validation does not, so it is aggregated here — with
    :func:`modeling.metrics.aggregate`, which is the project's population sd
    (ddof=0), so this page cannot disagree with the report by a convention.
    """
    from modeling.metrics import aggregate

    headline = block.get("headline")
    summaries: list[dict[str, Any]] = []
    for model in _MODELS:
        mine = [row for row in rows if row.get("model") == model]
        if not mine:
            continue
        if headline and model in headline:
            published = headline[model]
            summaries.append({
                "model": model,
                "auprc": published["auprc"],
                "auroc": published.get("auroc"),
                "random_baseline": published["random_baseline"]["mean"],
                "lift_over_random": published["lift_over_random"]["mean"],
                "context_days": published.get("context_days"),
                "n_parameters": published.get("n_parameters"),
                "rows_flagged_per_seed": [row.get("n_flagged") for row in mine],
                "from": "the report's own aggregate",
            })
            continue
        summaries.append({
            "model": model,
            "auprc": aggregate([row["auprc"] for row in mine]),
            "auroc": aggregate(
                [row["auroc"] for row in mine if row.get("auroc") is not None]
            ),
            "random_baseline": aggregate(
                [row["random_baseline"] for row in mine]
            )["mean"],
            "lift_over_random": aggregate(
                [row["lift_over_random"] for row in mine]
            )["mean"],
            "context_days": mine[0].get("context_days"),
            "n_parameters": mine[0].get("n_parameters"),
            "rows_flagged_per_seed": [row.get("n_flagged") for row in mine],
            "from": "aggregated here from the per-seed rows",
        })
    return summaries


def _pooled(
    block: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Per-scenario results as pooled counts, never as averaged recalls.

    Scenario 3 has four positive rows per split; a mean of three recalls over
    four rows each is a number with no denominator a reader can hold. The test
    block publishes these already; validation is pooled here from its per-seed
    ``per_scenario`` lists, summing hits and attack days.
    """
    published = block.get("per_scenario_pooled")
    pooled: dict[str, list[dict[str, Any]]] = {}
    for model in _MODELS:
        if published and model in published:
            pooled[model] = [
                {
                    "scenario": int(scenario),
                    "hits": values["hits_summed"],
                    "attack_days": values["attack_days_summed"],
                }
                for scenario, values in sorted(published[model].items())
            ]
            continue
        counts: dict[int, list[int]] = {}
        for row in rows:
            if row.get("model") != model:
                continue
            for entry in row.get("per_scenario", []):
                totals = counts.setdefault(int(entry["scenario"]), [0, 0])
                totals[0] += int(entry.get("hits", 0))
                totals[1] += int(entry.get("attack_days", 0))
        pooled[model] = [
            {"scenario": scenario, "hits": hits, "attack_days": days}
            for scenario, (hits, days) in sorted(counts.items())
        ]
    return pooled


def trained_models(
    *, results_path: Path = FINAL_RESULTS_PATH
) -> list[dict[str, Any]]:
    """Every stored run, from its manifest. Empty is a normal answer.

    Each run is also compared against the frozen Stage 5/6 validation numbers,
    because the first thing a reader asks on seeing two tables of identical
    figures is whether the page is showing the same thing twice. It is not —
    and saying so turns a confusing coincidence into the strongest
    reproducibility claim the project has.
    """
    report = _read_results(results_path) or {}
    published = {
        (row["model"], row["seed"]): row
        for row in report.get("validation", {}).get("validation_metrics", [])
    }
    epochs = _published_epochs(report)
    return [
        manifest.as_dict() | {
            "stale": _feature_config_drifted(manifest),
            "reproduces_frozen": _compare_to_frozen(
                manifest.validation_metrics, published, epochs
            ),
        }
        for manifest in store.list_runs()
    ]


def _published_epochs(report: dict[str, Any]) -> dict[tuple[str, int], int]:
    """Best epoch per (model, seed) from the frozen training metrics block."""
    found: dict[tuple[str, int], int] = {}
    for model, block in report.get("validation_training_metrics", {}).items():
        for row in block.get("per_seed", []):
            if row.get("best_epoch") is not None:
                found[(model, int(row["seed"]))] = int(row["best_epoch"])
    return found


def _compare_to_frozen(
    entries: list[dict[str, Any]],
    published: dict[tuple[str, int], dict[str, Any]],
    epochs: dict[tuple[str, int], int],
) -> str | None:
    """Does this run's validation match the frozen one? A sentence, or None.

    None when there is nothing to compare — a run on different seeds, or a
    repository with no report. Silence is the honest answer there; a
    "differs" that only means "these are different experiments" would be
    noise on every legitimately different run.
    """
    pairs = [(e, published[(e["model"], e["seed"])])
             for e in entries if (e["model"], e["seed"]) in published]
    if not pairs:
        return None

    deltas = [
        abs(mine["best_val_auprc"] - theirs["auprc"])
        for mine, theirs in pairs
        if mine.get("best_val_auprc") is not None
    ]
    if not deltas:
        return None
    worst = max(deltas)

    if worst > UI_REPRODUCTION_TOLERANCE:
        offender = max(
            pairs, key=lambda pair: abs(
                pair[0]["best_val_auprc"] - pair[1]["auprc"]
            )
        )[0]
        return (
            f"Differs from the frozen Stage 5/6 run — largest difference "
            f"{worst:.1e}, on {offender['model']} seed {offender['seed']}. "
            f"Expected if this run used a different folder, seeds or epoch cap; "
            f"investigate only if it was meant to be the same experiment."
        )

    same_threshold = all(
        mine["threshold"] == theirs.get("threshold_applied")
        for mine, theirs in pairs
    )
    same_epoch = all(
        epochs.get((mine["model"], mine["seed"])) == mine["best_epoch"]
        for mine, theirs in pairs
    )
    extras = [
        label for label, held in
        (("thresholds", same_threshold), ("stopping epochs", same_epoch))
        if held
    ]
    tail = f", and identical {' and '.join(extras)}" if extras else ""
    return (
        f"Reproduces the frozen Stage 5/6 training run: all {len(pairs)} "
        f"model/seed pairs agree on validation AUPRC (largest difference "
        f"{worst:.1e}){tail}. The two tables above and below are two separate "
        f"training runs that produced the same numbers, not one table shown "
        f"twice."
    )


def _feature_config_drifted(manifest: store.Manifest) -> str | None:
    """Whether this run's eighteen features still mean what they meant.

    Shown on the listing rather than only at load time, so a run that can no
    longer be used says so before anyone selects it.
    """
    try:
        store.verify_feature_config(manifest)
    except store.StoreError as error:
        return str(error)
    return None


_CONTEXT_DAYS_MODEL_A: Final[int] = 1


def trained_runs() -> list[dict[str, Any]]:
    """Every model this UI has trained, newest first. Empty is a normal answer.

    The whole of what the Results tab shows, and the whole of what that tab is
    for: a record of the training sessions started from this page. It does not
    open ``final_results.json``, does not load a checkpoint and does not score
    a row. Every number here was written by the training run that produced it
    and kept in that run's own ``manifest.json``, which is why a run whose
    ``_work`` sandbox has been deleted still shows its results.

    A run trained before the manifest carried ``results`` comes back with an
    empty ``models`` list and ``measured`` false, rather than with zeros. The
    page says so and names the command that repairs it; inventing a number for
    a run that did not record one would be the worse answer.
    """
    return [_one_run(manifest) for manifest in store.list_runs()]


def _one_run(manifest: store.Manifest) -> dict[str, Any]:
    """One training session, reshaped into the tables the page draws."""
    results = manifest.results or {}
    window = manifest.feature_config.get("WINDOW_DAYS")
    models = [
        _model_summary(name, results[name], window)
        for name in _MODELS if results.get(name)
    ]
    return {
        "run_id": manifest.run_id,
        "release": manifest.release,
        "release_label": _release_label(manifest.release),
        "created": manifest.created,
        "seeds": list(manifest.seeds),
        "device": _device_used(results, manifest.device),
        "device_asked_for": manifest.device,
        "source_folder": manifest.source_folder,
        "git_commit": manifest.git_commit,
        "window_days": window,
        "seconds_total": _total_seconds(results),
        "models": models,
        "scenarios": _scenario_rows(manifest.release, results),
        "measured": bool(models),
        "stale": _feature_config_drifted(manifest),
        "warnings": list(manifest.warnings),
    }


def _model_summary(
    name: str, block: dict[str, Any], window: Any
) -> dict[str, Any]:
    """One model's row in the averaged table, plus its per-seed rows."""
    aggregate = block.get("aggregate") or {}

    def mean(key: str) -> Any:
        return (aggregate.get(key) or {}).get("mean")

    auprc, random = mean("auprc"), mean("random_baseline")
    return {
        "model": name,
        "context_days": window if name == MODEL_B_NAME else _CONTEXT_DAYS_MODEL_A,
        "n_parameters": block.get("n_parameters"),
        "seconds_total": block.get("seconds_total"),
        "auprc": aggregate.get("auprc"),
        "auroc": aggregate.get("auroc"),
        "precision_at_10": mean("precision_at_10"),
        "precision_at_50": mean("precision_at_50"),
        "random_baseline": random,
        "lift_over_random": auprc / random if auprc and random else None,
        "per_seed": block.get("per_seed") or [],
    }


def _scenario_rows(
    release: str, results: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per attack scenario, with a column of hits per model.

    Merged here rather than in the page so that the two models land in one
    table instead of two, and so the label comes from
    :data:`config.PROFILES` -- which means an r5.2 run is labelled with r5.2's
    own day ranges and carries scenario 4, without this module knowing that
    scenario 4 exists.
    """
    profile = PROFILES.get(release)
    windows = profile.scenario_profiles if profile else {}
    rows: dict[int, dict[str, Any]] = {}
    for name in _MODELS:
        for entry in (results.get(name) or {}).get("per_scenario", []):
            number = int(entry["scenario"])
            row = rows.setdefault(number, {
                "scenario": number,
                "label": _scenario_label(number, windows.get(number)),
                "attack_days": entry.get("attack_days"),
                "hits": {},
            })
            row["hits"][name] = entry.get("hits")
            if row["attack_days"] is None:
                row["attack_days"] = entry.get("attack_days")
    return [rows[number] for number in sorted(rows)]


def _scenario_label(number: int, window: Any) -> str:
    """``2 — drift, 45-59 days``, or the bare number for a release we lack."""
    if window is None:
        return str(number)
    span = (
        f"{window.min_days} days" if window.min_days == window.max_days
        else f"{window.min_days}\u2013{window.max_days} days"
    )
    return f"{number} \u2014 {window.kind}, {span}"


def _release_label(release: str) -> str:
    profile = PROFILES.get(release)
    if profile is None:
        return f"CERT r{release}"
    return (
        f"CERT r{release} \u2014 "
        f"{profile.expected_employee_count:,} employees"
    )


def _device_used(results: dict[str, Any], asked_for: str) -> str:
    """What the training actually ran on, not what the page asked for.

    The manifest's ``device`` is the request -- usually ``auto``, which is not
    a device. The metrics files record what the request resolved to.
    """
    for name in _MODELS:
        found = (results.get(name) or {}).get("device")
        if found:
            return str(found)
    return asked_for


def _total_seconds(results: dict[str, Any]) -> float | None:
    total = sum(
        (results.get(name) or {}).get("seconds_total") or 0.0
        for name in _MODELS
    )
    return total or None


def everything() -> dict[str, Any]:
    """Both sections, for one page load. Reads files; computes nothing."""
    return {"frozen": frozen_models(), "trained": trained_models()}


def format_report(view: dict[str, Any]) -> str:
    """The same content as the page, for a terminal."""
    lines: list[str] = []
    frozen = view["frozen"]
    lines.append(f"frozen checkpoints ({frozen['source']})")
    for entry in frozen["checkpoints"]:
        lines.append(
            f"  {entry['model']} seed {entry['seed']:<4} "
            f"{entry['architecture']:7} {entry['context_days']:>2}d "
            f"{entry['n_parameters']:>6,} params  "
            f"threshold {entry['threshold']:.6f}  trained on {entry['trained_on']}"
        )
    for split in ("test", "validation"):
        block = frozen.get(split)
        if not block:
            lines.append(f"  no {split} block in the report")
            continue
        lines.append(f"  {split}:")
        for summary in block["aggregate"]:
            auprc = summary["auprc"]
            lines.append(
                f"    {summary['model']}: AUPRC {auprc['mean']:.4f} +/- "
                f"{auprc['std']:.4f} (random {summary['random_baseline']:.5f}, "
                f"{summary['lift_over_random']:.0f}x)"
            )
        for model, entries in block["per_scenario_pooled"].items():
            for entry in entries:
                lines.append(
                    f"    {model} scenario {entry['scenario']}: "
                    f"{entry['hits']} of {entry['attack_days']} attack days"
                )
    for problem in frozen["problems"]:
        lines.append(f"  ! {problem}")

    lines.append("")
    trained = view["trained"]
    lines.append(f"{len(trained)} model(s) trained by this UI")
    for manifest in trained:
        lines.append(
            f"  {manifest['run_id']} — r{manifest['release']}, "
            f"{len(manifest['seeds'])} seeds, {manifest['device']}"
            + (" [STALE]" if manifest["stale"] else "")
        )
        for entry in manifest["validation_metrics"]:
            lines.append(
                f"    {entry['model']} seed {entry['seed']}: val AUPRC "
                f"{entry['best_val_auprc']:.4f}, threshold {entry['threshold']:.6f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the stored record. No arguments; it reads and returns."""
    import argparse

    argparse.ArgumentParser(
        prog="python -m pipeline.results",
        description="Show every trained model and what it already measured.",
    ).parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(message)s")
    print(format_report(everything()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
