"""Deliverables 3 and 4 — the two figures, in the style Stages 5-6 already use.

Deliberately matching ``modeling.runner.plot_curves``: matplotlib on the Agg
backend, 150 dpi, ``grid(alpha=0.3)``, the default property cycle for series
colour. A report whose Stage 7 figures look like they came from a different
project invites the question of whether they came from different data.

Three rules from §11 are enforced rather than remembered:

* **Every axis is labelled**, including units.
* **`n` is stated in the caption**, because a recall bar over four attack days
  and a recall bar over 990 look identical otherwise.
* **The random baseline is drawn as a horizontal line on any AUPRC axis.** At
  a 0.4% positive rate an AUPRC of 0.68 is meaningless without it; with it,
  the bar and the line together are the whole result.

Figures are the first deliverables the cut order sacrifices, so every function
here returns ``None`` rather than raising if matplotlib is unavailable. A
missing figure must never cost a run.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    ACTIVE_RELEASE,
    FIGURE_DIR,
    MODEL_A_NAME,
    MODEL_B_NAME,
    SCENARIO_PROFILES,
    SCENARIOS,
)

logger = logging.getLogger(__name__)

_MODEL_LABEL = {MODEL_A_NAME: "Model A (1 day)", MODEL_B_NAME: "Model B (28 days)"}
_MODEL_HATCH = {MODEL_A_NAME: "", MODEL_B_NAME: "//"}
_BUCKET_ORDER = ("days 1-7", "days 8-14", "days 15-28", "days 29+")


def _pyplot():
    """Import matplotlib on the Agg backend, or return None."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping figures")
        return None
    return plt


def comparison(
    phase2: dict[str, Any],
    val_a: dict[str, Any] | None,
    val_b: dict[str, Any] | None,
    *,
    path: Path | None = None,
) -> Path | None:
    """Deliverable 3: training curves side by side, and per-scenario recall.

    Four panels, because the two questions a reader has are "did they train
    properly" and "what did that buy". The curves answer the first; the bars
    answer the second.
    """
    plt = _pyplot()
    if plt is None:
        return None

    path = path or FIGURE_DIR / "stage7_comparison.png"
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    _curves_panel(axes[0][0], val_a, "Model A — training curves", colours)
    _curves_panel(axes[0][1], val_b, "Model B — training curves", colours)
    _headline_panel(axes[1][0], phase2)
    _scenario_panel(axes[1][1], phase2)

    seeds = phase2.get("seeds", [])
    figure.suptitle(
        f"InnerSight Stage 7 — Model A vs Model B, CERT r{ACTIVE_RELEASE}, "
        f"seeds {seeds}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure  -> %s", path)
    return path


def _curves_panel(axis, metrics: dict[str, Any] | None, title: str, colours) -> None:
    """Validation AUPRC per epoch, one line per seed, baseline dotted."""
    axis.set_title(title)
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation average precision (AUPRC)")
    axis.grid(alpha=0.3)
    if metrics is None:
        axis.text(0.5, 0.5, "metrics file not found", ha="center", va="center",
                  transform=axis.transAxes)
        return

    for i, run in enumerate(metrics.get("per_seed", [])):
        colour = colours[i % len(colours)]
        history = run.get("history", [])
        epochs = [record["epoch"] for record in history]
        axis.plot(
            epochs, [record["val_auprc"] for record in history],
            color=colour, label=f"seed {run['seed']}",
        )
        axis.scatter(
            [run["best_epoch"]],
            [run["validation"]["auprc"]],
            color=colour, zorder=5, s=40,
        )
    baseline = metrics["aggregate"]["random_baseline"]["mean"]
    axis.axhline(
        baseline, color="black", linewidth=1.0, linestyle=":",
        label=f"random baseline {baseline:.4f}",
    )
    axis.legend(fontsize=7)


def _headline_panel(axis, phase2: dict[str, Any]) -> None:
    """Test AUPRC per model, mean +/- sd, with the baseline drawn on."""
    axis.set_title("Test AUPRC (mean ± sd over seeds)")
    axis.set_ylabel("average precision")
    axis.grid(alpha=0.3, axis="y")

    models = [m for m in (MODEL_A_NAME, MODEL_B_NAME) if phase2["headline"].get(m)]
    means = [phase2["headline"][m]["auprc"]["mean"] for m in models]
    errors = [phase2["headline"][m]["auprc"]["std"] for m in models]
    baseline = phase2["headline"][models[0]]["random_baseline"]["mean"]

    positions = np.arange(len(models))
    palette = ("C0", "C1")
    bars = axis.bar(
        positions, means, yerr=errors, capsize=6, width=0.5,
        color=[palette[i % len(palette)] for i in range(len(models))],
    )
    for bar, model in zip(bars, models):
        bar.set_hatch(_MODEL_HATCH[model])
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{_MODEL_LABEL[m]}\n{phase2['headline'][m]['n_parameters']:,} params"
            for m in models
        ],
        fontsize=8,
    )
    axis.axhline(
        baseline, color="black", linewidth=1.0, linestyle=":",
        label=f"random baseline {baseline:.5f}",
    )
    for position, mean, error in zip(positions, means, errors):
        axis.text(
            position, mean + error, f"{mean:.3f}", ha="center", va="bottom",
            fontsize=9,
        )
    n_rows = phase2["per_seed"][0]["n_rows"] if phase2.get("per_seed") else 0
    axis.set_xlabel(f"n = {n_rows:,} test rows per seed")
    axis.legend(fontsize=7)


def _scenario_panel(axis, phase2: dict[str, Any]) -> None:
    """Pooled per-scenario recall, as counts. n stated on every bar."""
    axis.set_title("Recall by attack scenario (test, pooled across seeds)")
    axis.set_ylabel("recall at each checkpoint's own threshold")
    axis.grid(alpha=0.3, axis="y")

    scenarios = tuple(str(scenario) for scenario in SCENARIOS)
    labels = {scenario: scenario_label(scenario) for scenario in scenarios}
    width = 0.38
    positions = np.arange(len(scenarios))

    for offset, model in enumerate((MODEL_A_NAME, MODEL_B_NAME)):
        values, annotations = scenario_series(phase2, model, scenarios)
        bars = axis.bar(
            positions + (offset - 0.5) * width, values, width,
            label=_MODEL_LABEL[model], hatch=_MODEL_HATCH[model],
        )
        for bar, text in zip(bars, annotations):
            axis.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                text, ha="center", va="bottom", fontsize=7,
            )

    axis.set_xticks(positions)
    axis.set_xticklabels([labels[s] for s in scenarios], fontsize=8)
    axis.set_xlabel("counts above each bar: attack days flagged / attack days")
    axis.set_ylim(0, 1.08)
    axis.legend(fontsize=7)


def scenario_series(
    phase2: dict[str, Any], model: str, scenarios: Sequence[str]
) -> tuple[list[float], list[str]]:
    """One model's pooled recall per scenario, plus the ``hits/total`` labels.

    Separated from the drawing so the figure's *data* can be tested without
    matplotlib and without a display: the bug this replaced was in the tuple of
    scenarios, not in any call to ``bar()``.

    A scenario absent from ``per_scenario_pooled`` plots as a zero-height bar
    annotated ``0/0``, which reads correctly as "nothing to detect here" rather
    than disappearing from the axis.
    """
    pooled = phase2["per_scenario_pooled"].get(model, {})
    values, annotations = [], []
    for scenario in scenarios:
        entry = pooled.get(scenario, {})
        total = entry.get("attack_days_summed", 0)
        hits = entry.get("hits_summed", 0)
        values.append(hits / total if total else 0.0)
        annotations.append(f"{hits}/{total}")
    return values, annotations


def scenario_label(scenario: str) -> str:
    """The x-axis label for one scenario: its id, its shape, its day range.

    Built from ``config.SCENARIO_PROFILES``, which is measured per release.
    This used to be a hand-typed dict, and it was right -- about r4.2, which is
    the only release that existed when it was written. Drawing r5.2 with it
    would have labelled a 33-to-161-day escalation as a 45-to-59-day drift and
    left scenario 4 off the chart entirely.

    A scenario the release profile does not describe still gets a bar and a
    number; only the prose is missing.
    """
    profile = SCENARIO_PROFILES.get(int(scenario))
    if profile is None:
        return str(scenario)
    span = (
        f"{profile.min_days}"
        if profile.min_days == profile.max_days
        else f"{profile.min_days}–{profile.max_days}"
    )
    return f"{scenario} — {profile.kind}\n({span} days)"


def latency(
    phase2: dict[str, Any], *, path: Path | None = None
) -> Path | None:
    """Deliverable 4: recall by position within the attack window, both models.

    The figure that decides how §3 of the report is worded. Flat bars mean
    onset detection; bars climbing left to right mean a persistence detector.
    """
    plt = _pyplot()
    if plt is None:
        return None

    buckets = [
        b for b in phase2["detection_latency"]["position_buckets"]
        if b["scenario"] is None
    ]
    if not buckets:
        logger.warning("no position buckets to plot")
        return None

    path = path or FIGURE_DIR / "stage7_detection_latency.png"
    figure, axis = plt.subplots(figsize=(8.5, 5))
    width = 0.38
    positions = np.arange(len(_BUCKET_ORDER))

    total_days = 0
    for offset, model in enumerate((MODEL_A_NAME, MODEL_B_NAME)):
        values, annotations = [], []
        for label in _BUCKET_ORDER:
            entry = next(
                (b for b in buckets if b["model"] == model and b["bucket"] == label),
                None,
            )
            total = entry["attack_days"] if entry else 0
            hits = entry["hits"] if entry else 0
            values.append(hits / total if total else 0.0)
            annotations.append(f"{hits}/{total}")
            if offset == 0:
                total_days += total
        bars = axis.bar(
            positions + (offset - 0.5) * width, values, width,
            label=_MODEL_LABEL[model], hatch=_MODEL_HATCH[model],
        )
        for bar, text in zip(bars, annotations):
            axis.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                text, ha="center", va="bottom", fontsize=7,
            )

    axis.set_xticks(positions)
    axis.set_xticklabels(_BUCKET_ORDER)
    axis.set_xlabel(
        f"position of the day within its insider's attack window "
        f"(n = {total_days:,} attack days, pooled across seeds)"
    )
    axis.set_ylabel("recall at each checkpoint's own threshold")
    axis.set_title(
        "Detection latency — flat bars mean onset detection, rising bars mean "
        "a persistence detector",
        fontsize=10,
    )
    axis.set_ylim(0, 1.08)
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure  -> %s", path)
    return path
