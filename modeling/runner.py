from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from config import (
    LEAKAGE_AUPRC,
    PRECISION_AT_K,
    SCENARIOS,
    SEEDS,
    WINDOW_DAYS,
    checkpoint_path,
    ensure_directories,
    figure_path,
    metrics_path,
)
from modeling.data import (
    PreparedData,
    SplitData,
    log_preparation,
    prepare,
)
from modeling.metrics import Evaluation, aggregate
from modeling.models import count_parameters
from modeling.train import (
    InputView,
    TrainingConfig,
    TrainingResult,
    seed_everything,
    train_model,
)

logger = logging.getLogger(__name__)

ModelFactory = Callable[[], nn.Module]
ViewSelector = Callable[[SplitData], InputView]


@dataclass(slots=True)
class SeedRun:
    """One seed's worth of result, kept whole so the report can be rebuilt."""

    seed: int
    result: TrainingResult
    data: PreparedData
    checkpoint: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "checkpoint": str(self.checkpoint),
            "pos_weight": self.result.pos_weight,
            "n_parameters": self.result.n_parameters,
            "device": self.result.device,
            "best_epoch": self.result.best_epoch,
            "epochs_run": len(self.result.history),
            "stopped_early": self.result.stopped_early,
            "collapse_guard_fired": self.result.collapse_fired,
            "seconds_total": self.result.seconds_total,
            "seconds_per_epoch": self.result.seconds_per_epoch,
            "train_rows": self.data.train.n_targets,
            "train_positives": self.data.train.n_positives,
            "val_rows": self.data.val.n_targets,
            "val_positives": self.data.val.n_positives,
            "positives_lost_to_row_mask": self.data.positives_lost_to_mask,
            "rows_masked_out": self.data.rows_masked_out,
            "validation": self.result.evaluation.as_dict(),
            "history": self.result.history_as_dicts(),
        }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    run: TrainingResult,
    data: PreparedData,
    config: TrainingConfig,
    input_kind: str,
) -> None:
    """Weights, threshold and standardiser in one file.

    All three, together, deliberately. A model whose decision threshold lives
    in a notebook and whose input scaling lives in a dataframe is not a
    working system — it is a matrix. Stage 7 must apply *exactly* the
    transform these weights were trained under and *exactly* the threshold
    validation chose, and the only way to guarantee that months later is for
    the file to carry them.
    """
    threshold = run.evaluation.threshold
    torch.save(
        {
            "model_class": type(model).__name__,
            "state_dict": run.best_state,
            "input_kind": input_kind,
            "window_days": WINDOW_DAYS if input_kind == "window" else 1,
            "threshold": threshold.threshold if threshold else 0.5,
            "threshold_selection": threshold.as_dict() if threshold else None,
            "standardiser": data.standardiser.to_dict(),
            "seed": config.seed,
            "device": config.device,
            "pos_weight": run.pos_weight,
            "best_epoch": run.best_epoch,
            "best_val_auprc": run.best_auprc,
            "val_random_baseline": data.val.positive_rate,
            "n_parameters": run.n_parameters,
            "training_config": config.as_dict(),
        },
        path,
    )
    logger.info("checkpoint -> %s", path)


def run_experiment(
    *,
    model_name: str,
    model_factory: ModelFactory,
    view_selector: ViewSelector,
    input_kind: str,
    seeds: Sequence[int] = SEEDS,
    sample: bool = False,
    device: str = "cpu",
    use_pos_weight: bool = True,
    tag: str = "",
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Train one architecture across every seed and write all the outputs.

    ``tag`` keeps a variant run (the ``--no-pos-weight`` ablation) off the
    real result's output paths. So does ``sample``: a 7-second smoke run and
    a real run must never share a filename, because in Stage 1 they did once
    and a dev run silently replaced the 1,000-user table.
    """
    ensure_directories()
    runs: list[SeedRun] = []

    for seed in seeds:
        logger.info("=" * 72)
        logger.info(
            "%s | seed %d | %s data%s", model_name, seed,
            "SAMPLE" if sample else "full",
            "" if use_pos_weight else " | ABLATION: pos_weight disabled",
        )
        logger.info("=" * 72)

        data = prepare(seed, sample=sample)
        log_preparation(data)

        config = TrainingConfig(
            seed=seed,
            device=device,
            use_pos_weight=use_pos_weight,
            **({"max_epochs": max_epochs} if max_epochs else {}),
        )
        seed_everything(seed)
        model = model_factory()
        logger.info(
            "%s: %s trainable parameters",
            type(model).__name__, f"{count_parameters(model):,}",
        )

        result = train_model(
            model,
            view_selector(data.train),
            data.train.labels,
            view_selector(data.val),
            data.val.labels,
            config,
            val_scenarios=data.val.scenarios.numpy(),
            pos_weight=data.pos_weight() if use_pos_weight else None,
        )

        path = checkpoint_path(
            f"{model_name}{'_' + tag if tag else ''}", seed, sample=sample
        )
        save_checkpoint(path, model, result, data, config, input_kind)
        runs.append(SeedRun(seed=seed, result=result, data=data, checkpoint=path))

    summary = summarise(runs, model_name=model_name, sample=sample, tag=tag,
                        device=device, use_pos_weight=use_pos_weight)

    out = metrics_path(model_name, sample=sample, tag=tag)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info("metrics -> %s", out)

    figure = figure_path(model_name, sample=sample, tag=tag)
    if plot_curves(runs, figure, model_name):
        logger.info("figure  -> %s", figure)

    log_summary(summary)
    return summary


def summarise(
    runs: Sequence[SeedRun],
    *,
    model_name: str,
    sample: bool,
    tag: str,
    device: str,
    use_pos_weight: bool,
) -> dict[str, Any]:
    """Per-seed detail plus mean +/- sd across seeds.

    Per-scenario recall is aggregated as *counts summed across seeds* as well
    as a mean rate, because scenario 3 contributes a handful of attack days
    per split and a mean of three percentages computed on 2-4 rows each is not
    a number anyone should quote.
    """
    evaluations = [run.result.evaluation for run in runs]
    per_scenario = aggregate_per_scenario(evaluations)

    return {
        "model": model_name,
        "tag": tag or None,
        "sample": sample,
        "device": device,
        "pos_weight_enabled": use_pos_weight,
        "split": "validation",
        "seeds": [run.seed for run in runs],
        "aggregate": {
            "auprc": aggregate([e.auprc for e in evaluations]),
            "random_baseline": aggregate([e.random_baseline for e in evaluations]),
            "auroc": aggregate([e.auroc for e in evaluations if e.auroc is not None]),
            **{
                f"precision_at_{k}": aggregate(
                    [e.precision_at_k[k] for e in evaluations]
                )
                for k in PRECISION_AT_K
            },
            "mean_predicted_prob": aggregate([e.stats.mean for e in evaluations]),
            "distinct_predicted_values": aggregate(
                [float(e.stats.distinct) for e in evaluations]
            ),
            "threshold": aggregate(
                [e.threshold.threshold for e in evaluations if e.threshold]
            ),
            "seconds_per_epoch": aggregate(
                [run.result.seconds_per_epoch for run in runs]
            ),
        },
        "per_scenario": per_scenario,
        "collapse_guard_fired_any_seed": any(
            run.result.collapse_fired for run in runs
        ),
        "leakage_suspected_any_seed": any(
            e.suspiciously_good for e in evaluations
        ),
        "per_seed": [run.as_dict() for run in runs],
    }


def aggregate_per_scenario(
    evaluations: Sequence[Evaluation],
    scenarios: Sequence[int] = SCENARIOS,
) -> dict[str, Any]:
    """Pool one entry per scenario across every seed's evaluation.

    Split out of :func:`summarise` on 2026-08-21 for two reasons. The first is
    that it was the fourth and worst place in the project that hard-typed
    ``(1, 2, 3)``: ``modeling.metrics`` evaluates each run against
    ``config.SCENARIOS`` and therefore *did* carry scenario 4, and then this
    loop dropped it on the way into the metrics JSON. Nothing downstream could
    tell, because a ``per_scenario`` block with three entries looks complete.
    On r5.2 that silently hid 56% of the release's positive rows.

    The second is that it is now testable in milliseconds. :func:`summarise`
    takes whole ``SeedRun`` objects, so pinning its per-scenario arithmetic
    used to mean building a training run.

    A scenario with no attack days in any seed still gets an entry, with a
    ``pooled_recall`` of ``None``. Absent and zero are different facts and the
    file should not conflate them.
    """
    per_scenario: dict[str, Any] = {}
    for scenario in scenarios:
        hits, total, rates = 0, 0, []
        for evaluation in evaluations:
            for entry in evaluation.per_scenario:
                if entry.scenario == scenario:
                    hits += entry.hits
                    total += entry.total
                    if entry.recall is not None:
                        rates.append(entry.recall)
        per_scenario[str(scenario)] = {
            "hits_summed": hits,
            "attack_days_summed": total,
            "pooled_recall": hits / total if total else None,
            "recall_across_seeds": aggregate(rates),
        }
    return per_scenario


def log_summary(summary: dict[str, Any]) -> None:
    """The block that goes in the report, printed at the end of the run."""
    agg = summary["aggregate"]
    logger.info("=" * 72)
    logger.info(
        "%s | %s | validation over seeds %s",
        summary["model"],
        "SAMPLE (numbers meaningless)" if summary["sample"] else "full data",
        summary["seeds"],
    )
    logger.info(
        "AUPRC        %.4f +/- %.4f    random baseline %.5f",
        agg["auprc"]["mean"], agg["auprc"]["std"],
        agg["random_baseline"]["mean"],
    )
    if agg["auroc"]["n"]:
        logger.info("AUROC        %.4f +/- %.4f", agg["auroc"]["mean"], agg["auroc"]["std"])
    for k in PRECISION_AT_K:
        entry = agg[f"precision_at_{k}"]
        logger.info("P@%-3d        %.3f +/- %.3f", k, entry["mean"], entry["std"])
    for scenario, entry in sorted(summary["per_scenario"].items()):
        pooled = entry["pooled_recall"]
        logger.info(
            "scenario %s   %d of %d attack days across all seeds%s",
            scenario, entry["hits_summed"], entry["attack_days_summed"],
            f"  ({pooled:.1%})" if pooled is not None else "",
        )
    if summary["collapse_guard_fired_any_seed"]:
        logger.warning(
            "the collapse guard fired on at least one seed — see the per-epoch "
            "distinct counts above"
        )
    if summary["leakage_suspected_any_seed"]:
        logger.warning(
            "at least one seed scored AUPRC >= %.2f. Treat that as leakage, "
            "not a result.", LEAKAGE_AUPRC,
        )


def plot_curves(runs: Sequence[SeedRun], path: Path, model_name: str) -> bool:
    """Train/val loss and validation AUPRC per epoch, one line per seed.

    Returns False rather than raising if matplotlib is unavailable — a missing
    figure is not a reason to lose a training run.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping the curves figure")
        return False

    figure, (loss_axis, auprc_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, run in enumerate(runs):
        colour = colours[i % len(colours)]
        epochs = [record.epoch for record in run.result.history]
        loss_axis.plot(
            epochs, [r.train_loss for r in run.result.history],
            color=colour, label=f"seed {run.seed} train",
        )
        loss_axis.plot(
            epochs, [r.val_loss for r in run.result.history],
            color=colour, linestyle="--", label=f"seed {run.seed} val",
        )
        auprc_axis.plot(
            epochs, [r.val_auprc for r in run.result.history],
            color=colour, label=f"seed {run.seed}",
        )
        auprc_axis.scatter(
            [run.result.best_epoch], [run.result.best_auprc],
            color=colour, zorder=5, s=40,
        )
        auprc_axis.axhline(
            run.data.val.positive_rate, color=colour, linewidth=0.8,
            linestyle=":", alpha=0.7,
        )

    loss_axis.set_title("Loss (solid: train, dashed: validation)")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("weighted BCE")
    loss_axis.legend(fontsize=7)
    loss_axis.grid(alpha=0.3)

    auprc_axis.set_title("Validation AUPRC (dot: best epoch, dotted: random)")
    auprc_axis.set_xlabel("epoch")
    auprc_axis.set_ylabel("average precision")
    auprc_axis.legend(fontsize=7)
    auprc_axis.grid(alpha=0.3)

    figure.suptitle(f"{model_name} — training curves")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True
