"""Train Model A and Model B on a dataset release, and keep what they produced.

Use case two, and the whole of it::

    python -m pipeline.train --release 4.2
    python -m pipeline.train --release 5.2 --seeds 42 --max-epochs 2

It identifies the release from five header lines, runs extraction, labelling,
baseline and splitting over that folder, then Stages 5 and 6 — all six inside
the sandbox :mod:`pipeline.sandbox` builds — and copies the checkpoints into
``models/<run_id>/`` with a manifest saying what they were trained on.

**Everything happens inside the sandbox, and that is the whole reason it
exists.** Stages 5 and 6 have no output-path flag: ``modeling/runner.py`` calls
``config.checkpoint_path()``, whose directory is unconditionally
``outputs/checkpoints/``. Run them with the repository as their working
directory and a training run started from a web page would overwrite the frozen
headline checkpoints. Run them with a per-run directory holding a copy of
``config.py``, and they cannot see them at all.

The stage module names below are literals in this file, and there are six of
them. Nothing here reads a name from a request, an environment variable or a
config table, so there is no input to this process that could turn into a
seventh command.

Exit codes are the project's: 0 clean, 4 refused, otherwise whatever the stage
that failed returned.
"""



from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from config import (
    DEVICE_CHOICES,
    DEVICE_PREFERENCE,
    MAX_EPOCHS,
    MODEL_A_NAME,
    MODEL_B_NAME,
    RELEASE_ID,
    SEEDS,
    UI_DATASET_CHOICES,
    child_environment,
)
from pipeline import commands, detect, sandbox, store
from pipeline.commands import Reporter, configure_logging

logger = logging.getLogger(__name__)


def _train(
    report: Reporter,
    *,
    release: str,
    device: str,
    seeds: tuple[int, ...],
    max_epochs: int | None,
) -> None:
    detection = detect.identify_or_refuse(report, release) #4.2 / 5.2
    if detection is None:
        return

    if commands.require_answers(report) is None:
        return

    detected = detection.release or RELEASE_ID
    stage_env = child_environment(detected)
    run_id = store.new_run_id(detected)

    work = sandbox.prepare(report, run_id)

    data = store.make_dir(sandbox.release_root(work, detected) / "data")
    stages: tuple[tuple[str, list[Any]], ...] = (
        ("extraction.extract",
         ["--data-dir", detection.folder, "--out", data / "daily_counts.parquet"]),
        ("labeling.label",
         ["--counts", data / "daily_counts.parquet",
          "--out", data / "labelled_counts.parquet"]),
        ("baseline.baseline",
         ["--counts", data / "labelled_counts.parquet",
          "--out", data / "dataset.parquet"]),
        ("splitting.split",
         ["--dataset", data / "dataset.parquet", "--out", data / "splits.json",
          "--seeds", *seeds]),
        ("modeling.run_model_a", _training_args(device, seeds, max_epochs)),
        ("modeling.run_model_b", _training_args(device, seeds, max_epochs)),
    )
    for module, args in stages:
        report.say("")
        finished = report.run(
            module, [str(a) for a in args], cwd=work, env=stage_env
        )
        if not finished.ok:
            report.fail(f"{module} {finished.meaning()}", exit_code=finished.exit_code)
            return

    manifest = _keep(report, run_id, work, detection, seeds, device)
    report.succeed({"run_id": run_id, "manifest": manifest.as_dict()})


def _training_args(
    device: str, seeds: tuple[int, ...], max_epochs: int | None
) -> list[Any]:
    args: list[Any] = ["--device", device, "--seeds", *seeds]
    if max_epochs:
        args += ["--max-epochs", int(max_epochs)]
    return args


def _keep(
    report: Reporter,
    run_id: str,
    work: Path,
    detection: detect.Detection,
    seeds: tuple[int, ...],
    device: str,
) -> store.Manifest:
    """Copy the sandbox's checkpoints into the store and write the manifest.

    The manifest carries everything Stages 5 and 6 measured, not a summary of
    it, because ``_work`` is the largest thing a run leaves behind and is the
    first thing deleted when the disk fills. A run whose sandbox is gone must
    still be able to show its own results.
    """
    inside = sandbox.release_root(work, detection.release or RELEASE_ID)
    kept: dict[str, str] = {}
    for model in (MODEL_A_NAME, MODEL_B_NAME):
        for seed in seeds:
            source = inside / "outputs" / "checkpoints" / f"{model}_seed{seed}.pt"
            target = store.stored_checkpoint_path(run_id, model, seed)
            store.copy_into_store(source, target)
            kept[f"{model}_seed{seed}"] = target.name

    metrics, problems = store.read_validation_metrics(inside)
    results, trouble = store.read_run_results(run_id)
    problems += trouble
    manifest = store.build_manifest(
        run_id=run_id,
        release=detection.release or RELEASE_ID,
        source_folder=detection.folder,
        seeds=seeds,
        device=device,
        validation_metrics=metrics,
        checkpoints=kept,
        results=results,
        warnings=problems,
    )
    store.write_manifest(manifest)
    report.say("")
    report.say(store.format_report(manifest))
    for problem in problems:
        report.say(f"WARNING — validation metrics: {problem}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.train",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--release", required=True, metavar="RELEASE",
        help=f"which release's folder to train on: one of "
             f"{', '.join(UI_DATASET_CHOICES)}. The choice names a place to "
             f"look and nothing more — the folder's own headers decide what it "
             f"is, and a disagreement stops the run before a stage starts. "
             f"Checked by pipeline.detect.locate, which refuses an unknown "
             f"release with the same sentence the page shows.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS), metavar="SEED",
        help="default: %(default)s. All three are reported as mean +/- sd; "
             "one seed is one draw of the split lottery, not a result.",
    )
    parser.add_argument(
        "--device", choices=DEVICE_CHOICES, default=DEVICE_PREFERENCE,
        help="auto prefers MPS and falls back to CPU. Both models must use "
             "the SAME device, or a second variable has entered a two-model "
             "comparison (default: %(default)s).",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help=f"override config.MAX_EPOCHS (currently {MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--result-json", type=Path, default=None, metavar="PATH",
        help="write the structured verdict here as JSON. This is how the web "
             "layer gets a result block out of a process it only streams; a "
             "person running this by hand has no use for it. The path must be "
             "inside models/ or runs/, because store.guard_write says so.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train, and return what happened as an exit code."""
    args = build_parser().parse_args(argv)
    configure_logging()

    report = Reporter()
    try:
        _train(
            report,
            release=args.release,
            device=args.device,
            seeds=tuple(args.seeds),
            max_epochs=args.max_epochs,
        )
    except Exception as error:
        logger.exception("training failed")
        report.fail(f"{type(error).__name__}: {error}")

    if not report.finished:
        report.fail("finished without reporting an outcome")
    if args.result_json is not None:
        store.write_text(args.result_json, json.dumps(report.envelope, indent=2))
    return report.exit_code or 0


if __name__ == "__main__":
    sys.exit(main())
