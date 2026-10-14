"""Where the UI is allowed to write, and what it keeps when it trains.

**One rule, enforced in one function.** Every filesystem write this package
makes goes through :func:`guard_write`, and :func:`guard_write` permits exactly
two directories: ``models/`` and ``runs/``. Everything else is refused,
including — named individually so the message is useful — the frozen
checkpoints, the audit marker and the results file.

That is stronger than the brief asked for, and deliberately so. "Never write to
``outputs/checkpoints/``" is a rule about one directory that has to be
remembered every time a new write appears. "The UI writes in two directories"
is a rule that a reader can check by reading one function, and a test can check
by snapshotting the rest of the tree.

**Stored models cannot be mistaken for the frozen ones.** A run's checkpoints
are named ``model_a_<run_id>_seed42.pt``, which does not match the headline
regex in :func:`evaluation.checkpoints.load_checkpoint` and therefore cannot
reach a results table by accident. They are loaded through
:func:`~evaluation.checkpoints.load_variant_checkpoint`, which is the escape
hatch built for exactly this.

**A model whose feature list has changed is refused, not adapted.** The
manifest records ``FEATURE_NAMES``, ``WINDOW_DAYS``, ``Z_CLIP`` and
``STD_FLOOR_MULTIPLIER`` at training time, and :func:`load_stored` compares
them against the current ones. Eighteen slots that mean something different
from what the weights were trained on is the same failure as running r4.2 code
on r5.2: same shape, different meaning, no crash.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from config import (
    ALL_CHECKPOINT_DIRS,
    ALL_FINAL_RESULTS_PATHS,
    ALL_TEST_SPLIT_OPENED_PATHS,
    MODEL_A_NAME,
    FEATURE_NAMES,
    MODEL_B_NAME,
    MODEL_STORE_DIR,
    PROJECT_ROOT,
    STD_FLOOR_MULTIPLIER,
    UI_RUN_DIR,
    WINDOW_DAYS,
    Z_CLIP,
)
from evaluation.checkpoints import Checkpoint, load_variant_checkpoint

logger = logging.getLogger(__name__)


class StoreError(RuntimeError):
    """A write the UI must not make, or a stored model it must not load."""


_WRITABLE_ROOTS: Final[tuple[Path, ...]] = (MODEL_STORE_DIR, UI_RUN_DIR)

_FROZEN_PATHS: Final[dict[Path, str]] = {
    **{
        path: (
            "the audit trail of a test-split opening — r4.2's is stamped "
            "2026-08-21T11:02:25+00:00. The UI does not open test splits and "
            "must not appear in this record"
        )
        for path in ALL_TEST_SPLIT_OPENED_PATHS
    },
    **{
        path: (
            "the frozen result the reproduction guard checks against. A UI "
            "that could rewrite it could make any run agree with itself"
        )
        for path in ALL_FINAL_RESULTS_PATHS
    },
}

MANIFEST_NAME: Final[str] = "manifest.json"


def guard_write(path: Path | str) -> Path:
    """Return ``path`` resolved, or raise if the UI may not write there.

    Call this immediately before every write, and pass its return value to the
    write. Splitting the check from the write — checking one path and writing
    another — is the only way to get this wrong, so there is no variant of this
    function that merely returns a boolean.

    **The whitelist is what does the protecting; the named refusals explain.**
    The last clause below permits ``models/`` and ``runs/`` and refuses
    everything else, so no release's frozen tree is ever writable no matter
    what ``INNERSIGHT_RELEASE`` says. The two clauses above it exist to make
    the message say *which* frozen thing was nearly overwritten, and they are
    built from every release's paths rather than the active one's so that they
    keep saying it when the active release is not the one being written to.
    """
    resolved = Path(path).expanduser().resolve()

    for frozen, why in _FROZEN_PATHS.items():
        if resolved == frozen.resolve():
            raise StoreError(
                f"refusing to write {resolved}: it is {why}."
            )

    for checkpoints in ALL_CHECKPOINT_DIRS:
        if _within(resolved, checkpoints):
            raise StoreError(
                f"refusing to write {resolved}: {checkpoints} holds the frozen "
                f"checkpoints a published result came from. Models the UI "
                f"trains go to {MODEL_STORE_DIR}."
            )

    if not any(_within(resolved, root) for root in _WRITABLE_ROOTS):
        roots = " or ".join(str(root) for root in _WRITABLE_ROOTS)
        raise StoreError(
            f"refusing to write {resolved}: the UI writes inside {roots} and "
            f"nowhere else. Everything outside those two directories is either "
            f"the pipeline's frozen output or somebody's actual files."
        )

    return resolved


def _within(path: Path, root: Path) -> bool:
    return path == root.resolve() or root.resolve() in path.parents


def write_text(path: Path | str, text: str) -> Path:
    """Guarded text write, creating parent directories."""
    target = guard_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    logger.info("wrote %s", target)
    return target


def make_dir(path: Path | str) -> Path:
    """Guarded ``mkdir -p``. The only way this package creates a directory."""
    target = guard_write(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def copy_into_store(source: Path, destination: Path | str) -> Path:
    """Guarded copy. The source may be anywhere; the destination may not."""
    target = guard_write(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    logger.info("copied %s -> %s", source.name, target)
    return target


def new_run_id(release: str, *, now: datetime | None = None) -> str:
    """A run id like ``20260825-r42-143022``: date, release, time.

    ``now`` is an argument so a test can assert the shape without waiting for
    the clock to be interesting.
    """
    stamp = now or datetime.now()
    return (
        f"{stamp:%Y%m%d}-r{release.replace('.', '')}-{stamp:%H%M%S}"
    )


def store_dir(run_id: str) -> Path:
    """Where a trained run's checkpoints and manifest live."""
    return MODEL_STORE_DIR / run_id


def run_dir(run_id: str) -> Path:
    """Where a run's intermediate tables live — never ``data/``."""
    return UI_RUN_DIR / run_id


def stored_checkpoint_path(run_id: str, model: str, seed: int) -> Path:
    """``models/<run_id>/model_a_<run_id>_seed42.pt``.

    The run id sits *between* the model name and the seed on purpose: that is
    what stops the filename matching the headline pattern
    ``^(model_a|model_b)_seed\\d+(\\.sample)?\\.pt$``, so
    :func:`evaluation.checkpoints.load_checkpoint` refuses it and only the
    deliberate variant loader can open it.
    """
    return store_dir(run_id) / f"{model}_{run_id}_seed{seed}.pt"


@dataclass(frozen=True, slots=True)
class Manifest:
    """What a stored model was trained on, and under which meanings.

    ``feature_config`` is the part that is checked rather than merely recorded.
    Everything else is provenance a reader may want; those four values are what
    make the weights mean anything.
    """

    run_id: str
    release: str
    source_folder: str
    seeds: tuple[int, ...]
    created: str
    device: str
    git_commit: str | None
    feature_config: dict[str, Any]
    validation_metrics: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: dict[str, str] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        return payload


def current_feature_config() -> dict[str, Any]:
    """The constants that decide what the eighteen input slots mean."""
    return {
        "FEATURE_NAMES": list(FEATURE_NAMES),
        "WINDOW_DAYS": WINDOW_DAYS,
        "Z_CLIP": Z_CLIP,
        "STD_FLOOR_MULTIPLIER": STD_FLOOR_MULTIPLIER,
    }


def build_manifest(
    *,
    run_id: str,
    release: str,
    source_folder: Path,
    seeds: tuple[int, ...],
    device: str,
    validation_metrics: list[dict[str, Any]] | None = None,
    checkpoints: dict[str, str] | None = None,
    results: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    now: datetime | None = None,
) -> Manifest:
    """Assemble a manifest. Nothing here is measured; it is all recorded."""
    stamp = now or datetime.now()
    return Manifest(
        run_id=run_id,
        release=release,
        source_folder=str(source_folder),
        seeds=tuple(seeds),
        created=stamp.isoformat(timespec="seconds"),
        device=device,
        git_commit=git_commit(),
        feature_config=current_feature_config(),
        validation_metrics=validation_metrics or [],
        checkpoints=checkpoints or {},
        results=results or {},
        warnings=list(warnings or []),
        notes=(
            "Trained by the Stage 8 UI. These weights are NOT the frozen "
            "r4.2 result; the published numbers come from "
            "outputs/checkpoints/ and were produced by the command line."
        ),
    )


def write_manifest(manifest: Manifest) -> Path:
    """Write ``models/<run_id>/manifest.json``."""
    return write_text(
        store_dir(manifest.run_id) / MANIFEST_NAME,
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
    )


def read_manifest(run_id: str) -> Manifest:
    """Read one stored run's manifest, or say which run has none."""
    path = store_dir(run_id) / MANIFEST_NAME
    if not path.is_file():
        raise StoreError(
            f"no manifest at {path}. A stored run without one cannot be "
            f"loaded, because nothing records what its eighteen features meant."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seeds"] = tuple(payload.get("seeds", ()))
    known = {f for f in Manifest.__dataclass_fields__}
    return Manifest(**{k: v for k, v in payload.items() if k in known})


def git_commit() -> str | None:
    """The commit the code was at, or None outside a git checkout."""
    try:
        finished = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return finished.stdout.strip() or None


def read_validation_metrics(sandbox: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The per-seed validation numbers Stages 5 and 6 wrote in the sandbox.

    Returns ``(entries, problems)``. A problem is a metrics file whose shape is
    not what this function expects; it is reported rather than swallowed.

    **The keys are nested, and this function got that wrong once.** The first
    version read ``best_val_auprc`` and ``threshold`` from the top of each
    per-seed block with ``.get()``. They are not there — they are at
    ``validation.auprc`` and ``validation.threshold.threshold`` — so every
    entry recorded ``null``, the page rendered a table of ``n/a``, and nothing
    complained, because a missing key and a genuine null are the same value.
    That is why this reads through :func:`_dig`, which raises with the path it
    was walking, and why ``ui/test_ui.py`` now asserts the fields are non-null
    rather than merely present.
    """
    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    for model in (MODEL_A_NAME, MODEL_B_NAME):
        path = sandbox / "outputs" / f"{model}_metrics.json"
        if not path.is_file():
            problems.append(f"{path.name} was not written")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for block in payload.get("per_seed", []):
            try:
                entries.append({
                    "model": model,
                    "seed": _dig(block, "seed"),
                    "best_val_auprc": _dig(block, "validation", "auprc"),
                    "random_baseline": _dig(block, "validation", "random_baseline"),
                    "threshold": _dig(block, "validation", "threshold", "threshold"),
                    "best_epoch": _dig(block, "best_epoch"),
                })
            except KeyError as error:
                problems.append(f"{path.name} seed {block.get('seed')}: {error}")
    for problem in problems:
        logger.error("validation metrics: %s", problem)
    return entries, problems


def _dig(payload: dict[str, Any], *path: str) -> Any:
    """Walk a nested dict, raising with the full key path if it dead-ends.

    The point is that it raises. ``payload.get("a", {}).get("b")`` returns
    ``None`` for a typo, a renamed key and a genuine null alike, and a manifest
    full of nulls looks like a finished record.
    """
    here: Any = payload
    for index, key in enumerate(path):
        if not isinstance(here, dict) or key not in here:
            raise KeyError(
                f"no {'.'.join(path)} (stopped at {'.'.join(path[:index + 1])})"
            )
        here = here[key]
    if here is None:
        raise KeyError(f"{'.'.join(path)} is null")
    return here


_AGGREGATE_KEYS: Final[tuple[str, ...]] = (
    "auprc", "auroc", "precision_at_10", "precision_at_50",
    "random_baseline", "seconds_per_epoch",
)


def run_metrics_paths(run_id: str) -> dict[str, Path]:
    """Where a finished run's two metrics files are, whichever release it was.

    Stages 5 and 6 write ``outputs/`` *under the release root*, and
    :func:`pipeline.sandbox.release_root` puts that at ``_work`` for r4.2 but
    at ``_work/releases/r52`` for r5.2. Searching for the filename is simpler
    than re-deriving that arithmetic here, and it keeps working if a later
    release moves its root again. A run whose ``_work`` is gone returns ``{}``
    -- an empty answer, not an error, because the checkpoints are still good.
    """
    work = store_dir(run_id) / "_work"
    found: dict[str, Path] = {}
    if not work.is_dir():
        return found
    for model in (MODEL_A_NAME, MODEL_B_NAME):
        hits = sorted(work.glob(f"**/outputs/{model}_metrics.json"))
        if hits:
            found[model] = hits[0]
    return found


def sandbox_release_root(run_id: str) -> Path:
    """The directory inside a run's sandbox that holds its ``outputs/``.

    For r4.2 that is ``_work``; for r5.2 it is ``_work/releases/r52``, because
    :func:`pipeline.sandbox.release_root` moves the release root under whatever
    the sandbox made ``PROJECT_ROOT``. Found by locating a metrics file rather
    than by repeating that arithmetic here, so a later release that moves its
    root again needs no change.

    **This is what :func:`backfill_manifest` got wrong.** It passed
    ``_work`` straight to :func:`read_validation_metrics`, which appends
    ``outputs/``, so backfilling an r5.2 run looked in a directory that only
    exists for r4.2 and reported that Stages 5 and 6 had written nothing. The
    fallback below keeps that error message pointing at a real directory when
    the sandbox genuinely is gone.
    """
    found = run_metrics_paths(run_id)
    if found:
        return next(iter(found.values())).parent.parent
    return store_dir(run_id) / "_work"


def read_run_results(run_id: str) -> tuple[dict[str, Any], list[str]]:
    """Everything Stages 5 and 6 measured about a run, read from its sandbox.

    Returns ``(results, problems)``, keyed by model name. This is the richer
    sibling of :func:`read_validation_metrics`, which keeps four fields per
    model and seed because that was all the first version of the page showed.
    The numbers here were all written by the training run; nothing is computed.

    **Read through :func:`_dig`, not ``.get()``, for the same reason.** The
    per-seed figures are nested under ``validation``, and a ``.get()`` that
    walks one level too few returns ``None`` for a missing key, a renamed key
    and a real null alike -- which is how every entry in this run's manifest
    came out null the first time.
    """
    problems: list[str] = []
    results: dict[str, Any] = {}
    for model, path in run_metrics_paths(run_id).items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            problems.append(f"{path.name} is not readable JSON: {error}")
            continue
        block, trouble = _one_model(model, payload, path.name)
        problems.extend(trouble)
        results[model] = block
    for problem in problems:
        logger.error("run results: %s", problem)
    return results, problems


def _one_model(
    model: str, payload: dict[str, Any], filename: str
) -> tuple[dict[str, Any], list[str]]:
    """One metrics file, reshaped into what one model's tables need."""
    problems: list[str] = []
    source = payload.get("aggregate") or {}
    aggregate = {
        key: {
            "mean": (source.get(key) or {}).get("mean"),
            "std": (source.get(key) or {}).get("std"),
        }
        for key in _AGGREGATE_KEYS
    }

    seeds: list[dict[str, Any]] = []
    for entry in payload.get("per_seed", []):
        try:
            seeds.append({
                "seed": _dig(entry, "seed"),
                "auprc": _dig(entry, "validation", "auprc"),
                "auroc": _dig(entry, "validation", "auroc"),
                "random_baseline": _dig(entry, "validation", "random_baseline"),
                "threshold": _dig(entry, "validation", "threshold", "threshold"),
                "precision": _dig(entry, "validation", "threshold", "precision"),
                "recall": _dig(entry, "validation", "threshold", "recall"),
                "days_flagged": _dig(
                    entry, "validation", "threshold", "predicted_positives"
                ),
                "best_epoch": _dig(entry, "best_epoch"),
                "epochs_run": _dig(entry, "epochs_run"),
                "seconds_total": _dig(entry, "seconds_total"),
                "n_parameters": _dig(entry, "n_parameters"),
                "train_rows": _dig(entry, "train_rows"),
                "val_rows": _dig(entry, "val_rows"),
                "device": entry.get("device"),
            })
        except KeyError as error:
            problems.append(f"{filename} seed {entry.get('seed')}: {error}")

    scenarios = [
        {
            "scenario": int(number),
            "hits": values.get("hits_summed"),
            "attack_days": values.get("attack_days_summed"),
        }
        for number, values in sorted(
            (payload.get("per_scenario") or {}).items(), key=lambda kv: int(kv[0])
        )
    ]

    return {
        "model": model,
        "n_parameters": seeds[0]["n_parameters"] if seeds else None,
        "device": seeds[0]["device"] if seeds else payload.get("device"),
        "seconds_total": sum(s["seconds_total"] for s in seeds) or None,
        "aggregate": aggregate,
        "per_seed": seeds,
        "per_scenario": scenarios,
    }, problems


def backfill_manifest(run_id: str) -> Manifest:
    """Re-read a stored run's sandbox metrics and rewrite its manifest.

    Repair, not recomputation. The metrics files Stages 5 and 6 wrote are still
    in the run's sandbox, so a manifest whose ``validation_metrics`` or
    ``results`` came out wrong -- or, for a run trained before ``results``
    existed, was never written at all -- can be rebuilt from them without
    retraining anything. Every other field is carried across untouched.

    This exists because the first version of the manifest writer read the
    per-seed AUPRC and threshold from the wrong nesting level and recorded
    ``null`` for all of them, silently. Runs made before that fix are repaired
    by ``python -m pipeline.store --backfill <run_id>``.
    """
    manifest = read_manifest(run_id)
    sandbox = sandbox_release_root(run_id)
    if not sandbox.is_dir():
        raise StoreError(
            f"run {run_id} has no _work sandbox at {sandbox}, so its metrics "
            f"files are gone and there is nothing to backfill from. The "
            f"checkpoints are still usable; the validation numbers are not "
            f"recoverable without retraining."
        )

    metrics, problems = read_validation_metrics(sandbox)
    if not metrics:
        raise StoreError(
            f"run {run_id}: no readable validation metrics in {sandbox}. "
            f"{'; '.join(problems) or 'no per-seed blocks found'}"
        )

    results, trouble = read_run_results(run_id)
    repaired = replace(
        manifest,
        validation_metrics=metrics,
        results=results,
        warnings=problems + trouble,
    )
    write_manifest(repaired)
    logger.info(
        "backfilled %s: %d validation entries, %d model result block(s), "
        "%d problem(s)",
        run_id, len(metrics), len(results), len(problems) + len(trouble),
    )
    return repaired


def list_runs() -> tuple[Manifest, ...]:
    """Every stored run that has a readable manifest, newest first."""
    if not MODEL_STORE_DIR.is_dir():
        return ()
    found: list[Manifest] = []
    for candidate in sorted(MODEL_STORE_DIR.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        try:
            found.append(read_manifest(candidate.name))
        except (StoreError, ValueError, TypeError) as error:
            logger.warning("ignoring %s: %s", candidate.name, error)
    return tuple(found)


def load_stored(run_id: str, model: str, seed: int) -> Checkpoint:
    """Load one stored checkpoint, after checking its features still mean this.

    The feature check is the point of the manifest. A stored model trained when
    ``FEATURE_NAMES`` had a different order, or ``Z_CLIP`` a different value,
    has eighteen inputs that look right and mean something else — the same
    silent failure r5.2 would cause, one directory closer to home.
    """
    manifest = read_manifest(run_id)
    verify_feature_config(manifest)
    path = stored_checkpoint_path(run_id, model, seed)
    if not path.is_file():
        raise StoreError(f"run {run_id} has no checkpoint at {path}")
    expected_window = (
        int(manifest.feature_config["WINDOW_DAYS"])
        if model == MODEL_B_NAME
        else None
    )
    return load_variant_checkpoint(
        path, model_name=model, expected_window_days=expected_window
    )


def verify_feature_config(manifest: Manifest) -> None:
    """Refuse a stored model whose feature meanings have since changed."""
    current = current_feature_config()
    differences = [
        f"{key}: stored {manifest.feature_config.get(key)!r}, "
        f"current {current[key]!r}"
        for key in current
        if manifest.feature_config.get(key) != current[key]
    ]
    if differences:
        raise StoreError(
            f"run {manifest.run_id} was trained under different feature "
            f"meanings and will not be loaded: {'; '.join(differences)}. The "
            f"input vector would have the same eighteen slots carrying "
            f"different quantities, which produces a confident wrong number "
            f"rather than an error. Retrain, or check out the commit named in "
            f"the manifest ({manifest.git_commit})."
        )


def format_report(manifest: Manifest) -> str:
    """The block the page shows for one stored run."""
    return "\n".join([
        f"run {manifest.run_id}",
        f"  release          r{manifest.release}",
        f"  trained          {manifest.created} on {manifest.device}",
        f"  source folder    {manifest.source_folder}",
        f"  seeds            {', '.join(str(s) for s in manifest.seeds)}",
        f"  git commit       {manifest.git_commit or 'not a git checkout'}",
        f"  window days      {manifest.feature_config.get('WINDOW_DAYS')}",
    ])


def main(argv: list[str] | None = None) -> int:
    """List stored runs, or repair one's manifest from its sandbox metrics."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pipeline.store",
        description="List the UI's stored models, or backfill one's manifest.",
    )
    parser.add_argument(
        "--backfill", metavar="RUN_ID", default=None,
        help="rewrite this run's manifest from its _work/outputs metrics.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.backfill:
        manifest = backfill_manifest(args.backfill)
        print(format_report(manifest))
        for entry in manifest.validation_metrics:
            print(
                f"  {entry['model']} seed {entry['seed']}: "
                f"val AUPRC {entry['best_val_auprc']:.4f} "
                f"(random {entry['random_baseline']:.5f}), "
                f"threshold {entry['threshold']:.6f}, "
                f"best epoch {entry['best_epoch']}"
            )
        return 0

    runs = list_runs()
    print(f"{len(runs)} stored run(s) in {MODEL_STORE_DIR}")
    for item in runs:
        print(format_report(item))
    return 0


if __name__ == "__main__":
    sys.exit(main())
