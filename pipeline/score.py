"""Score a dataset release with a stored model, and check the guards.

Use case one, and the whole of it::

    python -m pipeline.score --release 4.2 --model frozen
    python -m pipeline.score --release 4.2 --model 20260821-r42-143022
    python -m pipeline.score --release 4.2 --model 20260822-r52-200603 --allow-transfer
    python -m pipeline.score --sample

It identifies the release from five header lines, derives the daily counts,
labels, z-scores and splits from that folder into ``runs/<run_id>/``, and
scores the test split with the checkpoints it was pointed at. It touches
``data/`` and ``outputs/`` not at all — the derived tables land beside the run,
and :func:`pipeline.store.guard_write` refuses anything else.

**Why the run derives its own tables rather than reading ``data/``.** The
button that calls this says "run on a folder", and a button that ignores the
folder it was given is a button that proves nothing. Deriving them also
produces the comparison in :func:`_compare_splits`: if the freshly built splits
match the frozen ones user for user, then these are the same rows the report was
written from and the reproduction guard means something. If they do not, this
says so and the guard reports itself as not applicable rather than failing for
the wrong reason.

**The third line is a transfer run**, and it is opt-in for one reason: a
model whose release is not the folder's release is scoring people it was not
fitted on, and that is either a deliberate experiment or a mistake. Nothing
here guesses which. A mismatch with no ``--allow-transfer`` is refused --
before any table is derived, so the answer arrives in a second rather than
after three minutes of extraction -- and a mismatch with the flag runs under a
STRICTER standardiser guard, not a quieter one: see
:func:`evaluation.guards.standardiser_from_checkpoint`. On a matched run the
flag does nothing at all.

**No sandbox here, and that is not an oversight.** Scoring writes no
checkpoint and no metrics file, so there is nothing for a per-run project root
to protect. :mod:`pipeline.sandbox` exists for training, which does.

Exit codes are the project's: 0 clean, 3 a guard failed, 4 refused. Every line
it prints is a line the web layer puts on the page unchanged; ``--result-json``
is how the structured verdict gets back to a page that is in another process.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Final

from config import (
    ACTIVE_RELEASE,
    DEVICE_CHOICES,
    DEVICE_PREFERENCE,
    MODEL_A_NAME,
    MODEL_B_NAME,
    RELEASE_ID,
    SPLITS_PATH,
    UI_DATASET_CHOICES,
    child_environment,
)
from pipeline import commands, detect, inference, store
from pipeline.commands import Reporter, configure_logging

logger = logging.getLogger(__name__)

FROZEN_MODEL: Final[str] = "frozen"


def _infer(
    report: Reporter,
    *,
    release: str,
    device: str,
    sample: bool,
    run_id: str | None,
    allow_transfer: bool = False,
) -> None:
    """Identify, derive, score, and check that it reproduces the report."""
    if sample:
        _infer_sample(report, device=device)
        return

    detection = detect.identify_or_refuse(report, release)
    if detection is None:
        return

    data_release = detection.release or RELEASE_ID
    transfer, proceed = _transfer_or_refuse(
        report,
        model_release=_model_release(run_id),
        data_release=data_release,
        allowed=allow_transfer,
    )
    if not proceed:
        return

    derived = _derive_tables(report, detection.folder, data_release)
    if derived is None:
        return
    dataset_file, splits_file = derived

    same_rows, detail = _compare_splits(splits_file)
    report.say("")
    report.say(detail)
    if not same_rows and transfer is None:
        report.say(
            "Expect two guards to fail below. The standardiser guard compares "
            "the checkpoint's transform against a refit on THIS folder's "
            "training rows, and the leakage ceiling assumes r4.2's 0.4% "
            "positive rate. Both are statements about the frozen release; on "
            "a different set of people they fail by design. That is the "
            "guards working, not the run breaking."
        )
    elif not same_rows:
        report.say(
            "The standardiser guard below runs in its transfer form: it "
            "passes by proving the checkpoint's transform is NOT a refit on "
            "these rows, and fails if it turns out to be one. The leakage "
            "ceiling still assumes r4.2's 0.4% positive rate, so read it as a "
            "statement about the frozen release rather than about this run."
        )

    report.say("")
    report.say("scoring the test split — nothing is chosen, everything is loaded")
    run = inference.run_inference(
        device=device,
        dataset_file=dataset_file,
        splits_file=splits_file,
        checkpoints_for_seed=_stored_loader(run_id) if run_id else None,
        frozen_results=(
            None
            if not same_rows or run_id or transfer is not None
            else inference.FINAL_RESULTS_PATH
        ),
        transfer=transfer,
        on_line=report.say,
    )
    _finish(report, run)


def _infer_sample(report: Reporter, *, device: str) -> None:
    """The 30-user rehearsal: the same code, meaningless numbers, seconds.

    It reads the frozen ``.sample`` tables instead of deriving new ones,
    because the point of this mode is to prove the wiring end to end quickly,
    not to exercise extraction a second time.
    """
    report.say("SAMPLE MODE — 30 users at roughly a 30% insider rate.")
    report.say("These numbers prove the code runs. They are not results.")
    run = inference.run_inference(sample=True, device=device, on_line=report.say)
    _finish(report, run)


def _finish(report: Reporter, run: inference.InferenceRun) -> None:
    if not run.passed:
        failed = [g.name for g in run.guards if not g.passed and g.fatal]
        report.result = run.as_dict()
        report.fail(f"guard(s) failed: {'; '.join(failed)}", exit_code=3)
        return
    report.succeed(run.as_dict())


def _derive_tables(
    report: Reporter, folder: Path, release: str
) -> tuple[Path, Path] | None:
    """Stages 1-4 over the chosen folder, into ``runs/<run_id>/``.

    None means a stage failed and the run has already been marked. The
    outputs go beside the run and never to ``data/``: a demo that overwrote
    the project's tables would be a demo that could only be given once.
    """
    answers = commands.require_answers(report)
    if answers is None:
        return None
    run_id = store.new_run_id(release)
    where = store.make_dir(store.run_dir(run_id))
    report.say("")
    report.say(f"deriving tables into {where} — data/ is not touched")

    counts = store.guard_write(where / "daily_counts.parquet")
    labelled = store.guard_write(where / "labelled_counts.parquet")
    dataset = store.guard_write(where / "dataset.parquet")
    splits = store.guard_write(where / "splits.json")

    stages = (
        ("extraction.extract", ["--data-dir", folder, "--out", counts]),
        ("labeling.label",
         ["--counts", counts, "--answers", answers, "--out", labelled]),
        ("baseline.baseline", ["--counts", labelled, "--out", dataset]),
        ("splitting.split", ["--dataset", dataset, "--out", splits]),
    )
    stage_env = child_environment(release)
    for module, args in stages:
        report.say("")
        finished = report.run(module, [str(a) for a in args], env=stage_env)
        if not finished.ok:
            report.fail(f"{module} {finished.meaning()}", exit_code=finished.exit_code)
            return None
    return dataset, splits


def _compare_splits(derived: Path, frozen: Path = SPLITS_PATH) -> tuple[bool, str]:
    """Are the freshly built splits the same users as the frozen ones?

    This is what licenses the reproduction claim. Identical user lists mean
    the rows about to be scored are the rows the report was written from, so
    reproducing its AUPRC is a meaningful assertion. Different lists mean a
    different folder — the 30-user sample, say — and the honest thing is to
    say so and not claim a reproduction at all.
    """
    if not frozen.is_file():
        return False, f"no frozen splits at {frozen}; nothing to compare against."
    try:
        left = json.loads(derived.read_text(encoding="utf-8"))
        right = json.loads(frozen.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return False, f"could not compare splits: {error}"

    if left == right:
        return True, (
            f"the splits derived from this folder are identical to {frozen.name} "
            f"— same users, same seeds. These are the rows the report was "
            f"written from, so the reproduction guard applies."
        )
    return False, (
        f"the splits derived from this folder DIFFER from {frozen.name}. That "
        f"is not an error — a different folder produces different users — but "
        f"it means these are not the rows the published numbers came from, so "
        f"nothing is claimed to be reproduced."
    )


def _stored_loader(run_id: str) -> Any:
    """Load both models of a stored run for one seed, like the headline one."""

    def load(seed: int) -> dict[str, Any]:
        return {
            name: store.load_stored(run_id, name, seed)
            for name in (MODEL_A_NAME, MODEL_B_NAME)
        }

    return load


def _model_release(run_id: str | None) -> str:
    """Which CERT release the model about to be used was trained on.

    For a stored model it is recorded: ``pipeline.store.Manifest.release``,
    written when the model was trained.

    **For ``--model frozen`` it is this process's own release**, and that is a
    ruling rather than an accident, so it is written down. The frozen
    checkpoints are ``config.CHECKPOINT_DIR``, which is
    ``outputs/checkpoints/`` under r4.2 and ``releases/r52/outputs/checkpoints/``
    under r5.2 -- the release variable picks the directory, so the checkpoints
    that will be loaded are the active release's by construction and there is
    nothing to guess. The alternative, refusing transfer for ``frozen``
    outright, was rejected: the r4.2-frozen-checkpoints-against-the-r5.2-folder
    run is one a person can already start from the page today, and it would
    then be the ONE cross-release combination handled by a different rule --
    silently computing every number and then failing a guard, instead of
    refusing early or being labelled. One rule, both kinds of model.
    """
    if run_id is None:
        return ACTIVE_RELEASE
    return store.read_manifest(run_id).release


def _transfer_or_refuse(
    report: Reporter,
    *,
    model_release: str,
    data_release: str,
    allowed: bool,
) -> tuple[inference.Transfer | None, bool]:
    """Decide matched or transfer, and refuse an undeclared mismatch.

    Returns ``(transfer, proceed)``. ``(None, True)`` is a matched run and is
    the ordinary case; ``(Transfer(...), True)`` is a declared transfer;
    ``(None, False)`` means the run has been refused and has already said so.

    **The mode is declared, never inferred.** Detecting that the releases
    differ is not permission to switch modes -- a person who picked the wrong
    entry in a dropdown and a person running a deliberate transfer experiment
    produce exactly the same two release strings, and only one of them wants
    numbers. So a mismatch without the opt-in stops here, with exit code 4,
    which is the page's refusal colour and means "the system worked" rather
    than "the run broke".

    **It stops HERE, before ``_derive_tables``.** That is three minutes of
    extraction on r5.2, and a refusal that arrives after it is a refusal
    nobody thanks you for -- the same bad-failure shape brief R10 removed from
    ``pipeline.train``.
    """
    if model_release == data_release:
        if allowed:
            report.say(
                "--allow-transfer was given and is inert: the model and this "
                f"folder are both CERT r{data_release}, so this is an "
                f"ordinary matched run."
            )
        return None, True

    if not allowed:
        report.refuse(
            f"this model was trained on CERT r{model_release} and you have "
            f"pointed it at CERT r{data_release}'s folder. That is a transfer "
            f"run: the standardiser, the threshold and the weights were all "
            f"fitted on a different population, and three of the eighteen "
            f"features do not even mean the same thing across the two "
            f"releases. It may well be worth measuring, but it is not a "
            f"result for either release and this command will not do it by "
            f"accident. Re-run with --allow-transfer to ask for it "
            f"deliberately; every number will then be labelled as a transfer "
            f"measurement. Nothing has been derived or scored."
        )
        return None, False

    transfer = inference.Transfer(
        model_release=model_release, data_release=data_release
    )
    report.say("")
    report.say(transfer.headline())
    return transfer, True


def _resolve_model(report: Reporter, model: str) -> tuple[bool, str | None]:
    """``(True, None)`` for the frozen checkpoints, ``(True, run_id)`` for a
    stored one, ``(False, None)`` when there is no such model and the run has
    already been refused.

    Checked before anything is read, because "there is no model called that"
    is a sentence that should arrive in a tenth of a second rather than after
    ninety seconds of extraction.
    """
    if model == FROZEN_MODEL:
        return True, None
    if (store.store_dir(model) / store.MANIFEST_NAME).is_file():
        return True, model
    known = [manifest.run_id for manifest in store.list_runs()]
    report.refuse(
        f"there is no stored model called {model!r}. "
        + (f"This store holds: {', '.join(known)}." if known else
           f"This store is empty — nothing has been trained through "
           f"{store.MODEL_STORE_DIR} yet.")
        + f" Use --model {FROZEN_MODEL} for the published checkpoints."
    )
    return False, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.score",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--release", default=None, metavar="RELEASE",
        help=f"which release's folder to score: one of "
             f"{', '.join(UI_DATASET_CHOICES)}. The choice names a place to "
             f"look and nothing more — the folder's own headers decide what it "
             f"is, and a disagreement stops the run. Not used with --sample. "
             f"Checked by pipeline.detect.locate, which refuses an unknown "
             f"release with the same sentence the page shows.",
    )
    parser.add_argument(
        "--model", default=FROZEN_MODEL, metavar="{frozen|RUN_ID}",
        help=f"'{FROZEN_MODEL}' scores with the published checkpoints and "
             f"checks the answer against outputs/final_results.json. A run id "
             f"scores with a model this project trained and stored, which has "
             f"no published number to reproduce (default: %(default)s).",
    )
    parser.add_argument(
        "--allow-transfer", action="store_true",
        help="permit a model trained on one release to score another "
             "release's folder. Without it, a mismatch is refused before "
             "anything is derived. With it the run is labelled a transfer "
             "measurement on every line, names the features that are "
             "structurally degenerate on each release, and the standardiser "
             "guard inverts: it then passes only by proving the checkpoint's "
             "transform is foreign to these rows. Inert on a matched run.",
    )
    parser.add_argument(
        "--device", choices=DEVICE_CHOICES, default=DEVICE_PREFERENCE,
        help="auto prefers MPS and falls back to CPU (default: %(default)s).",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="the 30-user rehearsal. Seconds, not minutes, and it reads the "
             "frozen .sample tables rather than deriving anything. Its numbers "
             "are meaningless and the run says so on every line.",
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
    """Score, and return what happened as an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.sample and args.release is None:
        parser.error("--release is required unless --sample is given")
    configure_logging()

    report = Reporter()
    known, run_id = _resolve_model(report, args.model)
    if known:
        try:
            _infer(
                report,
                release=args.release or RELEASE_ID,
                device=args.device,
                sample=args.sample,
                run_id=run_id,
                allow_transfer=args.allow_transfer,
            )
        except Exception as error:
            logger.exception("scoring failed")
            report.fail(f"{type(error).__name__}: {error}")

    if not report.finished:
        report.fail("finished without reporting an outcome")
    if args.result_json is not None:
        store.write_text(args.result_json, json.dumps(report.envelope, indent=2))
    return report.exit_code or 0


if __name__ == "__main__":
    sys.exit(main())
