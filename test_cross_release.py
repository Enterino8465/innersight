"""Scoring one release's data with another release's model — brief R11.

Three things are being asserted here, and only the first is new behaviour.

**One: the guard got stricter, not looser.** ``standardiser_from_checkpoint``
now answers one of two questions. A matched run asks "is this standardiser the
train-only refit?" and passes when it matches. A transfer run asks "is this
standardiser genuinely foreign?" and passes when it does *not*. The inverted
half is the one a hurried implementation skips, so it is tested first and from
both sides: a foreign checkpoint passes in transfer mode, and a checkpoint that
turns out to match FAILS in transfer mode.

**Two: nothing matched moved.** The
``test_a_matched_run_still_prints_the_published_detail`` cases rebuild r4.2's
corpus and compare the guard's detail string, character for character, against
the block already in ``outputs/final_results.json`` — digests, standardiser
statistics and all. That file is the project's published
result; if this test passes, a matched run's guard output is the same output it
was before brief R11.

**Three: a transfer run says the thing a green guard block does not.** Three of
the eighteen features are structurally degenerate on r4.2 and informative on
r5.2, so a model trained on one and scored on the other holds learned weights
for dead inputs. Every transfer run names them, in both directions.

**This file lives at the project root on purpose.** ``pytest evaluation/
modeling/ -q`` is used across this project as a fingerprint of the frozen
science and must keep printing **101**. Tests added here rather than under
``evaluation/`` leave it there, which is the same choice ``test_shape_guard.py``
and ``test_scenario_coverage.py`` already made.

Run it with::

    python -m pytest test_cross_release.py -q
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from config import (
    DATASET_PATH,
    DEGENERATE_FEATURES,
    FEATURE_NAMES,
    N_FEATURES,
    RELEASE_R42,
    RELEASE_R52,
    SEEDS,
    SPLITS_PATH,
)
from evaluation.guards import standardiser_from_checkpoint
from modeling.data import Standardiser
from pipeline import commands, detect, inference, score, store

ROOT = Path(__file__).resolve().parent

#: The published guard block. Read from the file rather than pasted in, so that
#: a regenerated report is what this compares against and the two cannot
#: silently disagree — the same reasoning `inference.load_frozen_test_auprc`
#: gives for reading the AUPRCs rather than hardcoding them.
FINAL_RESULTS = ROOT / "outputs" / "final_results.json"


# --------------------------------------------------------------------------
# Fakes — the guard touches exactly three attributes, so three is what it gets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeCheckpoint:
    """Everything ``standardiser_from_checkpoint`` reads off a checkpoint."""

    path: Path
    standardiser: Standardiser


@dataclass(frozen=True)
class FakeCorpus:
    """A corpus whose refit is whatever the test says it is."""

    refit: Standardiser

    def refit_train_standardiser(self, seed: int) -> Standardiser:
        return self.refit


def _standardiser(offset: float = 0.0) -> Standardiser:
    """A plausible standardiser. A non-zero ``offset`` moves BOTH statistics.

    Both, because a foreign transform differs in both and the matched guard
    reports them separately — an offset that moved only the mean would let a
    half-implemented guard pass this file.
    """
    mean = np.arange(N_FEATURES, dtype=np.float64) / 100.0 + offset
    std = np.linspace(0.5, 1.5, N_FEATURES, dtype=np.float64) * (1.0 + offset)
    return Standardiser(mean=mean, std=std)


def _checkpoints(standardiser: Standardiser, n: int = 2) -> list[FakeCheckpoint]:
    return [
        FakeCheckpoint(Path(f"model_{letter}_seed42.pt"), standardiser)
        for letter in "ab"[:n]
    ]


# --------------------------------------------------------------------------
# The signature — evaluation/test_evaluation.py may not change
# --------------------------------------------------------------------------


def test_the_new_parameter_is_keyword_only_and_defaulted() -> None:
    """Three existing call sites pass three positional arguments and stop.

    ``evaluation/test_evaluation.py`` calls this guard at three places and is
    frozen; ``evaluation/phase1.py`` and ``evaluation/testpass.py`` call it
    once each. All five pass ``(corpus, seed, checkpoints)`` and nothing more,
    so the mode has to be keyword-only with a default or those five break.
    """
    signature = inspect.signature(standardiser_from_checkpoint)
    parameter = signature.parameters["transfer"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False
    positional = [
        name
        for name, p in signature.parameters.items()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == ["corpus", "seed", "checkpoints"]


# --------------------------------------------------------------------------
# The inverted half
# --------------------------------------------------------------------------


def test_the_transfer_guard_fails_when_the_standardiser_matches() -> None:
    """§8.4. The half a hurried implementation skips.

    If an r5.2 model's standardiser did equal a train-only refit on r4.2's
    training users, the two releases would not be two releases, or the wrong
    checkpoint was loaded. Either way the run is not what it says it is, and a
    transfer run that quietly passed here would be unguarded.
    """
    refit = _standardiser()
    result = standardiser_from_checkpoint(
        FakeCorpus(refit), 42, _checkpoints(refit), transfer=True
    )
    assert not result.passed
    assert result.fatal
    assert "MATCHES" in result.detail
    assert "model_a_seed42.pt" in result.detail


def test_the_transfer_guard_passes_on_a_genuinely_foreign_standardiser() -> None:
    """And prints the size of the gap, which is the evidence it is foreign."""
    foreign = _standardiser(offset=0.25)
    result = standardiser_from_checkpoint(
        FakeCorpus(_standardiser()), 42, _checkpoints(foreign), transfer=True
    )
    assert result.passed
    assert "mean differs by up to" in result.detail
    assert "std by up to" in result.detail
    # The refit's own statistics are printed too: §13 asks for the values, and
    # a transfer run needs them as much as a matched one.
    assert "mean[0]" in result.detail


def test_the_two_modes_do_not_share_a_name() -> None:
    """A transfer verdict read later must not look like a matched one."""
    refit = _standardiser()
    foreign = _standardiser(offset=0.25)
    matched = standardiser_from_checkpoint(
        FakeCorpus(refit), 42, _checkpoints(refit)
    )
    transferred = standardiser_from_checkpoint(
        FakeCorpus(refit), 42, _checkpoints(foreign), transfer=True
    )
    assert matched.passed and transferred.passed
    assert matched.name != transferred.name
    assert matched.name == "seed 42: standardiser loaded from checkpoint, not refit"
    assert "foreign" in transferred.name


# --------------------------------------------------------------------------
# The half that already existed, unchanged
# --------------------------------------------------------------------------


def test_the_matched_guard_still_fails_on_a_foreign_standardiser() -> None:
    """§8.5. The behaviour that refused michael's run on 2026-08-22."""
    result = standardiser_from_checkpoint(
        FakeCorpus(_standardiser()), 42, _checkpoints(_standardiser(offset=0.25))
    )
    assert not result.passed
    assert "mean differs by up to" in result.detail
    assert "std differs by up to" in result.detail


@pytest.mark.parametrize("transfer", [False, True])
def test_two_checkpoints_scaled_differently_are_caught_in_both_modes(
    transfer: bool,
) -> None:
    """Two models on different scalings are not comparable, either way round.

    This check is about the pair of checkpoints rather than about the data, so
    the transfer question does not touch it.
    """
    refit = _standardiser()
    a, b = _checkpoints(_standardiser(offset=0.25))
    mismatched = [a, FakeCheckpoint(b.path, _standardiser(offset=0.5))]
    result = standardiser_from_checkpoint(
        FakeCorpus(refit), 42, mismatched, transfer=transfer
    )
    assert not result.passed
    assert "different standardisers" in result.detail


# --------------------------------------------------------------------------
# §7 — a matched run's guard output has not moved a character
# --------------------------------------------------------------------------


def _published_guard(name: str) -> dict[str, Any]:
    payload = json.loads(FINAL_RESULTS.read_text(encoding="utf-8"))
    for guard in payload["test"]["guards"]:
        if guard["guard"] == name:
            return guard
    raise AssertionError(f"no guard called {name!r} in {FINAL_RESULTS}")


@pytest.mark.parametrize("seed", SEEDS)
def test_a_matched_run_still_prints_the_published_detail(seed: int) -> None:
    """§7, as a test rather than a claim.

    Rebuilds r4.2's corpus, refits the standardiser on this seed's training
    users, hands the guard checkpoints carrying exactly that — which is what
    the frozen checkpoints carry, since the published run says they match to
    1e-9 — and compares the detail string character for character against
    ``outputs/final_results.json``. Digests and standardiser statistics
    included.
    """
    if not (DATASET_PATH.is_file() and SPLITS_PATH.is_file()):
        pytest.skip("r4.2's derived tables are not on this machine")

    from evaluation.corpus import Corpus

    corpus = Corpus.load()
    refit = corpus.refit_train_standardiser(seed)
    result = standardiser_from_checkpoint(
        corpus, seed,
        [
            FakeCheckpoint(Path(f"model_a_seed{seed}.pt"), refit),
            FakeCheckpoint(Path(f"model_b_seed{seed}.pt"), refit),
        ],
    )
    published = _published_guard(
        f"seed {seed}: standardiser loaded from checkpoint, not refit"
    )
    assert result.passed is published["passed"]
    assert result.fatal is published["fatal_if_failed"]
    assert result.detail == published["detail"]


# --------------------------------------------------------------------------
# §4 — the degenerate features, named out loud, in both directions
# --------------------------------------------------------------------------


def test_the_degenerate_table_only_names_real_features() -> None:
    """A typo here would print a caveat about a feature that does not exist."""
    for release, features in DEGENERATE_FEATURES.items():
        for name in features:
            assert name in FEATURE_NAMES, f"r{release} names {name!r}"


def test_the_degenerate_table_matches_extractions_own_sanity_checks() -> None:
    """§4 says source these from ``extraction/extract.py``, so prove it.

    Stage 1 runs three release-aware checks over the whole release on every
    extraction, and those checks are where these three names were measured.
    This asserts the table did not drift away from them — extraction is frozen
    and this file is not, so drift would happen here.
    """
    source = (ROOT / "extraction" / "extract.py").read_text(encoding="utf-8")
    checks = source[source.index("def _release_check_lines"):]
    checks = checks[: checks.index("def _check(")]
    for name in DEGENERATE_FEATURES[RELEASE_R42]:
        assert name in checks
    assert DEGENERATE_FEATURES[RELEASE_R42].keys() == {
        "file_to_removable_count", "unique_filenames", "large_attachment_count",
    }
    # Measured, not assumed: r5.2's three checks assert the SEPARATION of the
    # same three features and pass, so nothing is degenerate there.
    assert DEGENERATE_FEATURES[RELEASE_R52] == {}


@pytest.mark.parametrize(
    ("model_release", "data_release"),
    [(RELEASE_R52, RELEASE_R42), (RELEASE_R42, RELEASE_R52)],
)
def test_both_directions_are_labelled_and_name_all_three_features(
    model_release: str, data_release: str,
) -> None:
    """§8.8. The direction has to be right, not merely present."""
    transfer = inference.Transfer(
        model_release=model_release, data_release=data_release
    )
    banner = "\n".join(transfer.banner())

    assert f"trained on CERT r{model_release}" in banner
    assert f"scoring CERT r{data_release}" in banner
    for name in DEGENERATE_FEATURES[RELEASE_R42]:
        assert name in banner
    # Both sides are named on every transfer run, including the side whose
    # answer is "none": stating an empty list is a measurement.
    assert f"degenerate on CERT r{RELEASE_R42}" in banner
    assert "None of the eighteen features" in banner
    assert f"degenerate on CERT r{RELEASE_R52}" in banner


def test_an_unknown_release_says_so_rather_than_claiming_none() -> None:
    """The failure mode worth avoiding is a silent, confident "none"."""
    banner = "\n".join(inference.Transfer("9.9", RELEASE_R42).banner())
    assert "NOT KNOWN" in banner


def test_the_marker_reaches_the_json_the_page_reads() -> None:
    """The page renders `result.transfer.banner`; a matched run has none."""
    transfer = inference.Transfer(RELEASE_R52, RELEASE_R42)
    payload = transfer.as_dict()
    assert payload["model_release"] == RELEASE_R52
    assert payload["data_release"] == RELEASE_R42
    assert payload["banner"] == list(transfer.banner())
    assert json.loads(json.dumps(payload))["banner"][0].startswith("TRANSFER RUN")


def test_a_transfer_run_compares_itself_against_no_published_number() -> None:
    """§3.3. The confident-wrong-number failure this project exists to stop.

    ``reproduces_frozen_results`` is handed an empty frozen mapping whenever
    ``frozen_results`` is None, and reports itself not applicable rather than
    passing on a comparison it did not make.
    """
    result = inference.reproduces_frozen_results([], {})
    assert result.passed
    assert not result.fatal
    assert "not applicable" in result.detail


# --------------------------------------------------------------------------
# The opt-in
# --------------------------------------------------------------------------


def test_the_opt_in_is_inert_on_a_matched_run() -> None:
    """§8.7. A matched run is a matched run whatever was typed."""
    report = commands.Reporter()
    transfer, proceed = score._transfer_or_refuse(
        report, model_release=RELEASE_R42, data_release=RELEASE_R42, allowed=True
    )
    assert transfer is None
    assert proceed
    assert report.state == "running"


def test_a_mismatch_without_the_opt_in_is_refused() -> None:
    report = commands.Reporter()
    transfer, proceed = score._transfer_or_refuse(
        report, model_release=RELEASE_R52, data_release=RELEASE_R42, allowed=False
    )
    assert transfer is None
    assert not proceed
    assert report.state == "refused"
    assert report.exit_code == 4
    assert "--allow-transfer" in (report.error or "")


def test_a_mismatch_with_the_opt_in_becomes_a_labelled_transfer() -> None:
    report = commands.Reporter()
    transfer, proceed = score._transfer_or_refuse(
        report, model_release=RELEASE_R52, data_release=RELEASE_R42, allowed=True
    )
    assert proceed
    assert transfer == inference.Transfer(RELEASE_R52, RELEASE_R42)
    assert report.state == "running"


def test_frozen_means_this_process_release() -> None:
    """The ``--model frozen`` ruling, as a test.

    ``frozen`` loads ``config.CHECKPOINT_DIR``, which the release variable
    already picks, so the frozen checkpoints' release is this process's active
    release and nothing has to be guessed.
    """
    from config import ACTIVE_RELEASE

    assert score._model_release(None) == ACTIVE_RELEASE


def test_a_stored_models_release_comes_off_its_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "20260822-r52-200603"
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path)
    (tmp_path / run_id).mkdir()
    (tmp_path / run_id / store.MANIFEST_NAME).write_text(
        json.dumps(_manifest_payload(run_id, RELEASE_R52)), encoding="utf-8"
    )
    assert score._model_release(run_id) == RELEASE_R52


def _manifest_payload(run_id: str, release: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "release": release,
        "source_folder": "/nowhere",
        "seeds": [42],
        "created": "2026-08-22T20:15:26",
        "device": "cpu",
        "git_commit": None,
        "feature_config": store.current_feature_config(),
    }


# --------------------------------------------------------------------------
# §8.6 — refused BEFORE anything is derived
# --------------------------------------------------------------------------


def test_an_undeclared_mismatch_refuses_before_a_single_table_is_derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of checking the releases where it is checked.

    ``_derive_tables`` is stages 1-4 over the raw release — three minutes on
    r5.2. Discovering the mismatch after it is the bad-failure shape brief R10
    took out of ``pipeline.train``. So this replaces the one function every
    derived table has to go through with one that fails the test if it is
    called at all.
    """
    run_id = "20260822-r52-200603"
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path)
    (tmp_path / run_id).mkdir()
    (tmp_path / run_id / store.MANIFEST_NAME).write_text(
        json.dumps(_manifest_payload(run_id, RELEASE_R52)), encoding="utf-8"
    )

    def _never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a stage was launched after a release mismatch")

    monkeypatch.setattr(commands, "launch", _never)
    monkeypatch.setattr(
        detect, "identify_or_refuse",
        lambda report, release: detect.Detection(
            folder=tmp_path, headers=(), release=RELEASE_R42,
            capability="SUPPORTED", refusal=None,
        ),
    )

    exit_code = score.main(["--release", RELEASE_R42, "--model", run_id])
    assert exit_code == 4


def test_the_same_pair_with_the_opt_in_gets_past_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror of the test above: with the flag, derivation is reached.

    ``_derive_tables`` is stubbed out here rather than run — this asserts where
    the run got to, not what stages 1-4 produce.
    """
    run_id = "20260822-r52-200603"
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path)
    (tmp_path / run_id).mkdir()
    (tmp_path / run_id / store.MANIFEST_NAME).write_text(
        json.dumps(_manifest_payload(run_id, RELEASE_R52)), encoding="utf-8"
    )
    monkeypatch.setattr(
        detect, "identify_or_refuse",
        lambda report, release: detect.Detection(
            folder=tmp_path, headers=(), release=RELEASE_R42,
            capability="SUPPORTED", refusal=None,
        ),
    )

    reached: list[str] = []

    def _stub(report: Any, folder: Path, release: str) -> None:
        reached.append(release)
        report.fail("stopped here on purpose", exit_code=1)
        return None

    monkeypatch.setattr(score, "_derive_tables", _stub)
    exit_code = score.main(
        ["--release", RELEASE_R42, "--model", run_id, "--allow-transfer"]
    )
    assert reached == [RELEASE_R42]
    assert exit_code == 1
