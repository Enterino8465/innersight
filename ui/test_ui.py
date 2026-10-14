"""The §10 acceptance tests. Every one of them runs in seconds.

The three structural refusals of §4.1 get one test each, and they are the
tests worth reading first — they are the difference between "we will not open
the test split" and "this code cannot".

Nothing here needs the 40 GB release folders mounted. Release detection is
tested against header-only fixtures written into a temporary directory, which
is the whole of what detection reads anyway.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from config import (
    CHECKPOINT_DIR,
    FINAL_RESULTS_PATH,
    LOG_NAMES,
    LOG_SPECS,
    MODEL_A_NAME,
    MODEL_B_NAME,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RELEASE_CAPABILITY,
    RELEASE_LOG_COLUMNS,
    RELEASE_NOT_YET_ALIGNED,
    RELEASE_SUPPORTED,
    TEST_SPLIT_OPENED_PATH,
)
from pipeline import detect, inference, results, store
from ui import jobs

UI_DIR = Path(__file__).resolve().parent

#: The three names no module under ui/ may reach. The first two are modules;
#: the third is a function, and the difference matters — see
#: test_no_ui_source_names_sweep_threshold.
FORBIDDEN_MODULES = ("evaluation.run_phase2", "evaluation.testpass")
FORBIDDEN_NAME = "sweep_threshold"


# --------------------------------------------------------------------------
# Fixtures — header lines only, which is all detection reads
# --------------------------------------------------------------------------


def _write_release(folder: Path, release: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for log in LOG_NAMES:
        columns = RELEASE_LOG_COLUMNS[release][log]
        (folder / LOG_SPECS[log].filename).write_text(
            ",".join(columns) + "\n1,01/02/2010 06:49:00,AAA0001,PC-1234\n",
            encoding="utf-8",
        )
    return folder


@pytest.fixture
def r42_folder(tmp_path: Path) -> Path:
    return _write_release(tmp_path / "r4.2", "4.2")


@pytest.fixture
def r52_folder(tmp_path: Path) -> Path:
    return _write_release(tmp_path / "r5.2", "5.2")


# --------------------------------------------------------------------------
# §10.2 — forbidden imports
# --------------------------------------------------------------------------


def _ui_modules() -> list[Path]:
    return sorted(p for p in UI_DIR.glob("*.py") if p.name != "test_ui.py")


def _imports_of(path: Path) -> set[str]:
    """Every module name imported by one file, as written."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _project_module_path(name: str) -> Path | None:
    """Where a first-party module lives, or None if it is not one of ours."""
    candidate = PROJECT_ROOT / Path(*name.split("."))
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    return None


def _transitive_imports(start: list[Path]) -> set[str]:
    """Every first-party module reachable from ``start``, following imports.

    Stops at anything that is not a file in this repository, so the walk does
    not wander into site-packages. That is exactly the scope the refusal is
    about: what *this project's* code can reach.
    """
    seen_files = set(start)
    reachable: set[str] = set()
    queue = list(start)
    while queue:
        for name in _imports_of(queue.pop()):
            reachable.add(name)
            path = _project_module_path(name)
            if path is not None and path not in seen_files:
                seen_files.add(path)
                queue.append(path)
    return reachable


def test_ui_cannot_reach_the_test_pass_modules() -> None:
    """No module under ui/ imports run_phase2 or testpass, however indirectly."""
    reachable = _transitive_imports(_ui_modules())
    offences = [name for name in FORBIDDEN_MODULES if name in reachable]
    assert not offences, (
        f"ui/ can reach {offences} through its import graph. The UI must have "
        f"no path to the code that opens a test split."
    )


def test_no_ui_source_names_sweep_threshold() -> None:
    """``sweep_threshold`` appears nowhere in ui/, as a name or an attribute.

    Checked against the source rather than the import graph, and deliberately
    so. ``pipeline.inference`` must import ``evaluation.scoring``, which
    imports ``modeling.metrics``, which is where ``sweep_threshold`` is
    *defined* — so "the graph never reaches sweep_threshold" is not a
    satisfiable statement for any module that scores anything. What is
    satisfiable, and is the thing that matters, is that no line of UI code
    refers to it.
    """
    for path in _ui_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            named = (
                isinstance(node, ast.Name) and node.id == FORBIDDEN_NAME
                or isinstance(node, ast.Attribute) and node.attr == FORBIDDEN_NAME
                or isinstance(node, ast.alias) and node.name == FORBIDDEN_NAME
            )
            assert not named, f"{path.name} refers to {FORBIDDEN_NAME}"


# --------------------------------------------------------------------------
# §10.3 — the allowlist
# --------------------------------------------------------------------------


def test_allowlist_excludes_the_test_pass() -> None:
    from config import UI_ALLOWED_COMMANDS

    assert "evaluation.run_phase2" not in UI_ALLOWED_COMMANDS
    assert "evaluation.run_stage7" not in UI_ALLOWED_COMMANDS


def test_disallowed_command_raises_before_any_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing is spawned. Not a killed process — no process."""

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.Popen was reached for a refused command")

    monkeypatch.setattr(subprocess, "Popen", explode)
    with pytest.raises(jobs.CommandNotAllowed):
        jobs.launch("evaluation.run_phase2", ["--sample"])
    with pytest.raises(jobs.CommandNotAllowed):
        jobs.launch("os", ["-c", "pass"])


# --------------------------------------------------------------------------
# §10.4 — nothing writes to the frozen outputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        CHECKPOINT_DIR / f"{MODEL_A_NAME}_seed42.pt",
        CHECKPOINT_DIR / "anything.pt",
        TEST_SPLIT_OPENED_PATH,
        OUTPUT_DIR / "final_results.json",
        PROJECT_ROOT / "data" / "dataset.parquet",
    ],
)
def test_guard_write_refuses_the_frozen_result(target: Path) -> None:
    with pytest.raises(store.StoreError):
        store.guard_write(target)


def test_guard_write_allows_only_the_store_and_run_dirs() -> None:
    from config import MODEL_STORE_DIR, UI_RUN_DIR

    assert store.guard_write(MODEL_STORE_DIR / "x" / "manifest.json")
    assert store.guard_write(UI_RUN_DIR / "x" / "dataset.parquet")
    with pytest.raises(store.StoreError):
        store.guard_write(Path("/tmp/somewhere-else.json"))


def _tree_state(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / f"{MODEL_A_NAME}_seed42.sample.pt").is_file(),
    reason="sample checkpoints not present",
)
def test_a_whole_inference_run_changes_nothing_under_outputs() -> None:
    """The strong form of §10.4: run it, then diff the tree.

    A guard function can be bypassed by code that does not call it. Comparing
    every file under ``outputs/`` before and after a complete run is the check
    that does not depend on anyone having remembered.
    """
    before = _tree_state(OUTPUT_DIR)
    inference.run_inference(sample=True, device="cpu")
    assert _tree_state(OUTPUT_DIR) == before


# --------------------------------------------------------------------------
# §10.5 — release detection
# --------------------------------------------------------------------------


def test_r42_headers_are_identified_and_allowed(r42_folder: Path) -> None:
    detection = detect.identify(r42_folder)
    assert detection.release == "4.2"
    assert detection.allowed
    assert detection.refusal is None


def test_r52_headers_are_identified_and_allowed(r52_folder: Path) -> None:
    """The mirror of the r4.2 test above, since 2026-08-22.

    This replaced ``test_r52_headers_are_identified_and_stopped``, which was
    the same five header lines with the opposite verdict. The two tests below
    are the other half of that replacement: the refusal *mechanism* it used to
    prove had to move somewhere, because it is a safety property of this
    project and not a fact about r5.2.
    """
    detection = detect.identify(r52_folder)
    assert detection.release == "5.2"
    assert detection.allowed
    assert detection.refusal is None


def test_a_refused_release_is_still_recognised_by_name(
    r52_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inverse of ``test_the_gate_opens``, and why it exists.

    Half of the gate is that a supported release runs. The other half is that
    an unsupported one is *recognised, named, and stopped with a reason a
    human can act on* — not silently skipped and not crashed on. Every release
    this project ships is SUPPORTED today, so without this test that half
    would be proved by nothing at all.

    The capability row is synthetic. That is the point: the mechanism is
    asserted without any release depending on being refused, which is what
    makes both this test and ``test_the_gate_opens`` survive the next flip in
    either direction.
    """
    monkeypatch.setitem(
        RELEASE_CAPABILITY, "5.2", RELEASE_NOT_YET_ALIGNED
    )
    detection = detect.identify(r52_folder)
    assert detection.release == "5.2", "a refused release must still be named"
    assert detection.capability == RELEASE_NOT_YET_ALIGNED
    assert not detection.allowed
    reason = detection.refusal or ""
    assert "email.csv" in reason, "a refusal that names no file is not actionable"
    assert "queued" in reason, "the refusal must read as pending work, not a verdict"


def test_a_release_with_no_written_reason_gets_the_fallback(
    r42_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal still happens when nobody wrote a paragraph for it.

    ``pipeline.detect._NOT_YET_ALIGNED_REASON`` has one entry, for r5.2. r4.2
    has none, so flipping *its* row exercises the generic fallback with no
    fictional release and no second table to monkeypatch. Together with the
    test above this says the gate does not depend on that dict: the bespoke
    paragraph makes a refusal more useful, never more likely.
    """
    monkeypatch.setitem(
        RELEASE_CAPABILITY, "4.2", RELEASE_NOT_YET_ALIGNED
    )
    detection = detect.identify(r42_folder)
    assert detection.release == "4.2"
    assert not detection.allowed
    reason = detection.refusal or ""
    assert RELEASE_NOT_YET_ALIGNED in reason, (
        "the fallback must name the state, so a reader can grep for it"
    )


def test_a_missing_log_is_refused_by_name(r42_folder: Path) -> None:
    (r42_folder / LOG_SPECS["email"].filename).unlink()
    detection = detect.identify(r42_folder)
    assert detection.release is None
    assert "email.csv" in (detection.refusal or "")


def test_an_unrecognised_header_names_what_was_expected(r42_folder: Path) -> None:
    (r42_folder / LOG_SPECS["logon"].filename).write_text(
        "id,date,user,pc,banana\n", encoding="utf-8"
    )
    detection = detect.identify(r42_folder)
    assert detection.release is None
    reason = detection.refusal or ""
    assert "logon.csv" in reason
    assert "activity" in reason, "the expected columns must be shown"


def test_an_empty_folder_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not detect.identify(empty).allowed
    assert not detect.identify(tmp_path / "does-not-exist").allowed


def test_detection_reads_only_the_header(r42_folder: Path, monkeypatch) -> None:
    """A 30 GB body must never be read. Proven by refusing to allow it.

    ``readline`` with a byte cap is the mechanism; this asserts the cap is
    passed, which is what stops a file with no newlines being slurped whole.
    """
    seen: list[object] = []
    real = Path.open

    def spy(self: Path, *args: object, **kwargs: object) -> object:
        handle = real(self, *args, **kwargs)
        if self.suffix == ".csv":
            original = handle.readline

            def capped(*a: object) -> str:
                seen.append(a)
                return original(*a)

            handle.readline = capped  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(Path, "open", spy)
    detect.identify(r42_folder)
    assert seen, "no header was read"
    assert all(a and isinstance(a[0], int) for a in seen), (
        "readline was called without a byte cap"
    )


# --------------------------------------------------------------------------
# §10.6 — the gate opens
# --------------------------------------------------------------------------


def test_the_gate_opens(r52_folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabling r5.2 later is one table entry. This is the evidence.

    If this test ever needs more than the single ``setitem`` below to pass,
    something in ``ui/`` has started branching on the release string and §5
    has quietly become a wall again.
    """
    monkeypatch.setitem(RELEASE_CAPABILITY, "5.2", RELEASE_SUPPORTED)
    detection = detect.identify(r52_folder)
    assert detection.release == "5.2"
    assert detection.allowed, "flipping the capability entry must open the gate"
    assert detection.refusal is None


def test_every_release_with_headers_has_a_capability_row() -> None:
    detect.verify_tables()
    assert set(RELEASE_LOG_COLUMNS) == set(RELEASE_CAPABILITY)


def test_the_r42_header_table_is_the_extractors_own() -> None:
    """The detector and the extractor must not be able to disagree."""
    for log in LOG_NAMES:
        assert RELEASE_LOG_COLUMNS["4.2"][log] == LOG_SPECS[log].columns


def test_r52_email_is_the_column_that_matters() -> None:
    """The measured difference the refusal talks about, asserted as data."""
    assert "activity" in RELEASE_LOG_COLUMNS["5.2"]["email"]
    assert "activity" not in RELEASE_LOG_COLUMNS["4.2"]["email"]


# --------------------------------------------------------------------------
# The folder menu — two entries, and neither of them is evidence
# --------------------------------------------------------------------------


def test_the_menu_offers_exactly_the_two_releases() -> None:
    from config import UI_DATASET_CHOICES

    assert list(UI_DATASET_CHOICES) == ["4.2", "5.2"]
    assert set(UI_DATASET_CHOICES) == set(RELEASE_CAPABILITY), (
        "a menu entry with no capability row would raise instead of refusing"
    )


def test_a_release_resolves_to_the_first_place_that_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import config

    missing, present = tmp_path / "drive", tmp_path / "mirror"
    present.mkdir()
    monkeypatch.setitem(config.UI_DATASET_CHOICES, "4.2", (missing, present))
    assert config.resolve_release_folder("4.2") == present

    monkeypatch.setitem(config.UI_DATASET_CHOICES, "4.2", (missing,))
    assert config.resolve_release_folder("4.2") is None, (
        "an unplugged drive is a thing to grey out, not an exception"
    )


def test_an_unavailable_release_refuses_with_the_places_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import config
    from ui import app

    monkeypatch.setitem(config.UI_DATASET_CHOICES, "5.2", (tmp_path / "nowhere",))
    folder, message = app.locate("5.2")
    assert folder is None
    assert "nowhere" in message
    assert "not on this machine" in message

    assert app.locate("6.4")[0] is None, "an off-menu release must not raise"


def test_the_menu_label_is_not_used_as_the_release(
    r52_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Choose "r4.2" but point it at r5.2 data, and the headers win.

    This is the property that keeps the dropdown honest. It is a shortcut to a
    location; what the folder *is* stays a question answered by reading it.

    Until 2026-08-22 this ended on ``assert not detection.allowed``, which read
    as part of the property and was not: it held because r5.2 happened to be
    refused, not because the menu was ignored. Both releases are SUPPORTED now,
    so the verdict half is asserted where it belongs — against a capability row
    this test sets itself, on the release the *headers* named rather than the
    one the menu did.
    """
    import config

    monkeypatch.setitem(config.UI_DATASET_CHOICES, "4.2", (r52_folder,))
    folder, _ = _app_locate("4.2")
    assert folder == r52_folder

    detection = detect.identify(folder)
    assert detection.release == "5.2", "the menu said 4.2; the headers said 5.2"
    assert detection.allowed

    # The menu still says 4.2, which is SUPPORTED. Refusing anyway is the whole
    # claim: the capability looked up is the detected release's, not the label's.
    monkeypatch.setitem(RELEASE_CAPABILITY, "5.2", RELEASE_NOT_YET_ALIGNED)
    refused = detect.identify(folder)
    assert refused.release == "5.2"
    assert not refused.allowed, "the menu label must not decide the capability"


def _app_locate(release: str) -> tuple[Path | None, str]:
    from ui import app

    return app.locate(release)


# --------------------------------------------------------------------------
# §10.7 — the sample smoke test, end to end
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / f"{MODEL_B_NAME}_seed42.sample.pt").is_file(),
    reason="sample checkpoints not present",
)
def test_sample_inference_runs_and_says_it_is_meaningless() -> None:
    run = inference.run_inference(sample=True, device="cpu")

    assert run.sample and run.as_dict()["meaningless"] is True
    assert len(run.per_seed) == 6, "two models x three seeds"
    assert run.models and all(m.n_parameters > 0 for m in run.models)
    assert "meaningless" in inference.format_report(run)

    reproduction = [g for g in run.guards if "reproduces" in g.name]
    assert len(reproduction) == 1
    assert reproduction[0].passed
    assert "not applicable" in reproduction[0].detail, (
        "a sample run has no published number to reproduce and must say so "
        "rather than compare 30 users against the 1,000-user result"
    )


# --------------------------------------------------------------------------
# The reproduction guard itself, without needing the real data
# --------------------------------------------------------------------------


class _FakeScored:
    """Just enough of :class:`evaluation.scoring.Scored` for the guard."""

    def __init__(self, model: str, seed: int, auprc: float) -> None:
        self.model_name = model
        self.seed = seed
        self.evaluation = type("E", (), {"auprc": auprc})()


def test_reproduction_guard_passes_inside_four_decimals() -> None:
    frozen = {(MODEL_A_NAME, 42): 0.150602}
    guard = inference.reproduces_frozen_results(
        [_FakeScored(MODEL_A_NAME, 42, 0.15060213)], frozen  # type: ignore[list-item]
    )
    assert guard.passed


def test_reproduction_guard_fails_in_the_fourth_decimal() -> None:
    frozen = {(MODEL_A_NAME, 42): 0.150602}
    guard = inference.reproduces_frozen_results(
        [_FakeScored(MODEL_A_NAME, 42, 0.1512)], frozen  # type: ignore[list-item]
    )
    assert not guard.passed
    assert guard.fatal, "a run that does not reproduce the report must stop"
    assert "0.151200" in guard.detail and "0.150602" in guard.detail


def test_frozen_values_are_read_from_the_report_not_hardcoded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "final_results.json"
    path.write_text(
        json.dumps({"test": {"headline": {MODEL_A_NAME: {"per_seed_auprc": {"7": 0.5}}}}}),
        encoding="utf-8",
    )
    assert inference.load_frozen_test_auprc(path) == {(MODEL_A_NAME, 7): 0.5}
    assert inference.load_frozen_test_auprc(tmp_path / "absent.json") == {}


# --------------------------------------------------------------------------
# The model store
# --------------------------------------------------------------------------


def test_stored_checkpoints_cannot_be_loaded_as_headline_ones() -> None:
    from evaluation.checkpoints import CheckpointError, load_checkpoint

    path = store.stored_checkpoint_path("20260825-r42-143022", MODEL_A_NAME, 42)
    assert path.name == "model_a_20260825-r42-143022_seed42.pt"
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_a_manifest_with_different_features_is_refused() -> None:
    manifest = store.build_manifest(
        run_id="x", release="4.2", source_folder=Path("/tmp"),
        seeds=(42,), device="cpu",
    )
    store.verify_feature_config(manifest)  # the current config agrees with itself

    drifted = store.Manifest(
        **(manifest.as_dict() | {
            "seeds": (42,),
            "feature_config": manifest.feature_config | {"Z_CLIP": 3.0},
        })
    )
    with pytest.raises(store.StoreError, match="different feature meanings"):
        store.verify_feature_config(drifted)


# --------------------------------------------------------------------------
# The manifest's validation metrics — the fields that were silently null
# --------------------------------------------------------------------------

#: One per-seed block in the shape Stages 5 and 6 actually write. The values
#: that matter are nested under "validation"; an earlier version of the reader
#: looked for them at the top level and recorded null for every one.
_REAL_SEED_BLOCK = {
    "seed": 42,
    "best_epoch": 16,
    "validation": {
        "auprc": 0.12372246379332585,
        "random_baseline": 0.00425,
        "threshold": {"threshold": 0.9582074284553528, "f1": 0.187},
    },
}


def _metrics_sandbox(tmp_path: Path, block: dict) -> Path:
    sandbox = tmp_path / "_work"
    (sandbox / "outputs").mkdir(parents=True)
    for model in (MODEL_A_NAME, MODEL_B_NAME):
        (sandbox / "outputs" / f"{model}_metrics.json").write_text(
            json.dumps({"per_seed": [block]}), encoding="utf-8"
        )
    return sandbox


def test_validation_metrics_are_read_and_are_not_null(tmp_path: Path) -> None:
    """Every recorded field carries a value. Presence is not the assertion.

    The test this replaces checked that a manifest could be built. It could —
    with `best_val_auprc: null` and `threshold: null` in every row, because
    `.get()` cannot tell a renamed key from a genuine null. Assert the values.
    """
    entries, problems = store.read_validation_metrics(
        _metrics_sandbox(tmp_path, _REAL_SEED_BLOCK)
    )
    assert not problems
    assert len(entries) == 2, "one per model"
    for entry in entries:
        for key, value in entry.items():
            assert value is not None, f"{key} came back null"
    assert entries[0]["best_val_auprc"] == pytest.approx(0.1237224637)
    assert entries[0]["threshold"] == pytest.approx(0.9582074284)
    assert entries[0]["best_epoch"] == 16


def test_the_old_wrong_shape_is_reported_not_recorded_as_null(
    tmp_path: Path,
) -> None:
    """A per-seed block missing the nested keys produces a problem, not a null.

    This is the regression that matters. The failure mode was not a crash — it
    was a table of `n/a` that looked like a finished record.
    """
    flat = {"seed": 42, "best_epoch": 16, "best_val_auprc": 0.12, "threshold": 0.95}
    entries, problems = store.read_validation_metrics(
        _metrics_sandbox(tmp_path, flat)
    )
    assert not entries
    assert len(problems) == 2
    assert "validation.auprc" in problems[0]


def test_a_null_value_is_a_problem_too(tmp_path: Path) -> None:
    """Present-but-null must not pass. `key in payload` is not enough."""
    nulled = json.loads(json.dumps(_REAL_SEED_BLOCK))
    nulled["validation"]["auprc"] = None
    entries, problems = store.read_validation_metrics(
        _metrics_sandbox(tmp_path, nulled)
    )
    assert not entries
    assert "is null" in problems[0]


def test_backfill_repairs_a_manifest_written_with_nulls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run stored before the fix can be repaired without retraining."""
    import config

    monkeypatch.setattr(config, "MODEL_STORE_DIR", tmp_path / "models")
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path / "models")
    monkeypatch.setattr(store, "_WRITABLE_ROOTS", (tmp_path / "models",))

    run_id = "20260825-r42-143022"
    _metrics_sandbox(store.store_dir(run_id), _REAL_SEED_BLOCK)
    broken = store.build_manifest(
        run_id=run_id, release="4.2", source_folder=Path("/tmp"),
        seeds=(42,), device="cpu",
        validation_metrics=[
            {"model": MODEL_A_NAME, "seed": 42, "best_val_auprc": None,
             "threshold": None, "best_epoch": None}
        ],
    )
    store.write_manifest(broken)

    repaired = store.backfill_manifest(run_id)
    assert repaired.validation_metrics
    for entry in repaired.validation_metrics:
        assert entry["best_val_auprc"] is not None
        assert entry["threshold"] is not None
    assert not repaired.warnings
    assert store.read_manifest(run_id).validation_metrics == repaired.validation_metrics


def test_backfill_says_so_when_the_sandbox_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path / "models")
    monkeypatch.setattr(store, "_WRITABLE_ROOTS", (tmp_path / "models",))
    store.write_manifest(
        store.build_manifest(
            run_id="gone", release="4.2", source_folder=Path("/tmp"),
            seeds=(42,), device="cpu",
        )
    )
    with pytest.raises(store.StoreError, match="no _work sandbox"):
        store.backfill_manifest("gone")


def test_run_ids_carry_the_date_and_the_release() -> None:
    from datetime import datetime

    run_id = store.new_run_id("4.2", now=datetime(2026, 8, 25, 14, 30, 22))
    assert run_id == "20260825-r42-143022"


# --------------------------------------------------------------------------
# The trained-models view — reads the record, runs nothing
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not FINAL_RESULTS_PATH.is_file(), reason="final_results.json not present"
)
def test_the_view_reports_what_the_report_says_and_nothing_else() -> None:
    """Every headline number is the one in final_results.json, unmodified."""
    published = json.loads(FINAL_RESULTS_PATH.read_text(encoding="utf-8"))
    frozen = results.frozen_models()

    assert not frozen["problems"]
    assert len(frozen["checkpoints"]) == 6, "two models x three seeds"

    for summary in frozen["test"]["aggregate"]:
        head = published["test"]["headline"][summary["model"]]
        assert summary["auprc"] == head["auprc"]
        assert summary["n_parameters"] == head["n_parameters"]

    rows = {(r["model"], r["seed"]): r["auprc"] for r in frozen["test"]["per_seed"]}
    for model, block in published["test"]["headline"].items():
        for seed, value in block["per_seed_auprc"].items():
            assert rows[(model, int(seed))] == value


@pytest.mark.skipif(
    not FINAL_RESULTS_PATH.is_file(), reason="final_results.json not present"
)
def test_validation_is_aggregated_with_the_projects_own_convention() -> None:
    """The report has no validation headline, so this view aggregates it.

    It must use the population sd (ddof=0) like everything else in the
    project, or the page would print a different spread from the report for
    the same three numbers.
    """
    import numpy as np

    block = results.frozen_models()["validation"]
    for summary in block["aggregate"]:
        assert summary["from"] == "aggregated here from the per-seed rows"
        mine = [
            row["auprc"]
            for row in block["per_seed"]
            if row["model"] == summary["model"]
        ]
        assert summary["auprc"]["std"] == pytest.approx(float(np.std(mine, ddof=0)))


@pytest.mark.skipif(
    not FINAL_RESULTS_PATH.is_file(), reason="final_results.json not present"
)
def test_scenario_three_keeps_its_denominator() -> None:
    """Counts, never a bare percentage. Four positives per split is why."""
    pooled = results.frozen_models()["test"]["per_scenario_pooled"]
    third = [e for e in pooled[MODEL_B_NAME] if e["scenario"] == 3]
    assert len(third) == 1
    assert third[0]["attack_days"] == 12
    assert set(third[0]) == {"scenario", "hits", "attack_days"}, (
        "no precomputed rate: the page must state the denominator"
    )


def test_a_missing_report_is_a_problem_not_an_exception(tmp_path: Path) -> None:
    """A fresh checkout has no final_results.json. Say so; do not 500."""
    frozen = results.frozen_models(results_path=tmp_path / "absent.json")
    assert frozen["test"] is None
    assert frozen["validation"] is None
    assert any("run_stage7" in p for p in frozen["problems"])
    assert frozen["checkpoints"], "checkpoint identity does not depend on the report"


def test_the_view_writes_nothing() -> None:
    """The strong form: build the whole view, then diff the outputs tree."""
    before = _tree_state(OUTPUT_DIR)
    results.everything()
    assert _tree_state(OUTPUT_DIR) == before


@pytest.mark.skipif(
    not FINAL_RESULTS_PATH.is_file(), reason="final_results.json not present"
)
def test_a_run_matching_the_frozen_one_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two tables of identical numbers must explain themselves.

    A stored run that reproduces the frozen training exactly renders beside a
    frozen table showing the same figures. Without a sentence saying they are
    two runs rather than one table twice, it reads as a rendering fault — which
    is exactly how it was first reported.
    """
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path / "models")
    monkeypatch.setattr(store, "_WRITABLE_ROOTS", (tmp_path / "models",))

    published = json.loads(FINAL_RESULTS_PATH.read_text(encoding="utf-8"))
    rows = published["validation"]["validation_metrics"]
    store.write_manifest(store.build_manifest(
        run_id="twin", release="4.2", source_folder=Path("/tmp"),
        seeds=(42, 123, 456), device="mps",
        validation_metrics=[
            {
                "model": row["model"], "seed": row["seed"],
                "best_val_auprc": row["auprc"],
                "random_baseline": row["random_baseline"],
                "threshold": row["threshold_applied"],
                "best_epoch": None,
            }
            for row in rows
        ],
    ))

    verdict = results.trained_models()[0]["reproduces_frozen"]
    assert verdict is not None
    assert verdict.startswith("Reproduces")
    assert "not one table shown twice" in verdict


@pytest.mark.skipif(
    not FINAL_RESULTS_PATH.is_file(), reason="final_results.json not present"
)
def test_a_run_that_differs_says_that_instead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path / "models")
    monkeypatch.setattr(store, "_WRITABLE_ROOTS", (tmp_path / "models",))
    store.write_manifest(store.build_manifest(
        run_id="different", release="4.2", source_folder=Path("/tmp"),
        seeds=(42,), device="cpu",
        validation_metrics=[{
            "model": MODEL_A_NAME, "seed": 42, "best_val_auprc": 0.4,
            "random_baseline": 0.004, "threshold": 0.5, "best_epoch": 1,
        }],
    ))
    verdict = results.trained_models()[0]["reproduces_frozen"]
    assert verdict is not None and verdict.startswith("Differs")


def test_an_unrelated_run_claims_nothing_either_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No overlapping seeds means nothing to compare. Silence, not a verdict."""
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path / "models")
    monkeypatch.setattr(store, "_WRITABLE_ROOTS", (tmp_path / "models",))
    store.write_manifest(store.build_manifest(
        run_id="elsewhere", release="4.2", source_folder=Path("/tmp"),
        seeds=(7,), device="cpu",
        validation_metrics=[{
            "model": MODEL_A_NAME, "seed": 7, "best_val_auprc": 0.4,
            "random_baseline": 0.004, "threshold": 0.5, "best_epoch": 1,
        }],
    ))
    assert results.trained_models()[0]["reproduces_frozen"] is None


def test_a_stale_stored_model_says_so_before_it_is_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Feature drift is shown on the listing, not only at load time."""
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path / "models")
    monkeypatch.setattr(store, "_WRITABLE_ROOTS", (tmp_path / "models",))
    manifest = store.build_manifest(
        run_id="drifted", release="4.2", source_folder=Path("/tmp"),
        seeds=(42,), device="cpu",
    )
    store.write_manifest(
        store.Manifest(**(manifest.as_dict() | {
            "seeds": (42,),
            "feature_config": manifest.feature_config | {"WINDOW_DAYS": 7},
        }))
    )
    listed = results.trained_models()
    assert len(listed) == 1
    assert listed[0]["stale"], "a run whose features drifted must be flagged"
    assert "WINDOW_DAYS" in listed[0]["stale"]


# --------------------------------------------------------------------------
# The page's contract with the server
# --------------------------------------------------------------------------

#: Exactly the field names `ui/static/app.js` reads off an inference result.
#: **Update this and the JavaScript together.** It exists because the page read
#: `model.rows_flagged` while the server sent `rows_flagged_per_seed`, and a
#: missing key in JavaScript is `undefined` rather than an error — so the
#: "Across seeds" section silently vanished and every HTTP-level check still
#: passed. The payload was tested; the page was not.
_PAGE_READS_RUN = {"split", "device", "meaningless", "per_seed", "models", "guards"}
_PAGE_READS_SEED = {
    "model", "seed", "auprc", "auroc", "precision_at_k", "random_baseline",
    "lift_over_random", "n_rows", "n_positives", "n_flagged",
}
_PAGE_READS_MODEL = {
    "model", "seeds", "auprc", "random_baseline", "lift_mean", "context_days",
    "n_parameters", "rows_flagged_per_seed", "per_scenario_pooled",
    "median_first_detection_day", "insiders_detected", "insiders_total",
}
_PAGE_READS_GUARD = {"guard", "passed", "detail"}
_PAGE_READS_SCENARIO = {"scenario", "hits", "attack_days"}


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / f"{MODEL_A_NAME}_seed42.sample.pt").is_file(),
    reason="sample checkpoints not present",
)
def test_every_field_the_page_reads_is_in_the_payload() -> None:
    """The page and the server must agree on names, not merely on shapes."""
    payload = inference.run_inference(sample=True, device="cpu").as_dict()

    assert _PAGE_READS_RUN <= set(payload)
    assert _PAGE_READS_SEED <= set(payload["per_seed"][0])
    assert {"10", "50"} <= set(payload["per_seed"][0]["precision_at_k"])
    assert _PAGE_READS_MODEL <= set(payload["models"][0])
    assert {"mean", "std"} <= set(payload["models"][0]["auprc"])
    assert _PAGE_READS_SCENARIO <= set(payload["models"][0]["per_scenario_pooled"][0])
    assert _PAGE_READS_GUARD <= set(payload["guards"][0])


#: Exactly the field names ``ui/static/app.js`` reads off a trained run.
#: **Update this and the JavaScript together**, for the reason recorded above
#: the inference set: a name the page reads and the server does not send is
#: ``undefined`` in JavaScript, not an error, so the section vanishes and every
#: HTTP-level check still passes.
_TAB_READS_RUN = {
    "run_id", "release", "release_label", "created", "seeds", "device",
    "device_asked_for", "source_folder", "git_commit", "window_days",
    "seconds_total", "models", "scenarios", "measured", "stale", "warnings",
}
_TAB_READS_MODEL = {
    "model", "context_days", "n_parameters", "auprc", "auroc",
    "precision_at_10", "precision_at_50", "random_baseline",
    "lift_over_random", "per_seed",
}
_TAB_READS_SEED = {
    "seed", "auprc", "auroc", "threshold", "precision", "recall",
    "days_flagged", "best_epoch", "epochs_run", "seconds_total",
}
_TAB_READS_SCENARIO = {"scenario", "label", "hits", "attack_days"}


def _a_results_block() -> dict[str, object]:
    """One model's measurements, shaped as :func:`store.read_run_results` emits."""
    return {
        MODEL_A_NAME: {
            "model": MODEL_A_NAME,
            "n_parameters": 641,
            "device": "mps",
            "seconds_total": 20.0,
            "aggregate": {
                "auprc": {"mean": 0.1557, "std": 0.0224},
                "auroc": {"mean": 0.9410, "std": 0.0109},
                "precision_at_10": {"mean": 0.83, "std": 0.1},
                "precision_at_50": {"mean": 0.57, "std": 0.1},
                "random_baseline": {"mean": 0.00429, "std": 0.0},
                "seconds_per_epoch": {"mean": 0.83, "std": 0.0},
            },
            "per_seed": [{
                "seed": 42, "auprc": 0.1240, "auroc": 0.9406,
                "random_baseline": 0.00425, "threshold": 0.9632,
                "precision": 0.127, "recall": 0.373, "days_flagged": 1059,
                "best_epoch": 16, "epochs_run": 23, "seconds_total": 19.9,
                "n_parameters": 641, "train_rows": 264609, "val_rows": 85191,
                "device": "mps",
            }],
            "per_scenario": [{"scenario": 1, "hits": 47, "attack_days": 121}],
        }
    }


def _store_one_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs) -> None:
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path / "models")
    monkeypatch.setattr(store, "_WRITABLE_ROOTS", (tmp_path / "models",))
    store.write_manifest(store.build_manifest(
        run_id=kwargs.pop("run_id", "20260101-r42-000000"),
        release="4.2", source_folder=Path("/tmp/r4.2"), seeds=(42,),
        device="auto", **kwargs,
    ))


def test_the_results_tab_is_empty_until_something_is_trained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tab's defining behaviour: no training here, nothing to show.

    It used to render the command line's own results whether or not anything
    had been trained on this page, which made the tab impossible to read as an
    answer to "what have I trained here".
    """
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path / "models")
    assert results.trained_runs() == []


def test_every_field_the_results_tab_reads_is_in_the_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Names *and* values. A key that is present and null renders as n/a."""
    _store_one_run(tmp_path, monkeypatch, results=_a_results_block())
    run = results.trained_runs()[0]

    assert _TAB_READS_RUN <= set(run)
    assert run["measured"] is True
    assert run["release_label"] == "CERT r4.2 — 1,000 employees"
    assert run["device"] == "mps", "the device that ran, not the one requested"
    assert run["device_asked_for"] == "auto"

    model = run["models"][0]
    assert _TAB_READS_MODEL <= set(model)
    assert {"mean", "std"} <= set(model["auprc"])
    assert model["auprc"]["mean"] == 0.1557
    assert model["context_days"] == 1, "model_a sees one day"
    assert round(model["lift_over_random"]) == 36

    seed = model["per_seed"][0]
    assert _TAB_READS_SEED <= set(seed)
    assert seed["days_flagged"] == 1059

    scenario = run["scenarios"][0]
    assert _TAB_READS_SCENARIO <= set(scenario)
    assert scenario["label"] == "1 — burst, 1–12 days", "labelled from config"
    assert scenario["hits"][MODEL_A_NAME] == 47


def test_a_run_that_recorded_nothing_says_so_rather_than_showing_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run trained before the manifest kept results has none. Say that."""
    _store_one_run(tmp_path, monkeypatch)
    run = results.trained_runs()[0]
    assert run["measured"] is False
    assert run["models"] == []
    assert run["scenarios"] == []
    assert run["seconds_total"] is None


def test_run_metrics_are_found_under_a_release_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r5.2 keeps its outputs at ``_work/releases/r52/outputs``, not ``_work``.

    :func:`store.backfill_manifest` assumed the r4.2 shape and so had never
    worked on an r5.2 run: it reported that Stages 5 and 6 wrote nothing.
    Point :func:`store.run_metrics_paths` back at ``_work / "outputs"`` and
    this test fails, which is the only reason to believe it.
    """
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path / "models")
    outputs = tmp_path / "models" / "r52run" / "_work" / "releases" / "r52" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / f"{MODEL_A_NAME}_metrics.json").write_text("{}", encoding="utf-8")

    found = store.run_metrics_paths("r52run")
    assert set(found) == {MODEL_A_NAME}
    assert store.sandbox_release_root("r52run") == outputs.parent


# --------------------------------------------------------------------------
# The job layer
# --------------------------------------------------------------------------


def test_allowed_command_runs_and_streams_lines() -> None:
    lines: list[str] = []
    finished = jobs.launch("pipeline.score", ["--help"], on_line=lines.append)
    assert finished.ok
    assert finished.meaning() == "completed cleanly"
    assert any("--model" in line for line in lines), "stdout did not reach the caller"


def test_a_task_that_raises_becomes_a_failure_not_a_hang() -> None:
    def work(task: jobs.Task) -> None:
        raise ValueError("deliberate")

    task = jobs.start("boom", work)
    assert task.thread is not None
    task.thread.join(timeout=10)
    assert task.state == "failed"
    assert "deliberate" in (task.error or "")
    assert task.stream.closed, "an unclosed stream leaves the page spinning"


def test_sse_frames_split_multiline_payloads() -> None:
    def work(task: jobs.Task) -> None:
        task.say("one\ntwo")
        task.succeed({})

    task = jobs.start("frames", work)
    assert task.thread is not None
    task.thread.join(timeout=10)
    body = "".join(jobs.sse_frames(task))
    assert "data: one\ndata: two\n" in body
    assert body.endswith("event: done\ndata: succeeded\n\n")

