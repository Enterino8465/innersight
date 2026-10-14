"""The two entry points: their shape, and what they refuse. Brief S1 §8.

Every test here runs in well under a second and none of them opens a dataset.
That is the point of them: the three ways a run can be asked for something
impossible -- an unknown release, an unknown stored model, no ground truth --
should all produce a sentence before any file is read, not a stack trace three
minutes into an extraction.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from config import (
    LOG_NAMES,
    LOG_SPECS,
    MODEL_A_NAME,
    MODEL_B_NAME,
    PROFILES,
    PROJECT_ROOT,
    RELEASE_LOG_COLUMNS,
    UI_ALLOWED_COMMANDS,
    UI_DATASET_CHOICES,
    DatasetNotFoundError,
    UnknownReleaseError,
    child_environment,
)
from pipeline import commands, sandbox, score, store, train

PIPELINE_DIR = Path(__file__).resolve().parent

#: The two modules nothing in this package may reach, and the one name.
#: ``evaluation.run_phase2`` opens the test split and writes the audit marker;
#: ``evaluation.testpass`` is the machinery it uses. ``sweep_threshold`` is a
#: function rather than a module, and the difference matters -- see the
#: docstring of test_no_pipeline_source_names_sweep_threshold.
FORBIDDEN_MODULES = ("evaluation.run_phase2", "evaluation.testpass")
FORBIDDEN_NAME = "sweep_threshold"


def _sources() -> list[Path]:
    return sorted(
        path for path in PIPELINE_DIR.glob("*.py")
        if not path.name.startswith("test_")
    )


def _imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _project_module_path(name: str) -> Path | None:
    candidate = PROJECT_ROOT / Path(*name.split("."))
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    return None


def _transitive_imports(start: list[Path]) -> set[str]:
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


# --------------------------------------------------------------------------
# The commands the UI is allowed to launch cannot reach the test pass
# --------------------------------------------------------------------------


def test_the_two_allowed_commands_cannot_reach_the_test_pass() -> None:
    """The other half of the allowlist argument, and the half that is new.

    Two permitted commands is only a shorter argument than seven if neither of
    the two can reach what the seven could not. This walks the import graph
    from exactly the modules named in ``config.UI_ALLOWED_COMMANDS``.
    """
    entry_points = [
        _project_module_path(module) for module in UI_ALLOWED_COMMANDS
    ]
    assert all(path is not None for path in entry_points), (
        f"an allowlisted command has no module: {UI_ALLOWED_COMMANDS}"
    )
    reachable = _transitive_imports([p for p in entry_points if p is not None])
    offences = [name for name in FORBIDDEN_MODULES if name in reachable]
    assert not offences, (
        f"{list(UI_ALLOWED_COMMANDS)} can reach {offences}. The UI would then "
        f"be one process away from the code that opens a test split."
    )


def test_no_pipeline_source_names_sweep_threshold() -> None:
    """The threshold comes off the checkpoint. Nothing here re-chooses one.

    Checked against the source rather than the import graph, deliberately:
    :mod:`pipeline.inference` must import :mod:`evaluation.scoring`, which
    imports :mod:`modeling.metrics`, which is where ``sweep_threshold`` is
    *defined*. "The graph never reaches it" is not a satisfiable statement for
    any module that scores anything. What is satisfiable is that no line of
    this package refers to it.
    """
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            named = (
                isinstance(node, ast.Name) and node.id == FORBIDDEN_NAME
                or isinstance(node, ast.Attribute) and node.attr == FORBIDDEN_NAME
                or isinstance(node, ast.alias) and node.name == FORBIDDEN_NAME
            )
            assert not named, f"{path.name} refers to {FORBIDDEN_NAME}"


# --------------------------------------------------------------------------
# The shape every entry point in this project has
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module", [score, train])
def test_every_entry_point_has_the_shape(module: object) -> None:
    """Docstring first line, ``build_parser()``, ``main(argv) -> int``.

    The docstring is not decoration: ``build_parser`` passes
    ``__doc__.splitlines()[0]`` as the description, and a module with no
    docstring dies on that line. This project has shipped that bug once.
    """
    assert module.__doc__, f"{module.__name__} has no docstring"
    assert module.__doc__.splitlines()[0].strip()
    parser = module.build_parser()
    assert parser.description == module.__doc__.splitlines()[0]
    assert callable(module.main)


@pytest.mark.parametrize("module", [score, train])
def test_help_exits_cleanly_without_touching_anything(module: object) -> None:
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    assert exit_info.value.code == 0


# --------------------------------------------------------------------------
# §8.6 — bad input is refused before any data is opened
# --------------------------------------------------------------------------


def test_score_refuses_an_unknown_release() -> None:
    assert score.main(["--release", "9.9", "--model", "frozen"]) == 4


def test_train_refuses_an_unknown_release() -> None:
    assert train.main(["--release", "9.9"]) == 4


def test_train_will_not_start_without_a_release() -> None:
    """No default, on purpose: which folder is not a thing to assume."""
    with pytest.raises(SystemExit) as exit_info:
        train.main([])
    assert exit_info.value.code == 2


def test_score_refuses_an_unknown_stored_model() -> None:
    """And says what it does have, because "no" is only half an answer."""
    report = commands.Reporter()
    known, run_id = score._resolve_model(report, "20990101-r42-000000")
    assert not known and run_id is None
    assert report.state == "refused" and report.exit_code == 4
    assert "20990101-r42-000000" in (report.error or "")
    assert "--model frozen" in (report.error or "")


def test_score_accepts_the_frozen_checkpoints_by_name() -> None:
    report = commands.Reporter()
    known, run_id = score._resolve_model(report, score.FROZEN_MODEL)
    assert known and run_id is None
    assert report.state == "running", "asking for the frozen models is not an event"


def test_a_missing_answers_file_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A folder of raw logs has no labels in it, and AUPRC needs labels.

    This was ``test_score_refuses_a_missing_answers_file`` and monkeypatched
    ``score.insiders_csv_path``. Brief R8 moved the check to
    :mod:`pipeline.commands`, because ``train`` needs the same sentence and one
    entry point importing the other is not how they should get it.
    """
    def no_drive() -> Path:
        raise DatasetNotFoundError("answers/ is not on this machine")

    monkeypatch.setattr(commands, "insiders_csv_path", no_drive)
    report = commands.Reporter()
    assert commands.require_answers(report) is None
    assert report.state == "refused" and report.exit_code == 4
    assert "INNERSIGHT_ANSWERS_DIR" in (report.error or "")

    monkeypatch.setattr(commands, "insiders_csv_path", lambda: tmp_path / "gone.csv")
    report = commands.Reporter()
    assert commands.require_answers(report) is None
    assert "does not exist" in (report.error or "")


def test_train_refuses_a_missing_answers_file_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug this replaced: ninety seconds of extraction, then a stack trace.

    ``train`` runs ``labeling.label`` second, so an unplugged drive used to be
    discovered *inside labelling* -- long after extraction had finished, and
    reading like a crash rather than like a drive nobody plugged in.

    The ordering is what is asserted, not the sentence: ``commands.launch`` is
    replaced with something that fails the test if it is ever reached. Exit 4,
    no subprocess, no dataset, milliseconds.
    """
    def no_drive() -> Path:
        raise DatasetNotFoundError("answers/ is not on this machine")

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("a stage was launched before the answers check")

    monkeypatch.setattr(commands, "insiders_csv_path", no_drive)
    monkeypatch.setattr(commands, "launch", never)
    monkeypatch.setattr(
        train.detect,
        "identify_or_refuse",
        lambda report, release: train.detect.Detection(
            folder=Path("/nowhere"),
            headers=(),
            release=release,
            capability="SUPPORTED",
            refusal=None,
        ),
    )
    assert train.main(["--release", "4.2"]) == 4


# --------------------------------------------------------------------------
# The verdict that crosses the process boundary
# --------------------------------------------------------------------------


def test_a_refusal_reports_itself_as_one() -> None:
    """Refused is not failed. The page colours them differently."""
    report = commands.Reporter()
    report.refuse("the drive is not plugged in")
    assert report.envelope == {
        "state": "refused",
        "exit_code": 4,
        "error": "the drive is not plugged in",
        "result": None,
    }


def test_a_guard_failure_keeps_the_result_block() -> None:
    """Exit 3 with a payload: the page must still be able to show what failed."""
    report = commands.Reporter()
    report.result = {"guards": [{"guard": "leakage", "passed": False}]}
    report.fail("guard(s) failed: leakage", exit_code=3)
    envelope = report.envelope
    assert envelope["state"] == "failed" and envelope["exit_code"] == 3
    assert envelope["result"] is not None, (
        "a failed guard is exactly when the page most needs the numbers"
    )


# --------------------------------------------------------------------------
# Brief R10 — the release reaches the stages, and the stages agree on where
#
# Two bugs, one root. `pipeline.train` identified an r5.2 folder and then
# launched its six stages with the environment it had inherited; the web UI's
# process has no release chosen, so every stage ran as r4.2 and
# `extraction.extract` refused the run — "device.csv is not r4.2's device log".
#
# Fixing that exposed the second: with the release finally reaching the
# children, Stages 5 and 6 — which have no path flag and resolve
# `config.DATASET_PATH` for themselves — started looking under the RELEASE
# root inside the sandbox, while Stages 1-4 were still being told to write to
# the sandbox root. Run 20260822-r52-194234 died three minutes in with
# "dataset not found at .../_work/releases/r52/data/dataset.parquet".
#
# Under r4.2 the two directories are the same one, which is why neither bug
# could be seen until r5.2 became runnable from the UI on 2026-08-22.
#
# Naming the release variable is allowed here and nowhere else in this
# package: `test_config.test_only_config_reads_the_release_variable` exempts
# test files, because setting it is exactly what they are for.
# --------------------------------------------------------------------------

RELEASE_VARIABLE = "INNERSIGHT_RELEASE"

#: The six stages, in order, and the four `score` derives with.
TRAIN_STAGES = (
    "extraction.extract", "labeling.label", "baseline.baseline",
    "splitting.split", "modeling.run_model_a", "modeling.run_model_b",
)
SCORE_STAGES = (
    "extraction.extract", "labeling.label", "baseline.baseline",
    "splitting.split",
)

#: One per-seed block in the shape `modeling/runner.py` actually writes, with
#: the numbers nested under "validation" — copied from `ui/test_ui.py`, which
#: records why: an earlier reader looked for them at the top level and stored
#: null for every one.
SEED_BLOCK = {
    "seed": 42,
    "best_epoch": 16,
    "validation": {
        "auprc": 0.12372246379332585,
        "random_baseline": 0.00425,
        "threshold": {"threshold": 0.9582074284553528, "f1": 0.187},
    },
}


@dataclass(frozen=True)
class Launch:
    """One subprocess that would have been started, and how."""

    module: str
    args: list[str]
    env: dict[str, str] | None

    def flag(self, name: str) -> Path:
        """The value of one command-line flag, as a path."""
        return Path(self.args[self.args.index(name) + 1])


def _write_release(folder: Path, release: str) -> Path:
    """A folder that is nothing but five header lines, which is all detection
    reads. The same fixture ``ui/test_ui.py`` uses, and the reason these tests
    need no dataset mounted."""
    folder.mkdir(parents=True, exist_ok=True)
    for log in LOG_NAMES:
        columns = RELEASE_LOG_COLUMNS[release][log]
        (folder / LOG_SPECS[log].filename).write_text(
            ",".join(columns) + "\n1,01/02/2010 06:49:00,AAA0001,PC-1234\n",
            encoding="utf-8",
        )
    return folder


def _record_launches(monkeypatch: pytest.MonkeyPatch) -> list[Launch]:
    """Replace the launcher with something that records and runs nothing.

    The whole point of these tests: "what would the child have been told, and
    where would it have written" is answerable in milliseconds. Reproducing
    either bug for real costs three and a quarter minutes to reach Stage 1's
    answer and about twenty to reach Stage 5's.
    """
    recorded: list[Launch] = []

    def record(
        module: str,
        args: Any = (),
        *,
        cwd: Any = None,
        env: dict[str, str] | None = None,
        on_line: Any = None,
        timeout: int = 0,
    ) -> commands.Completed:
        recorded.append(Launch(module, list(args), env))
        return commands.Completed(module, (module,), 0, 0)

    monkeypatch.setattr(commands, "launch", record)
    return recorded


def _use_a_temporary_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Point the model store at ``tmp_path`` so the REAL sandbox can be built.

    These tests run `sandbox.prepare` itself rather than a stub: it is the
    thing that decides what a child process thinks the project is, and a stub
    of it could not be asked where a child would look.
    """
    monkeypatch.setattr(store, "MODEL_STORE_DIR", tmp_path / "models")
    monkeypatch.setattr(store, "_WRITABLE_ROOTS", (tmp_path,))


def _what_a_child_would_read(work: Path, release: str) -> dict[str, str]:
    """Ask a real child process, in a real sandbox, where it will look.

    A subprocess and not arithmetic, deliberately. Recomputing the rebase in
    the test would only prove that two copies of one expression agree. This
    imports the sandbox's own copy of ``config.py`` with the release set,
    exactly as ``modeling.run_model_a`` does, and asks it. Nothing is read but
    ``config``; it costs about a fifth of a second and opens no dataset.
    """
    body = (
        "import json, config\n"
        "print(json.dumps({\n"
        "  'project_root': str(config.PROJECT_ROOT),\n"
        "  'dataset': str(config.DATASET_PATH),\n"
        "  'splits': str(config.SPLITS_PATH),\n"
        "  'checkpoints': str(config.CHECKPOINT_DIR),\n"
        "  'metrics_a': str(config.metrics_path(config.MODEL_A_NAME)),\n"
        "  'metrics_b': str(config.metrics_path(config.MODEL_B_NAME)),\n"
        "}))\n"
    )
    environment = child_environment(release)
    # Otherwise a PYTHONPATH pointing at the repository would import the
    # repository's config and the answer would be about the wrong project.
    environment.pop("PYTHONPATH", None)
    finished = subprocess.run(
        [sys.executable, "-c", body],
        cwd=str(work), env=environment,
        capture_output=True, text=True, check=False,
    )
    assert finished.returncode == 0, finished.stderr
    answers = json.loads(finished.stdout)
    assert Path(answers["project_root"]) == work.resolve(), (
        "the child imported some other config; this whole probe is then a lie"
    )
    return answers


def _train_launches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    menu: str,
    folder_holds: str,
) -> tuple[list[Launch], Path]:
    """Run `train`'s stage loop with every subprocess replaced by a recorder.

    ``menu`` is what the dropdown asked for; ``folder_holds`` is what the
    folder's headers actually say. Detection is the real thing, and so is the
    sandbox — the two are allowed to disagree, and which of them wins is the
    subject of one of the tests below.
    """
    folder = _write_release(tmp_path / f"folder-r{folder_holds}", folder_holds)
    monkeypatch.delenv("INNERSIGHT_DATASET_FULL", raising=False)
    monkeypatch.setitem(UI_DATASET_CHOICES, menu, (folder,))
    monkeypatch.setattr(
        commands, "require_answers", lambda report: tmp_path / "insiders.csv"
    )
    _use_a_temporary_store(monkeypatch, tmp_path)

    prepared: list[Path] = []
    really_prepare = sandbox.prepare

    def remember(report: commands.Reporter, run_id: str) -> Path:
        work = really_prepare(report, run_id)
        prepared.append(work)
        return work

    class _NothingKept:
        def as_dict(self) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(sandbox, "prepare", remember)
    monkeypatch.setattr(train, "_keep", lambda *a, **k: _NothingKept())

    recorded = _record_launches(monkeypatch)
    assert train.main(["--release", menu, "--seeds", "42"]) == 0
    return recorded, prepared[0]


def _score_launches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, release: str
) -> list[Launch]:
    """The same, for the four stages `score` derives its tables with."""
    monkeypatch.setattr(
        commands, "require_answers", lambda report: tmp_path / "insiders.csv"
    )
    monkeypatch.setattr(store, "UI_RUN_DIR", tmp_path / "runs")
    monkeypatch.setattr(store, "_WRITABLE_ROOTS", (tmp_path / "runs",))
    recorded = _record_launches(monkeypatch)
    report = commands.Reporter()
    assert score._derive_tables(report, tmp_path / "folder", release) is not None
    return recorded


# --------------------------------------------------------------------------
# The child environment is a COPY of the parent's, not a replacement
# --------------------------------------------------------------------------


def test_the_child_environment_keeps_everything_this_one_has() -> None:
    """The mistake this fix invites, asserted rather than hoped for.

    ``subprocess.Popen(env=...)`` REPLACES the environment; it does not merge.
    A helper that returned ``{RELEASE_VARIABLE: "5.2"}`` would launch a child
    with no PATH, no HOME and no virtualenv — and that failure looks nothing
    like the one this fix is for, which is why it is worth a test of its own.
    """
    child = child_environment("5.2")
    assert child[RELEASE_VARIABLE] == "5.2"
    assert "PATH" in child, "the child would have no PATH at all"
    assert child["PATH"] == os.environ["PATH"]
    missing = set(os.environ) - set(child) - {RELEASE_VARIABLE}
    assert not missing, f"the child would lose {sorted(missing)}"


def test_a_child_with_no_release_is_told_nothing_about_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not ``None``, and above all not ``""``.

    ``config._select_profile`` reads a blank value as unset, so an empty string
    would turn a deliberate choice back into the default without saying so.
    """
    monkeypatch.delenv(RELEASE_VARIABLE, raising=False)
    child = child_environment()
    assert RELEASE_VARIABLE not in child
    assert child == dict(os.environ)


def test_a_release_with_no_profile_is_refused_by_the_parent() -> None:
    """Cheaper than six identical tracebacks, and it names what is supported."""
    with pytest.raises(UnknownReleaseError):
        child_environment("9.9")


# --------------------------------------------------------------------------
# Every stage is told the release the HEADERS named
# --------------------------------------------------------------------------


@pytest.mark.parametrize("release", ["4.2", "5.2"])
def test_every_training_stage_is_told_the_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, release: str
) -> None:
    launches, _ = _train_launches(
        monkeypatch, tmp_path, menu=release, folder_holds=release
    )
    assert [launch.module for launch in launches] == list(TRAIN_STAGES)
    for launch in launches:
        assert launch.env is not None, f"{launch.module} got no environment"
        assert launch.env[RELEASE_VARIABLE] == release, launch.module
        assert launch.env["PATH"] == os.environ["PATH"], launch.module


@pytest.mark.parametrize("release", ["4.2", "5.2"])
def test_every_scoring_stage_is_told_the_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, release: str
) -> None:
    launches = _score_launches(monkeypatch, tmp_path, release)
    assert [launch.module for launch in launches] == list(SCORE_STAGES)
    for launch in launches:
        assert launch.env is not None, f"{launch.module} got no environment"
        assert launch.env[RELEASE_VARIABLE] == release, launch.module
        assert launch.env["PATH"] == os.environ["PATH"], launch.module


def test_the_detected_release_beats_the_menus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The menu names a place to look; the headers say what is there.

    A "4.2" choice pointed at an r5.2 folder must run the stages as r5.2 —
    the same rule that already decides what the stored model is called, and
    it matters more here: naming a model wrongly is embarrassing, running
    extraction under the wrong column contract is the bug this fixes.
    """
    launches, work = _train_launches(
        monkeypatch, tmp_path, menu="4.2", folder_holds="5.2"
    )
    assert len(launches) == len(TRAIN_STAGES)
    assert {launch.env[RELEASE_VARIABLE] for launch in launches if launch.env} == {
        "5.2"
    }
    assert sandbox.release_root(work, "5.2") == work / "releases" / "r52", (
        "the stages were told 5.2, so their files must land in r5.2's tree"
    )


def test_a_sample_scoring_run_launches_no_stage_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--sample`` runs no subprocess whatsoever.

    It reads the frozen ``.sample`` tables in this process, so there is no
    child to tell anything to. The assertion worth making is therefore that
    nothing is launched, not that a child was launched without the variable.
    """

    class _Passed:
        passed = True

        def as_dict(self) -> dict[str, Any]:
            return {}

    recorded = _record_launches(monkeypatch)
    monkeypatch.setattr(
        score.inference, "run_inference", lambda **kwargs: _Passed()
    )
    report = commands.Reporter()
    score._infer(report, release="4.2", device="cpu", sample=True, run_id=None)
    assert report.state == "succeeded"
    assert recorded == [], "a sample run launched a subprocess"


# --------------------------------------------------------------------------
# ...and every stage of one run agrees on where the files are
# --------------------------------------------------------------------------


def test_the_rebase_is_arithmetic_and_r42_falls_out_of_it(tmp_path: Path) -> None:
    """r4.2 is a no-op, proved rather than asserted in a comment.

    ``release_root`` has no branch on a release string in it. It does not need
    one: r4.2's release root IS the project root, so ``relative_to`` hands
    back ``Path(".")``, and joining ``Path(".")`` onto a path is the identity.
    Both halves of that are checked here, because the whole claim that the
    r4.2 demo cannot have moved rests on them.
    """
    assert PROFILES["4.2"].release_root.relative_to(PROJECT_ROOT) == Path(".")
    assert tmp_path / Path(".") == tmp_path
    assert sandbox.release_root(tmp_path, "4.2") == tmp_path

    assert PROFILES["5.2"].release_root.relative_to(PROJECT_ROOT) == Path(
        "releases/r52"
    )
    assert sandbox.release_root(tmp_path, "5.2") == tmp_path / "releases" / "r52"


@pytest.mark.parametrize("release", ["4.2", "5.2"])
def test_stage_five_reads_the_file_stage_three_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, release: str
) -> None:
    """The bug that killed 20260822-r52-194234, as a two-second test.

    Stages 1-4 are handed explicit ``--out`` paths. Stages 5 and 6 are handed
    no path at all and resolve ``config.DATASET_PATH`` in their own process.
    Nothing in the code made those two agree; under r4.2 they happened to, and
    under r5.2 they did not. So: record what Stage 3 and Stage 4 were told to
    write, then ask a real child process in the real sandbox what Stage 5
    would open, and require them to be the same file.
    """
    launches, work = _train_launches(
        monkeypatch, tmp_path, menu=release, folder_holds=release
    )
    written = {launch.module: launch for launch in launches}
    child = _what_a_child_would_read(work, release)

    assert written["baseline.baseline"].flag("--out").resolve() == Path(
        child["dataset"]
    ), "Stage 5 would open a dataset.parquet that Stage 3 never wrote"
    assert written["splitting.split"].flag("--out").resolve() == Path(
        child["splits"]
    ), "Stage 5 would open a splits.json that Stage 4 never wrote"

    # And the chain that gets there: each stage reads what the one before it
    # wrote. Cheap to assert now that the paths are in front of us.
    assert (
        written["labeling.label"].flag("--counts")
        == written["extraction.extract"].flag("--out")
    )
    assert (
        written["baseline.baseline"].flag("--counts")
        == written["labeling.label"].flag("--out")
    )
    assert (
        written["splitting.split"].flag("--dataset")
        == written["baseline.baseline"].flag("--out")
    )


@pytest.mark.parametrize("release", ["4.2", "5.2"])
def test_what_is_kept_is_what_the_models_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, release: str
) -> None:
    """The other half of the same mistake, and the one that would have been
    worst to hit: it arrives after both models have finished training.

    The checkpoints and the metrics files are placed where a real child
    process running as this release says it writes them — not where this test
    thinks it does — and then `_keep` has to find them.
    """
    _use_a_temporary_store(monkeypatch, tmp_path)
    report = commands.Reporter()
    run_id = store.new_run_id(release)
    work = sandbox.prepare(report, run_id)
    child = _what_a_child_would_read(work, release)

    Path(child["checkpoints"]).mkdir(parents=True, exist_ok=True)
    for model in (MODEL_A_NAME, MODEL_B_NAME):
        (Path(child["checkpoints"]) / f"{model}_seed42.pt").write_bytes(
            b"not a real checkpoint"
        )
    for key in ("metrics_a", "metrics_b"):
        Path(child[key]).parent.mkdir(parents=True, exist_ok=True)
        Path(child[key]).write_text(
            json.dumps({"per_seed": [SEED_BLOCK]}), encoding="utf-8"
        )

    manifest = train._keep(
        report,
        run_id,
        work,
        train.detect.Detection(
            folder=tmp_path / "folder", headers=(), release=release,
            capability="SUPPORTED", refusal=None,
        ),
        (42,),
        "cpu",
    )
    assert manifest.warnings == [], (
        f"_keep looked in the wrong place: {manifest.warnings}"
    )
    assert sorted(manifest.checkpoints) == ["model_a_seed42", "model_b_seed42"]
    assert len(manifest.validation_metrics) == 2, "one row per model"
    for row in manifest.validation_metrics:
        assert row["best_val_auprc"] is not None
