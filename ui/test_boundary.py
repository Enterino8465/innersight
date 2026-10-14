"""Where the web layer stops. Brief S1 §8, acceptance tests 4 and 5.

These are the tests that make "the frontend is only the UI" a fact about the
code rather than a promise in a docstring, and they are deliberately in a file
of their own: ``test_ui.py`` tests what the UI *does*, and this file tests what
it *cannot* do.

Everything here reads source files. Nothing here imports :mod:`pipeline.score`
or :mod:`pipeline.train` — a test that imported the modules it is asserting are
unreachable would be the one file in ``ui/`` that could reach them.
"""

from __future__ import annotations

import ast
from pathlib import Path

from config import PROJECT_ROOT, UI_ALLOWED_COMMANDS

UI_DIR = Path(__file__).resolve().parent

#: The four modules no module under ui/ may reach, however indirectly. Two of
#: them are the use cases the UI launches; the other two are what those use
#: cases run with. The UI starts a process and reads its output — the moment it
#: can call any of these in its own interpreter, "no logic in the frontend" is
#: back to being something a person has to remember.
FORBIDDEN_FROM_UI = (
    "pipeline.score",
    "pipeline.train",
    "pipeline.inference",
    "pipeline.sandbox",
)

#: Method names that put bytes on a disk. Not exhaustive against a determined
#: author -- nothing short of a sandbox is -- but exhaustive against the way
#: this code has ever actually written a file.
#:
#: ``remove``, ``rename``, ``replace``, ``move`` and ``copy`` are deliberately
#: NOT here: every one of them is also an ordinary method of a list or a dict,
#: and a check that fires on ``self._subscribers.remove(...)`` is a check
#: somebody deletes within a week. What covers their filesystem meanings is
#: test_no_ui_module_imports_a_file_moving_library below -- ``os.remove`` and
#: ``shutil.move`` need an import, and that import is refused.
WRITE_METHODS = frozenset({
    "write_text", "write_bytes", "mkdir", "makedirs", "touch", "unlink",
    "rmdir", "symlink_to", "chmod", "copy2", "copyfile", "copytree",
    "rmtree", "dump", "savefig", "to_parquet", "to_csv", "to_json",
})

#: The one call in ui/ that touches the filesystem, and the reason it is here
#: rather than in the list above. ``config.ensure_directories()`` creates the
#: project's standard empty directories when the server starts; it writes no
#: content and it predates this brief. It is named rather than ignored so that
#: a second such call cannot arrive quietly.
PERMITTED_DIRECTORY_CALL = "ensure_directories"


def _ui_sources() -> list[Path]:
    """Every module of the web layer itself. Test files are not the web layer."""
    return sorted(
        path for path in UI_DIR.glob("*.py") if not path.name.startswith("test_")
    )


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
    """Every first-party module reachable from ``start``, following imports."""
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
# §8.4 — the UI launches the use cases, it does not call them
# --------------------------------------------------------------------------


def test_the_ui_cannot_reach_the_two_use_cases() -> None:
    reachable = _transitive_imports(_ui_sources())
    offences = sorted(name for name in FORBIDDEN_FROM_UI if name in reachable)
    assert not offences, (
        f"ui/ can reach {offences} through its import graph. The two use cases "
        f"are things the UI starts as a process and streams; an import of one "
        f"is a path by which the work could happen inside the web server."
    )


def test_what_the_ui_may_still_read_is_only_the_three_read_only_modules() -> None:
    """The other half of the line: named, so that widening it is a visible act.

    ``pipeline.commands`` is the subprocess launcher ``ui.jobs`` delegates to
    after it has checked the allowlist; the other three back the three views
    that open a file and show what is in it.
    """
    reached = {
        name for name in _transitive_imports(_ui_sources())
        if name.split(".")[0] == "pipeline"
    }
    modules = {name for name in reached if name.count(".") <= 1}
    assert modules <= {
        "pipeline",
        "pipeline.commands",
        "pipeline.detect",
        "pipeline.results",
        "pipeline.store",
    }, f"ui/ reaches {sorted(modules)}, which is more of pipeline/ than agreed"


def test_the_allowlist_holds_exactly_the_two_use_cases() -> None:
    """§8.3. The mechanism is tested in test_ui.py; this is the contents."""
    assert UI_ALLOWED_COMMANDS == ("pipeline.train", "pipeline.score")


# --------------------------------------------------------------------------
# §8.5 — no module under ui/ writes to the filesystem
# --------------------------------------------------------------------------


def _write_calls(path: Path) -> list[str]:
    """Every call in one file that would put something on a disk."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in WRITE_METHODS:
            found.append(f"{path.name}:{node.lineno} .{function.attr}()")
        elif isinstance(function, ast.Name) and function.id == "open":
            modes = [
                argument.value for argument in node.args[1:]
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            ]
            if any(set("wax+") & set(mode) for mode in modes):
                found.append(f"{path.name}:{node.lineno} open(..., {modes})")
    return found


def test_no_ui_module_writes_to_the_filesystem() -> None:
    """Every write in this project belongs to pipeline.store's guard.

    Not "the UI writes only where it is allowed to" -- it does not write at
    all. The guard still exists and still refuses; what changed under brief S1
    is that nothing on this side of the boundary has a reason to call it.
    """
    offences = [call for path in _ui_sources() for call in _write_calls(path)]
    assert offences == [], (
        f"ui/ writes to the filesystem at {offences}. Every write belongs to "
        f"pipeline.store.guard_write, which permits models/ and runs/ and "
        f"refuses everything else."
    )


def test_the_one_directory_call_is_still_the_only_one() -> None:
    """``ensure_directories()`` at server start, and nothing else like it."""
    calls = [
        f"{path.name}:{node.lineno}"
        for path in _ui_sources()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == PERMITTED_DIRECTORY_CALL
    ]
    assert len(calls) == 1, (
        f"{PERMITTED_DIRECTORY_CALL} is called at {calls}. One call, when the "
        f"server starts, is the documented exception to the test above; a "
        f"second one is a decision somebody should have to defend."
    )


def test_no_ui_module_imports_a_file_moving_library() -> None:
    """``shutil`` left ui/ with ``_prepare_sandbox``. It should not come back."""
    imported = {name for path in _ui_sources() for name in _imports_of(path)}
    assert not {"shutil", "os"} & imported, (
        f"ui/ imports {sorted({'shutil', 'os'} & imported)}. Moving and "
        f"copying files is pipeline/'s work, behind the write guard."
    )
