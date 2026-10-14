"""The write guard must protect BOTH releases, whichever one is active.

Brief R1 §6.1. ``config.FINAL_RESULTS_PATH``, ``config.TEST_SPLIT_OPENED_PATH``
and ``config.CHECKPOINT_DIR`` all follow ``INNERSIGHT_RELEASE``. A refusal list
built from those would name ``releases/r52/outputs/`` when the variable says
5.2, and r4.2's frozen files would drop off it.

**What was actually at risk, stated honestly.** ``guard_write`` ends with a
whitelist: writes are permitted inside ``models/`` and ``runs/`` and refused
everywhere else. Those two do not move with the release, so a write to r4.2's
frozen tree under an active r5.2 was *already* refused, by the catch-all. What
would have been lost is the explanation -- "outside the writable roots" instead
of "that is the audit trail of a test-split opening". A guard that stops saying
what it is protecting is a guard on its way to being deleted by somebody who
cannot see the point of it.

So these tests assert two different things, and the distinction matters:

* the write is refused at all -- which was true before and must stay true;
* the refusal *names the frozen artifact* -- which is what §6.1 fixes.

Everything runs in a subprocess because ``config`` reads the release once, at
import, and ``pipeline.store`` copies the values it needs at its own import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _guard_under(release: str | None, targets: list[str]) -> dict[str, str]:
    """Try to guard each target in a process running as ``release``.

    Returns ``{target: error message}``, with the empty string for a target
    that was permitted -- which for everything this file passes in would be a
    failure.
    """
    body = (
        "import json, sys\n"
        "from pipeline import store\n"
        f"targets = {targets!r}\n"
        "out = {}\n"
        "for t in targets:\n"
        "    try:\n"
        "        store.guard_write(t)\n"
        "        out[t] = ''\n"
        "    except store.StoreError as error:\n"
        "        out[t] = str(error)\n"
        "print(json.dumps(out))\n"
    )
    environment = dict(os.environ)
    environment.pop("INNERSIGHT_RELEASE", None)
    if release is not None:
        environment["INNERSIGHT_RELEASE"] = release
    completed = subprocess.run(
        [sys.executable, "-c", body],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"guard subprocess under INNERSIGHT_RELEASE={release!r} exited "
            f"{completed.returncode}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


_R42_FROZEN = [
    str(ROOT / "outputs" / "checkpoints" / "model_a_seed42.pt"),
    str(ROOT / "outputs" / "checkpoints" / "anything.pt"),
    str(ROOT / "outputs" / "TEST_SPLIT_OPENED.json"),
    str(ROOT / "outputs" / "final_results.json"),
    str(ROOT / "STAGE7_REPORT.md"),
    str(ROOT / "data" / "dataset.parquet"),
]

_R52_FROZEN = [
    str(ROOT / "releases" / "r52" / "outputs" / "checkpoints" / "model_a_seed42.pt"),
    str(ROOT / "releases" / "r52" / "outputs" / "TEST_SPLIT_OPENED.json"),
    str(ROOT / "releases" / "r52" / "outputs" / "final_results.json"),
    str(ROOT / "releases" / "r52" / "data" / "dataset.parquet"),
]


@pytest.mark.parametrize("release", [None, "5.2"])
def test_r42_frozen_paths_are_refused_under_either_release(
    release: str | None,
) -> None:
    results = _guard_under(release, _R42_FROZEN)
    permitted = [target for target, error in results.items() if not error]
    assert permitted == [], (
        f"under INNERSIGHT_RELEASE={release!r} the UI was allowed to write "
        f"r4.2's frozen files: {permitted}"
    )


@pytest.mark.parametrize("release", [None, "5.2"])
def test_r52_frozen_paths_are_refused_under_either_release(
    release: str | None,
) -> None:
    results = _guard_under(release, _R52_FROZEN)
    permitted = [target for target, error in results.items() if not error]
    assert permitted == [], (
        f"under INNERSIGHT_RELEASE={release!r} the UI was allowed to write "
        f"r5.2's tree: {permitted}"
    )


def test_the_refusal_still_names_r42s_audit_trail_under_r52() -> None:
    """The §6.1 fix itself, as distinct from the refusal happening at all."""
    marker = str(ROOT / "outputs" / "TEST_SPLIT_OPENED.json")
    results = _guard_under("5.2", [marker])
    assert "audit trail" in results[marker], (
        "under r5.2 the guard refused r4.2's audit marker but no longer said "
        f"what it was: {results[marker]!r}"
    )


def test_the_refusal_still_names_r42s_published_result_under_r52() -> None:
    results_file = str(ROOT / "outputs" / "final_results.json")
    results = _guard_under("5.2", [results_file])
    assert "reproduction guard" in results[results_file], (
        "under r5.2 the guard refused r4.2's final_results.json but no longer "
        f"said what it was: {results[results_file]!r}"
    )


def test_the_refusal_still_names_r42s_checkpoints_under_r52() -> None:
    checkpoint = str(ROOT / "outputs" / "checkpoints" / "model_a_seed42.pt")
    results = _guard_under("5.2", [checkpoint])
    assert "frozen checkpoints" in results[checkpoint], (
        "under r5.2 the guard refused r4.2's checkpoint directory but no "
        f"longer said what it was: {results[checkpoint]!r}"
    )


@pytest.mark.parametrize("release", [None, "5.2"])
def test_the_writable_roots_do_not_move_with_the_release(
    release: str | None,
) -> None:
    """``models/`` and ``runs/`` are the guard's fixed point.

    If they followed ``INNERSIGHT_RELEASE``, the one function in this project
    that is supposed to be unmovable would move with an environment variable.
    """
    allowed = [
        str(ROOT / "models" / "somerun" / "manifest.json"),
        str(ROOT / "runs" / "somerun" / "dataset.parquet"),
    ]
    results = _guard_under(release, allowed)
    refused = {t: e for t, e in results.items() if e}
    assert refused == {}, (
        f"under INNERSIGHT_RELEASE={release!r} the UI could not write to its "
        f"own scratch directories: {refused}"
    )
