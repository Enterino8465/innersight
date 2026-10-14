"""A directory that looks like the whole project to the code running inside it.

Moved out of ``ui/app.py`` by brief S1. Per-*run* isolation, which is a
different thing from the per-*release* isolation ``INNERSIGHT_RELEASE`` gives:
that variable moves ``data/`` and ``outputs/`` to ``releases/r52/``, this moves
them under ``models/<run_id>/_work/``. Both are needed and they compose.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Final

from config import PROFILES, PROJECT_ROOT
from pipeline import store
from pipeline.commands import Reporter

_SANDBOX_PACKAGES: Final[tuple[str, ...]] = (
    "extraction", "labeling", "baseline", "splitting", "modeling", "evaluation",
)


def prepare(report: Reporter, run_id: str) -> Path:
    """A directory that looks like the project to the code running inside it.

    ``config.PROJECT_ROOT`` is ``Path(config.__file__).resolve().parent``, and
    that is the only place in the whole repository that derives a path from
    ``__file__``. So a directory holding a *copy* of ``config.py`` and
    *symlinks* to the six stage packages is, from the child process's point of
    view, a complete project whose ``outputs/`` and ``data/`` are its own.

    That is what stops a training run overwriting the frozen checkpoints.
    Stages 5 and 6 have no output-path flag — ``modeling/runner.py`` writes to
    ``config.checkpoint_path()``, which is always
    ``outputs/checkpoints/model_a_seed42.pt`` — and this project may not be
    edited to add one. Moving the root moves every output with it, and touches
    no line of the six stage packages.

    The packages are symlinks and not copies deliberately: the code that runs
    must be the repository's code, not a snapshot of it that could drift.
    """
    sandbox = store.make_dir(store.store_dir(run_id) / "_work")
    shutil.copy2(PROJECT_ROOT / "config.py", store.guard_write(sandbox / "config.py"))
    for package in _SANDBOX_PACKAGES:
        link = sandbox / package
        if not link.exists():
            store.guard_write(link).symlink_to(
                PROJECT_ROOT / package, target_is_directory=True
            )
    report.say(f"sandbox: {sandbox}")
    report.say(
        "the training run's project root is that directory, so its outputs/ "
        "and data/ are its own. The frozen checkpoints are not reachable."
    )
    return sandbox


def release_root(work: Path, release: str) -> Path:
    """Where a stage running as ``release`` inside ``work`` keeps its files.

    The composition rule this module's docstring promises, finally written
    down. The two isolations MULTIPLY rather than sit side by side:
    :func:`prepare` moves ``config.PROJECT_ROOT`` to ``work``, and the release
    profile then moves ``config.RELEASE_ROOT`` to somewhere *under whatever
    ``PROJECT_ROOT`` currently is*. So a stage's ``data/`` and ``outputs/``
    are at this path, not at ``work``, and the difference appears only on a
    release whose root is not the project root.

    For r4.2 that difference is nothing: its release root IS the project root,
    ``relative_to`` gives ``Path(".")``, and this returns ``work`` unchanged.
    Not by a special case -- by arithmetic, which is why
    ``test_pipeline.py`` asserts it rather than claiming it here. For r5.2 it
    returns ``work/releases/r52``.

    **What it cost to learn.** `pipeline.train` wrote Stages 1-4 to
    ``work/data`` through explicit ``--out`` flags, then let Stages 5 and 6 --
    which have no path flag at all, and resolve ``config.DATASET_PATH``
    instead -- find their input for themselves. Under r4.2 those are one
    directory and the mismatch could not be seen. Brief R10 made the release
    reach the children on 2026-08-22, and run ``20260822-r52-194234`` died
    three minutes and seventeen seconds in::

        modeling.data.DataError: dataset not found at
          .../_work/releases/r52/data/dataset.parquet

    Stage 3 had written that table one directory up, and no line of code
    anywhere said which of the two was right.

    A release whose root sits outside the project would raise ``ValueError``
    here, on the first second of a run rather than the thirty-second minute.
    `config.release_banner` makes the same assumption and says so too.
    """
    return work / PROFILES[release].release_root.relative_to(PROJECT_ROOT)
