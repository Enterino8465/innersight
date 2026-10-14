"""Stage 8 — the local web UI. A front door, and nothing behind it.

Two buttons over two commands: run a stored model on a folder, and train a new
model on a folder. Since brief S1 this package holds no science at all — the
work lives in :mod:`pipeline`, which is a thing you can run without a browser,
and everything here is routes, request parsing, output streaming and templates.

**This package cannot open a test split.** Not by convention — by construction,
in three independent ways, each with a test:

1. No module here imports :mod:`evaluation.run_phase2` or
   :mod:`evaluation.testpass`, directly or transitively, and the name
   ``sweep_threshold`` appears in no source file here.
2. :mod:`ui.jobs` will only launch a module named in
   ``config.UI_ALLOWED_COMMANDS``, which since brief S1 holds exactly two —
   ``pipeline.train`` and ``pipeline.score``. The check happens before a
   command line is even built, and neither of those two can reach the test
   pass either.
3. Nothing here writes to the filesystem. Every write belongs to
   :func:`pipeline.store.guard_write`, which permits ``models/`` and ``runs/``
   and refuses everything else — including, by name, ``outputs/checkpoints/``,
   the frozen results file and the test-split audit marker, for both releases.

**What the page may still read.** Three views open a file and show what is in
it: which release a folder is, which models have been stored, and what the
published checkpoints already measured. Those are calls, not commands, because
an instant answer is worth having in front of an examiner. The boundary that is
enforced is the one that matters — no module here can reach
:mod:`pipeline.score`, :mod:`pipeline.train`, :mod:`pipeline.inference` or
:mod:`pipeline.sandbox`. The UI launches those; it does not call them.

Module map, one job each:

=================  ==========================================================
``jobs``           the allowlist, the subprocess it permits, and its line stream
``app``            Flask routes and Server-Sent Events; no science
=================  ==========================================================
"""

from __future__ import annotations

__all__ = [
    "app",
    "jobs",
]
