"""The operation interface: one command per use case, and no web server.

The six stage packages -- ``extraction``, ``labeling``, ``baseline``,
``splitting``, ``modeling``, ``evaluation`` -- are the *development* interface.
Each writes a file, the next reads it, nothing recomputes, and every one of
them is a command a person can type while building the thing. That interface
stays exactly as it is.

What this package adds is the other one. There are two things this project
does, and until now neither of them had a name you could type:

.. code-block:: bash

    python -m pipeline.train --release {4.2|5.2} [--seeds ...] [--device ...]
    python -m pipeline.score --release {4.2|5.2} --model {frozen|<run_id>}

Both are complete without a browser. Both write their progress to stdout as
they go, the way every stage does, and both return a meaningful exit code.
Their other job is to be the only two things ``ui/`` ever launches, which is
what lets ``config.UI_ALLOWED_COMMANDS`` hold two entries instead of seven.

**Why "pipeline" here means something narrower than usual.** Elsewhere in this
repository "the pipeline packages" means the six frozen stage packages above.
This package is not one of them and does not belong to that set: it orchestrates
them and owns nothing they own. Every docstring in here says "the six stage
packages" when it means those, and never just "the pipeline".

**This package cannot open a test split**, by the same three constructions the
web layer used to rely on, each with a test:

1. No module here imports :mod:`evaluation.run_phase2` or
   :mod:`evaluation.testpass`, directly or transitively, and the name
   ``sweep_threshold`` appears in no source file here.
2. :func:`pipeline.commands.launch` is only ever handed the six stage modules,
   as literals written in :mod:`pipeline.train` and :mod:`pipeline.score`. The
   web layer's own allowlist sits one level above and holds two names.
3. Every filesystem write goes through :func:`pipeline.store.guard_write`,
   which permits ``models/`` and ``runs/`` and refuses everything else -- by
   name for the frozen checkpoints, the frozen results file and the test-split
   audit marker, for both releases.

Module map, one job each:

=================  ==========================================================
``detect``         which CERT release a folder is, from five header lines
``commands``       the subprocess launcher and what its exit codes mean
``sandbox``        the per-run working directory that keeps outputs apart
``inference``      the read-only scorer and the reproduction guard
``results``        what the stored models already measured; reads files only
``store``          the write guard, the model store and its manifest
``score``          use case one: score a folder with a stored model
``train``          use case two: train a new model on a folder and keep it
=================  ==========================================================
"""

from __future__ import annotations

__all__ = [
    "commands",
    "detect",
    "inference",
    "results",
    "sandbox",
    "score",
    "store",
    "train",
]
