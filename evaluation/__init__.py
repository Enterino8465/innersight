"""Stage 7 — evaluate and compare. The stage the project is submitted on.

This package reads checkpoints and computes numbers. It does not change how
anything was trained: ``modeling/data.py``, ``models.py``, ``metrics.py``,
``train.py`` and ``runner.py`` are imported and never edited, so Stage 7
cannot retroactively alter the experiment it is measuring.

The order is not negotiable, and the code enforces it::

    python -m evaluation.run_phase1     # validation: latency, P@10 inspection, guards
    python -m evaluation.run_b7         # validation: the capacity control (trains)
    python -m modeling.run_model_a --no-pos-weight   # validation: the ablation (trains)
    python -m evaluation.run_phase2     # THE TEST SPLIT — once, both models
    python -m evaluation.run_stage7     # report, JSON, figures

**Phase 1 first, and all of it.** Once the test split is opened no model may
change, because an adjustment made after seeing test numbers is fitting to the
test set and the reported result stops meaning anything.
:func:`evaluation.testpass.require_phase1` refuses to open test until Phase 1's
artifact exists, and every opening is appended to
``outputs/TEST_SPLIT_OPENED.json`` so a second pass is visible in the record
rather than only in someone's memory.

**Nothing is refitted and nothing is re-swept.** The standardiser and the
decision threshold travel inside each checkpoint, put there by Stage 5's
``runner.save_checkpoint`` precisely so that Stage 7 cannot recompute them by
accident. :mod:`evaluation.scoring` never imports ``sweep_threshold``, and
:mod:`evaluation.corpus` has no code path that fits a standardiser on the rows
it is about to score. :mod:`evaluation.guards` proves both, by value, in the
log and in the JSON.

**Model B v2 is not part of the result.** The tuning attempt lost — AUPRC
0.6767 → 0.6022 with double the spread — and by brief §3 it appears nowhere
except one discussion paragraph about the pooling operator, which needs no
weights. :func:`evaluation.checkpoints.load_checkpoint` will not open a
``*_v2_*`` file.

Module map, one job each:

===========================  ================================================
``corpus``                   the row set, built once, sliced per split
``checkpoints``              one loader for both models, with the §3 refusal
``scoring``                  checkpoint + rows -> probabilities and metrics
``guards``                   the §13 list, as checks that run and print values
``latency``                  first-detection day and recall by position (§5)
``falsepositives``           the top-scoring benign rows, inspected (§8)
``phase1``                   the validation analyses, orchestrated
``testpass``                 the single test pass (§9), with its audit trail
``report`` / ``figures``     the deliverables (§11, §12)
``run_*``                    CLI entry points
===========================  ================================================
"""

from __future__ import annotations

__all__ = [
    "checkpoints",
    "corpus",
    "falsepositives",
    "figures",
    "guards",
    "latency",
    "phase1",
    "report",
    "scoring",
    "testpass",
]
