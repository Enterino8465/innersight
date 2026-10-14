"""Stage 2 — calendar fill and ground-truth labelling.

Takes Stage 1's sparse table of ``(user, day)`` counts, expands each user onto
every calendar day they were observed across (zero-filling the days they
generated no events), and attaches the two ground-truth columns from
``answers/insiders.csv``:

``label``          per day  — is this day inside an attack window (0/1)
``user_scenario``  per user — 0 for benign, 1/2/3 for an insider

Writes ``data/labelled_counts.parquet``. Run it from the project root::

    python -m labeling.label                  # full table
    python -m labeling.label --sample         # 30-user sample, ~1 s
    python -m labeling.label --user KPC0073   # inspect one user's window

No z-scores, no windowing, no splitting, and no rows are dropped — those are
Stages 3 and 4.
"""

from __future__ import annotations

__all__ = ["label"]
