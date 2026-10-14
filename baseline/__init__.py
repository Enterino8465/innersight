"""Stage 3 — the personal baseline.

Turns every raw daily count into a z-score against that same person's own
earlier days, so that "3 USB connections" becomes "+20 sigma for this person"
or "-0.7 sigma for that one" depending on who did it. This is the stage the
thesis lives in; the models downstream only test whether temporal context
helps once the question has been posed this way.

Reads ``data/labelled_counts.parquet``, writes ``data/dataset.parquet`` with
the same rows, the same order and the same columns. Run it from the project
root::

    python -m baseline.baseline                   # full table
    python -m baseline.baseline --sample          # 30-user sample, ~1 s
    python -m baseline.baseline --user KPC0073    # raw counts beside z-scores

No windowing, no splitting, no models — those are Stages 4 onward.
"""

from __future__ import annotations

__all__ = ["baseline"]
