"""Stage 1 — feature extraction.

Turns the five raw CERT r4.2 activity logs into one row per ``(user, day)``
carrying the 18 daily counts defined by ``config.FEATURE_NAMES``.

This is the only slow stage in the project and the only one that touches the
16 GB release. It runs once; every later stage reads the small table it writes
to ``data/daily_counts.parquet``.

Run it from the project root::

    python -m extraction.extract                 # 30-user sample, ~7 s
    python -m extraction.extract --full          # full r4.2, ~1-3 h
    python -m extraction.extract --user KPC0073  # inspect one user's days

Nothing here labels, z-scores, windows or splits anything. Those belong to
Stages 2-4 so that this expensive pass never has to be repeated when a
downstream rule changes.
"""

from __future__ import annotations

__all__ = ["extract"]
