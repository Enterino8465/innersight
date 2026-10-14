"""Stage 4 — the train / validation / test split.

Assigns every one of the 1,000 users to exactly one of train, val and test,
for each of the three seeds in ``config.SEEDS``, and writes the assignment to
``data/splits.json``. Nothing is copied: the split is an *index* into
``data/dataset.parquet``, not a new table.

Two decisions carry the stage. The split is by **user**, never by day — a
person's Monday in training and their Tuesday in test measures memorisation,
not generalisation. And it is stratified by ``user_scenario`` rather than by
insider/benign, because scenario 3 has only ten insiders in the whole release
and a coarser stratification can hand the test set none of them.

Run it from the project root::

    python -m splitting.split              # the 1,000-user table
    python -m splitting.split --sample     # the 30-user sample

No z-scores, no windowing, no models — those are Stages 5 onward.
"""

from __future__ import annotations

__all__ = ["split"]
