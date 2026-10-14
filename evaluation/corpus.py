"""The row set Stage 7 scores, built once and shared by every analysis.

Stage 5's :func:`modeling.data.prepare` deliberately refuses to build the test
split, so Stage 7 needs its own way in. This module is that way in, and it is
written to be *the same rows* Stage 5 and 6 trained on rather than a second
pipeline that happens to agree.

Three properties make that true, and each is enforced rather than trusted:

**The same functions.** ``load_dataset``, ``clip_z_scores``, ``target_mask``,
``load_splits`` and ``_build_split`` are imported from :mod:`modeling.data`.
Nothing is reimplemented here. If Stage 3 ever changes the parquet, this
module breaks in the same place training would.

**The standardiser is supplied, never fitted.** :meth:`Corpus.build` takes a
:class:`~modeling.data.Standardiser` as a required argument — the caller
passes the one it read out of the checkpoint. There is no code path in this
module that fits one on evaluation rows, which is the leakage §13 asks to be
guarded against. :meth:`Corpus.refit_train_standardiser` exists only so a
guard can *compare* the checkpoint's values against a fresh fit and prove
they match; its result is never applied to anything.

**Calendar days travel with the rows.** :class:`EvalSplit` carries the ``day``
of every target row alongside the tensors, because detection latency (§5) is a
question about dates and :class:`~modeling.data.SplitData` does not keep them.
The day array is built from the same boolean mask and the same positions as
the features, and :meth:`Corpus.build` asserts the user column derived that
way matches the one ``_build_split`` produced — so an alignment slip is an
exception rather than a plausible latency table.

The raw (clipped, *un*-standardised) z-scores are kept too. §8 asks for the
eighteen z-scores of the highest-scoring false positives, and standardised
values are the model's units, not a reader's.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch

from config import (
    DAY_COLUMN,
    FEATURE_NAMES,
    LABEL_COLUMN,
    SCENARIO_COLUMN,
    USER_COLUMN,
    WINDOW_DAYS,
)
from modeling.data import (
    DataError,
    SplitData,
    Standardiser,
    _build_split,
    clip_z_scores,
    dataset_path,
    load_dataset,
    load_splits,
    splits_path,
    target_mask,
)

logger = logging.getLogger(__name__)

SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "val", "test")


class EvaluationError(RuntimeError):
    """A Stage 7 problem that must stop the run rather than produce a number."""


@dataclass(frozen=True, slots=True)
class EvalSplit:
    """One split's target rows, with everything Stage 7 needs to explain them.

    ``data`` is the ordinary :class:`~modeling.data.SplitData` the models are
    scored through — identical in construction to the one training used. The
    other four arrays are per *target row*, in the same order as
    ``data.labels``, and exist because Stage 7 reports about rows rather than
    only aggregating over them.
    """

    data: SplitData
    days: np.ndarray
    users: np.ndarray
    labels: np.ndarray
    scenarios: np.ndarray
    raw_features: np.ndarray

    @property
    def name(self) -> str:
        return self.data.name

    @property
    def n_rows(self) -> int:
        return int(self.labels.size)

    @property
    def n_positives(self) -> int:
        return int(self.labels.sum())

    @property
    def random_baseline(self) -> float:
        """The AUPRC a coin flip scores here. Print it beside every AUPRC."""
        return float(self.labels.mean()) if self.n_rows else 0.0

    def fingerprint(self) -> str:
        """A stable digest of *which rows these are*.

        Guard §13 asks that both models be scored on identical test rows. The
        cheap way to prove it is to hash the identity of every row — user, day
        and label — and compare the two digests, which catches a reordering
        that a row count would not.
        """
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.users.astype("U16").tobytes())
        digest.update(self.days.astype("datetime64[D]").astype(np.int64).tobytes())
        digest.update(self.labels.astype(np.int64).tobytes())
        return digest.hexdigest()[:16]

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.data.user_ids)} users, "
            f"{self.n_rows:,} target rows, {self.n_positives:,} positive "
            f"({self.random_baseline:.4%}), rows {self.fingerprint()}"
        )


@dataclass(frozen=True, slots=True)
class Corpus:
    """The whole clipped, masked table — loaded once, sliced many times.

    Reading ``dataset.parquet`` costs a couple of seconds and Stage 7 slices
    it six times (two models x three seeds) for validation and again for test.
    Holding the frame is worth it; recomputing the clip and the mask per slice
    would also risk the two differing, which is precisely the failure the
    experiment cannot survive.
    """

    frame: pd.DataFrame
    is_target: np.ndarray
    rows_clipped: int
    sample: bool
    splits_file: Path
    window_days: int = WINDOW_DAYS

    @classmethod
    def load(
        cls,
        *,
        sample: bool = False,
        window_days: int = WINDOW_DAYS,
        frame: pd.DataFrame | None = None,
        splits_file: Path | None = None,
    ) -> Corpus:
        """Load, clip and mask exactly as :func:`modeling.data.prepare` does.

        ``frame`` is an injection point for the tests; production passes none.
        """
        if frame is None:
            frame = load_dataset(dataset_path(sample))
        else:
            frame = frame.sort_values([USER_COLUMN, DAY_COLUMN], kind="stable")
            frame = frame.reset_index(drop=True)

        clipped, rows_clipped = clip_z_scores(frame)
        is_target = target_mask(clipped, window_days=window_days)

        labels = clipped[LABEL_COLUMN].to_numpy()
        lost = int(labels[~is_target].sum())
        if lost:
            logger.warning(
                "%d positive row(s) fall inside the %d-day warm-up and are "
                "scored by neither model", lost, window_days - 1,
            )

        return cls(
            frame=clipped,
            is_target=is_target,
            rows_clipped=rows_clipped,
            sample=sample,
            splits_file=splits_file or splits_path(sample),
            window_days=window_days,
        )

    def users_for(self, seed: int) -> dict[str, tuple[str, ...]]:
        """Stage 4's user lists for one seed, all three splits."""
        return load_splits(self.splits_file, seed)

    def build(
        self,
        name: str,
        seed: int,
        standardiser: Standardiser,
        *,
        users: Sequence[str] | None = None,
    ) -> EvalSplit:
        """Build one split's rows under a standardiser the caller supplies.

        The standardiser is an argument and not a computation on purpose: it
        comes out of the checkpoint, so the rows are scaled by exactly the
        transform the weights were trained under. Nothing in this module can
        fit one on the split it is about to score.
        """
        if name not in SPLIT_NAMES:
            raise EvaluationError(
                f"unknown split {name!r}; expected one of {', '.join(SPLIT_NAMES)}"
            )
        if users is None:
            users = self.users_for(seed)[name]
        if not users:
            raise EvaluationError(f"split {name!r} for seed {seed} lists no users")

        data = _build_split(name, self.frame, self.is_target, users, standardiser)

        belongs = self.frame[USER_COLUMN].isin(set(users)).to_numpy()
        rows = self.frame.loc[belongs]
        positions = np.flatnonzero(self.is_target[belongs])

        users_here = rows[USER_COLUMN].to_numpy()[positions]
        if not np.array_equal(users_here, data.users):
            raise EvaluationError(
                f"row alignment slipped while building split {name!r}: the day "
                f"array and the feature tensor disagree about who owns which "
                f"row. Every latency and per-row number would be wrong."
            )

        return EvalSplit(
            data=data,
            days=rows[DAY_COLUMN].to_numpy()[positions],
            users=users_here,
            labels=rows[LABEL_COLUMN].to_numpy()[positions].astype(np.int64),
            scenarios=rows[SCENARIO_COLUMN].to_numpy()[positions].astype(np.int64),
            raw_features=rows[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)[
                positions
            ],
        )

    def refit_train_standardiser(self, seed: int) -> Standardiser:
        """Refit on this seed's training target rows — **for comparison only**.

        §13 asks for proof that the standardiser came from the checkpoint
        rather than from the evaluation data. Proving it needs a second value
        to compare against, and this is that value. It is never applied to any
        split; :func:`evaluation.guards.standardiser_guard` is its only caller
        and it does nothing but assert equality and print both.
        """
        train_users = set(self.users_for(seed)["train"])
        in_train = self.frame[USER_COLUMN].isin(train_users).to_numpy()
        fit_rows = self.frame.loc[
            in_train & self.is_target, list(FEATURE_NAMES)
        ].to_numpy(dtype=np.float32)
        if fit_rows.size == 0:
            raise EvaluationError(
                f"seed {seed} has no training target rows to refit on; the "
                f"splits file and the dataset disagree about who exists"
            )
        return Standardiser.fit(fit_rows)


def as_tensor_labels(split: EvalSplit) -> torch.Tensor:
    """The float labels the loss functions want, from the int array we keep."""
    return torch.from_numpy(split.labels.astype(np.float32))


def positions_within(days: np.ndarray, first_day: np.datetime64) -> np.ndarray:
    """1-based day number of each date within a window starting at ``first_day``.

    Day 1 is the window's first day. Used by :mod:`evaluation.latency`, where
    the whole question is "how far into the attack was it noticed", so an
    off-by-one here is an off-by-one in the project's headline claim.
    """
    delta = (
        days.astype("datetime64[D]") - np.datetime64(first_day, "D")
    ).astype(np.int64)
    return delta + 1


def require_no_gaps(split: EvalSplit) -> None:
    """Assert every user's target rows are consecutive calendar days.

    :func:`modeling.data.day_index` already raises on a gap in the *full*
    record, which is the check that matters for Model B's windows. This is the
    narrower statement latency needs: within the target rows themselves, a
    missing day would make "day N of the attack" mean a different thing for
    one user than for another.
    """
    frame = pd.DataFrame({"user": split.users, "day": split.days})
    gaps = frame.groupby("user", sort=False)["day"].diff()
    irregular = gaps.notna() & (gaps != pd.Timedelta(days=1))
    if bool(irregular.any()):
        offenders = frame.loc[irregular, "user"].unique()[:5]
        raise DataError(
            f"{int(irregular.sum())} target row(s) break calendar-day density "
            f"in split {split.name!r}, first users: "
            f"{', '.join(map(str, offenders))}"
        )
