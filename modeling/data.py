from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch

from config import (
    DATASET_PATH,
    DATASET_SAMPLE_PATH,
    DAY_COLUMN,
    FEATURE_NAMES,
    LABEL_COLUMN,
    N_FEATURES,
    SCENARIO_COLUMN,
    SPLITS_PATH,
    SPLITS_SAMPLE_PATH,
    STD_FLOOR,
    USER_COLUMN,
    WINDOW_DAYS,
    Z_CLIP,
)

logger = logging.getLogger(__name__)

TRAINING_SPLITS: Final[tuple[str, ...]] = ("train", "val")

MODEL_A_CONTEXT_DAYS: Final[int] = 1


class DataError(RuntimeError):
    """A structural problem in the inputs that must stop the run."""


@dataclass(frozen=True, slots=True)
class Standardiser:

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        for name, array in (("mean", self.mean), ("std", self.std)):
            if array.shape != (N_FEATURES,):
                raise DataError(
                    f"standardiser {name} must have shape ({N_FEATURES},), "
                    f"got {array.shape}"
                )
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise DataError("standardiser statistics contain NaN or inf")
        if (self.std <= 0).any():
            raise DataError("standardiser std contains non-positive values")

    @classmethod
    def fit(cls, features: np.ndarray) -> Standardiser:

        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            raise DataError(
                f"expected (n, {N_FEATURES}) to fit on, got {features.shape}"
            )
        if features.shape[0] == 0:
            raise DataError("cannot fit a standardiser on zero rows")
        mean = features.mean(axis=0, dtype=np.float64)
        std = features.std(axis=0, dtype=np.float64)
        std = np.where(std < STD_FLOOR, 1.0, std)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def apply(self, features: np.ndarray) -> np.ndarray:
        """Standardise in float64, return float32.

        float32 throughout would be fine numerically, but MPS refuses float64
        outright, so the cast has to be explicit and here rather than
        discovered at the first Metal kernel.
        """
        return ((features.astype(np.float64) - self.mean) / self.std).astype(
            np.float32
        )

    def to_dict(self) -> StandardiserPayload:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": list(FEATURE_NAMES),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[float] | Sequence[str]]) -> Standardiser:
        names = list(payload.get("feature_names", FEATURE_NAMES))
        if names != list(FEATURE_NAMES):
            raise DataError(
                "checkpoint standardiser was fitted on a different feature "
                "order; config.FEATURE_NAMES has changed since it was saved"
            )
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class PointView:

    tensor: torch.Tensor

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def gather(self, index: torch.Tensor) -> torch.Tensor:
        return self.tensor.index_select(0, index.to(self.tensor.device))

    def to(self, device: str | torch.device) -> PointView:
        return PointView(self.tensor.to(device))

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape[1:])


@dataclass(frozen=True, slots=True)
class WindowView:

    features: torch.Tensor
    positions: torch.Tensor
    days: int = WINDOW_DAYS

    def __len__(self) -> int:
        return int(self.positions.numel())

    def gather(self, index: torch.Tensor) -> torch.Tensor:
        device = self.features.device
        selected = self.positions.index_select(0, index.to(self.positions.device))
        offsets = torch.arange(
            -(self.days - 1), 1, dtype=torch.int64, device=selected.device
        )
        rows = (selected.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
        gathered = self.features.index_select(0, rows.to(device))
        return gathered.reshape(-1, self.days, self.features.shape[1]).permute(
            0, 2, 1
        ).contiguous()

    def to(self, device: str | torch.device) -> WindowView:
        return WindowView(
            self.features.to(device), self.positions.to(device), self.days
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (int(self.features.shape[1]), self.days)


@dataclass(frozen=True, slots=True)
class SplitData:
    """Every row belonging to one split's users, plus which are targets.

    ``features`` holds *all* rows of these users in (user, day) order, already
    clipped and standardised — history included. ``target_positions`` indexes
    into it and marks the rows that are actually predicted.

    Keeping the two separate is what lets Stage 6 exist without touching this
    file. Model A takes ``features[target_positions]``. Model B takes
    ``features[p - 27 : p + 1]`` for each position ``p``, which is guaranteed
    to lie inside the same user because a target position is never within the
    first 27 rows of a user and rows of one user are contiguous.
    """

    name: str
    features: torch.Tensor
    target_positions: torch.Tensor
    labels: torch.Tensor
    scenarios: torch.Tensor
    users: np.ndarray
    user_ids: tuple[str, ...]

    @property
    def n_targets(self) -> int:
        return int(self.target_positions.numel())

    @property
    def n_positives(self) -> int:
        return int(self.labels.sum().item())

    @property
    def n_negatives(self) -> int:
        return self.n_targets - self.n_positives

    @property
    def positive_rate(self) -> float:
        """The random-baseline AUPRC. Print it beside every AUPRC reported."""
        return self.n_positives / self.n_targets if self.n_targets else 0.0

    def point_features(self) -> torch.Tensor:
        """Model A's input: one day per target row, (n_targets, 18)."""
        return self.features.index_select(0, self.target_positions)

    def window_features(self, days: int = WINDOW_DAYS) -> torch.Tensor:
        """Model B's input: (n_targets, 18, days), channels-first for Conv1d.

        Stage 6 calls this and Stage 5 does not. It lives here, next to the
        mask that makes it safe, so that Stage 6 adds an architecture and
        nothing else.
        """
        if days < 1 or days > WINDOW_DAYS:
            raise DataError(f"window of {days} days is outside 1..{WINDOW_DAYS}")
        offsets = torch.arange(-(days - 1), 1, dtype=torch.int64)
        rows = self.target_positions.unsqueeze(1) + offsets.unsqueeze(0)
        if int(rows.min().item()) < 0:
            raise DataError(
                "a window reached before the start of the array; the target "
                "mask and WINDOW_DAYS have gone out of step"
            )
        return self.features.index_select(0, rows.reshape(-1)).reshape(
            rows.shape[0], days, N_FEATURES
        ).permute(0, 2, 1).contiguous()

    def point_view(self) -> PointView:
        """Model A's view of this split."""
        return PointView(self.point_features())

    def window_view(self, days: int = WINDOW_DAYS) -> WindowView:
        """Model B's view of this split. Stage 6 calls this; Stage 5 does not."""
        if int(self.target_positions.min().item()) < days - 1:
            raise DataError(
                f"a target position sits inside the first {days - 1} rows of "
                f"its user; the mask and the window length disagree"
            )
        return WindowView(self.features, self.target_positions, days)

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.user_ids)} users, "
            f"{self.features.shape[0]:,} rows in memory, "
            f"{self.n_targets:,} target rows, "
            f"{self.n_positives:,} positive ({self.positive_rate:.4%})"
        )


@dataclass(frozen=True, slots=True)
class PreparedData:
    """What one seed's worth of preparation produces."""

    seed: int
    train: SplitData
    val: SplitData
    standardiser: Standardiser
    rows_total: int
    rows_masked_out: int
    positives_lost_to_mask: int
    rows_clipped: int
    sample: bool

    def pos_weight(self) -> float:
        """``n_negatives / n_positives`` of the TRAINING split.

        Computed here, from the training rows, every run. Never hardcoded and
        never taken from the whole table: the class balance of the test set is
        information about the test set, and putting it in the loss function is
        leakage even though nothing will ever complain.
        """
        if self.train.n_positives == 0:
            raise DataError(
                "the training split contains no positive rows; pos_weight is "
                "undefined and training would be meaningless"
            )
        return self.train.n_negatives / self.train.n_positives


def dataset_path(sample: bool) -> Path:
    return DATASET_SAMPLE_PATH if sample else DATASET_PATH


def splits_path(sample: bool) -> Path:
    return SPLITS_SAMPLE_PATH if sample else SPLITS_PATH


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the z-scored table and put it in (user, day) order.

    The sort is not cosmetic: every position-based operation below — the day
    index, the target mask, Model B's windows — assumes one user's rows are
    contiguous and chronological. Stage 3 happens to write it that way; this
    guarantees it.
    """
    if not path.is_file():
        raise DataError(
            f"dataset not found at {path}. Run Stage 3 (`python -m "
            f"baseline.baseline`) first, or pass --dataset."
        )
    frame = pd.read_parquet(path)
    missing = [
        column
        for column in (USER_COLUMN, DAY_COLUMN, SCENARIO_COLUMN, LABEL_COLUMN,
                       *FEATURE_NAMES)
        if column not in frame.columns
    ]
    if missing:
        raise DataError(
            f"{path.name} is missing {len(missing)} expected column(s): "
            f"{', '.join(missing)}"
        )
    frame = frame.sort_values([USER_COLUMN, DAY_COLUMN], kind="stable")
    return frame.reset_index(drop=True)


def clip_z_scores(frame: pd.DataFrame, limit: float = Z_CLIP) -> tuple[pd.DataFrame, int]:
    """Clip every feature to +/-``limit``. Returns the frame and rows touched.

    Before standardising, never after. See the module docstring.
    """
    values = frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float32, copy=True)
    touched = int((np.abs(values) > limit).any(axis=1).sum())
    np.clip(values, -limit, limit, out=values)
    clipped = frame.copy()
    clipped[list(FEATURE_NAMES)] = values
    return clipped, touched


def day_index(frame: pd.DataFrame) -> np.ndarray:
    """0-based position of each row within its user's record.

    Raises if a user's days are not calendar-dense. That check matters more
    than it looks: Model B builds its window by taking the 27 array rows
    before a target, so a missing calendar day would silently turn a "28-day
    window" into 28 rows spanning 40 days. Stage 3 guarantees density; this
    refuses to proceed if that ever stops being true.
    """
    index = frame.groupby(USER_COLUMN, sort=False).cumcount().to_numpy()
    gaps = frame.groupby(USER_COLUMN, sort=False)[DAY_COLUMN].diff()
    irregular = gaps.notna() & (gaps != pd.Timedelta(days=1))
    if bool(irregular.any()):
        offenders = frame.loc[irregular, USER_COLUMN].unique()[:5]
        raise DataError(
            f"{int(irregular.sum())} row(s) break calendar-day density, "
            f"first users: {', '.join(map(str, offenders))}. Model B's "
            f"windows assume one row per calendar day per user."
        )
    return index


def target_mask(frame: pd.DataFrame, window_days: int = WINDOW_DAYS) -> np.ndarray:
    """True for rows that may be *predicted*: day_index >= window_days - 1.

    Identical for Model A and Model B, for training and for evaluation. This
    is the rule that keeps the comparison honest; see the module docstring for
    why breaking it hands Model B about 6% of AUPRC for nothing.
    """
    return day_index(frame) >= (window_days - 1)


def load_splits(path: Path, seed: int) -> dict[str, tuple[str, ...]]:
    """Return ``{"train": (...), "val": (...), "test": (...)}`` for one seed.

    Stage 4 wrote this. Stage 5 reads it and does not re-split: generating a
    fresh split here would silently decouple Model A's users from Model B's.
    """
    if not path.is_file():
        raise DataError(
            f"splits not found at {path}. Run Stage 4 (`python -m "
            f"splitting.split`) first, or pass --splits."
        )
    payload = json.loads(path.read_text())
    table = payload.get("splits", payload)
    key = str(seed)
    if key not in table:
        raise DataError(
            f"seed {seed} is not in {path.name}; it holds "
            f"{', '.join(sorted(table))}"
        )
    return {name: tuple(users) for name, users in table[key].items()}


def _build_split(
    name: str,
    frame: pd.DataFrame,
    is_target: np.ndarray,
    users: Sequence[str],
    standardiser: Standardiser,
) -> SplitData:
    """Slice one split's users out of the full frame and tensorise it."""
    user_set = set(users)
    belongs = frame[USER_COLUMN].isin(user_set).to_numpy()
    rows = frame.loc[belongs]
    if rows.empty:
        raise DataError(f"split {name!r} matched no rows in the dataset")

    features = standardiser.apply(
        rows[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    )
    if not np.isfinite(features).all():
        raise DataError(f"split {name!r} contains NaN or inf after standardising")

    positions = np.flatnonzero(is_target[belongs])
    labels = rows[LABEL_COLUMN].to_numpy()[positions]
    scenarios = rows[SCENARIO_COLUMN].to_numpy()[positions]

    return SplitData(
        name=name,
        features=torch.from_numpy(features),
        target_positions=torch.from_numpy(positions.astype(np.int64)),
        labels=torch.from_numpy(labels.astype(np.float32)),
        scenarios=torch.from_numpy(scenarios.astype(np.int64)),
        users=rows[USER_COLUMN].to_numpy()[positions],
        user_ids=tuple(sorted(user_set)),
    )


def prepare(
    seed: int,
    *,
    sample: bool = False,
    frame: pd.DataFrame | None = None,
    splits: dict[str, Sequence[str]] | None = None,
    window_days: int = WINDOW_DAYS,
) -> PreparedData:
    """Run the whole pipeline for one seed and return train + validation.

    **The test split is not built.** Stage 5 trains on train, selects on
    validation and reports validation; the test users are read out of
    ``splits.json`` only to be ignored. Test data is opened once, in Stage 7,
    for both models together — which is also why there is exactly one place in
    this codebase that reads it. Saving test predictions "just to peek" is how
    a project ends up with a number it cannot defend.

    ``frame`` and ``splits`` are injection points for the tests; production
    calls pass neither.
    """
    if frame is None:
        frame = load_dataset(dataset_path(sample))
    else:
        frame = frame.sort_values([USER_COLUMN, DAY_COLUMN], kind="stable")
        frame = frame.reset_index(drop=True)
    if splits is None:
        splits = load_splits(splits_path(sample), seed)

    frame, rows_clipped = clip_z_scores(frame)
    is_target = target_mask(frame, window_days=window_days)

    labels_all = frame[LABEL_COLUMN].to_numpy()
    positives_lost = int(labels_all[~is_target].sum())
    if positives_lost:
        logger.warning(
            "%d positive row(s) fall inside the %d-day warm-up and are not "
            "scored by either model", positives_lost, window_days - 1,
        )

    missing = [name for name in TRAINING_SPLITS if name not in splits]
    if missing:
        raise DataError(f"splits for seed {seed} lack {', '.join(missing)}")

    train_users, val_users = splits["train"], splits["val"]
    overlap = set(train_users) & set(val_users)
    if overlap:
        raise DataError(
            f"{len(overlap)} user(s) appear in both train and val, e.g. "
            f"{sorted(overlap)[:3]} — the split is not by user"
        )

    in_train = frame[USER_COLUMN].isin(set(train_users)).to_numpy()
    fit_rows = frame.loc[in_train & is_target, list(FEATURE_NAMES)].to_numpy(
        dtype=np.float32
    )
    standardiser = Standardiser.fit(fit_rows)

    train = _build_split("train", frame, is_target, train_users, standardiser)
    val = _build_split("val", frame, is_target, val_users, standardiser)

    return PreparedData(
        seed=seed,
        train=train,
        val=val,
        standardiser=standardiser,
        rows_total=len(frame),
        rows_masked_out=int((~is_target).sum()),
        positives_lost_to_mask=positives_lost,
        rows_clipped=rows_clipped,
        sample=sample,
    )


def log_preparation(data: PreparedData) -> None:
    """The pre-flight block from the brief's guard list.

    Printed before a single gradient step, because every line of it is a
    failure that otherwise produces a plausible number instead of an error.
    """
    logger.info(
        "rows %s, masked out as warm-up %s (%.1f%%), positives lost to the "
        "mask %d",
        f"{data.rows_total:,}",
        f"{data.rows_masked_out:,}",
        100.0 * data.rows_masked_out / max(data.rows_total, 1),
        data.positives_lost_to_mask,
    )
    logger.info(
        "rows with any |z| > %.0f before clipping: %s (%.2f%%)",
        Z_CLIP, f"{data.rows_clipped:,}",
        100.0 * data.rows_clipped / max(data.rows_total, 1),
    )
    for split in (data.train, data.val):
        logger.info("%s", split.describe())
        if split.n_positives == 0 or split.n_negatives == 0:
            raise DataError(
                f"split {split.name!r} has only one class present "
                f"({split.n_positives} positive, {split.n_negatives} "
                f"negative); training or selection would be meaningless"
            )
    shared = set(data.train.user_ids) & set(data.val.user_ids)
    if shared:
        raise DataError(f"{len(shared)} user(s) are in both train and val")
    logger.info(
        "pos_weight (train only) = %.1f  [%s negatives / %s positives]",
        data.pos_weight(),
        f"{data.train.n_negatives:,}", f"{data.train.n_positives:,}",
    )
    logger.info(
        "random-baseline AUPRC on validation = %.5f",
        data.val.positive_rate,
    )
