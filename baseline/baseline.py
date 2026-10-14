"""Stage 3: labelled counts -> each day z-scored against that user's own past.

The thesis lives in this file. An insider does nothing forbidden -- they do
normal things at abnormal times and in abnormal amounts, with credentials that
are genuinely theirs. So no feature here asks "is this a lot of USB activity?"
Every one asks "is this a lot *for this person*?", and the answer is a z-score
against that same user's history.

**Causal, and only causal.** :func:`causal_z_scores` scores each day against
that user's *strictly earlier* days, using expanding aggregations rather than
`E[x^2] - E[x]^2`, which is unsafe at `total_email_size`'s ~1e8 magnitudes.
A day is never scored against the future, because the model is meant to work
on the morning it is deployed, not in hindsight. Rows before
:data:`config.MIN_HISTORY_DAYS` have no prior to divide by and are emitted as
exactly 0.

**The standard-deviation floor is the part that earns its keep.** Most
employees never do the things the insider signal is made of -- 73.5% of users
have a `usb_connect_count` standard deviation of exactly 0 -- so a naive
divide turns "was 0 every day for six months, then became 2" into infinity or
into nothing. The floor is
`STD_FLOOR_MULTIPLIER x median over users who actually exercise the feature of
that user's own std`, and taking that median over *active* users rather than
all of them took the validation gate from 54 of 70 insiders clearing 2 sigma
to 66 of 70, while benign days over 2 sigma barely moved (11.6% -> 13.4%).
See :data:`config.STD_FLOOR_ACTIVE_USERS_ONLY`; that deviation is deliberate
and must not be reverted.

**This stage writes raw, unclipped z-scores.** Clipping at
:data:`config.Z_CLIP` belongs to whoever builds the training tensors. The
whole point of the per-insider hand-check below is to see the un-clipped
number.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from config import (
    BENIGN_SCENARIO,
    DATASET_PATH,
    DATASET_SAMPLE_PATH,
    DAY_COLUMN,
    FEATURE_NAMES,
    LABEL_COLUMN,
    LABELLED_COUNTS_PATH,
    LABELLED_COUNTS_SAMPLE_PATH,
    MIN_HISTORY_DAYS,
    SCENARIO_COLUMN,
    SCENARIOS,
    STD_FLOOR,
    STD_FLOOR_ACTIVE_USERS_ONLY,
    STD_FLOOR_FALLBACK,
    STD_FLOOR_MULTIPLIER,
    USER_COLUMN,
    VALIDATION_Z_THRESHOLD,
    Z_CLIP,
    ensure_directories,
)

logger = logging.getLogger(__name__)

KEY_COLUMNS: Final[tuple[str, str]] = (USER_COLUMN, DAY_COLUMN)
_RULE: Final[str] = "=" * 78

DDOF: Final[int] = 1

DAY_MAX: Final[str] = "day_max_abs_z"
DAY_MEAN: Final[str] = "day_mean_abs_z"
WARMUP_COLUMN: Final[str] = "in_warmup"

MIN_DECAY_WINDOW_DAYS: Final[int] = 6

MIN_PRIOR_DAYS: Final[int] = 2


class BaselineError(AssertionError):
    """A z-scored table violates something that cannot be true of real data."""


def std_floors(
    counts: pd.DataFrame,
    features: Sequence[str] = FEATURE_NAMES,
    multiplier: float = STD_FLOOR_MULTIPLIER,
    fallback: float = STD_FLOOR_FALLBACK,
    active_only: bool = STD_FLOOR_ACTIVE_USERS_ONLY,
) -> pd.Series:
    """One lower bound per feature for the z-score denominator.

    ``floor[f] = multiplier * median over users of (that user's own std of f)``

    Read the shape of that quantity, not just the formula. It is *typical
    within-person day-to-day variability* in feature ``f`` -- the median person's
    own noise level -- deliberately not the population's spread, which would be
    dominated by the fact that different people do different jobs. Scaling it by
    0.1 bounds the resulting |z| at roughly ten typical-persons'-worth of
    variation, so the accountant's first USB connection scores large and finite
    instead of infinite.

    **The median is taken over users who exercise the feature at all** when
    *active_only* is set, and this is a measured correction rather than a taste.
    On r4.2 most employees never do the things an insider's signal is made of:
    73.5% of users have a ``usb_connect_count`` std of exactly zero, 81% for
    ``after_hours_logons``, 73.6% for ``file_count``. Over *all* users the
    median is therefore literally 0 for eight of the eighteen features, the
    floor collapses onto *fallback*, and "one USB connection" becomes one sigma
    on a count that is normally zero. Excluding the users for whom the feature
    is constant asks the better-posed question -- how much does this vary for
    somebody who does it? -- and takes the gate from 54/70 to 66/70 (scenario 1
    from 22/30 to 30/30) while moving the benign 2-sigma rate only from 11.6% to
    13.4%. See ``config.STD_FLOOR_ACTIVE_USERS_ONLY``.

    Args:
        counts: Raw counts with a ``user`` column; the whole table.
        features: Columns to compute a floor for.
        multiplier: Fraction of the median per-user std to use.
        fallback: Value substituted when the computed floor is 0 or NaN. That
            happens when a feature is constant for *every* user --
            ``large_attachment_count`` is identically zero on r4.2 -- and 1.0
            means "one unit of the raw count is one sigma", which keeps a dead
            feature pinned at z = 0 rather than exploding the day it moves.
        active_only: Take the median over users with a non-zero own-std only.

    Returns:
        A float64 Series indexed by feature name, in ``features`` order, with
        every value strictly positive.
    """
    columns = list(features)
    per_user_std = counts.groupby(USER_COLUMN, sort=False)[columns].std(ddof=DDOF)
    if active_only:
        per_user_std = per_user_std.where(per_user_std > 0)
    typical = per_user_std.median(skipna=True)

    floors = (multiplier * typical).astype("float64")
    dead = ~np.isfinite(floors.to_numpy()) | (floors.to_numpy() <= 0.0)
    if dead.any():
        logger.info(
            "std floor fell back to %.3f for %s (feature constant for every "
            "user, so there is no variability to scale)",
            fallback,
            ", ".join(floors.index[dead]),
        )
    floors[dead] = fallback
    return floors.reindex(columns)


def causal_z_scores(
    counts: pd.DataFrame,
    floors: pd.Series,
    features: Sequence[str] = FEATURE_NAMES,
    min_history: int = MIN_HISTORY_DAYS,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score every day against the same user's strictly-earlier days.

    Args:
        counts: One row per ``(user, day)``, calendar-dense per user. Row order
            is preserved in the output; the computation itself sorts internally,
            so an unsorted input is handled rather than silently mis-scored.
        floors: Per-feature denominator floors from :func:`std_floors`.
        features: The 18 columns to score, in ``FEATURE_NAMES`` order.
        min_history: Rows earlier than this position in a user's history are
            emitted as exactly 0. Raised to :data:`MIN_PRIOR_DAYS` if smaller,
            because below that there is no prior mean or spread to divide by.

    Returns:
        ``(z, warmup)`` -- a float32 frame of z-scores aligned to *counts*'
        index, and a boolean array marking the rows that were zeroed.

    The implementation uses pandas' expanding aggregations rather than
    cumulative sums on purpose. ``E[x^2] - E[x]^2`` is the obvious formulation
    and it is unsafe here: ``total_email_size`` runs to ~1e8, so both terms are
    ~1e16 and their difference can come out *negative* in floating point, which
    becomes NaN under the square root. The expanding kernel accumulates with a
    Welford-style update instead, and 465k x 18 is far too small for the speed
    difference to matter.
    """
    columns = list(features)
    _require_features(counts, columns)

    order = counts.sort_values(list(KEY_COLUMNS), kind="stable").index
    ordered = counts.loc[order, columns].astype("float64")
    users = counts.loc[order, USER_COLUMN]

    grouped = ordered.groupby(users, sort=False)
    mean_prior = grouped.expanding().mean().reset_index(level=0, drop=True)
    std_prior = grouped.expanding().std(ddof=DDOF).reset_index(level=0, drop=True)
    mean_prior = mean_prior.groupby(users, sort=False).shift(1)
    std_prior = std_prior.groupby(users, sort=False).shift(1)

    denominator = np.maximum(
        std_prior.fillna(0.0).to_numpy(),
        np.maximum(floors.reindex(columns).to_numpy(), STD_FLOOR),
    )
    scores = (ordered.to_numpy() - mean_prior.fillna(0.0).to_numpy()) / denominator

    warmup_ordered = _warmup_mask(users, max(min_history, MIN_PRIOR_DAYS))
    scores[warmup_ordered, :] = 0.0

    z_ordered = pd.DataFrame(
        scores.astype("float32"), index=ordered.index, columns=columns
    )
    z = z_ordered.reindex(counts.index)
    warmup = pd.Series(warmup_ordered, index=ordered.index).reindex(
        counts.index
    ).to_numpy()

    logger.info(
        "z-scored %d rows x %d features; %d row(s) zeroed as warm-up",
        len(z), len(columns), int(warmup.sum()),
    )
    return z, warmup


def _warmup_mask(users: pd.Series, min_history: int) -> np.ndarray:
    """True for each user's first *min_history* rows, in the given order."""
    position = users.groupby(users, sort=False).cumcount()
    return (position < min_history).to_numpy()


def _require_features(counts: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in counts.columns]
    if missing:
        raise BaselineError(
            f"input is missing feature column(s) {missing}; Stage 2 must emit "
            f"all {len(FEATURE_NAMES)} of FEATURE_NAMES"
        )


def build_dataset(
    labelled: pd.DataFrame,
    features: Sequence[str] = FEATURE_NAMES,
    min_history: int = MIN_HISTORY_DAYS,
    active_only: bool = STD_FLOOR_ACTIVE_USERS_ONLY,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """The whole stage: labelled counts in, z-scored dataset out.

    The output has the *same columns in the same order* as the input. Only the
    meaning of the 18 feature values changes. That is deliberate -- the column
    order is the CNN's channel order, and the two files are told apart by
    filename, never by schema.

    Returns:
        ``(dataset, floors, warmup)``.
    """
    columns = list(features)
    floors = std_floors(labelled, columns, active_only=active_only)
    z, warmup = causal_z_scores(labelled, floors, columns, min_history)

    dataset = labelled.copy()
    for name in columns:
        dataset[name] = z[name]
    return dataset[list(labelled.columns)], floors, warmup


def verify(
    dataset: pd.DataFrame,
    labelled: pd.DataFrame,
    warmup: np.ndarray,
    features: Sequence[str] = FEATURE_NAMES,
) -> None:
    """Raise on anything that cannot be true of a correct z-scored table.

    These are impossibilities, not surprises. A surprising *measurement* -- a
    weak insider, a big |z| count -- belongs in the report, where a human looks
    at it. A NaN, a lost row or a positive day hidden inside a warm-up is a bug,
    and the run stops.

    Raises:
        BaselineError: An invariant was violated.
    """
    columns = list(features)

    if len(dataset) != len(labelled):
        raise BaselineError(
            f"row count changed: {len(labelled):,} in, {len(dataset):,} out"
        )
    if list(dataset.columns) != list(labelled.columns):
        raise BaselineError(
            f"column layout changed.\n  in : {list(labelled.columns)}\n"
            f"  out: {list(dataset.columns)}"
        )

    values = dataset[columns].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        n_nan = int(np.isnan(values).sum())
        n_inf = int(np.isinf(values).sum())
        culprits = [
            name for name in columns
            if not np.isfinite(dataset[name].to_numpy(dtype="float64")).all()
        ]
        raise BaselineError(
            f"non-finite z-scores: {n_nan} NaN, {n_inf} inf, in column(s) "
            f"{culprits}. This is the stage where the std floor or the "
            f"variance formula shows up if either is wrong."
        )

    for column in (LABEL_COLUMN, SCENARIO_COLUMN):
        if not dataset[column].equals(labelled[column]):
            raise BaselineError(f"{column!r} changed during z-scoring")

    if not (values[warmup] == 0.0).all():
        raise BaselineError("warm-up rows are not exactly zero")

    hidden = int((labelled[LABEL_COLUMN].to_numpy() == 1)[warmup].sum())
    if hidden:
        raise BaselineError(
            f"{hidden} positive day(s) fall inside a user's {MIN_HISTORY_DAYS}-day "
            f"warm-up and were zeroed. On r4.2 this must be 0 -- the earliest "
            f"attack window opens 2010-06-10 and the data starts 2010-01-02."
        )

    if not (values < 0).any():
        raise BaselineError(
            "no negative z-scores anywhere, so the values are still raw counts"
        )


def collapse(
    dataset: pd.DataFrame, features: Sequence[str] = FEATURE_NAMES
) -> pd.DataFrame:
    """Reduce each day's 18 z-scores to a max and a mean of ``|z|``.

    Both, always, side by side. Collapsing 18 numbers into 1 is ambiguous and
    **the conclusion flips depending on which you pick**: the mean dilutes a
    sharp three-feature attack across fifteen features that had no reason to
    move, while the max is noisy enough that an ordinary day clears 2 sigma. The
    ambiguity is a real finding about the problem, so it is reported rather than
    resolved quietly, and every threshold below names the column it applies to.
    """
    magnitude = dataset[list(features)].abs()
    return pd.DataFrame(
        {DAY_MAX: magnitude.max(axis=1), DAY_MEAN: magnitude.mean(axis=1)},
        index=dataset.index,
    )


def validation_gate(
    dataset: pd.DataFrame,
    warmup: np.ndarray,
    features: Sequence[str] = FEATURE_NAMES,
    threshold: float = VALIDATION_Z_THRESHOLD,
) -> pd.DataFrame:
    """Per insider: does their attack window score above their own baseline?

    This is the most valuable thing Stage 3 produces. It is run *before* any
    model exists because no downstream architecture can invent a separation
    that is not in the data, and finding that out on the last day is the one
    failure mode with no recovery.

    Args:
        dataset: The z-scored table.
        warmup: The mask returned by :func:`causal_z_scores`.
        features: Columns to collapse over.
        threshold: The bar, applied to the ``attack_*`` columns.

    Returns:
        One row per insider: ``user``, ``scenario``, ``days``, ``window_days``,
        ``pre_attack_max``/``attack_max``/``lift_max``/``passes_max`` and the
        same four for ``_mean``, plus ``peak`` and ``late_ratio``.

        ``pre_attack_*`` **excludes the user's warm-up days.** Those are exactly
        0 by construction, and letting ~30 structural zeros into a ~150-day
        pre-attack average would deflate the baseline and inflate every lift in
        the table by a fifth. The gate should not flatter itself.

        ``late_ratio`` is the mean ``day_max`` over the last third of the window
        divided by the mean over the first third. Below 1 means the attack is
        being absorbed into the baseline it is measured against -- expected for
        scenario 2, and the reason detection latency matters.
    """
    collapsed = collapse(dataset, features)
    frame = pd.DataFrame(
        {
            USER_COLUMN: dataset[USER_COLUMN].to_numpy(),
            DAY_COLUMN: dataset[DAY_COLUMN].to_numpy(),
            SCENARIO_COLUMN: dataset[SCENARIO_COLUMN].to_numpy(),
            LABEL_COLUMN: dataset[LABEL_COLUMN].to_numpy(),
            WARMUP_COLUMN: warmup,
            DAY_MAX: collapsed[DAY_MAX].to_numpy(),
            DAY_MEAN: collapsed[DAY_MEAN].to_numpy(),
        }
    )

    insiders = frame.loc[frame[SCENARIO_COLUMN] != BENIGN_SCENARIO]
    rows = [
        summary
        for _, mine in insiders.groupby(USER_COLUMN, sort=True)
        if (summary := _gate_row(mine, threshold)) is not None
    ]
    return pd.DataFrame(rows)


def _gate_row(mine: pd.DataFrame, threshold: float) -> dict[str, object] | None:
    """One insider's line of the gate, or None if they have no window."""
    window = mine.loc[mine[LABEL_COLUMN] == 1]
    if window.empty:
        logger.warning(
            "insider %s has no label=1 rows and is absent from the gate",
            mine[USER_COLUMN].iloc[0],
        )
        return None

    before = mine.loc[
        (mine[DAY_COLUMN] < window[DAY_COLUMN].min()) & ~mine[WARMUP_COLUMN]
    ]
    row: dict[str, object] = {
        USER_COLUMN: mine[USER_COLUMN].iloc[0],
        "scenario": int(mine[SCENARIO_COLUMN].iloc[0]),
        "days": len(mine),
        "window_days": len(window),
        "peak": float(window[DAY_MAX].max()),
        "late_ratio": _late_ratio(window),
    }
    for suffix, column in (("max", DAY_MAX), ("mean", DAY_MEAN)):
        pre = float(before[column].mean()) if len(before) else float("nan")
        attack = float(window[column].mean())
        row[f"pre_attack_{suffix}"] = pre
        row[f"attack_{suffix}"] = attack
        row[f"lift_{suffix}"] = attack / pre if pre > 0 else float("nan")
        row[f"passes_{suffix}"] = bool(attack > threshold)
    return row


def benign_exceedance(
    dataset: pd.DataFrame,
    warmup: np.ndarray,
    features: Sequence[str] = FEATURE_NAMES,
    threshold: float = VALIDATION_Z_THRESHOLD,
) -> dict[str, float]:
    """What fraction of ordinary days already clears the bar.

    Without this number "54 of 70 insiders pass 2 sigma" is unreadable. If a
    tenth of every benign employee's days also clear it, passing means the
    insider is loud *and* the threshold is cheap, and any claim about the gate
    has to be made in that light. Computed over benign users' scored days only
    -- warm-up rows are structurally 0 and would flatter the rate.
    """
    collapsed = collapse(dataset, features)
    scored = (dataset[SCENARIO_COLUMN].to_numpy() == BENIGN_SCENARIO) & ~warmup
    if not scored.any():
        return {"days": 0, "max": float("nan"), "mean": float("nan")}
    return {
        "days": float(scored.sum()),
        "max": float((collapsed[DAY_MAX].to_numpy()[scored] > threshold).mean()),
        "mean": float((collapsed[DAY_MEAN].to_numpy()[scored] > threshold).mean()),
    }


def _late_ratio(window: pd.DataFrame) -> float:
    """Mean ``day_max`` over the window's last third, divided by its first third.

    The expanding baseline absorbs a long attack: a 56-day scenario-2 window
    against ~150 days of prior history measurably drags the mean it is being
    compared to, so late attack days score lower than early ones. This number
    measures that decay instead of asserting it, and it is a direct argument for
    Model B -- a 28-day context window still contains the loud early days after
    the late ones have faded.

    NaN for windows shorter than :data:`MIN_DECAY_WINDOW_DAYS`. Scenario 3's
    windows are two days long, where "last third over first third" degenerates
    into "day two over day one" -- a number that would print convincingly and
    mean nothing. Absorption is not measurable on a two-day attack, and saying
    so is better than reporting noise.
    """
    if len(window) < MIN_DECAY_WINDOW_DAYS:
        return float("nan")
    third = max(len(window) // 3, 1)
    ordered = window.sort_values(DAY_COLUMN)
    early = float(ordered[DAY_MAX].iloc[:third].mean())
    late = float(ordered[DAY_MAX].iloc[-third:].mean())
    return late / early if early > 0 else float("nan")


@dataclass(frozen=True, slots=True)
class BaselineStats:
    """The numbers the report prints, gathered in one pass."""

    rows_in: int
    rows_out: int
    users: int
    warmup_rows: int
    positives: int
    positives_in_warmup: int
    z_min: float
    z_max: float
    mean_abs_z: float
    rows_over_clip: int
    nan_count: int
    inf_count: int
    has_negative: bool


def compute_stats(
    dataset: pd.DataFrame,
    labelled: pd.DataFrame,
    warmup: np.ndarray,
    features: Sequence[str] = FEATURE_NAMES,
) -> BaselineStats:
    values = dataset[list(features)].to_numpy(dtype="float64")
    magnitude = np.abs(values)
    labels = labelled[LABEL_COLUMN].to_numpy()

    return BaselineStats(
        rows_in=len(labelled),
        rows_out=len(dataset),
        users=dataset[USER_COLUMN].nunique(),
        warmup_rows=int(warmup.sum()),
        positives=int(labels.sum()),
        positives_in_warmup=int((labels == 1)[warmup].sum()),
        z_min=float(np.nanmin(values)) if values.size else 0.0,
        z_max=float(np.nanmax(values)) if values.size else 0.0,
        mean_abs_z=float(np.nanmean(magnitude)) if values.size else 0.0,
        rows_over_clip=int((magnitude > Z_CLIP).any(axis=1).sum()),
        nan_count=int(np.isnan(values).sum()),
        inf_count=int(np.isinf(values).sum()),
        has_negative=bool((values < 0).any()),
    )


def format_report(
    dataset: pd.DataFrame,
    labelled: pd.DataFrame,
    warmup: np.ndarray,
    floors: pd.Series,
    gate: pd.DataFrame,
    out: Path | None = None,
    features: Sequence[str] = FEATURE_NAMES,
    active_only: bool = STD_FLOOR_ACTIVE_USERS_ONLY,
) -> str:
    """The whole stage's diagnostics, in one block, printed on every run."""
    stats = compute_stats(dataset, labelled, warmup, features)
    benign = benign_exceedance(dataset, warmup, features)
    lines = [
        _RULE,
        "InnerSight Stage 3 - personal baseline (causal z-scores)",
        _RULE,
        f"Rows in / out            : {stats.rows_in:,} -> {stats.rows_out:,}",
        f"Distinct users           : {stats.users:,}",
        f"Warm-up rows zeroed      : {stats.warmup_rows:,} "
        f"(first {MIN_HISTORY_DAYS} days of each user)",
        f"Positive rows            : {stats.positives:,}",
        f"Positives inside warm-up : {stats.positives_in_warmup}"
        f"{'   <- MUST be 0' if stats.positives_in_warmup else '   ok'}",
        f"z range                  : {stats.z_min:,.2f} .. {stats.z_max:,.2f}",
        f"mean |z|                 : {stats.mean_abs_z:.3f}",
        f"Rows with any |z| > {Z_CLIP:g}   : {stats.rows_over_clip:,} "
        f"({stats.rows_over_clip / max(stats.rows_out, 1):.2%}; if huge, the "
        f"floor is too small)",
        f"NaN / inf                : {stats.nan_count} / {stats.inf_count}",
        f"Negative values present  : {'yes' if stats.has_negative else 'NO'}"
        f"   <- proves z-scoring actually happened",
        "",
        f"Std floor per feature ({STD_FLOOR_MULTIPLIER:g} x median per-user std, "
        f"over {'users who exercise the feature' if active_only else 'ALL users'}):",
        *_floor_lines(floors, features),
        "",
        f"Validation gate ({len(gate)} insiders) - threshold |z| > "
        f"{VALIDATION_Z_THRESHOLD:g}",
        "  pre-attack averages exclude warm-up days (they are structurally 0)",
        *_gate_summary_lines(gate),
        "",
        "What a 'pass' is worth - the same threshold on BENIGN users' scored days:",
        f"  {benign['days']:,.0f} benign days: {benign['max']:.2%} already exceed "
        f"2 sigma on max|z|, {benign['mean']:.3%} on mean|z|",
        "  so the max collapse is a loose bar and the mean collapse is a strict "
        "one; neither is 'the' answer",
        "",
        "Per-insider table:",
        *_gate_table_lines(gate),
        _RULE,
    ]
    if out is not None:
        lines.append(f"Written to: {out}")
    return "\n".join(lines)


def _floor_lines(
    floors: pd.Series, features: Sequence[str] = FEATURE_NAMES
) -> list[str]:
    return [
        f"  {index:>2} {name:<26} {float(floors[name]):>12,.4f}"
        for index, name in enumerate(features)
    ]


def _gate_summary_lines(gate: pd.DataFrame) -> list[str]:
    """Per scenario: how many insiders clear the bar, under each collapse.

    Read the two "passing" columns as a pair. They disagree, and the
    disagreement is the point -- see :func:`collapse`.
    """
    header = (
        f"  {'scenario':<10}{'insiders':>10}{'pass (max)':>12}"
        f"{'pass (mean)':>13}{'median lift (max)':>20}{'median late/early':>20}"
    )
    lines = [header]
    if gate.empty:
        return [*lines, "  (no insiders in this table)"]

    for scenario in (*SCENARIOS, "all"):
        mine = gate if scenario == "all" else gate.loc[gate["scenario"] == scenario]
        if mine.empty:
            lines.append(f"  {str(scenario):<10}{0:>10}{'-':>12}{'-':>13}"
                         f"{'-':>20}{'-':>20}")
            continue
        lines.append(
            f"  {str(scenario):<10}{len(mine):>10}"
            f"{int(mine['passes_max'].sum()):>12}"
            f"{int(mine['passes_mean'].sum()):>13}"
            f"{mine['lift_max'].median():>20.2f}"
            f"{mine['late_ratio'].median():>20.2f}"
        )

    below = gate.loc[~gate["passes_max"], USER_COLUMN].tolist()
    lines.append(
        f"  below threshold on max|z|: {', '.join(below) if below else 'none'}"
    )
    return lines


def _gate_table_lines(gate: pd.DataFrame) -> list[str]:
    if gate.empty:
        return ["  (none)"]
    shown = gate[[
        USER_COLUMN, "scenario", "window_days",
        "pre_attack_max", "attack_max", "lift_max", "passes_max",
        "pre_attack_mean", "attack_mean", "lift_mean", "passes_mean",
        "peak", "late_ratio",
    ]].sort_values(["scenario", "attack_max"], ascending=[True, False])
    body = shown.to_string(index=False, float_format=lambda v: f"{v:,.2f}")
    return [f"  {line}" for line in body.splitlines()]


def format_user_report(
    dataset: pd.DataFrame,
    labelled: pd.DataFrame,
    user: str,
    rows: int = 20,
) -> str:
    """One user's most notable days: raw counts and z-scores, side by side.

    This view is the clearest single illustration of the whole project. The
    same row shows "2 USB connections" -- a nothing of a number -- and the
    z-score that says it is the most abnormal thing this person has done in
    seventeen months. Days are ranked by peak |z| rather than shown from the
    start, because an attack is rarely in January, and positive days are pinned
    into the selection so a quiet day inside the window is still visible.
    """
    mask = (dataset[USER_COLUMN] == user).to_numpy()
    if not mask.any():
        sample = ", ".join(sorted(dataset[USER_COLUMN].unique())[:5])
        return f"No rows for user {user!r}. Users look like: {sample}"

    z_rows = dataset.loc[mask]
    raw_rows = labelled.loc[mask]
    peak = z_rows[list(FEATURE_NAMES)].abs().max(axis=1)
    ranked = peak.where(z_rows[LABEL_COLUMN] == 0, peak + peak.max() + 1)
    chosen = ranked.sort_values(ascending=False).index[:rows]

    window = z_rows.loc[z_rows[LABEL_COLUMN] == 1, list(FEATURE_NAMES)]
    driver = (
        window.abs().mean().idxmax() if not window.empty
        else z_rows[list(FEATURE_NAMES)].abs().mean().idxmax()
    )

    peak_feature = z_rows.loc[chosen, list(FEATURE_NAMES)].abs().idxmax(axis=1)
    peak_raw = pd.Series(
        [
            raw_rows.loc[index, name]
            for index, name in zip(chosen, peak_feature, strict=True)
        ],
        index=chosen,
    )

    table = pd.DataFrame(
        {
            "day": z_rows.loc[chosen, DAY_COLUMN].dt.strftime("%Y-%m-%d"),
            "label": z_rows.loc[chosen, LABEL_COLUMN],
            f"raw_{driver}": raw_rows.loc[chosen, driver],
            f"z_{driver}": z_rows.loc[chosen, driver].round(2),
            "raw_usb": raw_rows.loc[chosen, "usb_connect_count"],
            "z_usb": z_rows.loc[chosen, "usb_connect_count"].round(2),
            "peak_feature": peak_feature,
            "peak_raw": peak_raw,
            "peak_abs_z": peak.loc[chosen].round(2),
        }
    ).sort_values("day")

    scenario = int(z_rows[SCENARIO_COLUMN].iloc[0])
    positives = int(z_rows[LABEL_COLUMN].sum())
    return "\n".join([
        _RULE,
        f"{user} - scenario {scenario}, {len(z_rows):,} days, "
        f"{positives} labelled 1, biggest in-window mover: {driver}",
        f"showing the {len(table)} days with the largest peak |z| "
        f"(labelled days always included)",
        _RULE,
        table.to_string(index=False),
        _RULE,
    ])


def main(argv: Sequence[str] | None = None) -> int:
    """Z-score the labelled counts, write the dataset, print the diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sample", action="store_true",
        help="Read the 30-user sample table and write a sample-named output.",
    )
    parser.add_argument(
        "--counts", type=Path, default=None,
        help="Labelled counts parquet to read. Defaults to the full/sample path.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Parquet to write. Defaults to the full/sample dataset path.",
    )
    parser.add_argument(
        "--user", default=None,
        help="After scoring, print this user's raw counts beside their z-scores.",
    )
    parser.add_argument(
        "--gate-csv", type=Path, default=None,
        help="Also write the per-insider validation gate table to this CSV.",
    )
    parser.add_argument(
        "--floor-all-users", action="store_true",
        help=(
            "Compute the std floor over all users instead of only those who "
            "exercise the feature. Reproduces the weaker variant discussed in "
            "config.STD_FLOOR_ACTIVE_USERS_ONLY; use with --out to compare."
        ),
    )
    args = parser.parse_args(argv)
    active_only = not args.floor_all_users

    counts_path = args.counts or (
        LABELLED_COUNTS_SAMPLE_PATH if args.sample else LABELLED_COUNTS_PATH
    )
    out_path = args.out or (
        DATASET_SAMPLE_PATH if args.sample else DATASET_PATH
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not counts_path.is_file():
        logger.error(
            "labelled counts not found: %s. Run `python -m labeling.label%s` first.",
            counts_path, " --sample" if args.sample else "",
        )
        return 1

    logger.info("reading %s", counts_path)
    labelled = pd.read_parquet(counts_path)

    dataset, floors, warmup = build_dataset(labelled, active_only=active_only)
    verify(dataset, labelled, warmup)
    gate = validation_gate(dataset, warmup)

    ensure_directories()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    logger.info("wrote %d rows to %s", len(dataset), out_path)

    if args.gate_csv is not None:
        args.gate_csv.parent.mkdir(parents=True, exist_ok=True)
        gate.to_csv(args.gate_csv, index=False)
        logger.info("wrote the validation gate to %s", args.gate_csv)

    print(
        format_report(
            dataset, labelled, warmup, floors, gate,
            out=out_path, active_only=active_only,
        )
    )
    if args.user:
        print()
        print(format_user_report(dataset, labelled, args.user))
    return 0


if __name__ == "__main__":
    sys.exit(main())
