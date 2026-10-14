"""Tests for Stage 3, chosen by the Stage 1/2 criterion.

Cover the failures that are **silent** -- no exception, no NaN, just a number
that is subtly wrong and survives all the way to the results table. A crash is
cheap; a baseline that quietly includes today in the mean it is measured
against costs a week and looks like a modelling problem.

Run from the project root::

    python -m pytest baseline/test_baseline.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baseline.baseline import (
    BaselineError,
    build_dataset,
    causal_z_scores,
    collapse,
    std_floors,
    validation_gate,
    verify,
)
from config import (
    DAY_COLUMN,
    FEATURE_NAMES,
    LABEL_COLUMN,
    MIN_HISTORY_DAYS,
    SCENARIO_COLUMN,
    STD_FLOOR_FALLBACK,
    USER_COLUMN,
)

FEATURES = list(FEATURE_NAMES)
PROBE = "usb_connect_count"
BIG = "total_email_size"


def make_user(
    user: str,
    values: dict[str, list[float]],
    days: int,
    start: str = "2010-01-01",
    scenario: int = 0,
    positives: tuple[int, ...] = (),
) -> pd.DataFrame:
    """A calendar-dense block of rows for one synthetic user.

    Every one of the 18 features is present and zero unless *values* names it,
    which keeps each test's intent readable: the column under test is the only
    one that moves.
    """
    frame = pd.DataFrame(
        {
            USER_COLUMN: user,
            DAY_COLUMN: pd.date_range(start, periods=days, freq="D"),
            **{name: np.zeros(days, dtype="float64") for name in FEATURES},
            SCENARIO_COLUMN: np.int8(scenario),
            LABEL_COLUMN: np.zeros(days, dtype="int8"),
        }
    )
    for name, series in values.items():
        frame[name] = np.asarray(series, dtype="float64")
    for position in positives:
        frame.loc[position, LABEL_COLUMN] = np.int8(1)
    return frame


def flat_floors(value: float = 1.0) -> pd.Series:
    """Explicit per-feature floors, so a test never depends on a population."""
    return pd.Series(value, index=FEATURES, dtype="float64")


# --------------------------------------------------------------------------
# 1. Score against the past, never including today
# --------------------------------------------------------------------------


def test_spike_is_scored_against_the_pre_spike_mean() -> None:
    """The single most important property of the stage.

    A constant history of 1.0 for 60 days, then 100 on day 61. If today leaks
    into the mean it is measured against, the numerator shrinks from 99 to
    99 * 60/61 -- a 1.6% error that no assertion downstream would ever notice,
    and one that grows to a factor of two on a short history.
    """
    history, spike = 1.0, 100.0
    days = 61
    counts = make_user(
        "A", {PROBE: [history] * (days - 1) + [spike]}, days=days
    )
    floors = flat_floors(0.5)

    z, _ = causal_z_scores(counts, floors, min_history=0)
    observed = float(z[PROBE].iloc[-1])

    # prior mean = 1.0 exactly, prior std = 0 -> the floor is the denominator.
    assert observed == pytest.approx((spike - history) / 0.5, rel=1e-5)
    # And explicitly not the value you get by including today.
    leaked_mean = (history * (days - 1) + spike) / days
    assert observed != pytest.approx((spike - leaked_mean) / 0.5, rel=1e-3)


def test_first_day_of_a_user_has_no_history_and_scores_zero() -> None:
    """With no prior days there is nothing to be unlike, so z must be 0.

    NaN would be defensible mathematically and fatal in practice -- it would
    reach the loss function of a model that has no idea it is looking at an
    undefined statistic.
    """
    counts = make_user("A", {PROBE: [7.0, 7.0, 7.0]}, days=3)
    z, _ = causal_z_scores(counts, flat_floors(), min_history=0)
    assert float(z[PROBE].iloc[0]) == 0.0


# --------------------------------------------------------------------------
# 2. Zero-variance history
# --------------------------------------------------------------------------


def test_constant_history_gives_a_finite_score_when_it_changes() -> None:
    """The accountant who has never touched a USB drive.

    Prior std is exactly 0 for 180 days. Without a floor this is a division by
    zero; the point of the floor is that the answer is large *and finite*,
    because a large finite number is the thing worth looking at.
    """
    days = 181
    counts = make_user("A", {PROBE: [0.0] * (days - 1) + [2.0]}, days=days)
    z, _ = causal_z_scores(counts, flat_floors(0.1), min_history=0)

    observed = float(z[PROBE].iloc[-1])
    assert np.isfinite(observed)
    assert observed == pytest.approx(20.0, rel=1e-5)


def test_a_feature_constant_for_everybody_stays_at_zero() -> None:
    """``large_attachment_count`` is identically zero on r4.2.

    Its floor falls back to 1.0 rather than to 0, so a dead feature contributes
    exactly nothing instead of dividing by nothing.
    """
    counts = pd.concat(
        [make_user(u, {}, days=40) for u in ("A", "B", "C")], ignore_index=True
    )
    floors = std_floors(counts)
    assert float(floors["large_attachment_count"]) == STD_FLOOR_FALLBACK

    z, _ = causal_z_scores(counts, floors, min_history=0)
    assert (z["large_attachment_count"] == 0.0).all()


def test_the_floor_binds_only_when_it_exceeds_the_prior_std() -> None:
    """A user with real variability is scored by their own std, not the floor."""
    values = [0.0, 10.0] * 30 + [10.0]
    counts = make_user("A", {PROBE: values}, days=len(values))
    z, _ = causal_z_scores(counts, flat_floors(0.001), min_history=0)

    prior = np.asarray(values[:-1])
    expected = (values[-1] - prior.mean()) / prior.std(ddof=1)
    assert float(z[PROBE].iloc[-1]) == pytest.approx(expected, rel=1e-4)


# --------------------------------------------------------------------------
# 3. Warm-up
# --------------------------------------------------------------------------


def test_warmup_rows_are_exactly_zero_and_the_next_row_is_not() -> None:
    """Invisible, not wrong. And the boundary is off-by-one bait."""
    days = MIN_HISTORY_DAYS + 5
    rng = np.random.default_rng(0)
    counts = make_user("A", {PROBE: rng.integers(0, 9, days).astype(float)}, days)

    z, warmup = causal_z_scores(counts, flat_floors(0.5))

    assert int(warmup.sum()) == MIN_HISTORY_DAYS
    assert (z.iloc[:MIN_HISTORY_DAYS].to_numpy() == 0.0).all()
    assert (z.iloc[MIN_HISTORY_DAYS:][PROBE].to_numpy() != 0.0).any()


def test_a_positive_day_inside_the_warmup_is_a_hard_failure() -> None:
    """It would be zeroed and silently undetectable, so the run must stop.

    On r4.2 this cannot happen -- the earliest window opens 2010-06-10 against
    data starting 2010-01-02 -- which is exactly why an assertion is the right
    shape: it costs nothing and it catches a changed release.
    """
    counts = make_user(
        "A", {PROBE: [5.0] * 40}, days=40, scenario=1, positives=(3,)
    )
    dataset, _, warmup = build_dataset(counts)
    with pytest.raises(BaselineError, match="warm-up"):
        verify(dataset, counts, warmup)


# --------------------------------------------------------------------------
# 4. No cross-user leakage
# --------------------------------------------------------------------------


def test_users_are_scored_only_against_themselves() -> None:
    """Two users at wildly different levels, interleaved in the table.

    The heavy user's ordinary day must not look anomalous because the light
    user exists, and the light user's small bump must still be loud. If the
    grouping were dropped, both would be scored against a pooled mean of ~500
    and the light user's spike would vanish.
    """
    heavy = make_user("HEAVY", {PROBE: [1000.0] * 40}, days=40)
    light = make_user("LIGHT", {PROBE: [1.0] * 39 + [3.0]}, days=40)
    counts = pd.concat([heavy, light], ignore_index=True)

    z, _ = causal_z_scores(counts, flat_floors(0.5), min_history=0)
    scored = z.assign(user=counts[USER_COLUMN])

    assert (scored.loc[scored["user"] == "HEAVY", PROBE] == 0.0).all()
    light_spike = float(scored.loc[scored["user"] == "LIGHT", PROBE].iloc[-1])
    assert light_spike == pytest.approx((3.0 - 1.0) / 0.5, rel=1e-5)


def test_row_order_survives_an_unsorted_input() -> None:
    """Causality is defined by date order; the output contract is row order.

    Both have to hold at once, so the stage sorts internally and restores. A
    version that only sorted would silently reorder Stage 4's rows; one that
    only preserved order would mis-score a shuffled table.
    """
    counts = pd.concat(
        [
            make_user("A", {PROBE: [1.0] * 39 + [9.0]}, days=40),
            make_user("B", {PROBE: [2.0] * 40}, days=40),
        ],
        ignore_index=True,
    )
    shuffled = counts.sample(frac=1.0, random_state=7).reset_index(drop=True)

    z_ordered, _ = causal_z_scores(counts, flat_floors(0.5), min_history=0)
    z_shuffled, _ = causal_z_scores(shuffled, flat_floors(0.5), min_history=0)

    rejoined = (
        shuffled[[USER_COLUMN, DAY_COLUMN]]
        .assign(z=z_shuffled[PROBE].to_numpy())
        .merge(
            counts[[USER_COLUMN, DAY_COLUMN]].assign(z=z_ordered[PROBE].to_numpy()),
            on=[USER_COLUMN, DAY_COLUMN],
            suffixes=("_shuffled", "_ordered"),
        )
    )
    assert np.allclose(rejoined["z_shuffled"], rejoined["z_ordered"], atol=1e-5)


# --------------------------------------------------------------------------
# 5. The output contract
# --------------------------------------------------------------------------


def test_shape_column_order_and_labels_survive_unchanged() -> None:
    """Same rows, same order, same layout. Only the 18 values change meaning.

    Column order is the CNN's channel order, so a reordering here raises
    nothing and invalidates every checkpoint later.
    """
    # The values must actually vary: verify() rejects a table with no negative
    # z-score anywhere, and a constant feature produces exactly zeros.
    counts = pd.concat(
        [
            make_user(
                "A", {PROBE: [1.0, 3.0] * 20}, days=40, scenario=1, positives=(35,)
            ),
            make_user("B", {PROBE: [4.0, 9.0] * 20}, days=40),
        ],
        ignore_index=True,
    )
    dataset, _, warmup = build_dataset(counts)

    assert len(dataset) == len(counts)
    assert list(dataset.columns) == list(counts.columns)
    assert dataset[LABEL_COLUMN].equals(counts[LABEL_COLUMN])
    assert dataset[SCENARIO_COLUMN].equals(counts[SCENARIO_COLUMN])
    assert dataset[DAY_COLUMN].equals(counts[DAY_COLUMN])
    assert dataset[USER_COLUMN].equals(counts[USER_COLUMN])
    assert all(dataset[name].dtype == "float32" for name in FEATURES)
    verify(dataset, counts, warmup)


def test_verify_rejects_a_table_that_was_never_z_scored() -> None:
    """No negative value anywhere means the raw counts were written through.

    A pass-through bug produces a file of exactly the right shape that trains
    exactly as well as noise, and nothing else in the pipeline would object.
    """
    counts = make_user("A", {PROBE: [1.0] * 40}, days=40)
    warmup = np.zeros(len(counts), dtype=bool)
    with pytest.raises(BaselineError, match="still raw counts"):
        verify(counts.copy(), counts, warmup)


# --------------------------------------------------------------------------
# 6. Numerical stability
# --------------------------------------------------------------------------


def test_values_around_1e8_produce_no_nan() -> None:
    """``total_email_size`` runs to ~1e8 on the real data.

    Under ``E[x^2] - E[x]^2`` the two terms are ~1e16 and their difference can
    come out negative in floating point, which becomes NaN under the square
    root. This is the test that fails if anybody rewrites the expanding
    aggregation as cumulative sums for speed.
    """
    rng = np.random.default_rng(1)
    values = 1e8 + rng.normal(0.0, 1e3, 200)
    counts = make_user("A", {BIG: values}, days=200)

    z, _ = causal_z_scores(counts, flat_floors(1.0), min_history=0)
    assert np.isfinite(z[BIG].to_numpy()).all()
    # And the scores are of a sane magnitude, not 1e8-sized garbage.
    assert np.abs(z[BIG].to_numpy()).max() < 100.0


# --------------------------------------------------------------------------
# The floor, and the gate
# --------------------------------------------------------------------------


def test_active_only_ignores_users_who_never_exercise_the_feature() -> None:
    """The measured correction: on r4.2, 73.5% of users never touch a USB port.

    Over all users the median per-user std of such a feature is 0 and the floor
    collapses to the fallback, which flattens the very signal the project is
    about. Over active users only it reflects real variability.
    """
    users = [make_user(f"Q{i}", {}, days=40) for i in range(8)]
    varied = make_user("VARIED", {PROBE: [0.0, 4.0] * 20}, days=40)
    counts = pd.concat([*users, varied], ignore_index=True)

    over_all = std_floors(counts, active_only=False)
    over_active = std_floors(counts, active_only=True)

    assert float(over_all[PROBE]) == STD_FLOOR_FALLBACK  # median std is 0
    assert 0.0 < float(over_active[PROBE]) < STD_FLOOR_FALLBACK


def test_the_gate_measures_the_window_against_the_days_before_it() -> None:
    """An insider whose window is loud must show lift; the warm-up must not.

    ``pre_attack`` deliberately excludes warm-up rows. They are exactly 0 by
    construction, and letting ~30 structural zeros into a pre-attack average
    would deflate the baseline and inflate every lift in the table.
    """
    days = 80
    # An ordinary life with a little day-to-day noise, then a loud fortnight.
    # The noise matters: with a perfectly constant history pre_attack is 0 and
    # the lift is infinite, which would make the test pass for the wrong reason.
    values = [1.0, 2.0] * 35 + [50.0] * 10
    counts = pd.concat(
        [
            make_user(
                "SPY", {PROBE: values}, days=days, scenario=1,
                positives=tuple(range(70, 80)),
            ),
            make_user("CALM", {PROBE: [1.0, 2.0] * 40}, days=days),
        ],
        ignore_index=True,
    )
    dataset, _, warmup = build_dataset(counts)
    gate = validation_gate(dataset, warmup)

    assert list(gate[USER_COLUMN]) == ["SPY"]  # benign users are not in the gate
    row = gate.iloc[0]
    assert row["window_days"] == 10
    assert row["attack_max"] > row["pre_attack_max"] > 0
    assert bool(row["passes_max"])


def test_collapse_reports_both_max_and_mean() -> None:
    """The two summaries disagree, and the disagreement is a finding.

    A sharp one-feature attack is loud under max and diluted to ~1/18 of that
    under mean. Reporting only one of them would let the write-up choose its
    own conclusion.
    """
    counts = make_user("A", {PROBE: [1.0] * 39 + [37.0]}, days=40)
    z, _ = causal_z_scores(counts, flat_floors(1.0), min_history=0)
    collapsed = collapse(z.assign(**{USER_COLUMN: "A"}))

    last = collapsed.iloc[-1]
    assert last.iloc[0] == pytest.approx(36.0, rel=1e-4)          # max
    assert last.iloc[1] == pytest.approx(36.0 / len(FEATURES), rel=1e-4)  # mean


def test_an_empty_input_does_not_crash() -> None:
    """Edge case worth one line: --counts pointed at a filtered, empty table."""
    empty = make_user("A", {}, days=0)
    z, warmup = causal_z_scores(empty, flat_floors())
    assert len(z) == 0
    assert warmup.size == 0
