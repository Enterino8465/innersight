"""Tests for Stage 2.

Short on purpose, and aimed at the failures that are wrong *silently* -- no
exception, no NaN, just a label column that is subtly off and a model that
appears to train fine:

1. The window boundary. Inclusive at both ends, compared at day granularity, so
   a 65-minute window still owns its whole calendar day.
2. The release filter. ``dataset`` read as a float rather than a string labels
   users from other CERT releases and misses this one's.
3. The calendar fill. Right number of inserted days, per user, zeros not NaNs,
   and nothing beyond a user's own first and last observed day.
4. The two columns meaning different things: an insider's ordinary day is
   ``label=0`` with ``user_scenario`` still non-zero.

Run with ``python -m pytest labeling/test_label.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import FEATURE_NAMES
from labeling.label import (
    GroundTruthError,
    LabellingError,
    attach_labels,
    build_labelled_counts,
    calendar_fill,
    load_insiders,
    verify,
)

FEATURES = list(FEATURE_NAMES)


def _counts(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Rows of (user, day) as a Stage 1 table with all features set to 1."""
    frame = pd.DataFrame(rows, columns=["user", "day"])
    frame["day"] = pd.to_datetime(frame["day"])
    for feature in FEATURES:
        frame[feature] = 1
    frame["total_email_size"] = frame["total_email_size"].astype("float64")
    return frame


def _windows(rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    """Rows of (user, scenario, start_day, end_day) as parsed windows."""
    frame = pd.DataFrame(rows, columns=["user", "scenario", "start_day", "end_day"])
    for column in ("start_day", "end_day"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def _insiders_csv(path, rows: list[tuple[str, str, str, str, str]]):
    """Write a minimal insiders.csv of (dataset, scenario, user, start, end)."""
    frame = pd.DataFrame(
        [(d, s, "r.csv", u, a, b) for d, s, u, a, b in rows],
        columns=["dataset", "scenario", "details", "user", "start", "end"],
    )
    frame.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Labelling rule
# --------------------------------------------------------------------------


def test_window_is_inclusive_at_both_ends() -> None:
    """The first and last day of a window are positives, the neighbours are not."""
    daily = _counts([("AAA0001", f"2010-07-{d:02d}") for d in range(5, 12)])
    windows = _windows([("AAA0001", 1, "2010-07-07", "2010-07-09")])

    labelled = attach_labels(daily, windows)
    by_day = dict(zip(labelled["day"].dt.strftime("%m-%d"), labelled["label"]))

    assert by_day["07-06"] == 0
    assert by_day["07-07"] == 1, "start day must be inside the window"
    assert by_day["07-08"] == 1
    assert by_day["07-09"] == 1, "end day must be inside the window"
    assert by_day["07-10"] == 0


def test_a_window_shorter_than_a_day_still_labels_that_day() -> None:
    """BIH0745's real window is 65 minutes on 2010-07-13. It owns the whole day."""
    daily = _counts([("BIH0745", f"2010-07-{d:02d}") for d in (12, 13, 14)])
    windows = _windows([("BIH0745", 1, "2010-07-13", "2010-07-13")])

    labelled = attach_labels(daily, windows)

    assert labelled["label"].tolist() == [0, 1, 0]


def test_insiders_ordinary_day_is_negative_but_still_carries_the_scenario() -> None:
    """The two columns answer different questions and must not agree."""
    daily = _counts([("AAF0535", "2010-03-02"), ("AAF0535", "2010-07-01")])
    windows = _windows([("AAF0535", 2, "2010-06-28", "2010-08-20")])

    labelled = attach_labels(daily, windows)

    assert labelled["label"].tolist() == [0, 1]
    assert labelled["user_scenario"].tolist() == [2, 2], (
        "user_scenario is per USER and stays set on benign days -- Stage 4 "
        "stratifies the split by it"
    )


def test_benign_user_is_all_zeros_in_both_columns() -> None:
    daily = _counts([("ZZZ9999", "2010-07-07"), ("ZZZ9999", "2010-07-08")])
    windows = _windows([("AAA0001", 1, "2010-07-07", "2010-07-09")])

    labelled = attach_labels(daily, windows)

    assert labelled["label"].sum() == 0
    assert labelled["user_scenario"].sum() == 0
    assert labelled["user_scenario"].dtype == "int8"


def test_a_window_only_labels_its_own_user() -> None:
    """Same day, two users: only the named one is positive."""
    daily = _counts([("AAA0001", "2010-07-07"), ("BBB0002", "2010-07-07")])
    windows = _windows([("AAA0001", 3, "2010-07-07", "2010-07-08")])

    labelled = attach_labels(daily, windows).set_index("user")

    assert labelled.loc["AAA0001", "label"] == 1
    assert labelled.loc["BBB0002", "label"] == 0


# --------------------------------------------------------------------------
# Ground truth loading
# --------------------------------------------------------------------------


def test_release_filter_compares_dataset_as_a_string(tmp_path) -> None:
    """4.2 and 4.20 are different releases; a float read cannot tell them apart."""
    path = _insiders_csv(
        tmp_path / "insiders.csv",
        [
            ("4.2", "1", "KPC0073", "07/07/2010 20:05:29", "07/15/2010 03:08:16"),
            ("5.2", "2", "OTHER001", "07/07/2010 20:05:29", "07/15/2010 03:08:16"),
            ("4.20", "3", "OTHER002", "07/07/2010 20:05:29", "07/15/2010 03:08:16"),
        ],
    )
    windows = load_insiders(path)

    assert windows["user"].tolist() == ["KPC0073"]
    assert windows["scenario"].tolist() == [1]


def test_timestamps_are_normalised_to_midnight(tmp_path) -> None:
    path = _insiders_csv(
        tmp_path / "insiders.csv",
        [("4.2", "1", "KPC0073", "07/07/2010 20:05:29", "07/15/2010 03:08:16")],
    )
    window = load_insiders(path).iloc[0]

    assert window["start_day"] == pd.Timestamp("2010-07-07")
    assert window["end_day"] == pd.Timestamp("2010-07-15")


def test_a_release_with_no_rows_is_an_error_not_an_empty_table(tmp_path) -> None:
    """Silently labelling nothing would train a model on 0 positives."""
    path = _insiders_csv(
        tmp_path / "insiders.csv",
        [("5.2", "2", "OTHER001", "07/07/2010 20:05:29", "07/15/2010 03:08:16")],
    )
    with pytest.raises(GroundTruthError, match="4.2"):
        load_insiders(path)


def test_one_user_under_two_scenarios_is_rejected(tmp_path) -> None:
    path = _insiders_csv(
        tmp_path / "insiders.csv",
        [
            ("4.2", "1", "AAA0001", "07/07/2010 20:05:29", "07/09/2010 03:08:16"),
            ("4.2", "2", "AAA0001", "09/07/2010 20:05:29", "09/09/2010 03:08:16"),
        ],
    )
    with pytest.raises(GroundTruthError, match="more than one scenario"):
        load_insiders(path)


# --------------------------------------------------------------------------
# Calendar fill
# --------------------------------------------------------------------------


def test_fill_inserts_exactly_the_missing_calendar_days() -> None:
    """Fri, Mon, Tue observed -> Sat and Sun inserted, nothing else."""
    daily = _counts(
        [("AAA0001", "2010-01-08"), ("AAA0001", "2010-01-11"), ("AAA0001", "2010-01-12")]
    )
    filled = calendar_fill(daily)

    assert len(filled) == 5
    assert filled["day"].tolist() == list(
        pd.date_range("2010-01-08", "2010-01-12", freq="D")
    )
    weekend = filled.loc[filled["day"].isin(pd.to_datetime(["2010-01-09", "2010-01-10"]))]
    assert (weekend[FEATURES] == 0).all().all(), "inserted days are zeros, not NaNs"
    assert filled.isna().sum().sum() == 0


def test_fill_is_per_user_and_never_beyond_their_own_span() -> None:
    """A user who left in January is not expanded to someone else's May."""
    daily = _counts(
        [
            ("EARLY001", "2010-01-04"),
            ("EARLY001", "2010-01-06"),
            ("LATER002", "2010-05-10"),
            ("LATER002", "2010-05-12"),
        ]
    )
    filled = calendar_fill(daily).groupby("user")["day"].agg(["min", "max", "size"])

    assert filled.loc["EARLY001", "size"] == 3
    assert filled.loc["EARLY001", "max"] == pd.Timestamp("2010-01-06")
    assert filled.loc["LATER002", "size"] == 3
    assert filled.loc["LATER002", "min"] == pd.Timestamp("2010-05-10")


def test_fill_preserves_dtypes_and_existing_values() -> None:
    daily = _counts([("AAA0001", "2010-01-04"), ("AAA0001", "2010-01-06")])
    filled = calendar_fill(daily)

    assert filled["logon_count"].dtype == daily["logon_count"].dtype
    assert filled["total_email_size"].dtype == "float64"
    observed = filled.loc[filled["day"] != pd.Timestamp("2010-01-05")]
    assert (observed[FEATURES] == 1).all().all()


def test_fill_makes_a_silent_day_labellable() -> None:
    """The whole reason the fill runs before the labelling.

    An insider who has stopped coming in generates no events, so Stage 1 emits
    no row -- and an unfilled table cannot mark a day that does not exist.
    """
    daily = _counts([("AAF0535", "2010-06-28"), ("AAF0535", "2010-07-05")])
    windows = _windows([("AAF0535", 2, "2010-06-28", "2010-07-05")])

    unfilled = attach_labels(daily, windows)
    filled = build_labelled_counts(daily, windows)

    assert unfilled["label"].sum() == 2
    assert filled["label"].sum() == 8, "the six silent days are inside the window"


def test_duplicate_user_days_are_rejected_rather_than_multiplied() -> None:
    daily = _counts([("AAA0001", "2010-01-04"), ("AAA0001", "2010-01-04")])

    with pytest.raises(LabellingError, match="duplicate"):
        calendar_fill(daily)


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------


def test_verify_rejects_a_positive_day_on_a_benign_user() -> None:
    """The join error that would poison every downstream metric."""
    daily = _counts([("AAA0001", "2010-07-07")])
    labelled = attach_labels(daily, _windows([]))
    labelled.loc[0, "label"] = 1

    with pytest.raises(LabellingError, match="user_scenario=0"):
        verify(labelled, daily)


def test_verify_accepts_a_correctly_labelled_table() -> None:
    daily = _counts([("AAA0001", f"2010-07-{d:02d}") for d in range(5, 12)])
    windows = _windows([("AAA0001", 1, "2010-07-07", "2010-07-09")])

    labelled = build_labelled_counts(daily, windows)
    verify(labelled, daily)  # must not raise

    assert list(labelled.columns) == ["user", "day", *FEATURE_NAMES, "user_scenario", "label"]
