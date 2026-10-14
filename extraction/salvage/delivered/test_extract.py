"""Tests for Stage 1.

Deliberately short. These cover the three behaviours that are wrong *silently*
-- no exception, no NaN, just a number that is a bit too big -- and would not be
noticed until the results made no sense:

1. A distinct count that straddles two chunks must not be double-counted.
2. A ``(user, day)`` present in one log and absent from another must come out as
   zero, not as a missing row and not as NaN.
3. External-recipient detection must look at ``cc`` and ``bcc``, and must not be
   fooled by a lookalike domain.

Run with ``python -m pytest test_extract.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import FEATURE_NAMES
from extract import (
    aggregate_log,
    build_email_partial,
    combine_daily_counts,
    daily_counts,
)

DAY = "2010-01-04"


def _logon_chunk(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Rows of (timestamp, user, pc, activity) as a parsed logon chunk."""
    frame = pd.DataFrame(rows, columns=["date", "user", "pc", "activity"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_distinct_pcs_are_not_double_counted_across_chunks() -> None:
    """PC-1 in two chunks of the same user-day is one distinct PC, not two."""
    chunks = [
        _logon_chunk([(f"{DAY} 08:00:00", "AAA0001", "PC-1", "Logon")]),
        _logon_chunk([(f"{DAY} 13:00:00", "AAA0001", "PC-1", "Logon")]),
        _logon_chunk([(f"{DAY} 15:00:00", "AAA0001", "PC-2", "Logon")]),
    ]
    result = aggregate_log("logon", chunks, compact_every=2)

    assert result["unique_pcs_used"].iloc[0] == 2
    assert result["logon_count"].iloc[0] == 3


def test_after_hours_and_weekend_count_logons_only() -> None:
    """A 02:00 logoff is not an after-hours logon; 2010-01-02 is a Saturday."""
    chunks = [
        _logon_chunk(
            [
                (f"{DAY} 02:00:00", "AAA0001", "PC-1", "Logoff"),
                (f"{DAY} 22:00:00", "AAA0001", "PC-1", "Logon"),
                ("2010-01-02 10:00:00", "AAA0001", "PC-1", "Logon"),
            ]
        )
    ]
    result = aggregate_log("logon", chunks).sort_index()

    assert result["after_hours_logons"].sum() == 1
    assert result["weekend_logons"].sum() == 1


def test_missing_log_days_become_zero_not_nan() -> None:
    """The key universe is the union of all logs; absence means zero."""
    logon = _logon_chunk([(f"{DAY} 08:00:00", "AAA0001", "PC-1", "Logon")])
    http = pd.DataFrame(
        {
            "date": pd.to_datetime(["2010-01-05 09:00:00"]),
            "user": ["BBB0002"],
            "url": ["http://indeed.com/jobs"],
        }
    )
    daily = daily_counts({"logon": [logon], "http": [http]})

    assert len(daily) == 2, "one row per (user, day) across the union of logs"
    assert daily.isna().sum().sum() == 0
    # BBB0002 has no logon log at all, so its logon features are a real zero.
    other = daily.loc[daily["user"] == "BBB0002"].iloc[0]
    assert other["logon_count"] == 0
    assert other["job_search_visits"] == 1


def test_column_order_is_the_config_vector_layout() -> None:
    """Reordering FEATURE_NAMES silently invalidates every checkpoint."""
    daily = daily_counts(
        {"logon": [_logon_chunk([(f"{DAY} 08:00:00", "AAA0001", "PC-1", "Logon")])]}
    )
    assert list(daily.columns) == ["user", "day", *FEATURE_NAMES]


@pytest.mark.parametrize(
    ("to", "cc", "bcc", "expected"),
    [
        ("a@dtaa.com", "", "", 0),
        ("a@dtaa.com", "", "me@gmail.com", 1),
        ("a@dtaa.com", "b@gmail.com", "", 1),
        # A lookalike host is external, not internal.
        ("a@dtaa.com.evil.net", "", "", 1),
    ],
)
def test_external_recipients_span_to_cc_and_bcc(
    to: str, cc: str, bcc: str, expected: int
) -> None:
    chunk = pd.DataFrame(
        {
            "date": pd.to_datetime([f"{DAY} 09:00:00"]),
            "user": ["AAA0001"],
            "to": [to],
            "cc": [cc],
            "bcc": [bcc],
            "size": ["1000"],
        }
    )
    partial = build_email_partial(chunk)
    assert partial.sums["email_to_external_count"].iloc[0] == expected


def test_empty_input_gives_an_empty_but_correctly_shaped_table() -> None:
    daily = combine_daily_counts({})
    assert list(daily.columns) == ["user", "day", *FEATURE_NAMES]
    assert daily.empty
