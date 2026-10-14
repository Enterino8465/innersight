"""Tests for Stage 1.

Deliberately short. These cover the behaviours that are wrong *silently* -- no
exception, no NaN, just a number that is a bit too big -- and would not be
noticed until the results made no sense:

1. A distinct count that straddles two chunks must not be double-counted.
2. A ``(user, day)`` present in one log and absent from another must come out as
   zero, not as a missing row and not as NaN.
3. External-recipient detection must look at ``cc`` and ``bcc``, and must not be
   fooled by a lookalike domain.

Brief R2 added a second class of silent wrongness, and the second half of this
file is about it. r5.2 ships columns r4.2 does not, and the danger is not that
the code breaks on them -- it is that it quietly computes a different feature
under the same name. So every R2 change is tested TWICE: once on an r4.2-shaped
frame, where it must be a no-op, and once on an r5.2-shaped one, where it must
bite. One test per change, so a failure names the culprit.

The r4.2-shaped tests are the load-bearing ones. Model A's AUPRC of 0.1903 and
Model B's 0.6663 describe the code that produced ``data/daily_counts.parquet``;
if any of these moves, those numbers stop describing this repository.

Run with ``python -m pytest extraction/ -q``. The tests that need a real
release folder live in ``test_extract_datasets.py`` next door.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from config import FEATURE_NAMES, LARGE_ATTACHMENT_BYTES, RELEASE_LOG_COLUMNS
from extraction.extract import (
    ExtractionError,
    _validated_header,
    aggregate_log,
    build_email_partial,
    combine_daily_counts,
    daily_counts,
    format_report,
)

DAY = "2010-01-04"
ROOT = Path(__file__).resolve().parent.parent


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


# --------------------------------------------------------------------------
# Brief R2: the four feature changes, each proved a no-op on r4.2's shape and
# proved to bite on r5.2's.
#
# Every one of them is driven by the COLUMNS the chunk carries, never by which
# release is active, so these frames are the whole story: an r4.2 email chunk
# is one with no `activity` column and an integer `attachments`, wherever it
# came from.
# --------------------------------------------------------------------------


def _email_chunk(rows: list[dict[str, str]]) -> pd.DataFrame:
    """An email chunk carrying whichever keys the rows carry.

    Absent keys stay absent, which is the point: leaving out ``activity`` is
    how a test says "r4.2" and putting it in is how it says "r5.2".
    """
    frame = pd.DataFrame(rows)
    frame.insert(0, "user", ["AAA0001"] * len(rows))
    frame.insert(0, "date", pd.to_datetime([f"{DAY} 09:00:00"] * len(rows)))
    return frame


def _file_chunk(rows: list[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.insert(0, "user", ["AAA0001"] * len(rows))
    frame.insert(0, "date", pd.to_datetime([f"{DAY} 09:00:00"] * len(rows)))
    return frame


def _email_totals(rows: list[dict[str, str]]) -> dict[str, float]:
    sums = build_email_partial(_email_chunk(rows)).sums
    return {column: sums[column].sum() for column in sums.columns}


def _file_totals(rows: list[dict[str, str]]) -> dict[str, float]:
    frame = aggregate_log("file", [_file_chunk(rows)])
    return {column: frame[column].sum() for column in frame.columns}


# --- Change 1: the Send filter (features 11, 12 and 14) --------------------


def test_without_an_activity_column_every_row_is_a_send() -> None:
    """r4.2's shape. Nothing here may move: this is what the frozen table is."""
    totals = _email_totals(
        [
            {"to": "a@dtaa.com", "cc": "", "bcc": "", "size": "100"},
            {"to": "b@gmail.com", "cc": "", "bcc": "", "size": "200"},
            {"to": "c@dtaa.com", "cc": "", "bcc": "", "size": "300"},
        ]
    )
    assert totals["email_sent_count"] == 3
    assert totals["email_to_external_count"] == 1
    assert totals["total_email_size"] == 600


def test_with_an_activity_column_only_sends_are_counted() -> None:
    """r5.2's shape. Unfiltered, email_sent_count is 3.1253x too large."""
    totals = _email_totals(
        [
            {"activity": "Send", "to": "a@dtaa.com", "size": "100"},
            {"activity": "Receive", "to": "b@dtaa.com", "size": "200"},
            {"activity": "Send", "to": "c@dtaa.com", "size": "300"},
        ]
    )
    assert totals["email_sent_count"] == 2
    assert totals["total_email_size"] == 400, "received bytes are not sent bytes"


def test_the_filter_runs_before_the_recipient_test() -> None:
    """Mail arriving FROM outside is not mail leaving the organisation.

    Filtering after the recipient test instead of before it would put the
    insider's inbox into email_to_external_count, which is the one feature
    whose whole job is outbound traffic.
    """
    totals = _email_totals(
        [
            {"activity": "Receive", "to": "victim@dtaa.com",
             "cc": "spy@gmail.com", "size": "100"},
            {"activity": "Send", "to": "spy@gmail.com", "cc": "", "size": "200"},
        ]
    )
    assert totals["email_to_external_count"] == 1


# --- Change 2: large_attachment_count (feature 13) -------------------------


def test_an_integer_attachments_column_yields_no_large_attachments() -> None:
    """r4.2's shape, and the sharpest test in the file.

    ``size`` here is five megabytes, far over the threshold. Before R2 that
    scored a hit, because the code tested the message size. It must not now:
    r4.2's ``attachments`` is a COUNT ("0" to "9", confirmed over all 2,629,979
    rows), there are no attachment sizes in it to find, and the feature is
    identically zero on r4.2 -- which is exactly what it was before.
    """
    totals = _email_totals(
        [
            {"to": "a@dtaa.com", "size": "5000000", "attachments": "3"},
            {"to": "b@dtaa.com", "size": "9000000", "attachments": "9"},
        ]
    )
    assert totals["large_attachment_count"] == 0


def test_a_missing_attachments_column_yields_no_large_attachments() -> None:
    """Defensive: a projection without the column must not raise."""
    totals = _email_totals([{"to": "a@dtaa.com", "size": "5000000"}])
    assert totals["large_attachment_count"] == 0


def test_the_largest_single_attachment_decides_not_their_total() -> None:
    """Ten ordinary documents are not one exfiltrated archive."""
    below = (
        f"a.doc({LARGE_ATTACHMENT_BYTES - 1});b.doc({LARGE_ATTACHMENT_BYTES - 1})"
    )
    totals = _email_totals(
        [{"activity": "Send", "to": "a@dtaa.com", "size": "9",
          "attachments": below}]
    )
    assert totals["large_attachment_count"] == 0


def test_an_attachment_at_exactly_the_threshold_counts() -> None:
    """LARGE_ATTACHMENT_BYTES is documented as "this size or larger"."""
    at = f"a.zip({LARGE_ATTACHMENT_BYTES})"
    totals = _email_totals(
        [{"activity": "Send", "to": "a@dtaa.com", "size": "9", "attachments": at}]
    )
    assert totals["large_attachment_count"] == 1


def test_a_large_attachment_that_arrived_does_not_count() -> None:
    """Receiving a big file is not sending one."""
    big = f"a.zip({LARGE_ATTACHMENT_BYTES * 3})"
    totals = _email_totals(
        [
            {"activity": "Receive", "to": "me@dtaa.com", "size": "9",
             "attachments": big},
            {"activity": "Send", "to": "you@dtaa.com", "size": "9",
             "attachments": big},
        ]
    )
    assert totals["large_attachment_count"] == 1


def test_a_chunk_with_no_parsable_attachment_at_all_is_zero() -> None:
    """The prefilter's empty case: no brackets anywhere, no crash, no hits."""
    totals = _email_totals(
        [
            {"activity": "Send", "to": "a@dtaa.com", "size": "9",
             "attachments": ""},
            {"activity": "Send", "to": "b@dtaa.com", "size": "9",
             "attachments": "0"},
        ]
    )
    assert totals["large_attachment_count"] == 0


# --- Change 3: unique_filenames (feature 10) -------------------------------


def test_bare_filenames_are_left_alone() -> None:
    """r4.2's shape: 0 of its 445,581 filenames contain a separator."""
    totals = _file_totals(
        [{"filename": "EYPC9Y08.doc"}, {"filename": "R4T7WPQ2.pdf"}]
    )
    assert totals["unique_filenames"] == 2
    assert totals["file_count"] == 2


def test_full_paths_are_counted_by_basename() -> None:
    """r5.2's shape. The same document fetched from two drives is one name."""
    totals = _file_totals(
        [
            {"filename": r"R:\79L99n6\H7RHJS5J.zip"},
            {"filename": r"C:\Users\bob\H7RHJS5J.zip"},
            {"filename": "R:/other/AB12CD34.doc"},
        ]
    )
    assert totals["unique_filenames"] == 2
    assert totals["file_count"] == 3


def test_a_missing_filename_is_not_counted_as_a_name() -> None:
    """The trap: converting to text first turns NaN into the string "nan",
    which then counts as a filename somebody used."""
    chunk = _file_chunk([{"filename": "A.doc"}, {"filename": None}])
    frame = aggregate_log("file", [chunk])
    assert frame["unique_filenames"].sum() == 1
    assert frame["file_count"].sum() == 2


# --- Change 4: file_count and file_to_removable_count (8 and 9) ------------


def test_without_the_column_every_file_row_is_a_removable_copy() -> None:
    """r4.2's documented degeneracy, unchanged by R2 and load-bearing."""
    totals = _file_totals([{"filename": "A.doc"}, {"filename": "B.doc"}])
    assert totals["file_count"] == totals["file_to_removable_count"] == 2


def test_with_the_column_only_writes_to_removable_media_count() -> None:
    """r5.2's shape: the two features separate, which is the point of r5.2."""
    totals = _file_totals(
        [
            {"filename": r"R:\a\A.doc", "to_removable_media": "True"},
            {"filename": r"R:\a\B.doc", "to_removable_media": "False"},
            {"filename": r"C:\a\C.doc", "to_removable_media": "False"},
        ]
    )
    assert totals["file_count"] == 3
    assert totals["file_to_removable_count"] == 1


# --------------------------------------------------------------------------
# Brief R2: header validation
#
# The hole this closes was measured, not imagined. r5.2's device, file and
# email headers are SUPERSETS of r4.2's, so the old subset check let an r4.2
# run read 39.7 GB of the wrong release and produce plausible numbers.
# --------------------------------------------------------------------------


def _write_header(directory: Path, name: str, columns: tuple[str, ...]) -> Path:
    path = directory / name
    path.write_text(",".join(columns) + "\n", encoding="utf-8")
    return path


def test_the_active_releases_header_is_accepted(tmp_path: Path) -> None:
    columns = RELEASE_LOG_COLUMNS["4.2"]["file"]
    path = _write_header(tmp_path, "file.csv", columns)
    assert _validated_header(path, "file") == list(columns)


def test_a_superset_header_is_refused_and_named(tmp_path: Path) -> None:
    """The measured failure: r5.2's file.csv under an r4.2 run."""
    path = _write_header(tmp_path, "file.csv", RELEASE_LOG_COLUMNS["5.2"]["file"])
    with pytest.raises(ExtractionError) as raised:
        _validated_header(path, "file")

    message = str(raised.value)
    assert "file.csv" in message
    assert "to_removable_media" in message, "the message must name what it found"
    assert "r5.2" in message, "and must say which release this folder actually is"


def test_a_reordered_header_is_refused(tmp_path: Path) -> None:
    """Same names, wrong order. Reading by name would survive it; a file whose
    layout is not the one the release shipped is still not that release."""
    columns = RELEASE_LOG_COLUMNS["4.2"]["email"]
    scrambled = (columns[1], columns[0], *columns[2:])
    path = _write_header(tmp_path, "email.csv", scrambled)
    with pytest.raises(ExtractionError, match="ORDER"):
        _validated_header(path, "email")


def test_an_unrecognisable_header_is_refused(tmp_path: Path) -> None:
    path = _write_header(tmp_path, "logon.csv", ("when", "who", "what"))
    with pytest.raises(ExtractionError, match="no release"):
        _validated_header(path, "logon")


_REFUSAL_SCRIPT = """
import sys
from pathlib import Path
from extraction.extract import ExtractionError, _validated_header

try:
    _validated_header(Path(sys.argv[1]), "email")
except ExtractionError as error:
    print(error)
else:
    print("ACCEPTED")
"""


def test_the_r52_profile_refuses_r42s_header(tmp_path: Path) -> None:
    """The other direction, which needs a differently configured process.

    ``config`` reads the release once, at import, so this cannot be done by
    reassigning anything -- it takes a subprocess, exactly as test_config.py
    does for the same reason.
    """
    path = _write_header(tmp_path, "email.csv", RELEASE_LOG_COLUMNS["4.2"]["email"])
    completed = subprocess.run(
        [sys.executable, "-c", _REFUSAL_SCRIPT, str(path)],
        cwd=ROOT,
        env={**os.environ, "INNERSIGHT_RELEASE": "5.2"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ACCEPTED" not in completed.stdout
    assert "r4.2" in completed.stdout, "it must name the release the file is"
    assert "activity" in completed.stdout, "and the column that is missing"


# --------------------------------------------------------------------------
# Brief R2: the verify block
#
# The three checks whose expected ANSWER depends on the release. On r4.2 they
# assert the degeneracies still hold; on r5.2 they assert the features have
# come apart, which is the reason the second release is worth running.
# --------------------------------------------------------------------------


def _one_row(**features: float) -> pd.DataFrame:
    row = {name: 0.0 for name in FEATURE_NAMES}
    row.update(features)
    frame = pd.DataFrame([row])
    frame.insert(0, "day", pd.to_datetime([DAY]))
    frame.insert(0, "user", ["AAA0001"])
    return frame


_DEGENERATE: dict[str, float] = {
    "file_count": 100, "file_to_removable_count": 100,
    "unique_filenames": 100, "large_attachment_count": 0,
}
_SEPARATED: dict[str, float] = {
    "file_count": 100, "file_to_removable_count": 45,
    "unique_filenames": 12, "large_attachment_count": 7,
}


def _three_release_lines(report: str) -> list[str]:
    """The three release-dependent CHECK lines, and not the totals above them.

    The feature-totals block names all three features too, so a naive search
    over the whole report finds six lines and the test passes or fails for the
    wrong reason. Scan the sanity-check section only.
    """
    checks = report.split("Sanity checks")[1]
    return [
        line for line in checks.splitlines()
        if "file_to_removable" in line
        or "unique_filenames" in line
        or "large_attachment" in line
    ]


def test_the_r42_verify_block_asserts_the_degeneracies() -> None:
    report = format_report(_one_row(**_DEGENERATE), release="4.2")
    lines = _three_release_lines(report)
    assert len(lines) == 3
    assert all(line.strip().startswith("ok") for line in lines), report


def test_the_r42_verify_block_complains_when_they_stop_holding() -> None:
    report = format_report(_one_row(**_SEPARATED), release="4.2")
    assert len(_three_release_lines(report)) == 3
    assert all("CHECK" in line for line in _three_release_lines(report)), report


def test_the_r52_verify_block_asserts_that_they_separated() -> None:
    report = format_report(_one_row(**_SEPARATED), release="5.2")
    lines = _three_release_lines(report)
    assert len(lines) == 3
    assert all(line.strip().startswith("ok") for line in lines), report
    assert "45.0%" in report, "the separation must be printed, not just asserted"


def test_the_r52_verify_block_complains_when_a_feature_stays_degenerate() -> None:
    """A degenerate r5.2 run means the alignment did not happen -- which is
    precisely the silent failure this brief exists to prevent."""
    report = format_report(_one_row(**_DEGENERATE), release="5.2")
    assert all("CHECK" in line for line in _three_release_lines(report)), report


def test_the_url_checks_are_kept_on_both_releases() -> None:
    """A zero there means URL matching never fired and everything downstream
    still runs perfectly happily. That is release-independent."""
    for release in ("4.2", "5.2"):
        report = format_report(_one_row(**_SEPARATED), release=release)
        for feature in ("job_search_visits", "cloud_upload_visits",
                        "http_request_count"):
            assert f"CHECK {feature} non-zero" in report, (release, feature)
