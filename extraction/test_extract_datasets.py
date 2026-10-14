"""Stage 1 against the real sample releases.

Separate from ``test_extract.py`` for two reasons. These read hundreds of
megabytes and take seconds rather than milliseconds, and they skip on a machine
that has no dataset folder, so a passing run of the fast file means something
different from a passing run of this one.

Two questions, one per release:

* **r4.2 must not move.** Brief R2 changed four feature computations, and every
  one of them is supposed to be a no-op here. The proof is that extracting the
  30-user sample still reproduces that sample's rows in
  ``data/daily_counts.parquet`` -- the 1,000-user table the frozen results are
  computed from -- value for value.
* **r5.2 must separate.** Three features are degenerate on r4.2 and the whole
  reason for a second release is that they stop being. If an r5.2 run still
  showed ``unique_filenames == file_count`` the port would have failed
  silently, and everything downstream would run perfectly happily on it.

Values are compared, never bytes. Parquet is not byte-reproducible: measured
2026-08-21, a full re-run produced a different md5 and identical values,
because ``day`` is ``datetime64[ns]`` in the committed files and
``datetime64[us]`` in a fresh one. ``check_dtype=False`` is not laziness, it is
the correct comparison. Byte equality is valid only for the ``.pt``
checkpoints.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from config import DAILY_COUNTS_PATH, PROFILES

ROOT = Path(__file__).resolve().parent.parent
LOGS = ("logon", "device", "file", "email", "http")


def _sample_dir(release: str) -> Path:
    return PROFILES[release].sample_dir


def _has_logs(release: str, logs: tuple[str, ...]) -> bool:
    directory = _sample_dir(release)
    return all((directory / f"{log}.csv").is_file() for log in logs)


# --------------------------------------------------------------------------
# r4.2: nothing moved
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_logs("4.2", LOGS) or not DAILY_COUNTS_PATH.is_file(),
    reason="needs the r4.2 sample release and data/daily_counts.parquet",
)
def test_the_r42_sample_reproduces_its_slice_of_the_frozen_table() -> None:
    """The regression that matters. ~8 seconds; it reads the 150 MB sample.

    The baseline is the FULL table's rows for the sample's 30 users, not
    ``data/daily_counts.sample.parquet``. That file is stale -- measured
    2026-08-21, its ``cloud_upload_visits`` totals 462 against today's 122,
    because it predates ``mediafire.com`` leaving CLOUD_UPLOAD_DOMAINS -- so a
    test anchored to it would fail on correct code. The full table is current:
    its rows for these 30 users match a fresh extraction exactly, which is also
    the check that the sample logs really are a slice of the full ones.
    """
    from extraction.extract import daily_counts_from_dir

    fresh = daily_counts_from_dir(_sample_dir("4.2"))
    full = pd.read_parquet(DAILY_COUNTS_PATH)
    slice_of_full = (
        full[full["user"].isin(set(fresh["user"]))]
        .sort_values(["user", "day"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(slice_of_full, fresh, check_dtype=False)


@pytest.mark.skipif(
    not _has_logs("4.2", LOGS),
    reason="needs the r4.2 sample release",
)
def test_the_three_r42_degeneracies_still_hold_on_the_sample() -> None:
    """Stated as an assertion rather than left to the report's eye test."""
    from extraction.extract import daily_counts_from_dir

    totals = daily_counts_from_dir(_sample_dir("4.2")).sum(numeric_only=True)

    assert totals["file_to_removable_count"] == totals["file_count"]
    assert totals["unique_filenames"] == totals["file_count"]
    assert totals["large_attachment_count"] == 0


# --------------------------------------------------------------------------
# r5.2: the features came apart
#
# `config` reads the release once, at import, so an r5.2 aggregation needs its
# own process -- the same reason test_config.py uses subprocesses. The script
# prints one JSON object of totals and the tests below read it.
# --------------------------------------------------------------------------

_R52_LOGS = ("file", "email")

_TOTALS_SCRIPT = """
import json
import sys
from pathlib import Path
from extraction.extract import aggregate_log, _stream_log

directory = Path(sys.argv[1])
totals = {}
for log in ("file", "email"):
    frame = aggregate_log(log, _stream_log(directory, log, chunk_size=50_000))
    totals |= {c: int(frame[c].sum()) for c in frame.columns}
print(json.dumps(totals))
"""


@pytest.fixture(scope="module")
def r52_totals() -> dict[str, int]:
    if not _has_logs("5.2", _R52_LOGS):
        pytest.skip("needs the r5.2 sample release")

    completed = subprocess.run(
        [sys.executable, "-c", _TOTALS_SCRIPT, str(_sample_dir("5.2"))],
        cwd=ROOT,
        env={**os.environ, "INNERSIGHT_RELEASE": "5.2"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_the_r52_file_features_separate(r52_totals: dict[str, int]) -> None:
    """Measured on the 8,439-row sample file.csv on 2026-08-21.

    ``to_removable_media == True`` is exactly ``activity == "File Write"``
    there, 3,822 rows of 8,439, so #9 is now a genuine subset of #8. And 4,731
    distinct basenames over 8,439 events means names repeat, which they never
    do on r4.2.
    """
    assert r52_totals["file_count"] == 8_439
    assert r52_totals["file_to_removable_count"] == 3_822
    assert r52_totals["unique_filenames"] == 4_731

    assert r52_totals["file_to_removable_count"] < r52_totals["file_count"]
    assert r52_totals["unique_filenames"] < r52_totals["file_count"]


def test_the_send_filter_is_real_on_the_r52_sample(
    r52_totals: dict[str, int],
) -> None:
    """81,520 of the sample's 237,495 email rows are sends.

    Without the filter this is the row count, 237,495 -- 2.91x too large on the
    sample and 3.1253x too large on the full release, with no error anywhere,
    because the number would still be a plausible count of something.
    """
    assert r52_totals["email_sent_count"] == 81_520
    assert 0.30 < r52_totals["email_sent_count"] / 237_495 < 0.40


def test_large_attachments_fire_on_r52_and_are_parsed_not_sized(
    r52_totals: dict[str, int],
) -> None:
    """7,012 sends, i.e. 8.60% of them. The feature was never dead on r5.2.

    Under the pre-R2 code -- ``size >= 1_000_000`` over every row, sent or
    received -- 25,611 of the sample's rows scored a hit. Under R2's definition
    it is 7,012. Both are non-zero, which is exactly why "it does not fire on
    r5.2" was the wrong premise: the feature was saturated and measuring
    something else, not asleep.
    """
    assert r52_totals["large_attachment_count"] == 7_012
    assert r52_totals["large_attachment_count"] > 0


def test_r52_total_email_size_counts_sent_bytes_only(
    r52_totals: dict[str, int],
) -> None:
    """The bytes of the 81,520 sends, measured 2026-08-21."""
    assert r52_totals["total_email_size"] == 24_163_135_922
