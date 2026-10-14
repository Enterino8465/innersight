"""Stage 1: the five raw CERT logs -> one row per (user, day) with 18 counts.

A rewrite of ``extraction/extract.py``, which stays on disk as the reference.
Nothing is labelled, z-scored or judged here; this stage only counts.

One intentional difference from the original: ``main()`` writes to
``data/daily_counts.ext2.parquet`` (or ``.sample.ext2.parquet``) so a run from
here can never overwrite the frozen data Stages 2-7 depend on.

The 18 counts, in the fixed order ``config.FEATURE_NAMES`` defines. The models
see 18 unnamed numbers, so slot 5 must be ``unique_pcs_used`` every time.

  from logon.csv
     1  logon_count               times they logged in
     2  logoff_count              times they logged off
     3  after_hours_logons        logons outside 07:00-19:00
     4  weekend_logons            logons on a Saturday or Sunday
     5  unique_pcs_used           how many DIFFERENT machines   (distinct)

  from device.csv
     6  usb_connect_count         USB drives plugged in
     7  usb_disconnect_count      USB drives pulled out
     8  after_hours_usb           USB events outside 07:00-19:00

  from file.csv
     9  file_count                files touched
    10  file_to_removable_count   of those, copies onto a USB drive
    11  unique_filenames          how many DIFFERENT names      (distinct)

  from email.csv
    12  email_sent_count          mails sent
    13  email_to_external_count   of those, sent outside the company
    14  large_attachment_count    mails with an attachment over ~1 MB
    15  total_email_size          bytes of mail sent          (the only float)

  from http.csv
    16  http_request_count        web pages visited
    17  job_search_visits         of those, job sites
    18  cloud_upload_visits       of those, file-sharing sites

Sixteen of these are sums and add up across chunks. Two are distinct counts --
slots 5 and 11 -- and cannot be added: the same PC seen in two chunks of the
same user-day is one PC, not two.

Three are degenerate on r4.2: slots 10 and 11 both equal slot 9, and slot 14 is
always 0. That is a property of the release, not a bug, so r4.2 carries 15
informative features rather than 18.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from config import (
    ACTIVE_RELEASE,
    BUSINESS_HOUR_END,
    BUSINESS_HOUR_START,
    CHUNK_SIZE,
    CLOUD_UPLOAD_DOMAINS,
    DAILY_COUNTS_PATH,
    DAILY_COUNTS_SAMPLE_PATH,
    DATE_FORMAT,
    DAY_COLUMN,
    DEVICE_CONNECT,
    DEVICE_DISCONNECT,
    EMAIL_SEND_ACTIVITY,
    FEATURE_NAMES,
    INTERNAL_DOMAIN,
    JOB_SEARCH_DOMAINS,
    LARGE_ATTACHMENT_BYTES,
    LOG_NAMES,
    LOG_SPECS,
    LOGOFF_ACTIVITY,
    LOGON_ACTIVITY,
    RELEASE_LOG_COLUMNS,
    RELEASE_R42,
    USER_COLUMN,
    WEEKEND_WEEKDAYS,
    ensure_directories,
    resolve_dataset,
)

logger = logging.getLogger(__name__)

KEY_COLUMNS: Final[list[str]] = [USER_COLUMN, DAY_COLUMN]

LOG_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "logon": (
        "logon_count",
        "logoff_count",
        "after_hours_logons",
        "weekend_logons",
        "unique_pcs_used",
    ),
    "device": (
        "usb_connect_count",
        "usb_disconnect_count",
        "after_hours_usb",
    ),
    "file": (
        "file_count",
        "file_to_removable_count",
        "unique_filenames",
    ),
    "email": (
        "email_sent_count",
        "email_to_external_count",
        "large_attachment_count",
        "total_email_size",
    ),
    "http": (
        "http_request_count",
        "job_search_visits",
        "cloud_upload_visits",
    ),
}

LOG_INPUT_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "logon": ("date", "user", "pc", "activity"),
    "device": ("date", "user", "activity"),
    "file": ("date", "user", "filename"),
    "email": ("date", "user", "to", "cc", "bcc", "size"),
    "http": ("date", "user", "url"),
}

_LOG_OPTIONAL_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "file": ("to_removable_media",),
    "email": ("activity", "attachments"),
}

RECIPIENT_COLUMNS: Final[tuple[str, ...]] = ("to", "cc", "bcc")

DISTINCT_FEATURES: Final[frozenset[str]] = frozenset(
    {"unique_pcs_used", "unique_filenames"}
)

FLOAT_FEATURES: Final[frozenset[str]] = frozenset({"total_email_size"})

_TRUE_STRINGS: Final[frozenset[str]] = frozenset({"true", "1", "yes", "y", "t"})

_COMPACT_EVERY: Final[int] = 64

_LOG_EVERY: Final[int] = 200

_RULE: Final[str] = "-" * 72


class ExtractionError(Exception):
    """Raised when a raw log cannot be read as Stage 1 requires."""


def _check_feature_partition() -> None:
    """Fail at import if LOG_FEATURES does not exactly partition FEATURE_NAMES."""
    owned: list[str] = [f for features in LOG_FEATURES.values() for f in features]
    duplicated = sorted({f for f in owned if owned.count(f) > 1})
    if duplicated:
        raise ExtractionError(f"features owned by more than one log: {duplicated}")
    if set(owned) != set(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(owned))
        extra = sorted(set(owned) - set(FEATURE_NAMES))
        raise ExtractionError(
            f"LOG_FEATURES does not match config.FEATURE_NAMES "
            f"(missing {missing}, unexpected {extra})"
        )


def _alternation(domains: Sequence[str], name: str) -> str:
    """Compile a domain tuple into a lowercase regex alternation.

    Raises on an empty tuple rather than returning ``''``, which as a regex
    matches every string -- the failure mode where a misconfigured list makes
    every web request look like a job-site visit.
    """
    if not domains:
        raise ExtractionError(
            f"config.{name} is empty. An empty URL list compiles to a pattern "
            f"that matches every URL, so every http row would count as a hit."
        )
    return "|".join(re.escape(domain.lower()) for domain in domains)


_check_feature_partition()

_JOB_URL_PATTERN: Final[str] = _alternation(JOB_SEARCH_DOMAINS, "JOB_SEARCH_DOMAINS")
_CLOUD_URL_PATTERN: Final[str] = _alternation(
    CLOUD_UPLOAD_DOMAINS, "CLOUD_UPLOAD_DOMAINS"
)

_INTERNAL_ADDRESS_PATTERN: Final[str] = (
    rf"@{re.escape(INTERNAL_DOMAIN.lower())}(?![\w.-])"
)

_ATTACHMENT_SIZE_PATTERN: Final[str] = r"\((\d+)\)"

_PATH_PREFIX_PATTERN: Final[str] = r"^.*[\\/]"


@dataclass(frozen=True)
class LogPartial:
    """What one chunk of one log contributes to the daily table.

    Attributes:
        sums: Additive features indexed by ``(user, day)``. Combining two of
            these is addition.
        distinct: ``{feature: DataFrame[user, day, value]}`` of unique triples.
            Combining two of these is concatenate-then-deduplicate; the count is
            taken once, at the very end.
    """

    sums: pd.DataFrame
    distinct: dict[str, pd.DataFrame]


def build_logon_partial(chunk: pd.DataFrame) -> LogPartial:
    """Partials from a chunk of ``logon.csv``.

    ``after_hours_logons`` and ``weekend_logons`` count logons only, never
    logoffs: the logon at 02:00 is the anomaly, and counting its matching logoff
    would report the same event twice.
    """
    keys = _key_frame(chunk)
    when = chunk["date"].dt
    is_logon = chunk["activity"] == LOGON_ACTIVITY

    sums = _sum_by_key(
        keys,
        {
            "logon_count": _count(is_logon),
            "logoff_count": _count(chunk["activity"] == LOGOFF_ACTIVITY),
            "after_hours_logons": _count(is_logon & _after_hours(chunk)),
            "weekend_logons": _count(is_logon & when.dayofweek.isin(WEEKEND_WEEKDAYS)),
        },
    )
    return LogPartial(
        sums=sums,
        distinct={"unique_pcs_used": _distinct_by_key(keys, chunk["pc"])},
    )


def build_device_partial(chunk: pd.DataFrame) -> LogPartial:
    """Partials from a chunk of ``device.csv``.

    ``after_hours_usb`` counts *any* device event outside business hours, not
    just connects: a drive plugged in at 18:55 and pulled at 19:30 is one
    after-hours episode either way.
    """
    sums = _sum_by_key(
        _key_frame(chunk),
        {
            "usb_connect_count": _count(chunk["activity"] == DEVICE_CONNECT),
            "usb_disconnect_count": _count(chunk["activity"] == DEVICE_DISCONNECT),
            "after_hours_usb": _count(_after_hours(chunk)),
        },
    )
    return LogPartial(sums=sums, distinct={})


def build_file_partial(chunk: pd.DataFrame) -> LogPartial:
    """Partials from a chunk of ``file.csv``.

    **The two counts are deliberately unchanged, and that is a decision rather
    than an omission.** ``file_count`` counts every file event; and
    ``file_to_removable_count`` counts ``to_removable_media == True`` where the
    release records it. Brief R2 re-examined both against r5.2 and confirmed
    that this -- what the code already did -- is what the data supports.

    A missing ``to_removable_media`` column means every row is a removable-media
    copy. The r4.2 readme states that each entry represents a file copy to a
    removable media device, which is why the release ships no such column, and
    the visible consequence is that ``file_to_removable_count`` equals
    ``file_count`` exactly on r4.2. **That is the readme's claim and nothing in
    the shipped r4.2 data confirms it** -- no activity column, no drive letter,
    no cross-reference -- and the same readme is demonstrably wrong about r5.2's
    copy of this file. Treat it as documentation, not measurement.

    Measured on the full 887,621-row r5.2 ``file.csv`` on 2026-08-21:

    * Only three combinations occur. ``File Write`` / to_removable / ``R:``
      403,966 rows; ``File Open`` / from_removable / ``R:`` 401,836; ``File
      Open`` / neither / ``C:`` 81,819. The log has three states, not eight.
    * ``to_removable_media == True`` is EXACTLY ``activity == "File Write"`` --
      the same 403,966 rows. The column carries no independent information, so
      reading it and reading the activity would give the same feature.
    * Local ``C:`` opens, the rows r4.2 cannot contain at all, are 9.2% of the
      log rather than the bulk of it.
    * Per employee the two releases agree within 0.4%: 445.6 events on r4.2
      against 443.8 on r5.2.

    A narrower "removable writes only" definition was considered and rejected:
    it is blind to 26 of the 30 scenario-4 insiders, and all-file-events is the
    strongest single signal in r5.2 (9.02x in-window concentration, 30 of 30
    insiders). The cost of keeping it broad is stated in the module docstring:
    slot 8 means something different on each release, and slot 9 is the one to
    compare across them.

    ``unique_filenames`` is counted over BASENAMES on both releases -- see
    :func:`_basename`.
    """
    keys = _key_frame(chunk)
    if "to_removable_media" in chunk.columns:
        to_removable = _count(_truthy(chunk["to_removable_media"]))
    else:
        to_removable = _ones(chunk)

    sums = _sum_by_key(
        keys,
        {"file_count": _ones(chunk), "file_to_removable_count": to_removable},
    )
    return LogPartial(
        sums=sums,
        distinct={
            "unique_filenames": _distinct_by_key(
                keys, _basename(chunk["filename"])
            )
        },
    )


def build_email_partial(chunk: pd.DataFrame) -> LogPartial:
    """Partials from a chunk of ``email.csv``.

    **All four email features count what the person SENT.** r4.2's email log has
    no ``activity`` column: every row is a message the user in the ``user``
    column sent, so a row count is already a sent count. r5.2 logs the inbox in
    the same file -- measured over all 17,361,575 rows, 32.00% Send and 68.00%
    Receive, with no third value -- so it needs the filter that r4.2's format
    made unnecessary. Without it ``email_sent_count`` is 3.1253x too large and
    counts an insider's incoming mail as evidence of exfiltration, silently,
    because the z-score step happily rescales a number that means the wrong
    thing.

    The filter is applied BEFORE the recipient test, not after: a received mail
    from an outside address would otherwise land in
    ``email_to_external_count``, which is meant to be "mail this person sent
    out of the organisation".

    ``large_attachment_count`` reads the ``attachments`` column rather than
    ``size``, and that is a correction rather than an extension. r5.2's ``size``
    INCLUDES attachment bytes -- correlation 0.999980 over the 3,738,336 mails
    carrying one, zero counterexamples, and a residual topping out at 139,887
    bytes against r4.2's largest-ever message of 141,909, so the residual is an
    r4.2-shaped body. r5.2's readme says the opposite and is wrong. The
    consequence is that the untouched pre-R2 code was ALREADY firing on 13.22%
    of all r5.2 mail: the feature was never dead on r5.2, it was saturated, and
    it was computing a different quantity from the one it computes on r4.2
    without anybody having edited a line. See :func:`_largest_attachment_bytes`.
    """
    size = pd.to_numeric(chunk["size"], errors="coerce")
    sent = _is_sent(chunk)
    largest_attachment = _largest_attachment_bytes(chunk)

    sums = _sum_by_key(
        _key_frame(chunk),
        {
            "email_sent_count": _count(sent),
            "email_to_external_count": _count(
                sent & _has_external_recipient(chunk)
            ),
            "large_attachment_count": _count(
                sent & (largest_attachment >= LARGE_ATTACHMENT_BYTES)
            ),
            "total_email_size": (
                size.where(sent, 0.0).fillna(0.0).astype("float64")
            ),
        },
    )
    return LogPartial(sums=sums, distinct={})


def build_http_partial(chunk: pd.DataFrame) -> LogPartial:
    """Partials from a chunk of ``http.csv``.

    All three features are pure sums, which is why the largest file in the
    release is the cheapest to aggregate: nothing carries between chunks beyond
    running totals. Matching is substring-against-lowercased-URL rather than
    host equality so a path-qualified entry would still work.
    """
    url = chunk["url"].astype("str").fillna("").str.lower()

    sums = _sum_by_key(
        _key_frame(chunk),
        {
            "http_request_count": _ones(chunk),
            "job_search_visits": _count(url.str.contains(_JOB_URL_PATTERN, regex=True)),
            "cloud_upload_visits": _count(
                url.str.contains(_CLOUD_URL_PATTERN, regex=True)
            ),
        },
    )
    return LogPartial(sums=sums, distinct={})


LOG_BUILDERS: Final[dict[str, Callable[[pd.DataFrame], LogPartial]]] = {
    "logon": build_logon_partial,
    "device": build_device_partial,
    "file": build_file_partial,
    "email": build_email_partial,
    "http": build_http_partial,
}


def aggregate_log(
    log: str,
    chunks: Iterable[pd.DataFrame],
    *,
    compact_every: int = _COMPACT_EVERY,
) -> pd.DataFrame:
    """Reduce a stream of chunks of one log to its per-(user, day) features.

    Takes any iterable of chunks, so a test can hand it two two-row frames and a
    production run can hand it a generator over 14.5 GB, down the same path.

    Args:
        log: One of :data:`config.LOG_NAMES`.
        chunks: Any iterable of raw chunks for that log.
        compact_every: Fold accumulated partials together every this many
            chunks.

    Returns:
        A frame indexed by ``(user, day)`` with exactly ``LOG_FEATURES[log]`` as
        columns; correctly shaped and empty if the stream yielded no rows.

    Raises:
        KeyError: If *log* is not a known log name.
        ValueError: If *compact_every* is not positive.
    """
    builder = LOG_BUILDERS[log]
    if compact_every <= 0:
        raise ValueError(f"compact_every must be positive, got {compact_every}")

    sum_parts: list[pd.DataFrame] = []
    distinct_parts: dict[str, list[pd.DataFrame]] = {}

    for number, chunk in enumerate(chunks, start=1):
        if chunk.empty:
            continue
        partial = builder(chunk)
        sum_parts.append(partial.sums)
        for feature, triples in partial.distinct.items():
            distinct_parts.setdefault(feature, []).append(triples)

        if number % compact_every == 0:
            sum_parts = [_compact_sums(sum_parts)]
            distinct_parts = {
                feature: [_compact_distinct(parts)]
                for feature, parts in distinct_parts.items()
            }

    if not sum_parts:
        return _empty_log_frame(log)

    features = _compact_sums(sum_parts)
    for feature, parts in distinct_parts.items():
        counts = (
            _compact_distinct(parts)
            .groupby(KEY_COLUMNS, sort=False)
            .size()
            .rename(feature)
        )
        features = features.join(counts, how="outer")

    return features.reindex(columns=list(LOG_FEATURES[log])).fillna(0)


def combine_daily_counts(per_log: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the per-log tables into the Stage 1 output.

    Three rules, each a decision rather than a detail:

    * The key universe is the **union** of ``(user, day)`` across every log, not
      any one log's keys. A day that exists only in http still produces a row.
    * Missing means **zero**, never NaN. "No USB events today" is an observation
      of zero, and a NaN here would poison the z-scores in Stage 3.
    * Columns come out in :data:`config.FEATURE_NAMES` order, always. That
      ordering is the vector layout every later stage assumes.

    Args:
        per_log: ``{log_name: features}`` as returned by :func:`aggregate_log`.
            Logs may be omitted; their features come out as zero.

    Returns:
        ``user``, ``day``, then the 18 counts, sorted by ``(user, day)``.
    """
    key_index = _union_key_index(per_log.values())

    columns: dict[str, pd.Series] = {}
    for frame in per_log.values():
        aligned = frame.reindex(key_index)
        for feature in frame.columns:
            columns[feature] = aligned[feature]

    absent = [feature for feature in FEATURE_NAMES if feature not in columns]
    if absent:
        logger.warning("no log supplied %s; emitting them as zero", absent)

    daily = pd.DataFrame(
        {feature: columns.get(feature, 0) for feature in FEATURE_NAMES},
        index=key_index,
    ).fillna(0)

    for feature in FEATURE_NAMES:
        daily[feature] = daily[feature].astype(
            "float64" if feature in FLOAT_FEATURES else "int64"
        )

    return daily.sort_index().reset_index()[[*KEY_COLUMNS, *FEATURE_NAMES]]


def daily_counts(
    per_log_chunks: Mapping[str, Iterable[pd.DataFrame]],
    *,
    compact_every: int = _COMPACT_EVERY,
) -> pd.DataFrame:
    """Aggregate every log's chunk stream and combine the results."""
    per_log = {
        log: aggregate_log(log, chunks, compact_every=compact_every)
        for log, chunks in per_log_chunks.items()
    }
    return combine_daily_counts(per_log)


def daily_counts_from_dir(
    data_dir: Path | str,
    *,
    chunk_size: int = CHUNK_SIZE,
    compact_every: int = _COMPACT_EVERY,
) -> pd.DataFrame:
    """Build the Stage 1 table by streaming a CERT release directory.

    Memory stays bounded by population x days, so this runs on a laptop against
    the full 16 GB release.

    Args:
        data_dir: Directory laid out like the ACTIVE release. A folder holding
            a different release is refused by the header check before a single
            data row is read.
        chunk_size: Rows per CSV chunk.
        compact_every: See :func:`aggregate_log`.

    Returns:
        The daily count table.

    Raises:
        FileNotFoundError: If the directory or one of the five logs is missing.
        ExtractionError: If a log's header does not match its
            :class:`config.LogSpec`.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    return daily_counts(
        {
            log: _stream_log(data_dir, log, chunk_size=chunk_size)
            for log in LOG_NAMES
        },
        compact_every=compact_every,
    )


def _stream_log(
    data_dir: Path,
    log: str,
    *,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """Yield header-validated, column-projected, date-parsed chunks of one log.

    The header is checked against the full :class:`config.LogSpec` contract
    before a single data row is read, and only then are the columns the builder
    needs materialised -- three out of six on http.csv, and in particular never
    ``content``.
    """
    path = data_dir / LOG_SPECS[log].filename
    header = _validated_header(path, log)

    wanted = [column for column in LOG_INPUT_COLUMNS[log] if column in header]
    wanted += [c for c in _LOG_OPTIONAL_COLUMNS.get(log, ()) if c in header]

    logger.info("%s: reading %s from %s", log, wanted, path.name)
    rows = 0
    reader = pd.read_csv(path, usecols=wanted, dtype="str", chunksize=chunk_size)
    with reader:
        for number, chunk in enumerate(reader, start=1):
            _parse_dates_in_place(chunk, path)
            rows += len(chunk)
            if number % _LOG_EVERY == 0:
                logger.info("%s: %d rows so far", log, rows)
            yield chunk
    logger.info("%s: %d rows total", log, rows)


def _validated_header(path: Path, log: str) -> list[str]:
    """Read a CSV's header and check it EXACTLY against the log's contract.

    Reading ``nrows=0`` means the contract is enforced in full even though the
    frame that follows is only a projection of it.

    **This used to be a subset check** -- it asked whether every contracted
    column was present and never whether an unexpected one had appeared. That
    is a real hole rather than a theoretical one, and it was open in this
    repository until 2026-08-21: r5.2's ``device``, ``file`` and ``email``
    headers are supersets of r4.2's, so pointing an r4.2 run at the r5.2 folder
    passed validation and read 39.7 GB of the wrong release without one
    complaint. The features would have come out plausible and wrong.

    Exact tuple equality closes it, and matching on ORDER as well as on names
    is the existing house rule rather than a new one: ``verify_setup.py`` has
    compared these headers with ``found == spec.columns`` since Stage 0. Both
    releases' shipped files match their spec exactly, order included, verified
    with ``head -1`` on r4.2's sample, r4.2's full mirror, r5.2's full release
    and r5.2's sample on 2026-08-21.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ExtractionError: If the header is not exactly the active release's.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Log CSV not found: {path}")

    header = tuple(pd.read_csv(path, nrows=0).columns)
    expected = LOG_SPECS[log].columns
    if header != expected:
        raise ExtractionError(_header_mismatch(path, log, header, expected))
    return list(header)


def _header_mismatch(
    path: Path,
    log: str,
    header: tuple[str, ...],
    expected: tuple[str, ...],
) -> str:
    """The message a wrong header gets: what was found, what was wanted, and --
    the part that saves the reader ten minutes -- which release this file is.
    """
    missing = [c for c in expected if c not in header]
    unexpected = [c for c in header if c not in expected]
    matches = [
        release
        for release, columns in RELEASE_LOG_COLUMNS.items()
        if columns.get(log) == header
    ]

    if matches:
        verdict = (
            f"That is r{matches[0]}'s {log}.csv, and this process is running "
            f"as r{ACTIVE_RELEASE}. Point it at the r{ACTIVE_RELEASE} folder, "
            f"or select r{matches[0]} and re-run -- config.release_banner() "
            f"names the release in force on every run's first line."
        )
    elif not missing and not unexpected:
        verdict = (
            "The column NAMES are right and their ORDER is not, so this is not "
            "the file the release shipped."
        )
    else:
        verdict = "It matches no release this project has a profile for."

    return (
        f"{path.name} is not r{ACTIVE_RELEASE}'s {log} log.\n"
        f"  expected: {', '.join(expected)}\n"
        f"  found:    {', '.join(header)}\n"
        f"  missing:  {missing or 'none'}\n"
        f"  extra:    {unexpected or 'none'}\n"
        f"  {verdict}"
    )


def _parse_dates_in_place(chunk: pd.DataFrame, path: Path) -> None:
    """Convert ``date`` to datetime64 using the explicit CERT format.

    The explicit format is tried first and per-row inference is used only as a
    fallback. That ordering is the whole point: ``format='mixed'`` re-infers the
    layout of every row, which is unmeasurable on a 100-row fixture and costs
    minutes on a 28M-row log.
    """
    try:
        chunk["date"] = pd.to_datetime(chunk["date"], format=DATE_FORMAT)
    except (ValueError, TypeError):
        logger.warning("%s: dates are not %s; falling back to inference",
                       path.name, DATE_FORMAT)
        chunk["date"] = pd.to_datetime(chunk["date"], format="mixed")


def _key_frame(chunk: pd.DataFrame) -> pd.DataFrame:
    """The ``(user, day)`` key for every row, days normalised to midnight."""
    return pd.DataFrame(
        {USER_COLUMN: chunk[USER_COLUMN], DAY_COLUMN: chunk["date"].dt.normalize()},
        index=chunk.index,
    )


def _after_hours(chunk: pd.DataFrame) -> pd.Series:
    """Rows outside ``[BUSINESS_HOUR_START, BUSINESS_HOUR_END)``."""
    hour = chunk["date"].dt.hour
    return (hour < BUSINESS_HOUR_START) | (hour >= BUSINESS_HOUR_END)


def _ones(chunk: pd.DataFrame) -> pd.Series:
    """A 1-per-row column, for features that are plain row counts."""
    return pd.Series(1, index=chunk.index, dtype="int64")


def _count(mask: pd.Series) -> pd.Series:
    """A possibly-nullable boolean mask as a 0/1 int64 column.

    Going through ``boolean`` first is what makes this safe for both a plain
    numpy mask and a nullable one: comparing against a column holding NA yields
    NA, and NA must count as "did not happen" rather than propagate.
    """
    return mask.astype("boolean").fillna(False).astype("int64")


def _truthy(series: pd.Series) -> pd.Series:
    """Interpret a CERT boolean-ish column as an actual boolean mask."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean").fillna(False)
    return series.astype("str").str.strip().str.lower().isin(_TRUE_STRINGS)


def _basename(paths: pd.Series) -> pd.Series:
    """Filenames with any directory prefix stripped, on BOTH releases.

    Applied unconditionally rather than only where a separator appears, because
    a rule with an exception is a rule somebody has to remember. It costs r4.2
    nothing: 0 of its 445,581 filenames contain a separator, so every name is
    already its own basename.

    On r5.2 it is also a no-op in VALUE and not in meaning, which is the part
    worth knowing. 22,047 distinct full paths reduce to 22,047 distinct
    basenames -- zero collisions -- so nothing is lost. But r4.2 has 445,581
    distinct names for 445,581 rows, one per event, while r5.2 has 22,047 names
    for 887,621 rows, each recurring about forty times. So
    ``unique_filenames`` is ``file_count`` restated on r4.2 and a genuine
    distinct-count over a small vocabulary on r5.2. **The feature changes
    character, not just magnitude.** All four numbers measured 2026-08-21.

    NaN in, NaN out -- :func:`_distinct_by_key` drops those, and converting to
    text first would turn a missing filename into the literal string "nan" and
    count it as a name somebody used.
    """
    return paths.str.replace(_PATH_PREFIX_PATTERN, "", regex=True)


def _is_sent(chunk: pd.DataFrame) -> pd.Series:
    """Rows that are mail this user sent, as a plain (never nullable) mask.

    No ``activity`` column means the release logs sent mail only, which is
    r4.2's contract, so everything is a send and every email feature is
    computed over the whole chunk exactly as it was before R2.
    """
    if "activity" not in chunk.columns:
        return pd.Series(True, index=chunk.index, dtype="bool")
    equals = chunk["activity"] == EMAIL_SEND_ACTIVITY
    return equals.astype("boolean").fillna(False).astype("bool")


def _largest_attachment_bytes(chunk: pd.DataFrame) -> pd.Series:
    """Bytes of the biggest single attachment on each row; 0 where there is none.

    The LARGEST single attachment rather than their total, because the feature
    asks "did this person send one big thing out", and because a total would
    make ten ordinary documents look like one exfiltrated archive.

    The parse is a no-op on r4.2 by construction: its ``attachments`` column is
    an integer count, confirmed over all 2,629,979 rows to be exactly the ten
    values "0" to "9" with no brackets anywhere, so the pattern finds nothing
    and every row scores zero. That leaves ``large_attachment_count``
    identically zero on r4.2, which is exactly what it already was.

    The ``(`` prefilter is a speed measure, not a correctness one: 80% of r5.2's
    mail carries no attachment, and ``extractall`` on the remainder is
    materially cheaper than on all of it.
    """
    zeros = pd.Series(0, index=chunk.index, dtype="int64")
    if "attachments" not in chunk.columns:
        return zeros

    text = chunk["attachments"].astype("str")
    sized = text.str.contains("(", regex=False).astype("boolean").fillna(False)
    if not sized.any():
        return zeros

    matches = text[sized.astype("bool")].str.extractall(_ATTACHMENT_SIZE_PATTERN)
    if matches.empty:
        return zeros

    largest = matches[0].astype("int64").groupby(level=0).max()
    return largest.reindex(chunk.index).fillna(0).astype("int64")


def _has_external_recipient(chunk: pd.DataFrame) -> pd.Series:
    """True where any address across ``to``/``cc``/``bcc`` is outside the org.

    Counting rather than splitting: every address contains exactly one ``@``, so
    "more ``@`` signs than internal-domain matches" means at least one recipient
    is external. That is one vectorised regex count per column instead of a
    per-row split, which matters on a 1.36 GB email log.
    """
    parts = [
        chunk[column].astype("str").fillna("").str.lower()
        for column in RECIPIENT_COLUMNS
        if column in chunk.columns
    ]
    if not parts:
        return pd.Series(False, index=chunk.index, dtype="bool")

    recipients = parts[0]
    for part in parts[1:]:
        recipients = recipients + ";" + part

    return recipients.str.count("@") > recipients.str.count(_INTERNAL_ADDRESS_PATTERN)


def _sum_by_key(keys: pd.DataFrame, columns: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Group *columns* by ``(user, day)`` and sum them.

    ``groupby`` drops NA keys by default, which is wanted: a row with no user or
    an unparseable date cannot be attributed to anyone's day, so it is counted
    for nobody rather than silently assigned to a NaN key.
    """
    return keys.assign(**columns).groupby(KEY_COLUMNS, sort=False).sum()


def _distinct_by_key(keys: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """Unique ``(user, day, value)`` triples, the input to a distinct count."""
    return keys.assign(value=values).dropna().drop_duplicates(ignore_index=True)


def _compact_sums(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Add a list of per-(user, day) sum frames together."""
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts).groupby(level=KEY_COLUMNS, sort=False).sum()


def _compact_distinct(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate and deduplicate triple frames -- never count in between."""
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts, ignore_index=True).drop_duplicates(ignore_index=True)


def _empty_log_frame(log: str) -> pd.DataFrame:
    """A correctly shaped, correctly indexed, zero-row frame for an empty log."""
    return pd.DataFrame(
        {feature: pd.Series(dtype="int64") for feature in LOG_FEATURES[log]},
        index=pd.MultiIndex.from_arrays(
            [pd.Index([], dtype="str"), pd.DatetimeIndex([])], names=KEY_COLUMNS
        ),
    )


def _union_key_index(frames: Iterable[pd.DataFrame]) -> pd.MultiIndex:
    """The union of every frame's ``(user, day)`` index, unique and sorted."""
    union: pd.Index | None = None
    for frame in frames:
        union = frame.index if union is None else union.union(frame.index)
    if union is None or len(union) == 0:
        return pd.MultiIndex.from_arrays(
            [pd.Index([], dtype="str"), pd.DatetimeIndex([])], names=KEY_COLUMNS
        )
    return union.unique().sort_values().set_names(KEY_COLUMNS)


def format_report(
    daily: pd.DataFrame,
    out: Path | None = None,
    release: str = ACTIVE_RELEASE,
) -> str:
    """Render the table's shape, totals and sanity checks.

    The checks are the point. Unit tests prove the code does what was written;
    these prove what was written is true of the world -- the class of bug that
    otherwise produces a beautiful model trained on nonsense. Nothing raises: a
    surprising number is something to look at, not necessarily something wrong.
    """
    totals = daily[list(FEATURE_NAMES)].sum()
    lines = [
        _RULE,
        "InnerSight Stage 1 - daily counts",
        _RULE,
        f"Rows (user-days)   : {len(daily):,}",
        f"Distinct users     : {daily[USER_COLUMN].nunique():,}",
        f"Distinct days      : {daily[DAY_COLUMN].nunique():,}",
    ]
    if not daily.empty:
        lines.append(
            f"Date range         : {daily[DAY_COLUMN].min():%Y-%m-%d} -> "
            f"{daily[DAY_COLUMN].max():%Y-%m-%d}"
        )
    lines += ["", "Feature totals:"]
    lines += [f"  {i:>2} {f:<26} {totals[f]:>18,.0f}"
              for i, f in enumerate(FEATURE_NAMES)]
    lines += [
        "",
        f"Sanity checks (r{release}):",
        *_check_lines(daily, totals, release),
        _RULE,
    ]
    if out is not None:
        lines.append(f"Written to: {out}")
    return "\n".join(lines)


def _check_lines(
    daily: pd.DataFrame,
    totals: pd.Series,
    release: str = ACTIVE_RELEASE,
) -> list[str]:
    """One line per check, each marked ok or CHECK."""
    negatives = [f for f in FEATURE_NAMES if (daily[f] < 0).any()]
    nan_columns = [c for c in daily.columns if daily[c].isna().any()]
    logons, logoffs = int(totals["logon_count"]), int(totals["logoff_count"])
    ratio = logoffs / logons if logons else 0.0

    return [
        _check("no negative counts", not negatives, negatives or "none"),
        _check("no NaNs anywhere", not nan_columns, nan_columns or "none"),
        _check("no duplicate (user, day)",
               not daily.duplicated(KEY_COLUMNS).any(),
               f"{int(daily.duplicated(KEY_COLUMNS).sum())} duplicates"),
        _check("logons approx logoffs, not equal",
               logons != logoffs and 0.8 <= ratio <= 1.25,
               f"{logons:,} logons vs {logoffs:,} logoffs"),
        _check("job_search_visits non-zero", totals["job_search_visits"] > 0,
               f"{int(totals['job_search_visits']):,} visits"),
        _check("cloud_upload_visits non-zero", totals["cloud_upload_visits"] > 0,
               f"{int(totals['cloud_upload_visits']):,} visits"),
        _check("http_request_count non-zero", totals["http_request_count"] > 0,
               f"{int(totals['http_request_count']):,} requests"),
        *_release_check_lines(totals, release),
    ]


def _release_check_lines(totals: pd.Series, release: str) -> list[str]:
    """The three checks whose expected ANSWER depends on the release.

    Everything else in this module decides by looking at the data. These three
    cannot: "is ``file_to_removable_count`` equal to ``file_count``" is a
    correct result on r4.2 and a bug on r5.2, and no amount of looking at the
    frame tells you which you are owed. So they branch on the release, and they
    are the only place in Stage 1 that does.

    They used to be printed as bare facts, on the reasoning that a degeneracy is
    a property of the release rather than of this code. That was right while
    there was one release. With two it is not enough: r5.2 exists in this
    project precisely BECAUSE these three separate there, so on r5.2 the
    separation is the thing being claimed and it should be asserted and shown.
    """
    files = int(totals["file_count"])
    removable = int(totals["file_to_removable_count"])
    names = int(totals["unique_filenames"])
    large = int(totals["large_attachment_count"])

    if release == RELEASE_R42:
        return [
            _check("file_to_removable == file_count", removable == files,
                   f"{removable:,} of {files:,} (degenerate on r4.2, expected)"),
            _check("unique_filenames == file_count", names == files,
                   f"{names:,} of {files:,} (degenerate on r4.2, expected)"),
            _check("large_attachment_count is zero", large == 0,
                   f"{large:,} (r4.2's largest mail is 141,909 bytes)"),
        ]

    separation = f"{100 * removable / files:.1f}%" if files else "n/a"
    vocabulary = f"1 name per {files / names:,.1f} events" if names else "n/a"
    return [
        _check("file_to_removable < file_count", removable < files,
               f"{removable:,} of {files:,} file events are removable "
               f"writes ({separation})"),
        _check("unique_filenames < file_count", names < files,
               f"{names:,} distinct names over {files:,} events "
               f"({vocabulary})"),
        _check("large_attachment_count non-zero", large > 0,
               f"{large:,} sends carry an attachment over the threshold"),
    ]


def _check(label: str, passed: bool, detail: object) -> str:
    return f"  {'ok   ' if passed else 'CHECK'} {label:<34} {detail}"


def format_user_report(daily: pd.DataFrame, user: str, rows: int = 30) -> str:
    """Print one user's days, so real numbers can be looked at directly."""
    preview_features = (
        "logon_count", "after_hours_logons", "unique_pcs_used",
        "usb_connect_count", "file_count", "email_to_external_count",
        "http_request_count", "job_search_visits", "cloud_upload_visits",
    )
    subset = daily.loc[daily[USER_COLUMN] == user]
    if subset.empty:
        sample = ", ".join(sorted(daily[USER_COLUMN].unique())[:5])
        return f"No rows for user {user!r}. Users look like: {sample}"

    signal_features = (
        "usb_connect_count", "after_hours_usb", "after_hours_logons",
        "weekend_logons", "file_count", "job_search_visits",
        "cloud_upload_visits",
    )
    signal = subset[list(signal_features)].sum(axis=1)
    quiet = int((signal == 0).sum())
    notable = subset.loc[signal.sort_values(ascending=False).index[:rows]]
    shown = notable.sort_values(DAY_COLUMN).copy()
    shown[DAY_COLUMN] = shown[DAY_COLUMN].dt.strftime("%Y-%m-%d")

    tail = f"(the other {quiet:,} days are quiet: no USB, after-hours, file or upload activity)"
    return "\n".join([
        _RULE,
        f"{user} - {len(subset):,} days, showing the {len(shown)} most notable",
        _RULE,
        shown[[DAY_COLUMN, *preview_features]].to_string(index=False),
        tail if quiet else f"({len(subset) - len(shown):,} further days not shown)",
        _RULE,
    ])


def main(argv: Sequence[str] | None = None) -> int:
    """Build the daily count table, write it, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--full", action="store_true",
                        help="Use the full release instead of the sample.")
    source.add_argument("--data-dir", type=Path, default=None,
                        help="Read this directory instead of a resolved dataset.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Parquet to write. Defaults to the sample path "
                             "for a sample run and the full path for --full.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Rows per CSV chunk (default: {CHUNK_SIZE})")
    parser.add_argument("--user", default=None,
                        help="After building, print this user's days.")
    args = parser.parse_args(argv)

    if args.out is None:
        args.out = DAILY_COUNTS_PATH if args.full else DAILY_COUNTS_SAMPLE_PATH

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = args.data_dir or resolve_dataset(sample=not args.full)
    logger.info("reading %s", data_dir)

    daily = daily_counts_from_dir(data_dir, chunk_size=args.chunk_size)

    ensure_directories()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(args.out, index=False)
    logger.info("wrote %d rows to %s", len(daily), args.out)

    print(format_report(daily, out=args.out))
    if args.user:
        print()
        print(format_user_report(daily, args.user))
    return 0


if __name__ == "__main__":
    sys.exit(main())
