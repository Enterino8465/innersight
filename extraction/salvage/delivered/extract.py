"""Stage 1: raw CERT logs -> one row per (user, day) with 18 integer counts.

This is the only stage that ever touches the raw release. Everything after it
reads ``data/daily_counts.parquet`` and nothing else.

The shape of this module is dictated by one number: r4.2's ``http.csv`` is
14.5 GB. So nothing here ever holds a log. Each log is streamed in chunks, each
chunk is reduced to per-``(user, day)`` partials immediately, and the partials
are combined at the end. The input is unbounded; the running state is bounded
by population x days -- ~500k rows for r4.2, which fits in memory trivially.

Sixteen of the eighteen features are counts or sums and combine by addition.
**Two are distinct counts** -- ``unique_pcs_used`` and ``unique_filenames`` --
and those cannot be added: PC-42 seen in two chunks of the same user-day is one
distinct PC, not two. Distinct features therefore keep unique
``(user, day, value)`` triples around until the end and count them once. The
triples stay small because they are bounded by real behaviour: a person uses a
handful of machines a day, not thousands. Conveniently the log that is actually
enormous needs no distinct counts at all -- http's three features are a row
count and two URL tallies, so its memory cost is constant.

No labelling happens here. ``label.py`` (Stage 2) joins the answer key onto this
table. Keeping them apart means the expensive pass can be reused when the
labelling rule changes, which it will.

Three of the eighteen columns are degenerate on r4.2, all three measured across
the whole release rather than assumed. They are kept because the definitions
are correct and come back to life on r5+:

* ``file_to_removable_count == file_count`` -- r4.2 ships no
  ``to_removable_media`` column because every ``file.csv`` row already *is* a
  copy to removable media (the release readme says so).
* ``unique_filenames == file_count`` -- file.csv has 445,581 rows and 445,581
  distinct filenames; CERT generates a fresh random name per event.
* ``large_attachment_count == 0`` everywhere -- the largest of 2,629,979 emails
  is 141,909 bytes against a ~1 MB threshold, because r4.2's ``size`` is the
  message body and attachments are a separate integer count.

r4.2 therefore carries 16 informative features, not 18, and file.csv
contributes one number three times. Say this before an examiner finds it in a
feature-importance plot.
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
    BUSINESS_HOUR_END,
    BUSINESS_HOUR_START,
    CHUNK_SIZE,
    CLOUD_UPLOAD_DOMAINS,
    DAILY_COUNTS_PATH,
    DATE_FORMAT,
    DAY_COLUMN,
    DEVICE_CONNECT,
    DEVICE_DISCONNECT,
    FEATURE_NAMES,
    INTERNAL_DOMAIN,
    JOB_SEARCH_DOMAINS,
    LARGE_ATTACHMENT_BYTES,
    LOG_NAMES,
    LOG_SPECS,
    LOGOFF_ACTIVITY,
    LOGON_ACTIVITY,
    USER_COLUMN,
    WEEKEND_WEEKDAYS,
    ensure_directories,
    resolve_dataset,
)

logger = logging.getLogger(__name__)

KEY_COLUMNS: Final[list[str]] = [USER_COLUMN, DAY_COLUMN]

#: Which of the 18 features each log produces. The union is exactly
#: FEATURE_NAMES and the sets are disjoint -- checked at import. A feature owned
#: by two logs would be silently overwritten in the join; a feature owned by
#: none would be silently zero.
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

#: Columns each builder reads. Passed to the CSV reader as a projection, which
#: is what makes streaming http.csv cheap: three columns instead of six, and in
#: particular never materialising ``content`` -- which is most of the bytes.
LOG_INPUT_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "logon": ("date", "user", "pc", "activity"),
    "device": ("date", "user", "activity"),
    "file": ("date", "user", "filename"),
    "email": ("date", "user", "to", "cc", "bcc", "size"),
    "http": ("date", "user", "url"),
}

#: Read as well if the release provides it. r5+ ships it; r4.2 does not, and its
#: absence is what makes ``file_to_removable_count == file_count``.
_FILE_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("to_removable_media",)

#: Recipient columns scanned for external addresses. ``to`` alone is not enough:
#: a mail addressed to a colleague and blind-copied to a personal account reads
#: as entirely internal, which is exactly the exfiltration pattern. Measured on
#: real r4.2 mail, cc/bcc contributes 6.3% of external mail ``to`` cannot see.
RECIPIENT_COLUMNS: Final[tuple[str, ...]] = ("to", "cc", "bcc")

#: The two distinct counts. A third one must go through the ``distinct`` side of
#: LogPartial -- a distinct count taken per chunk and then summed is wrong,
#: silently, and only in proportion to how many chunks a user-day straddles.
DISTINCT_FEATURES: Final[frozenset[str]] = frozenset(
    {"unique_pcs_used", "unique_filenames"}
)

#: ``total_email_size`` is a byte total that overflows int32 and arrives from a
#: nullable column, so it stays float64. Everything else is int64.
FLOAT_FEATURES: Final[frozenset[str]] = frozenset({"total_email_size"})

#: Strings :func:`_truthy` accepts as true.
_TRUE_STRINGS: Final[frozenset[str]] = frozenset({"true", "1", "yes", "y", "t"})

#: Chunks folded together before the partial lists are compacted. Without this,
#: a 14.5 GB http.csv leaves hundreds of thousands of small frames pinned until
#: the end. Purely a memory knob -- the result is identical either way.
_COMPACT_EVERY: Final[int] = 64

#: Chunks between progress log lines while streaming.
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

#: ``@dtaa.com`` not followed by more domain characters, so a lookalike host
#: such as ``@dtaa.com.example.net`` is correctly counted as external.
_INTERNAL_ADDRESS_PATTERN: Final[str] = (
    rf"@{re.escape(INTERNAL_DOMAIN.lower())}(?![\w.-])"
)


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


# --------------------------------------------------------------------------
# The five chunk builders
# --------------------------------------------------------------------------


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

    A missing ``to_removable_media`` column means every row is a removable-media
    copy. That is not a guess: the r4.2 readme states that each entry represents
    a file copy to a removable media device, which is why the release ships no
    such column. The visible consequence is that ``file_to_removable_count``
    equals ``file_count`` exactly on r4.2 -- correct, not a bug.
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
        distinct={"unique_filenames": _distinct_by_key(keys, chunk["filename"])},
    )


def build_email_partial(chunk: pd.DataFrame) -> LogPartial:
    """Partials from a chunk of ``email.csv``.

    r4.2's email log has no ``activity`` column: every row is a message the user
    in the ``user`` column sent, so ``email_sent_count`` is a row count.
    """
    size = pd.to_numeric(chunk["size"], errors="coerce")

    sums = _sum_by_key(
        _key_frame(chunk),
        {
            "email_sent_count": _ones(chunk),
            "email_to_external_count": _count(_has_external_recipient(chunk)),
            "large_attachment_count": _count(size >= LARGE_ATTACHMENT_BYTES),
            "total_email_size": size.fillna(0.0).astype("float64"),
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


# --------------------------------------------------------------------------
# Folding a stream of chunks
# --------------------------------------------------------------------------


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

    # Cast last, after filling: an int64 column cannot hold the NaN that
    # reindexing a missing (user, day) introduces.
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
        data_dir: Directory laid out like the r4.2 release.
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
    if log == "file":
        wanted += [c for c in _FILE_OPTIONAL_COLUMNS if c in header]

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
    """Read a CSV's header and check it against the log's contract.

    Reading ``nrows=0`` means the contract is enforced in full even though the
    frame that follows is only a projection of it.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ExtractionError: If a contracted column is absent.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Log CSV not found: {path}")

    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in LOG_SPECS[log].columns if c not in header]
    if missing:
        raise ExtractionError(
            f"{path.name} is missing contracted columns {missing}. "
            f"Header is {header}; config.LOG_SPECS[{log!r}] expects "
            f"{list(LOG_SPECS[log].columns)}."
        )
    return header


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


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def format_report(daily: pd.DataFrame, out: Path | None = None) -> str:
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
    lines += ["", "Sanity checks:", *_check_lines(daily, totals), _RULE]
    if out is not None:
        lines.append(f"Written to: {out}")
    return "\n".join(lines)


def _check_lines(daily: pd.DataFrame, totals: pd.Series) -> list[str]:
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
        # The one most likely to fail silently: scenario 2 is thirty insiders
        # whose entire signature is job hunting. A zero here means URL matching
        # never fired, and everything downstream still runs perfectly happily.
        _check("job_search_visits non-zero", totals["job_search_visits"] > 0,
               f"{int(totals['job_search_visits']):,} visits"),
        _check("cloud_upload_visits non-zero", totals["cloud_upload_visits"] > 0,
               f"{int(totals['cloud_upload_visits']):,} visits"),
        _check("http_request_count non-zero", totals["http_request_count"] > 0,
               f"{int(totals['http_request_count']):,} requests"),
        # The three r4.2 degeneracies. These REPORT rather than judge: all three
        # are properties of the release, not of this code, and an examiner will
        # find them in any feature-importance plot.
        (
            f"   -   file_to_removable_count == file_count: "
            f"{bool(totals['file_to_removable_count'] == totals['file_count'])} "
            f"(expected True on r4.2)"
        ),
        (
            f"   -   unique_filenames == file_count: "
            f"{bool(totals['unique_filenames'] == totals['file_count'])} "
            f"(expected True on r4.2)"
        ),
        (
            f"   -   large_attachment_count total: "
            f"{int(totals['large_attachment_count'])} (expected 0 on r4.2)"
        ),
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

    shown = subset.head(rows).copy()
    shown[DAY_COLUMN] = shown[DAY_COLUMN].dt.strftime("%Y-%m-%d")
    return "\n".join([
        _RULE,
        f"{user} - {len(subset):,} days",
        _RULE,
        shown[[DAY_COLUMN, *preview_features]].to_string(index=False),
        (
            f"(showing {len(shown)} of {len(subset)} days, "
            f"{len(preview_features)} of {len(FEATURE_NAMES)} features)"
        ),
        _RULE,
    ])


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Build the daily count table, write it, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--full", action="store_true",
                        help="Use the full r4.2 release instead of the sample.")
    source.add_argument("--data-dir", type=Path, default=None,
                        help="Read this directory instead of a resolved dataset.")
    parser.add_argument("--out", type=Path, default=DAILY_COUNTS_PATH,
                        help=f"Parquet to write (default: {DAILY_COUNTS_PATH})")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Rows per CSV chunk (default: {CHUNK_SIZE})")
    parser.add_argument("--user", default=None,
                        help="After building, print this user's days.")
    args = parser.parse_args(argv)

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
