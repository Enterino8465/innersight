"""The daily behaviour profile: 18 numbers per person per day.

This module is the waist of the whole project. Above it are tens of millions of
raw events in five differently-shaped logs; below it every module — the EMA
baseline, the windowing, the CNN, the graph, the API — reads one tidy table and
nothing else. One row per ``(user, day)``, columns in
:data:`~innersight.schema.FEATURE_NAMES` order, plus ``is_malicious``.

The shape of the code is dictated by one number: r4.2's ``http.csv`` is 13.5 GB.
So nothing here ever holds a log. Each log is read in chunks, each chunk is
reduced to per-``(user, day)`` partials immediately, and the partials are
combined at the end. The input is unbounded; the running state is bounded by
population x days, which is ~500k rows for r4.2 and fits in memory trivially.

Sixteen of the eighteen features are counts or sums and combine by addition.
**Two are distinct counts** — ``unique_pcs_used`` and ``unique_filenames`` — and
those cannot be added: PC-42 seen in two chunks of the same user-day is one
distinct PC, not two. Distinct features therefore keep unique ``(user, day,
value)`` triples around until the end and count them once. The triples stay
small because they are bounded by real behaviour: a person uses a handful of
machines a day, not thousands. Conveniently, the log that is actually enormous
needs no distinct counts at all — http's three features are a row count and two
URL tallies, so its memory cost is constant.

Two definitions in here were genuinely ambiguous and were settled deliberately;
both differ from the original InnerSight implementation and both are documented
where they are used:

* **External email counts ``to`` + ``cc`` + ``bcc``**, not just ``to``. See
  :data:`RECIPIENT_COLUMNS`.
* **URL matching uses real domains only**, not the generic ``job`` / ``career``
  / ``linkedin`` substrings. See :data:`~innersight.config.CERT_JOB_DOMAINS`.

Deliberately r4.2-only, per the project's standing decision to defer the
cross-version adapter layer. The one r4.2-specific fact encoded here is that
every ``file.csv`` row is a copy to removable media — see
:func:`build_file_partial`.

**Three of the eighteen columns are degenerate on r4.2**, all three measured
across the whole release rather than assumed:

* ``file_to_removable_count == file_count``: every file event is a
  removable-media copy (readme.txt), so the flag is constant.
* ``unique_filenames == file_count``: file.csv has 445,581 rows and 445,581
  distinct filenames — CERT generates a fresh random name per event, so no
  filename ever repeats and the distinct count is the row count.
* ``large_attachment_count == 0`` everywhere: the largest of 2,629,979 emails
  is 141,909 bytes against a 1 MiB threshold, because r4.2's ``size`` is the
  message body only and attachments are a separate integer count.

None of these is a bug in this module, and all three come back to life on r5+.
But it means r4.2 carries 16 informative features rather than 18, file.csv
contributes one number three times, and any feature-importance plot will show
the collinearity. Better stated than discovered.

Public API:
    LogPartial               — one chunk's per-(user, day) partial aggregates
    build_*_partial          — the five per-log chunk builders
    LOG_BUILDERS, LOG_FEATURES, LOG_INPUT_COLUMNS
    aggregate_log            — fold a stream of chunks into one log's features
    combine_daily_features   — join the five logs, zero-fill, label
    daily_features           — the two above, wired together
    daily_features_from_dataset / daily_features_from_dir — the two entry points
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from innersight.config import (
    BUSINESS_HOURS_END,
    BUSINESS_HOURS_START,
    CERT_CLOUD_DOMAINS,
    CERT_JOB_DOMAINS,
    CSV_CHUNK_SIZE,
    FEATURE_NAMES,
    INTERNAL_DOMAIN,
    LARGE_ATTACHMENT_BYTES,
)
from innersight.data.answers import load_insiders, malicious_day_set
from innersight.data.dataset import (
    LOG_DTYPES,
    LOG_NAMES,
    LOG_SCHEMAS,
    R42_VERSION,
    CertDataset,
    validated_header,
)
from innersight.data.loaders import iter_log_csv_chunks

logger = logging.getLogger(__name__)


# ── Output contract ──────────────────────────────────────────────────────────

#: The two key columns. Every downstream stage groups or joins on these.
USER_COLUMN: Final[str] = "user"
DAY_COLUMN: Final[str] = "day"
KEY_COLUMNS: Final[list[str]] = [USER_COLUMN, DAY_COLUMN]

#: Ground-truth label column appended by :func:`combine_daily_features`.
LABEL_COLUMN: Final[str] = "is_malicious"

#: Which of the 18 features each log produces. The union is exactly
#: FEATURE_NAMES and the sets are disjoint — asserted in tests, because a
#: feature owned by two logs would be silently overwritten in the join and a
#: feature owned by none would be silently zero.
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

#: The features that are distinct counts rather than sums. If you ever add a
#: third, it must go through the ``distinct`` side of :class:`LogPartial` — a
#: distinct count computed per chunk and then added is wrong, silently, and
#: only in proportion to how many chunks a user-day happens to straddle.
DISTINCT_FEATURES: Final[frozenset[str]] = frozenset(
    {"unique_pcs_used", "unique_filenames"}
)

#: Everything else is an integer count. ``total_email_size`` is a byte total
#: that can exceed a 32-bit int and arrives as float because ``size`` is
#: nullable, so it stays float64.
FLOAT_FEATURES: Final[frozenset[str]] = frozenset({"total_email_size"})

#: Columns each builder actually reads. Passed to the CSV reader as a
#: projection, which is what makes streaming http.csv cheap: three columns
#: instead of six, and in particular never materialising ``content``.
LOG_INPUT_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "logon": ("date", "user", "pc", "activity"),
    "device": ("date", "user", "activity"),
    "email": ("date", "user", "to", "cc", "bcc", "size"),
    "http": ("date", "user", "url"),
    "file": ("date", "user", "filename"),
}

#: Read too if the file provides it (r5+ ships it; r4.2 does not).
_FILE_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("to_removable_media",)

#: Recipient columns scanned for external addresses.
#:
#: DELIBERATE CHANGE FROM THE ORIGINAL PROJECT, which checked ``to`` alone. An
#: email addressed to a colleague and blind-copied to a personal account then
#: reads as entirely internal — which is precisely the scenario-4 exfiltration
#: pattern. Measured on 400k real r4.2 email rows, cc/bcc contributes 6.3% of
#: all external mail that ``to`` alone does not see.
RECIPIENT_COLUMNS: Final[tuple[str, ...]] = ("to", "cc", "bcc")

#: Strings :func:`_truthy` accepts. CERT writes ``True``/``False``; the extra
#: spellings cost nothing and stop a hand-edited fixture from silently reading
#: as all-false.
_TRUE_STRINGS: Final[frozenset[str]] = frozenset({"true", "1", "yes", "y", "t"})

#: Chunks folded together before the partial lists are compacted. Without this
#: a 13.5 GB http.csv at the default chunk size leaves ~280k small frames
#: pinned until the end; with it, at most this many.
_COMPACT_EVERY: Final[int] = 64

#: Chunks between progress log lines while streaming.
_LOG_EVERY: Final[int] = 100


def _alternation(domains: Sequence[str], name: str) -> str:
    """Build a lowercase regex alternation from a domain tuple.

    Raises on an empty tuple rather than returning ``''``, which as a regex
    matches every string — the failure mode where a misconfigured list makes
    every web request look like a job-site visit.
    """
    if not domains:
        raise ValueError(
            f"{name} is empty. An empty URL list would compile to a pattern that "
            f"matches every URL, so every http row would count as a hit."
        )
    return "|".join(re.escape(domain.lower()) for domain in domains)


#: Matched against a lowercased URL. See config.CERT_JOB_DOMAINS for why these
#: are full domains and not the generic substrings the original project used.
_JOB_URL_PATTERN: Final[str] = _alternation(CERT_JOB_DOMAINS, "CERT_JOB_DOMAINS")
_CLOUD_URL_PATTERN: Final[str] = _alternation(CERT_CLOUD_DOMAINS, "CERT_CLOUD_DOMAINS")

#: ``@dtaa.com`` not followed by more domain characters, so a lookalike host
#: such as ``@dtaa.com.example.net`` is correctly counted as external.
_INTERNAL_ADDRESS_PATTERN: Final[str] = rf"@{re.escape(INTERNAL_DOMAIN.lower())}(?![\w.-])"


# ── One chunk's contribution ─────────────────────────────────────────────────

@dataclass(frozen=True)
class LogPartial:
    """What one chunk of one log contributes to the daily table.

    Attributes:
        sums: Additive features, indexed by ``(user, day)``. Combining two of
            these is addition.
        distinct: ``{feature_name: DataFrame[user, day, value]}`` of unique
            triples. Combining two of these is concatenate-then-deduplicate;
            the count is only taken once, at the very end.
    """

    sums: pd.DataFrame
    distinct: dict[str, pd.DataFrame]


# ── The five builders ────────────────────────────────────────────────────────

def build_logon_partial(chunk: pd.DataFrame) -> LogPartial:
    """Per-(user, day) partials from a chunk of ``logon.csv``.

    ``after_hours_logons`` and ``weekend_logons`` count logons only, not
    logoffs: a logon at 02:00 is the anomaly, and counting the matching logoff
    would double-report the same event.
    """
    keys = _key_frame(chunk)
    when = chunk["date"].dt
    is_logon = chunk["activity"] == "Logon"
    after_hours = (when.hour < BUSINESS_HOURS_START) | (when.hour >= BUSINESS_HOURS_END)
    weekend = when.dayofweek >= 5

    sums = _sum_by_key(
        keys,
        {
            "logon_count": _count(is_logon),
            "logoff_count": _count(chunk["activity"] == "Logoff"),
            "after_hours_logons": _count(is_logon & after_hours),
            "weekend_logons": _count(is_logon & weekend),
        },
    )
    return LogPartial(
        sums=sums,
        distinct={"unique_pcs_used": _distinct_by_key(keys, chunk["pc"])},
    )


def build_device_partial(chunk: pd.DataFrame) -> LogPartial:
    """Per-(user, day) partials from a chunk of ``device.csv``.

    ``after_hours_usb`` counts *any* device event outside business hours, not
    just connects. A thumb drive plugged in at 18:55 and pulled at 19:30 is one
    after-hours event either way, and the CERT readme calls out after-hours
    device use — not after-hours connects specifically — as significant.
    """
    keys = _key_frame(chunk)
    when = chunk["date"].dt
    after_hours = (when.hour < BUSINESS_HOURS_START) | (when.hour >= BUSINESS_HOURS_END)

    sums = _sum_by_key(
        keys,
        {
            "usb_connect_count": _count(chunk["activity"] == "Connect"),
            "usb_disconnect_count": _count(chunk["activity"] == "Disconnect"),
            "after_hours_usb": _count(after_hours),
        },
    )
    return LogPartial(sums=sums, distinct={})


def build_file_partial(chunk: pd.DataFrame) -> LogPartial:
    """Per-(user, day) partials from a chunk of ``file.csv``.

    A missing ``to_removable_media`` column means every row is a removable-media
    copy. That is not a guess: the r4.2 readme states "each entry represents a
    file copy to a removable media device", which is why r4.2 ships no such
    column and why :func:`~innersight.data.dataset.load_r42` synthesises it as
    ``'True'``. Treating absence as all-true here is what keeps the streaming
    path and the CertDataset path in exact agreement.

    The visible consequence on r4.2 is that ``file_to_removable_count`` equals
    ``file_count`` exactly. That is correct, not a bug; the two diverge on r5+
    where the column ships explicitly.
    """
    keys = _key_frame(chunk)
    if "to_removable_media" in chunk.columns:
        to_removable = _count(_truthy(chunk["to_removable_media"]))
    else:
        to_removable = _ones(chunk)

    sums = _sum_by_key(
        keys,
        {
            "file_count": _ones(chunk),
            "file_to_removable_count": to_removable,
        },
    )
    return LogPartial(
        sums=sums,
        distinct={"unique_filenames": _distinct_by_key(keys, chunk["filename"])},
    )


def build_email_partial(chunk: pd.DataFrame) -> LogPartial:
    """Per-(user, day) partials from a chunk of ``email.csv``.

    r4.2's email log has no ``activity`` column — every row is a message that
    the user in the ``user`` column sent, so ``email_sent_count`` is a row
    count. See :data:`RECIPIENT_COLUMNS` for the cc/bcc decision.
    """
    keys = _key_frame(chunk)
    size = pd.to_numeric(chunk["size"], errors="coerce")

    sums = _sum_by_key(
        keys,
        {
            "email_sent_count": _ones(chunk),
            "email_to_external_count": _count(_has_external_recipient(chunk)),
            "large_attachment_count": _count(size > LARGE_ATTACHMENT_BYTES),
            "total_email_size": size.fillna(0.0).astype("float64"),
        },
    )
    return LogPartial(sums=sums, distinct={})


def build_http_partial(chunk: pd.DataFrame) -> LogPartial:
    """Per-(user, day) partials from a chunk of ``http.csv``.

    All three features are pure sums, which is why the largest file in the
    dataset is the cheapest to aggregate: no state carries between chunks
    beyond running totals.

    Matching is substring-against-lowercased-URL rather than host equality
    because one entry, ``aol.com/jobs``, is a path and not a host.
    """
    keys = _key_frame(chunk)
    url = chunk["url"].astype("str").fillna("").str.lower()

    sums = _sum_by_key(
        keys,
        {
            "http_request_count": _ones(chunk),
            "job_search_visits": _count(url.str.contains(_JOB_URL_PATTERN, regex=True)),
            "cloud_upload_visits": _count(
                url.str.contains(_CLOUD_URL_PATTERN, regex=True)
            ),
        },
    )
    return LogPartial(sums=sums, distinct={})


#: Builder per log name. Keyed the same way as
#: :data:`~innersight.data.dataset.LOG_SCHEMAS`, so the three registries line up.
LOG_BUILDERS: Final[dict[str, Callable[[pd.DataFrame], LogPartial]]] = {
    "logon": build_logon_partial,
    "device": build_device_partial,
    "file": build_file_partial,
    "email": build_email_partial,
    "http": build_http_partial,
}


# ── Folding a stream of chunks ───────────────────────────────────────────────

def aggregate_log(
    log: str,
    chunks: Iterable[pd.DataFrame],
    *,
    compact_every: int = _COMPACT_EVERY,
) -> pd.DataFrame:
    """Reduce a stream of chunks of one log to its per-(user, day) features.

    This is the testable core of the slice: it takes any iterable of chunks, so
    a test can hand it two two-row frames and a production run can hand it a
    generator over 13.5 GB, and the code path is identical.

    Args:
        log: Log name, one of :data:`~innersight.data.dataset.LOG_NAMES`.
        chunks: Any iterable of raw chunks for that log.
        compact_every: Fold accumulated partials together every this many
            chunks. Purely a memory knob; the result is identical either way.

    Returns:
        A DataFrame indexed by ``(user, day)`` with exactly
        ``LOG_FEATURES[log]`` as columns. Empty (but correctly shaped and
        typed) if the stream yielded no rows.

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


def combine_daily_features(
    per_log: Mapping[str, pd.DataFrame],
    *,
    malicious_days: Iterable[tuple[str, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Join the per-log tables into the daily feature table.

    Three rules, each of which is a decision rather than a detail:

    * The key universe is the **union** of ``(user, day)`` across every log, not
      any one log's keys. A day that exists only in http still produces a row.
    * Missing means **zero**, not NaN. "No USB events today" is an observation
      of zero, not an absence of data, and NaN here would poison the z-scores
      in Slice 3.
    * Columns come out in :data:`~innersight.schema.FEATURE_NAMES` order,
      always. That ordering is the vector layout every later stage assumes.

    Args:
        per_log: ``{log_name: features}`` as returned by :func:`aggregate_log`.
            Logs may be omitted; their features come out as zero.
        malicious_days: ``(user, midnight)`` pairs from
            :meth:`~innersight.data.dataset.CertDataset.malicious_days`. ``None``
            labels every row ``False``.

    Returns:
        ``user``, ``day``, the 18 features in canonical order, ``is_malicious``.
        Sorted by ``(user, day)`` with a fresh RangeIndex.
    """
    key_index = _union_key_index(per_log.values())

    columns: dict[str, pd.Series] = {}
    for frame in per_log.values():
        aligned = frame.reindex(key_index)
        for feature in frame.columns:
            columns[feature] = aligned[feature]

    absent = [feature for feature in FEATURE_NAMES if feature not in columns]
    if absent:
        logger.warning(
            "combine_daily_features | no log supplied %s; emitting them as zero",
            absent,
        )

    daily = pd.DataFrame(
        {feature: columns.get(feature, 0) for feature in FEATURE_NAMES},
        index=key_index,
    ).fillna(0)

    # Cast last, after filling: an int64 column cannot hold the NaN that
    # reindexing a missing (user, day) introduces.
    for feature in FEATURE_NAMES:
        target = "float64" if feature in FLOAT_FEATURES else "int64"
        daily[feature] = daily[feature].astype(target)

    daily = daily.sort_index().reset_index()
    daily[LABEL_COLUMN] = _label(daily, malicious_days)
    return daily[[*KEY_COLUMNS, *FEATURE_NAMES, LABEL_COLUMN]]


def daily_features(
    per_log_chunks: Mapping[str, Iterable[pd.DataFrame]],
    *,
    malicious_days: Iterable[tuple[str, pd.Timestamp]] | None = None,
    compact_every: int = _COMPACT_EVERY,
) -> pd.DataFrame:
    """Aggregate every log's chunk stream and combine the results.

    Args:
        per_log_chunks: ``{log_name: iterable of chunks}``.
        malicious_days: See :func:`combine_daily_features`.
        compact_every: See :func:`aggregate_log`.

    Returns:
        The daily feature table.
    """
    per_log = {
        log: aggregate_log(log, chunks, compact_every=compact_every)
        for log, chunks in per_log_chunks.items()
    }
    return combine_daily_features(per_log, malicious_days=malicious_days)


# ── The two entry points ─────────────────────────────────────────────────────

def daily_features_from_dataset(dataset: CertDataset) -> pd.DataFrame:
    """Build the daily table from an already-loaded :class:`CertDataset`.

    For the synthetic demo and any dataset small enough that Slice 1 could load
    it whole. Each log is handed to the aggregator as a single chunk, so this
    exercises exactly the same code as the streaming path.

    Args:
        dataset: A loaded dataset.

    Returns:
        The daily feature table, labelled from the dataset's answer key.
    """
    return daily_features(
        {log: [frame] for log, frame in dataset.logs.items()},
        malicious_days=dataset.malicious_days(),
    )


def daily_features_from_dir(
    data_dir: Path | str,
    *,
    chunk_size: int = CSV_CHUNK_SIZE,
    version: str = R42_VERSION,
    compact_every: int = _COMPACT_EVERY,
) -> pd.DataFrame:
    """Build the daily table by streaming a dataset directory.

    The path for real r4.2. Memory stays bounded by population x days, so this
    runs on a laptop against the full 16 GB release.

    Args:
        data_dir: Dataset directory, laid out like the r4.2 release.
        chunk_size: Rows per CSV chunk.
        version: Answer-key version to label against.
        compact_every: See :func:`aggregate_log`.

    Returns:
        The daily feature table, labelled from ``answers/insiders.csv``.

    Raises:
        FileNotFoundError: If the directory or a required file is missing.
        ValueError: If a log fails its schema contract, or the answer key has
            no rows for *version*.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. Set INNERSIGHT_DATA_DIR "
            f"or pass --data-dir explicitly."
        )

    insiders = load_insiders(data_dir / "answers" / "insiders.csv", version=version)
    per_log_chunks = {
        log: _stream_log(data_dir, log, chunk_size=chunk_size) for log in LOG_NAMES
    }
    return daily_features(
        per_log_chunks,
        malicious_days=malicious_day_set(insiders),
        compact_every=compact_every,
    )


def _stream_log(
    data_dir: Path,
    log: str,
    *,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """Yield validated, column-projected chunks of one log.

    The header is checked against the log's full Slice 0 contract before a
    single data row is read, then only the columns the builder needs are
    materialised. On http.csv that is three columns out of six, and in
    particular never ``content`` — which is most of the file's bytes.
    """
    path = data_dir / f"{log}.csv"
    header = validated_header(path, LOG_SCHEMAS[log])

    wanted = [column for column in LOG_INPUT_COLUMNS[log] if column in header]
    if log == "file":
        wanted += [c for c in _FILE_OPTIONAL_COLUMNS if c in header]

    logger.info("_stream_log | %s: reading %s from %s", log, wanted, path.name)
    rows = 0
    for number, chunk in enumerate(
        iter_log_csv_chunks(
            path, dtype=LOG_DTYPES[log], columns=wanted, chunk_size=chunk_size
        ),
        start=1,
    ):
        rows += len(chunk)
        if number % _LOG_EVERY == 0:
            logger.info("_stream_log | %s: %d rows so far", log, rows)
        yield chunk
    logger.info("_stream_log | %s: %d rows total", log, rows)


# ── Small shared helpers ─────────────────────────────────────────────────────

def _key_frame(chunk: pd.DataFrame) -> pd.DataFrame:
    """The ``(user, day)`` key for every row of *chunk*.

    Days are midnight-normalised so they match the ``(user, midnight)`` pairs
    :func:`~innersight.data.answers.malicious_day_set` produces.
    """
    return pd.DataFrame(
        {USER_COLUMN: chunk[USER_COLUMN], DAY_COLUMN: chunk["date"].dt.normalize()},
        index=chunk.index,
    )


def _ones(chunk: pd.DataFrame) -> pd.Series:
    """A 1-per-row column, for features that are plain row counts."""
    return pd.Series(1, index=chunk.index, dtype="int64")


def _count(mask: pd.Series) -> pd.Series:
    """A possibly-nullable boolean mask as a 0/1 int64 column.

    Going through ``boolean`` first is what makes this safe for both a plain
    numpy bool mask and a nullable one: a comparison against a column holding
    NA yields NA, and NA must count as "did not happen", not propagate.
    """
    return mask.astype("boolean").fillna(False).astype("int64")


def _truthy(series: pd.Series) -> pd.Series:
    """Interpret a CERT boolean-ish column as an actual boolean mask.

    Handles both a real bool column and the string ``'True'`` that r5+ ships and
    :func:`~innersight.data.dataset.load_r42` synthesises for r4.2. Never
    compares a dtype to a literal: under pandas 3.0 text columns have their own
    dtype and ``series.dtype == 'object'`` is False, so a dtype check written
    that way silently takes the wrong branch.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean").fillna(False)
    return series.astype("str").str.strip().str.lower().isin(_TRUE_STRINGS)


def _has_external_recipient(chunk: pd.DataFrame) -> pd.Series:
    """True where any address across ``to``/``cc``/``bcc`` is outside the org.

    Counting rather than splitting: every address contains exactly one ``@``, so
    "more ``@`` signs than internal-domain matches" means at least one recipient
    is external. That is one vectorised regex count per column set instead of a
    per-row split, which matters on a 1.36 GB email log.
    """
    parts = [
        chunk[column].astype("str").fillna("").str.lower()
        for column in RECIPIENT_COLUMNS
        if column in chunk.columns
    ]
    recipients = parts[0]
    for part in parts[1:]:
        recipients = recipients + ";" + part

    total = recipients.str.count("@")
    internal = recipients.str.count(_INTERNAL_ADDRESS_PATTERN)
    return total > internal


def _sum_by_key(keys: pd.DataFrame, columns: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Group *columns* by ``(user, day)`` and sum them.

    ``dropna=True`` is the default and is wanted: a row with no user or an
    unparseable date cannot be attributed to anyone's day, so it is counted for
    nobody rather than being silently assigned to a NaN key.
    """
    return keys.assign(**columns).groupby(KEY_COLUMNS, sort=False).sum()


def _distinct_by_key(keys: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """Unique ``(user, day, value)`` triples, the input to a distinct count."""
    return (
        keys.assign(value=values)
        .dropna()
        .drop_duplicates(ignore_index=True)
    )


def _compact_sums(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Add a list of per-(user, day) sum frames together."""
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts).groupby(level=KEY_COLUMNS, sort=False).sum()


def _compact_distinct(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate and deduplicate triple frames — never count in between."""
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
    """The union of every frame's ``(user, day)`` index, sorted and unique."""
    union: pd.Index | None = None
    for frame in frames:
        union = frame.index if union is None else union.union(frame.index)
    if union is None or len(union) == 0:
        return pd.MultiIndex.from_arrays(
            [pd.Index([], dtype="str"), pd.DatetimeIndex([])], names=KEY_COLUMNS
        )
    return union.unique().sort_values().set_names(KEY_COLUMNS)


def _label(
    daily: pd.DataFrame,
    malicious_days: Iterable[tuple[str, pd.Timestamp]] | None,
) -> pd.Series:
    """``is_malicious`` for every row, from the answer key's (user, day) pairs.

    Derived from :func:`~innersight.data.answers.malicious_day_set` rather than
    re-implemented, so there is exactly one definition of "inside an attack
    window" in the project.
    """
    if malicious_days is None:
        return pd.Series(False, index=daily.index, dtype="bool")
    pairs = list(malicious_days)
    if not pairs:
        return pd.Series(False, index=daily.index, dtype="bool")

    keys = pd.MultiIndex.from_arrays([daily[USER_COLUMN], daily[DAY_COLUMN]])
    return pd.Series(keys.isin(pairs), index=daily.index, dtype="bool")
