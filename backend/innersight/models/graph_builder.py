"""Graph node and edge feature builders for the InnerSight heterogeneous graph.

Pipeline position:
  load_raw_logs() → build_user_nodes()   ← Task 4 (this file)
                  → build_pc_nodes()     ← Task 5
                  → build_url_nodes()    ← Task 5
                  → build_file_nodes()   ← Task 5
                  → build_edges()        ← Tasks 6-7
                  → HeteroData           ← Task 8

All builders share the same call signature:
  build_<type>_nodes(logs, start_date, end_date) -> (tensor, <type>_to_idx)

The returned tensor is float32 with shape (num_nodes, NODE_FEATURE_DIM).
The index dict maps entity id → row index and is used by edge builders.
"""

from __future__ import annotations

import logging
import os
import sys

import time as _time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from innersight.config import (
    BUSINESS_HOURS_START,
    BUSINESS_HOURS_END,
    JOB_KEYWORDS,
    CLOUD_KEYWORDS,
    INTERNAL_DOMAIN,
    CERT_JOB_DOMAINS,
    CERT_CLOUD_DOMAINS,
    CERT_KEYLOGGER_DOMAINS,
)
from innersight.models.graph_schema import (
    USER_FEATURE_DIM,
    PC_FEATURE_DIM,
    URL_FEATURE_DIM,
    FILE_FEATURE_DIM,
    LOGON_EDGE_DIM,
    USB_EDGE_DIM,
    EMAIL_EDGE_DIM,
    HTTP_EDGE_DIM,
    FILE_EDGE_DIM,
    EDGE_LOGON,
    EDGE_USB,
    EDGE_EMAIL,
    EDGE_HTTP,
    EDGE_FILE_COPY,
    REV_EDGE_LOGON,
    REV_EDGE_USB,
    REV_EDGE_EMAIL,
    REV_EDGE_HTTP,
    REV_EDGE_FILE_COPY,
    NODE_USER,
    NODE_PC,
    NODE_URL,
    NODE_FILE,
    USER_TEMPORAL_DIM,
    WINDOWED_LOGON_EDGE_DIM,
    WINDOWED_USB_EDGE_DIM,
    WINDOWED_EMAIL_EDGE_DIM,
    WINDOWED_HTTP_EDGE_DIM,
    WINDOWED_FILE_EDGE_DIM,
)

logger = logging.getLogger(__name__)

# Canonical column order — must stay in sync with USER_FEATURE_DIM.
_USER_FEAT_COLS: list[str] = [
    'mean_daily_logons',       # avg logon events per active logon day
    'std_daily_logons',        # day-to-day variability in logon count
    'total_unique_pcs',        # distinct machines visited across the window
    'after_hours_logon_ratio', # fraction of logons outside business hours
    'weekend_logon_ratio',     # fraction of logons on Sat/Sun
    'mean_daily_usb',          # avg USB-connect events per active USB day
    'total_usb_days',          # number of days with any USB connect activity
    'mean_daily_emails_sent',  # avg emails sent per active email day
    'mean_email_size',         # mean email size in bytes across the window
    'external_email_ratio',    # fraction of emails addressed outside dtaa.com
    'mean_daily_http',         # avg HTTP requests per active HTTP day
    'job_search_ratio',        # fraction of HTTP visits to job-search URLs
    'cloud_upload_ratio',      # fraction of HTTP visits to cloud/leak sites
    'mean_daily_file_copies',  # avg file copies to removable media per active day
    'total_file_copies',       # total files copied to removable media
    'active_days',             # days with any event across all log types
]

assert len(_USER_FEAT_COLS) == USER_FEATURE_DIM, (
    f"_USER_FEAT_COLS has {len(_USER_FEAT_COLS)} columns but "
    f"USER_FEATURE_DIM={USER_FEATURE_DIM}. Update graph_schema.py."
)

_PC_FEAT_COLS: list[str] = [
    'num_distinct_users',      # unique users who logged in
    'total_logon_events',      # total logon events to this PC
    'is_shared',               # 1.0 if >5 distinct users (shared workstation)
    'after_hours_usage_ratio', # fraction of logons outside business hours
    'usb_event_count',         # total USB connects on this machine
    'mean_session_count',      # avg logon events per active day
    'weekend_usage_ratio',     # fraction of logons on Sat/Sun
    'unique_days_active',      # distinct days with any activity (logon or USB)
]

assert len(_PC_FEAT_COLS) == PC_FEATURE_DIM, (
    f"_PC_FEAT_COLS has {len(_PC_FEAT_COLS)} columns but "
    f"PC_FEATURE_DIM={PC_FEATURE_DIM}. Update graph_schema.py."
)

_URL_FEAT_COLS: list[str] = [
    'total_visits',            # total HTTP requests to this domain
    'unique_visitors',         # distinct users who visited
    'is_job_related',          # 1.0 if domain matches JOB_KEYWORDS
    'is_cloud_related',        # 1.0 if domain matches CLOUD_KEYWORDS
    'mean_daily_visits',       # avg visits per active day
    'visitor_concentration',   # max single-user visits / total visits
    'weekend_visit_ratio',     # fraction of visits on Sat/Sun
    'after_hours_visit_ratio', # fraction of visits outside business hours
]

assert len(_URL_FEAT_COLS) == URL_FEATURE_DIM, (
    f"_URL_FEAT_COLS has {len(_URL_FEAT_COLS)} columns but "
    f"URL_FEATURE_DIM={URL_FEATURE_DIM}. Update graph_schema.py."
)

_FILE_FEAT_COLS: list[str] = [
    'total_copies',        # times copied to removable media
    'unique_users',        # distinct users who copied this file
    'file_type_code',      # extension category: doc=1, pdf=2, exe=3, archive=4, other=0
    'avg_copy_hour',       # mean hour-of-day of copy events (0 if no copies)
    'is_after_hours_copy', # fraction of copies outside business hours
    'content_topic_count', # distinct suspicious keywords found in content field
]

assert len(_FILE_FEAT_COLS) == FILE_FEATURE_DIM, (
    f"_FILE_FEAT_COLS has {len(_FILE_FEAT_COLS)} columns but "
    f"FILE_FEATURE_DIM={FILE_FEATURE_DIM}. Update graph_schema.py."
)

# File extension → type code  (doc=1, pdf=2, exe=3, archive=4, other=0)
_EXT_CODE: dict[str, int] = {
    '.doc': 1, '.docx': 1, '.xls': 1, '.xlsx': 1,
    '.ppt': 1, '.pptx': 1, '.txt': 1, '.csv': 1,
    '.pdf': 2,
    '.exe': 3, '.dll': 3, '.bat': 3, '.cmd': 3, '.sh': 3,
    '.zip': 4, '.tar': 4, '.gz': 4, '.rar': 4, '.7z': 4, '.bz2': 4,
}

# Suspicious keywords checked in the file content field
_CONTENT_KEYWORDS: frozenset[str] = frozenset(
    [*JOB_KEYWORDS, *CLOUD_KEYWORDS, 'resume', 'salary', 'offer', 'confidential']
)

# Edge feature column names — order must match the tensors built below.
_LOGON_EDGE_COLS: list[str] = [
    'hour_normalized',  # hour_of_day / 24.0
    'is_weekend',       # 1.0 if Sat or Sun
    'is_after_hours',   # 1.0 if outside BUSINESS_HOURS_START/END
    'day_of_week',      # day_of_week / 7.0  (0=Mon … 6=Sun, normalised)
]

assert len(_LOGON_EDGE_COLS) == LOGON_EDGE_DIM, (
    f"_LOGON_EDGE_COLS has {len(_LOGON_EDGE_COLS)} entries but "
    f"LOGON_EDGE_DIM={LOGON_EDGE_DIM}. Update graph_schema.py."
)

_USB_EDGE_COLS: list[str] = [
    'hour_normalized',  # hour_of_day / 24.0
    'is_weekend',       # 1.0 if Sat or Sun
    'is_after_hours',   # 1.0 if outside business hours
]

assert len(_USB_EDGE_COLS) == USB_EDGE_DIM, (
    f"_USB_EDGE_COLS has {len(_USB_EDGE_COLS)} entries but "
    f"USB_EDGE_DIM={USB_EDGE_DIM}. Update graph_schema.py."
)

_EMAIL_EDGE_COLS: list[str] = [
    'email_size_normalized',  # size_bytes / 1_000_000 (MB scale)
    'attachment_count',       # attachments clipped to [0,10] then / 10.0
    'is_external_thread',     # 1.0 if ANY recipient is outside dtaa.com
    'hour_normalized',        # hour_of_day / 24.0
    'is_after_hours',         # 1.0 if outside business hours
]

assert len(_EMAIL_EDGE_COLS) == EMAIL_EDGE_DIM, (
    f"_EMAIL_EDGE_COLS has {len(_EMAIL_EDGE_COLS)} entries but "
    f"EMAIL_EDGE_DIM={EMAIL_EDGE_DIM}. Update graph_schema.py."
)

_HTTP_EDGE_COLS: list[str] = [
    'hour_normalized',  # hour_of_day / 24.0
    'is_weekend',       # 1.0 if Sat or Sun
    'is_after_hours',   # 1.0 if outside business hours
]

assert len(_HTTP_EDGE_COLS) == HTTP_EDGE_DIM, (
    f"_HTTP_EDGE_COLS has {len(_HTTP_EDGE_COLS)} entries but "
    f"HTTP_EDGE_DIM={HTTP_EDGE_DIM}. Update graph_schema.py."
)

_FILE_EDGE_COLS: list[str] = [
    'hour_normalized',  # hour_of_day / 24.0
    'is_weekend',       # 1.0 if Sat or Sun
    'is_after_hours',   # 1.0 if outside business hours
    'file_type_code',   # extension category: doc=1 pdf=2 exe=3 archive=4 other=0
]

assert len(_FILE_EDGE_COLS) == FILE_EDGE_DIM, (
    f"_FILE_EDGE_COLS has {len(_FILE_EDGE_COLS)} entries but "
    f"FILE_EDGE_DIM={FILE_EDGE_DIM}. Update graph_schema.py."
)


# ── Private helpers ───────────────────────────────────────────────────────────

def _filter_window(
    df: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> pd.DataFrame:
    """Slice rows where date ∈ [t_start, t_end]. Safe on empty DataFrames.

    Uses binary search (O(log n)) when the DataFrame carries
    ``df.attrs['_date_sorted'] = True`` (set by ``_presort_logs`` in
    ``train_temporal_graph``), which reduces the 28 M-row HTTP scan from
    ~10 s to microseconds per window.  Falls back to a linear boolean-mask
    scan for unsorted DataFrames.
    """
    if df.empty or 'date' not in df.columns:
        return df
    if df.attrs.get('_date_sorted'):
        dates = df['date'].values                               # numpy datetime64[ns]
        lo = int(np.searchsorted(dates, np.datetime64(t_start, 'ns'), side='left'))
        hi = int(np.searchsorted(dates, np.datetime64(t_end,   'ns'), side='right'))
        return df.iloc[lo:hi].copy()
    mask = (df['date'] >= t_start) & (df['date'] <= t_end)
    return df.loc[mask].copy()


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise num / den; returns 0.0 wherever den is zero or NaN."""
    safe_den = den.where(den != 0, other=np.nan)
    return (num / safe_den).fillna(0.0)


def _logon_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by user_id with five logon aggregate columns."""
    cols = [
        'mean_daily_logons', 'std_daily_logons', 'total_unique_pcs',
        'after_hours_logon_ratio', 'weekend_logon_ratio',
    ]
    if df.empty or 'user' not in df.columns:
        return pd.DataFrame(columns=cols)

    df = df.copy()
    df['_day']        = df['date'].dt.normalize()
    df['_is_logon']   = df['activity'].str.lower() == 'logon'
    h = df['date'].dt.hour
    df['_after_hours'] = (h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)
    df['_weekend']    = df['date'].dt.dayofweek >= 5   # 5=Sat, 6=Sun

    logons = df[df['_is_logon']].copy()
    if logons.empty:
        return pd.DataFrame(columns=cols)

    # daily logon count → per-user mean / std
    daily = logons.groupby(['user', '_day']).size().rename('cnt')
    stats = daily.groupby(level='user').agg(['mean', 'std'])
    stats.columns = ['mean_daily_logons', 'std_daily_logons']
    stats['std_daily_logons'] = stats['std_daily_logons'].fillna(0.0)  # 1-day users

    by_user = logons.groupby('user')
    n = by_user.size()

    result = stats.copy()
    result['total_unique_pcs']        = by_user['pc'].nunique()
    result['after_hours_logon_ratio'] = _ratio(by_user['_after_hours'].sum(), n)
    result['weekend_logon_ratio']     = _ratio(by_user['_weekend'].sum(),     n)

    return result[cols]


def _device_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by user_id with two USB aggregate columns."""
    cols = ['mean_daily_usb', 'total_usb_days']
    if df.empty or 'user' not in df.columns:
        return pd.DataFrame(columns=cols)

    df = df.copy()
    df['_day'] = df['date'].dt.normalize()
    connects = df[df['activity'].str.lower() == 'connect'].copy()
    if connects.empty:
        return pd.DataFrame(columns=cols)

    daily = connects.groupby(['user', '_day']).size().rename('cnt')
    by_user = daily.groupby(level='user')

    return pd.DataFrame({
        'mean_daily_usb': by_user.mean(),
        'total_usb_days': by_user.count(),   # number of days with ≥1 connect
    })[cols]


def _email_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by user_id with three email aggregate columns."""
    cols = ['mean_daily_emails_sent', 'mean_email_size', 'external_email_ratio']
    if df.empty or 'user' not in df.columns:
        return pd.DataFrame(columns=cols)

    df = df.copy()
    df['_day']      = df['date'].dt.normalize()
    df['_size']     = pd.to_numeric(df.get('size', 0), errors='coerce').fillna(0.0)

    def _has_external(to_field) -> bool:
        if pd.isna(to_field):
            return False
        return any(INTERNAL_DOMAIN not in addr for addr in str(to_field).split(';'))

    df['_external'] = df['to'].apply(_has_external)

    by_user  = df.groupby('user')
    n        = by_user.size()
    mean_daily = (
        df.groupby(['user', '_day']).size()
          .groupby(level='user').mean()
    )

    return pd.DataFrame({
        'mean_daily_emails_sent': mean_daily,
        'mean_email_size':        by_user['_size'].mean(),
        'external_email_ratio':   _ratio(by_user['_external'].sum(), n),
    })[cols]


def _http_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by user_id with three HTTP aggregate columns."""
    cols = ['mean_daily_http', 'job_search_ratio', 'cloud_upload_ratio']
    if df.empty or 'user' not in df.columns:
        return pd.DataFrame(columns=cols)

    df  = df.copy()
    df['_day'] = df['date'].dt.normalize()
    url = df['url'].fillna('').str.lower()
    df['_job']   = url.apply(lambda u: any(k in u for k in JOB_KEYWORDS))
    df['_cloud'] = url.apply(lambda u: any(k in u for k in CLOUD_KEYWORDS))

    by_user  = df.groupby('user')
    n        = by_user.size()
    mean_daily = (
        df.groupby(['user', '_day']).size()
          .groupby(level='user').mean()
    )

    return pd.DataFrame({
        'mean_daily_http':    mean_daily,
        'job_search_ratio':   _ratio(by_user['_job'].sum(),   n),
        'cloud_upload_ratio': _ratio(by_user['_cloud'].sum(), n),
    })[cols]


def _file_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by user_id with two file-copy aggregate columns."""
    cols = ['mean_daily_file_copies', 'total_file_copies']
    if df.empty or 'user' not in df.columns:
        return pd.DataFrame(columns=cols)

    df = df.copy()
    df['_day'] = df['date'].dt.normalize()

    removable = df.get('to_removable_media', pd.Series(False, index=df.index))
    if removable.dtype == object:
        removable = removable.str.lower() == 'true'
    df['_removable'] = removable.fillna(False)

    copies = df[df['_removable']].copy()
    if copies.empty:
        return pd.DataFrame(columns=cols)

    daily  = copies.groupby(['user', '_day']).size().rename('cnt')
    by_user = daily.groupby(level='user')

    return pd.DataFrame({
        'mean_daily_file_copies': by_user.mean(),
        'total_file_copies':      by_user.sum(),
    })[cols]


def _active_days_feature(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Count distinct active days across all log types per user."""
    parts = []
    for df in dfs:
        if df.empty or 'user' not in df.columns or 'date' not in df.columns:
            continue
        tmp = df[['user', 'date']].copy()
        tmp['_day'] = tmp['date'].dt.normalize()
        parts.append(tmp[['user', '_day']].drop_duplicates())

    if not parts:
        return pd.DataFrame(columns=['active_days'])

    combined    = pd.concat(parts, ignore_index=True).drop_duplicates()
    active_days = combined.groupby('user').size().rename('active_days')
    return active_days.to_frame()


def _extract_domain(url: str) -> str:
    """Return the netloc (domain) of a URL, lower-cased.

    Falls back to the raw string if urlparse cannot find a netloc —
    e.g. when the value is already a bare domain or is malformed.
    """
    from urllib.parse import urlparse
    try:
        netloc = urlparse(str(url)).netloc.lower()
        return netloc if netloc else str(url).lower()
    except Exception:
        return str(url).lower()


def _file_type_code(filename: str) -> int:
    """Map a filename extension to a numeric type code (see _EXT_CODE)."""
    ext = os.path.splitext(str(filename).lower())[1]
    return _EXT_CODE.get(ext, 0)


def _count_content_keywords(content) -> int:
    """Count distinct suspicious keywords present in a content string."""
    if content is None or (isinstance(content, float) and np.isnan(content)):
        return 0
    c = str(content).lower()
    return sum(1 for kw in _CONTENT_KEYWORDS if kw in c)


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the 'date' column is datetime64.

    load_raw_logs() already parses dates, but edge builders accept raw
    DataFrames so we defensively convert here.  The real CERT CSV stores
    dates as '01/02/2010 07:32:16'; pd.to_datetime handles both that format
    and ISO strings transparently.
    """
    if df.empty or 'date' not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    return df


def _build_edge_features_logon(df: pd.DataFrame) -> np.ndarray:
    """Vectorised feature matrix for logon/logoff rows — shape (n, LOGON_EDGE_DIM)."""
    h   = df['date'].dt.hour.to_numpy(dtype=np.float32)
    dow = df['date'].dt.dayofweek.to_numpy(dtype=np.float32)
    return np.column_stack([
        h / 24.0,                                                          # hour_normalized
        (dow >= 5).astype(np.float32),                                     # is_weekend
        ((h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)).astype(np.float32),  # is_after_hours
        dow / 7.0,                                                         # day_of_week
    ])


def _build_edge_features_usb(df: pd.DataFrame) -> np.ndarray:
    """Vectorised feature matrix for USB-connect rows — shape (n, USB_EDGE_DIM)."""
    h   = df['date'].dt.hour.to_numpy(dtype=np.float32)
    dow = df['date'].dt.dayofweek.to_numpy(dtype=np.float32)
    return np.column_stack([
        h / 24.0,                                                          # hour_normalized
        (dow >= 5).astype(np.float32),                                     # is_weekend
        ((h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)).astype(np.float32),  # is_after_hours
    ])


def _temporal3(df: pd.DataFrame) -> np.ndarray:
    """Return (n, 3) float32: [hour_normalized, is_weekend, is_after_hours].

    Shared basis for HTTP and file edge feature matrices.
    """
    h   = df['date'].dt.hour.to_numpy(dtype=np.float32)
    dow = df['date'].dt.dayofweek.to_numpy(dtype=np.float32)
    return np.column_stack([
        h / 24.0,
        (dow >= 5).astype(np.float32),
        ((h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)).astype(np.float32),
    ])


def _explode_and_map_recipients(
    df: pd.DataFrame,
    user_to_idx: dict[str, int],
) -> pd.DataFrame:
    """Expand email rows into one row per internal (sender, recipient) pair.

    Processes the ``to``, ``cc``, and ``bcc`` columns, splits semicolon-
    separated addresses, keeps only ``@<INTERNAL_DOMAIN>`` recipients that
    are present in ``user_to_idx``, and drops self-loops.

    Returns a DataFrame with columns:
        _sender_idx, _recip_idx, _size, _attach, _is_external, _hour, _is_after
    """
    if df.empty:
        return pd.DataFrame()

    user_map = pd.Series(user_to_idx)
    df = df.copy()

    # Per-row scalar features
    h = df['date'].dt.hour
    df['_hour']     = h.astype(float)
    df['_is_after'] = ((h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)).astype(float)
    df['_size']     = pd.to_numeric(df.get('size', 0), errors='coerce').fillna(0.0)
    df['_attach']   = (
        pd.to_numeric(df.get('attachments', 0), errors='coerce')
          .fillna(0.0).clip(0, 10)
    )

    # is_external_thread: any recipient address does NOT contain @INTERNAL_DOMAIN
    all_recip = df[['to', 'cc', 'bcc']].fillna('').agg(';'.join, axis=1)
    df['_is_external'] = all_recip.apply(
        lambda s: float(any(
            INTERNAL_DOMAIN not in addr
            for addr in s.split(';') if '@' in addr
        ))
    )

    # Map sender → index; drop rows where sender is unknown
    df['_sender_idx'] = df['user'].astype(str).map(user_map)
    df = df.dropna(subset=['_sender_idx'])
    df['_sender_idx'] = df['_sender_idx'].astype(int)

    keep = ['_sender_idx', '_size', '_attach', '_is_external', '_hour', '_is_after']
    domain_suffix = f'@{INTERNAL_DOMAIN.lower()}'
    results: list[pd.DataFrame] = []

    for col in ('to', 'cc', 'bcc'):
        if col not in df.columns:
            continue
        sub = df[keep + [col]].copy()
        # Split on ';', explode to one address per row
        sub[col] = sub[col].fillna('').str.split(';')
        sub = sub.explode(col)
        sub[col] = sub[col].str.strip().str.lower()
        # Keep only internal addresses with a known user
        sub = sub[sub[col].str.endswith(domain_suffix)]
        sub['_recip_user'] = sub[col].str.split('@').str[0]
        sub['_recip_idx']  = sub['_recip_user'].map(user_map)
        sub = sub.dropna(subset=['_recip_idx'])
        sub['_recip_idx'] = sub['_recip_idx'].astype(int)
        # Drop self-loops
        sub = sub[sub['_sender_idx'] != sub['_recip_idx']]
        results.append(sub[keep + ['_recip_idx']])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ── Public API ────────────────────────────────────────────────────────────────

def build_user_nodes(
    logs: dict,
    start_date: str,
    end_date: str,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Build user node feature tensors for the heterogeneous graph.

    Aggregates raw CERT log events over the given time window into a single
    float32 feature vector per user.  Users with no events in a particular
    log type receive 0.0 for those features.

    Args:
        logs:       Dict mapping log-type name → raw DataFrame, as returned by
                    ``pipeline.load_raw_logs()``.  Expected keys (any subset
                    is acceptable): ``'logon'``, ``'device'``, ``'email'``,
                    ``'http'``, ``'file'``.
        start_date: Inclusive window start as an ISO-8601 string, e.g.
                    ``'2010-01-01'``.
        end_date:   Inclusive window end,   e.g. ``'2010-09-30'``.

    Returns:
        A tuple ``(x, user_to_idx)`` where

        * ``x`` — ``float32`` tensor, shape ``(num_users, USER_FEATURE_DIM)``.
          Row ``i`` is the feature vector for the user whose id maps to ``i``
          in ``user_to_idx``.
        * ``user_to_idx`` — ``{user_id: row_index}`` dict used by edge builders
          to look up source/destination node indices.

    Feature columns in row order (16 total, matching USER_FEATURE_DIM):
        0   mean_daily_logons         avg logon events per active logon day
        1   std_daily_logons          day-to-day std of logon count
        2   total_unique_pcs          distinct machines visited
        3   after_hours_logon_ratio   fraction of logons outside business hours
        4   weekend_logon_ratio       fraction of logons on Sat/Sun
        5   mean_daily_usb            avg USB connects per active USB day
        6   total_usb_days            days with ≥1 USB connect
        7   mean_daily_emails_sent    avg emails sent per active email day
        8   mean_email_size           mean email size in bytes
        9   external_email_ratio      fraction of emails to non-dtaa.com
        10  mean_daily_http           avg HTTP requests per active HTTP day
        11  job_search_ratio          fraction of HTTP visits to job-search sites
        12  cloud_upload_ratio        fraction of HTTP visits to cloud/leak sites
        13  mean_daily_file_copies    avg file copies to removable media per day
        14  total_file_copies         total files copied to removable media
        15  active_days               days with any event across all logs
    """
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)

    # ── 1. Filter each log to the time window ─────────────────────────────────
    logon_w  = _filter_window(logs.get('logon',  pd.DataFrame()), t_start, t_end)
    device_w = _filter_window(logs.get('device', pd.DataFrame()), t_start, t_end)
    email_w  = _filter_window(logs.get('email',  pd.DataFrame()), t_start, t_end)
    http_w   = _filter_window(logs.get('http',   pd.DataFrame()), t_start, t_end)
    file_w   = _filter_window(logs.get('file',   pd.DataFrame()), t_start, t_end)

    # ── 2. Build the user roster from all log types ───────────────────────────
    all_users: set[str] = set()
    for df in (logon_w, device_w, email_w, http_w, file_w):
        if not df.empty and 'user' in df.columns:
            all_users.update(df['user'].dropna().astype(str).unique())

    sorted_users = sorted(all_users)
    user_to_idx  = {u: i for i, u in enumerate(sorted_users)}
    n_users      = len(sorted_users)

    logger.info(
        'build_user_nodes | window %s – %s | %d users found',
        start_date, end_date, n_users,
    )

    if n_users == 0:
        return torch.zeros((0, USER_FEATURE_DIM), dtype=torch.float32), user_to_idx

    # ── 3. Compute per-type feature DataFrames (all indexed by user_id) ───────
    feats_list = [
        _logon_user_features(logon_w),
        _device_user_features(device_w),
        _email_user_features(email_w),
        _http_user_features(http_w),
        _file_user_features(file_w),
        _active_days_feature(logon_w, device_w, email_w, http_w, file_w),
    ]

    # ── 4. Left-join all feature blocks onto the full user roster ─────────────
    base = pd.DataFrame(index=pd.Index(sorted_users, name='user'))
    for feats_df in feats_list:
        if not feats_df.empty:
            base = base.join(feats_df, how='left')

    # Ensure every expected column exists (guards against all-empty log types)
    for col in _USER_FEAT_COLS:
        if col not in base.columns:
            base[col] = 0.0

    base = base[_USER_FEAT_COLS].fillna(0.0)

    # ── 5. Convert to float32 tensor ─────────────────────────────────────────
    x = torch.tensor(base.values, dtype=torch.float32)
    assert x.shape == (n_users, USER_FEATURE_DIM), (
        f"Shape mismatch: expected ({n_users}, {USER_FEATURE_DIM}), got {tuple(x.shape)}"
    )
    return x, user_to_idx


def build_pc_nodes(
    logs: dict,
    start_date: str,
    end_date: str,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Build PC (workstation) node feature tensors for the heterogeneous graph.

    Identifies all unique PCs from logon and device logs, then aggregates
    event statistics over the given time window into one float32 feature
    vector per PC.

    Args:
        logs:       Dict from ``pipeline.load_raw_logs()``.
                    Uses ``'logon'`` and ``'device'`` keys.
        start_date: Inclusive window start, e.g. ``'2010-01-01'``.
        end_date:   Inclusive window end,   e.g. ``'2010-09-30'``.

    Returns:
        ``(x, pc_to_idx)`` — float32 tensor ``(num_pcs, PC_FEATURE_DIM)``
        and a ``{pc_id: row_index}`` lookup dict.

    Feature columns in row order (8 total, matching PC_FEATURE_DIM):
        0  num_distinct_users      unique users who logged in
        1  total_logon_events      total logon (not logoff) events
        2  is_shared               1.0 if >5 distinct users
        3  after_hours_usage_ratio fraction of logons outside business hours
        4  usb_event_count         total USB-connect events on this machine
        5  mean_session_count      avg logon events per active day
        6  weekend_usage_ratio     fraction of logons on Sat/Sun
        7  unique_days_active      distinct days with any activity
    """
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)

    logon_w  = _filter_window(logs.get('logon',  pd.DataFrame()), t_start, t_end)
    device_w = _filter_window(logs.get('device', pd.DataFrame()), t_start, t_end)

    # ── 1. PC roster from both log types ─────────────────────────────────────
    all_pcs: set[str] = set()
    for df in (logon_w, device_w):
        if not df.empty and 'pc' in df.columns:
            all_pcs.update(df['pc'].dropna().astype(str).unique())

    sorted_pcs = sorted(all_pcs)
    pc_to_idx  = {p: i for i, p in enumerate(sorted_pcs)}
    n_pcs      = len(sorted_pcs)

    logger.info(
        'build_pc_nodes | window %s – %s | %d PCs found',
        start_date, end_date, n_pcs,
    )

    if n_pcs == 0:
        return torch.zeros((0, PC_FEATURE_DIM), dtype=torch.float32), pc_to_idx

    base = pd.DataFrame(index=pd.Index(sorted_pcs, name='pc'))

    # ── 2. Logon-based features ───────────────────────────────────────────────
    if not logon_w.empty and 'pc' in logon_w.columns:
        logons = logon_w[logon_w['activity'].str.lower() == 'logon'].copy()
        if not logons.empty:
            logons['_day'] = logons['date'].dt.normalize()
            h = logons['date'].dt.hour
            logons['_after_hours'] = (h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)
            logons['_weekend']     = logons['date'].dt.dayofweek >= 5

            by_pc = logons.groupby('pc')
            n     = by_pc.size()

            num_distinct_users = by_pc['user'].nunique()
            daily_logon_counts = logons.groupby(['pc', '_day']).size()
            mean_session_count = daily_logon_counts.groupby(level='pc').mean()

            logon_feats = pd.DataFrame({
                'num_distinct_users':     num_distinct_users,
                'total_logon_events':     n,
                'is_shared':              (num_distinct_users > 5).astype(float),
                'after_hours_usage_ratio': _ratio(by_pc['_after_hours'].sum(), n),
                'mean_session_count':     mean_session_count,
                'weekend_usage_ratio':    _ratio(by_pc['_weekend'].sum(), n),
            })
            base = base.join(logon_feats, how='left')

    # ── 3. USB connect count (from device log) ────────────────────────────────
    if not device_w.empty and 'pc' in device_w.columns:
        connects = device_w[device_w['activity'].str.lower() == 'connect']
        if not connects.empty:
            usb_counts = connects.groupby('pc').size().rename('usb_event_count')
            base = base.join(usb_counts.to_frame(), how='left')

    # ── 4. Unique active days (union of logon + device) ───────────────────────
    day_parts = []
    for df in (logon_w, device_w):
        if not df.empty and 'pc' in df.columns:
            tmp = df[['pc', 'date']].copy()
            tmp['_day'] = tmp['date'].dt.normalize()
            day_parts.append(tmp[['pc', '_day']].drop_duplicates())

    if day_parts:
        combined    = pd.concat(day_parts, ignore_index=True).drop_duplicates()
        unique_days = combined.groupby('pc').size().rename('unique_days_active')
        base = base.join(unique_days.to_frame(), how='left')

    # ── 5. Fill missing columns and convert ───────────────────────────────────
    for col in _PC_FEAT_COLS:
        if col not in base.columns:
            base[col] = 0.0

    # is_shared is derived — recompute from the filled num_distinct_users
    base['is_shared'] = (base['num_distinct_users'].fillna(0) > 5).astype(float)

    base = base[_PC_FEAT_COLS].fillna(0.0)
    x = torch.tensor(base.values, dtype=torch.float32)
    assert x.shape == (n_pcs, PC_FEATURE_DIM)
    return x, pc_to_idx


def build_url_nodes(
    logs: dict,
    start_date: str,
    end_date: str,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Build URL (domain) node feature tensors for the heterogeneous graph.

    Deduplicates URLs to domain level (netloc) to keep the node count
    manageable.  All HTTP requests to the same domain share one node.

    Args:
        logs:       Dict from ``pipeline.load_raw_logs()``.
                    Uses the ``'http'`` key.
        start_date: Inclusive window start.
        end_date:   Inclusive window end.

    Returns:
        ``(x, url_to_idx)`` — float32 tensor ``(num_domains, URL_FEATURE_DIM)``
        and a ``{domain: row_index}`` lookup dict.

    Feature columns in row order (8 total, matching URL_FEATURE_DIM):
        0  total_visits            total HTTP requests to this domain
        1  unique_visitors         distinct users who visited
        2  is_job_related          1.0 if domain matches JOB_KEYWORDS
        3  is_cloud_related        1.0 if domain matches CLOUD_KEYWORDS
        4  mean_daily_visits       avg visits per active day
        5  visitor_concentration   max single-user visits / total visits
        6  weekend_visit_ratio     fraction of visits on Sat/Sun
        7  after_hours_visit_ratio fraction of visits outside business hours
    """
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)

    http_w = _filter_window(logs.get('http', pd.DataFrame()), t_start, t_end)

    if http_w.empty or 'url' not in http_w.columns:
        return torch.zeros((0, URL_FEATURE_DIM), dtype=torch.float32), {}

    # ── 1. Normalise to domain level ──────────────────────────────────────────
    http_w = http_w.copy()
    # Use pre-computed columns from _presort_logs where available, falling back
    # to on-the-fly computation only when the df wasn't pre-processed.
    if '_domain' not in http_w.columns:
        http_w['_domain'] = http_w['url'].apply(_extract_domain)
    if '_day' not in http_w.columns:
        http_w['_day'] = http_w['date'].dt.normalize()
    # _presort_logs stores the after-hours flag as '_after'; rename for this function.
    if '_after_hours' not in http_w.columns:
        if '_after' in http_w.columns:
            http_w['_after_hours'] = http_w['_after']
        else:
            h = http_w['date'].dt.hour
            http_w['_after_hours'] = (h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)
    if '_weekend' not in http_w.columns:
        http_w['_weekend'] = http_w['date'].dt.dayofweek >= 5

    sorted_domains = sorted(http_w['_domain'].unique())
    url_to_idx     = {d: i for i, d in enumerate(sorted_domains)}
    n_urls         = len(sorted_domains)

    logger.info(
        'build_url_nodes | window %s – %s | %d unique domains',
        start_date, end_date, n_urls,
    )

    # ── 2. Aggregate per domain ───────────────────────────────────────────────
    by_domain = http_w.groupby('_domain')
    n         = by_domain.size()

    # visitor concentration: max visits by a single user / total visits
    visits_per_user = http_w.groupby(['_domain', 'user']).size()
    max_user_visits = visits_per_user.groupby(level='_domain').max()

    mean_daily = (
        http_w.groupby(['_domain', '_day']).size()
              .groupby(level='_domain').mean()
    )

    # Keyword flags are domain-level constants — one check per domain
    domain_lower = pd.Index(sorted_domains).str.lower()
    is_job   = pd.Series(
        [any(k in d for k in JOB_KEYWORDS)   for d in sorted_domains],
        index=sorted_domains, name='is_job_related', dtype=float,
    )
    is_cloud = pd.Series(
        [any(k in d for k in CLOUD_KEYWORDS) for d in sorted_domains],
        index=sorted_domains, name='is_cloud_related', dtype=float,
    )

    base = pd.DataFrame({
        'total_visits':            n,
        'unique_visitors':         by_domain['user'].nunique(),
        'is_job_related':          is_job,
        'is_cloud_related':        is_cloud,
        'mean_daily_visits':       mean_daily,
        'visitor_concentration':   _ratio(max_user_visits, n),
        'weekend_visit_ratio':     _ratio(by_domain['_weekend'].sum(),     n),
        'after_hours_visit_ratio': _ratio(by_domain['_after_hours'].sum(), n),
    }, index=pd.Index(sorted_domains, name='_domain'))

    base = base[_URL_FEAT_COLS].fillna(0.0)
    x = torch.tensor(base.values, dtype=torch.float32)
    assert x.shape == (n_urls, URL_FEATURE_DIM)
    return x, url_to_idx


def build_file_nodes(
    logs: dict,
    start_date: str,
    end_date: str,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Build file node feature tensors for the heterogeneous graph.

    A file node represents a unique filename path.  All files that appear
    in the file log within the time window are included; files never copied
    to removable media will have ``total_copies = 0``.

    Args:
        logs:       Dict from ``pipeline.load_raw_logs()``.
                    Uses the ``'file'`` key.
        start_date: Inclusive window start.
        end_date:   Inclusive window end.

    Returns:
        ``(x, file_to_idx)`` — float32 tensor ``(num_files, FILE_FEATURE_DIM)``
        and a ``{filename: row_index}`` lookup dict.

    Feature columns in row order (6 total, matching FILE_FEATURE_DIM):
        0  total_copies        times copied to removable media
        1  unique_users        distinct users who copied this file
        2  file_type_code      extension category (doc=1 pdf=2 exe=3 archive=4 other=0)
        3  avg_copy_hour       mean hour-of-day of copy events
        4  is_after_hours_copy fraction of copies outside business hours
        5  content_topic_count distinct suspicious keywords found in content
    """
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)

    file_w = _filter_window(logs.get('file', pd.DataFrame()), t_start, t_end)

    if file_w.empty or 'filename' not in file_w.columns:
        return torch.zeros((0, FILE_FEATURE_DIM), dtype=torch.float32), {}

    file_w = file_w.copy()

    # ── 1. Normalise to_removable_media column ────────────────────────────────
    removable = file_w.get('to_removable_media', pd.Series(False, index=file_w.index))
    if removable.dtype == object:
        removable = removable.str.lower() == 'true'
    file_w['_removable'] = removable.fillna(False)

    sorted_files = sorted(file_w['filename'].dropna().astype(str).unique())
    file_to_idx  = {f: i for i, f in enumerate(sorted_files)}
    n_files      = len(sorted_files)

    logger.info(
        'build_file_nodes | window %s – %s | %d unique files',
        start_date, end_date, n_files,
    )

    if n_files == 0:
        return torch.zeros((0, FILE_FEATURE_DIM), dtype=torch.float32), file_to_idx

    base = pd.DataFrame(index=pd.Index(sorted_files, name='filename'))

    # ── 2. Copy-event features (subset where to_removable_media is True) ──────
    copies = file_w[file_w['_removable']].copy()
    if not copies.empty:
        h = copies['date'].dt.hour
        copies['_after_hours'] = (h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)
        copies['_hour']        = h.astype(float)

        by_file = copies.groupby('filename')
        n       = by_file.size()

        copy_feats = pd.DataFrame({
            'total_copies':        n,
            'unique_users':        by_file['user'].nunique(),
            'avg_copy_hour':       by_file['_hour'].mean(),
            'is_after_hours_copy': _ratio(by_file['_after_hours'].sum(), n),
        })
        base = base.join(copy_feats, how='left')

    # ── 3. Static per-filename features (computed from all events) ────────────
    # file_type_code — derived from filename extension alone
    type_codes = pd.Series(
        {f: float(_file_type_code(f)) for f in sorted_files},
        name='file_type_code',
    )
    base = base.join(type_codes.to_frame(), how='left')

    # content_topic_count — aggregate across all file events for this filename
    if 'content' in file_w.columns:
        topic_counts = (
            file_w.groupby('filename')['content']
                  .apply(lambda col: sum(_count_content_keywords(v) for v in col))
                  .rename('content_topic_count')
        )
        base = base.join(topic_counts.to_frame(), how='left')

    # ── 4. Fill missing columns and convert ───────────────────────────────────
    for col in _FILE_FEAT_COLS:
        if col not in base.columns:
            base[col] = 0.0

    base = base[_FILE_FEAT_COLS].fillna(0.0)
    x = torch.tensor(base.values, dtype=torch.float32)
    assert x.shape == (n_files, FILE_FEATURE_DIM)
    return x, file_to_idx


def build_logon_edges(
    logon_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    pc_to_idx: dict[str, int],
    start_date: str,
    end_date: str,
) -> dict:
    """Build (user → pc) logon edge tensors for the heterogeneous graph.

    One edge is created per *Logon* event (Logoff events are skipped).
    Each forward edge (user → pc) is paired with a reverse edge (pc → user)
    carrying identical features, enabling bidirectional message passing.

    Args:
        logon_df:    Raw logon DataFrame from ``pipeline.load_raw_logs()``.
                     Must have columns: ``date``, ``user``, ``pc``,
                     ``activity``.  Dates may be ISO strings or Timestamps.
        user_to_idx: ``{user_id: row_index}`` from ``build_user_nodes()``.
        pc_to_idx:   ``{pc_id:   row_index}`` from ``build_pc_nodes()``.
        start_date:  Inclusive window start.
        end_date:    Inclusive window end.

    Returns:
        Dict with keys:

        * ``'edge_index'``     — ``(2, n)`` long tensor, forward user→pc.
        * ``'edge_attr'``      — ``(n, LOGON_EDGE_DIM)`` float32 tensor.
        * ``'rev_edge_index'`` — ``(2, n)`` long tensor, reverse pc→user.
        * ``'rev_edge_attr'``  — ``(n, LOGON_EDGE_DIM)`` float32 tensor
          (identical values to ``edge_attr``).
        * ``'n_skipped'``      — edges dropped because user or pc was absent
          from the node-index mappings.

    Edge features (LOGON_EDGE_DIM = 4, in column order):
        hour_normalized  hour_of_day / 24.0
        is_weekend       1.0 if Saturday or Sunday
        is_after_hours   1.0 if outside business hours
        day_of_week      day_of_week / 7.0  (0=Mon … 6=Sun, normalised)
    """
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)

    df = _ensure_datetime(logon_df)
    df = _filter_window(df, t_start, t_end)

    if df.empty:
        _empty = torch.zeros((2, 0), dtype=torch.long)
        _eattr = torch.zeros((0, LOGON_EDGE_DIM), dtype=torch.float32)
        return {'edge_index': _empty, 'edge_attr': _eattr,
                'rev_edge_index': _empty, 'rev_edge_attr': _eattr, 'n_skipped': 0}

    # ── 1. Keep only Logon events ─────────────────────────────────────────────
    df = df[df['activity'].str.lower() == 'logon'].copy()

    # ── 2. Vectorised index lookup (NaN for entities absent from mappings) ────
    user_series = pd.Series(user_to_idx, name='_src')
    pc_series   = pd.Series(pc_to_idx,   name='_dst')
    df['_src'] = df['user'].astype(str).map(user_series)
    df['_dst'] = df['pc'].astype(str).map(pc_series)

    n_before  = len(df)
    df        = df.dropna(subset=['_src', '_dst'])
    n_skipped = n_before - len(df)

    if n_skipped:
        logger.warning(
            'build_logon_edges | %d rows skipped (user/pc not in node mappings)',
            n_skipped,
        )

    logger.info(
        'build_logon_edges | window %s – %s | %d edges  (%d skipped)',
        start_date, end_date, len(df), n_skipped,
    )

    if df.empty:
        _empty = torch.zeros((2, 0), dtype=torch.long)
        _eattr = torch.zeros((0, LOGON_EDGE_DIM), dtype=torch.float32)
        return {'edge_index': _empty, 'edge_attr': _eattr,
                'rev_edge_index': _empty, 'rev_edge_attr': _eattr,
                'n_skipped': n_skipped}

    # ── 3. Build tensors ──────────────────────────────────────────────────────
    src  = torch.tensor(df['_src'].astype(int).values, dtype=torch.long)
    dst  = torch.tensor(df['_dst'].astype(int).values, dtype=torch.long)
    feat = torch.tensor(_build_edge_features_logon(df), dtype=torch.float32)

    return {
        'edge_index':     torch.stack([src, dst]),
        'edge_attr':      feat,
        'rev_edge_index': torch.stack([dst, src]),
        'rev_edge_attr':  feat,           # same features; copy-on-write is fine
        'n_skipped':      n_skipped,
    }


def build_usb_edges(
    device_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    pc_to_idx: dict[str, int],
    start_date: str,
    end_date: str,
) -> dict:
    """Build (user → pc) USB-connect edge tensors for the heterogeneous graph.

    One edge is created per *Connect* event (Disconnect events are skipped).
    Reverse edges (pc → user) are returned with identical features.

    Args:
        device_df:   Raw device DataFrame from ``pipeline.load_raw_logs()``.
                     Must have columns: ``date``, ``user``, ``pc``,
                     ``activity``.
        user_to_idx: ``{user_id: row_index}`` from ``build_user_nodes()``.
        pc_to_idx:   ``{pc_id:   row_index}`` from ``build_pc_nodes()``.
        start_date:  Inclusive window start.
        end_date:    Inclusive window end.

    Returns:
        Dict with the same keys as ``build_logon_edges()``.

    Edge features (USB_EDGE_DIM = 3, in column order):
        hour_normalized  hour_of_day / 24.0
        is_weekend       1.0 if Saturday or Sunday
        is_after_hours   1.0 if outside business hours
    """
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)

    df = _ensure_datetime(device_df)
    df = _filter_window(df, t_start, t_end)

    if df.empty:
        _empty = torch.zeros((2, 0), dtype=torch.long)
        _eattr = torch.zeros((0, USB_EDGE_DIM), dtype=torch.float32)
        return {'edge_index': _empty, 'edge_attr': _eattr,
                'rev_edge_index': _empty, 'rev_edge_attr': _eattr, 'n_skipped': 0}

    # ── 1. Keep only Connect events ───────────────────────────────────────────
    df = df[df['activity'].str.lower() == 'connect'].copy()

    # ── 2. Vectorised index lookup ────────────────────────────────────────────
    user_series = pd.Series(user_to_idx, name='_src')
    pc_series   = pd.Series(pc_to_idx,   name='_dst')
    df['_src'] = df['user'].astype(str).map(user_series)
    df['_dst'] = df['pc'].astype(str).map(pc_series)

    n_before  = len(df)
    df        = df.dropna(subset=['_src', '_dst'])
    n_skipped = n_before - len(df)

    if n_skipped:
        logger.warning(
            'build_usb_edges | %d rows skipped (user/pc not in node mappings)',
            n_skipped,
        )

    logger.info(
        'build_usb_edges | window %s – %s | %d edges  (%d skipped)',
        start_date, end_date, len(df), n_skipped,
    )

    if df.empty:
        _empty = torch.zeros((2, 0), dtype=torch.long)
        _eattr = torch.zeros((0, USB_EDGE_DIM), dtype=torch.float32)
        return {'edge_index': _empty, 'edge_attr': _eattr,
                'rev_edge_index': _empty, 'rev_edge_attr': _eattr,
                'n_skipped': n_skipped}

    # ── 3. Build tensors ──────────────────────────────────────────────────────
    src  = torch.tensor(df['_src'].astype(int).values, dtype=torch.long)
    dst  = torch.tensor(df['_dst'].astype(int).values, dtype=torch.long)
    feat = torch.tensor(_build_edge_features_usb(df), dtype=torch.float32)

    return {
        'edge_index':     torch.stack([src, dst]),
        'edge_attr':      feat,
        'rev_edge_index': torch.stack([dst, src]),
        'rev_edge_attr':  feat,
        'n_skipped':      n_skipped,
    }


def build_email_edges(
    email_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    start_date: str,
    end_date: str,
) -> dict:
    """Build (user → user) email edge tensors for the heterogeneous graph.

    One edge is created per internal recipient per email.  External recipients
    are silently skipped — they have no node in the graph.  CC and BCC
    recipients generate the same edge type as TO recipients.  Self-loops
    (sender == recipient) are dropped.

    The ``is_external_thread`` feature captures whether the email also had
    external recipients, which is a meaningful anomaly signal even when the
    specific edge being built is internal.

    Args:
        email_df:    Raw email DataFrame from ``pipeline.load_raw_logs()``.
                     Must have columns: ``date``, ``user`` (sender),
                     ``to``, ``cc``, ``bcc``, ``size``, ``attachments``.
        user_to_idx: ``{user_id: row_index}`` from ``build_user_nodes()``.
        start_date:  Inclusive window start.
        end_date:    Inclusive window end.

    Returns:
        Dict with keys: ``edge_index``, ``edge_attr``, ``rev_edge_index``,
        ``rev_edge_attr``, ``n_skipped`` (rows with unknown sender).

    Edge features (EMAIL_EDGE_DIM = 5):
        email_size_normalized  size_bytes / 1_000_000
        attachment_count       attachments clipped [0,10] / 10.0
        is_external_thread     1.0 if any recipient is outside dtaa.com
        hour_normalized        hour_of_day / 24.0
        is_after_hours         1.0 if outside business hours
    """
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)

    df = _ensure_datetime(email_df)
    df = _filter_window(df, t_start, t_end)

    _empty_ei   = torch.zeros((2, 0), dtype=torch.long)
    _empty_attr = torch.zeros((0, EMAIL_EDGE_DIM), dtype=torch.float32)
    _empty      = {'edge_index': _empty_ei, 'edge_attr': _empty_attr,
                   'rev_edge_index': _empty_ei, 'rev_edge_attr': _empty_attr,
                   'n_skipped': 0}

    if df.empty:
        return _empty

    n_before = len(df)
    pairs    = _explode_and_map_recipients(df, user_to_idx)
    n_skipped = n_before - (0 if pairs.empty else
                            len(pairs['_sender_idx'].unique()))  # rough estimate

    logger.info(
        'build_email_edges | window %s – %s | %d edges from %d emails  (%d sender rows skipped)',
        start_date, end_date,
        len(pairs) if not pairs.empty else 0,
        n_before, max(0, n_skipped),
    )

    if pairs.empty:
        return {**_empty, 'n_skipped': n_skipped}

    src  = torch.tensor(pairs['_sender_idx'].values, dtype=torch.long)
    dst  = torch.tensor(pairs['_recip_idx'].values,  dtype=torch.long)

    feat = np.column_stack([
        pairs['_size'].to_numpy(dtype=np.float32) / 1_000_000.0,  # email_size_normalized
        pairs['_attach'].to_numpy(dtype=np.float32) / 10.0,        # attachment_count (normalised)
        pairs['_is_external'].to_numpy(dtype=np.float32),           # is_external_thread
        pairs['_hour'].to_numpy(dtype=np.float32) / 24.0,           # hour_normalized
        pairs['_is_after'].to_numpy(dtype=np.float32),              # is_after_hours
    ])
    edge_attr = torch.tensor(feat, dtype=torch.float32)

    return {
        'edge_index':     torch.stack([src, dst]),
        'edge_attr':      edge_attr,
        'rev_edge_index': torch.stack([dst, src]),
        'rev_edge_attr':  edge_attr,
        'n_skipped':      max(0, n_skipped),
    }


def build_http_edges(
    http_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    url_to_idx: dict[str, int],
    start_date: str,
    end_date: str,
) -> dict:
    """Build (user → url-domain) HTTP edge tensors for the heterogeneous graph.

    URLs are deduplicated to domain level (netloc) — the same key used by
    ``build_url_nodes()`` — so lookups always succeed for nodes that exist.

    Args:
        http_df:     Raw HTTP DataFrame from ``pipeline.load_raw_logs()``.
                     Must have columns: ``date``, ``user``, ``url``.
        user_to_idx: ``{user_id: row_index}`` from ``build_user_nodes()``.
        url_to_idx:  ``{domain:  row_index}`` from ``build_url_nodes()``.
        start_date:  Inclusive window start.
        end_date:    Inclusive window end.

    Returns:
        Dict with same keys as ``build_logon_edges()``.

    Edge features (HTTP_EDGE_DIM = 3):
        hour_normalized  hour_of_day / 24.0
        is_weekend       1.0 if Saturday or Sunday
        is_after_hours   1.0 if outside business hours
    """
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)

    df = _ensure_datetime(http_df)
    df = _filter_window(df, t_start, t_end)

    _empty_ei   = torch.zeros((2, 0), dtype=torch.long)
    _empty_attr = torch.zeros((0, HTTP_EDGE_DIM), dtype=torch.float32)
    _empty      = {'edge_index': _empty_ei, 'edge_attr': _empty_attr,
                   'rev_edge_index': _empty_ei, 'rev_edge_attr': _empty_attr,
                   'n_skipped': 0}

    if df.empty or 'url' not in df.columns:
        return _empty

    df = df.copy()
    # Use pre-computed _domain column from _presort_logs (avoids expensive
    # .apply(_extract_domain) on potentially 1 M+ rows per window).
    if '_domain' not in df.columns:
        df['_domain'] = df['url'].apply(_extract_domain)

    user_map = pd.Series(user_to_idx)
    url_map  = pd.Series(url_to_idx)
    df['_src'] = df['user'].astype(str).map(user_map)
    df['_dst'] = df['_domain'].map(url_map)

    n_before  = len(df)
    df        = df.dropna(subset=['_src', '_dst'])
    n_skipped = n_before - len(df)

    if n_skipped:
        logger.warning('build_http_edges | %d rows skipped', n_skipped)

    if df.empty:
        return {**_empty, 'n_skipped': n_skipped}

    # ── Aggregate to one edge per unique (user, domain) pair ──────────────────
    # A raw 28-day HTTP window has ~1.4 M rows → ~1 M edges after node-cap
    # filtering. Holding 3 200 such graphs in RAM would require ~200 GB.
    # Aggregating to unique (user, domain) pairs reduces this to ~50 K edges
    # per graph (~4 MB) while preserving all meaningful behavioural signals:
    # mean hour of access, fraction of visits on weekends, fraction after-hours.
    df['_src'] = df['_src'].astype(int)
    df['_dst'] = df['_dst'].astype(int)

    # Use precomputed flag columns when available (set by _presort_logs).
    if '_hour' not in df.columns:
        h = df['date'].dt.hour
        df['_hour'] = h.astype(float)
    if '_weekend' not in df.columns:
        df['_weekend'] = (df['date'].dt.dayofweek >= 5).astype(float)
    if '_after' not in df.columns:
        h = df['date'].dt.hour
        df['_after'] = ((h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)).astype(float)

    agg = (
        df.groupby(['_src', '_dst'], sort=False)[['_hour', '_weekend', '_after']]
        .mean()
        .reset_index()
    )
    agg['_hour'] = agg['_hour'] / 24.0  # normalise to [0, 1]

    n_edges = len(agg)
    logger.info(
        'build_http_edges | window %s – %s | %d unique (user,domain) edges '
        'aggregated from %d requests (%d skipped)',
        start_date, end_date, n_edges, n_before - n_skipped, n_skipped,
    )

    src  = torch.tensor(agg['_src'].values,  dtype=torch.long)
    dst  = torch.tensor(agg['_dst'].values,  dtype=torch.long)
    feat = torch.tensor(
        agg[['_hour', '_weekend', '_after']].values.astype(np.float32),
        dtype=torch.float32,
    )

    return {
        'edge_index':     torch.stack([src, dst]),
        'edge_attr':      feat,
        'rev_edge_index': torch.stack([dst, src]),
        'rev_edge_attr':  feat,
        'n_skipped':      n_skipped,
    }


def build_file_edges(
    file_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    file_to_idx: dict[str, int],
    start_date: str,
    end_date: str,
) -> dict:
    """Build (user → file) file-copy edge tensors for the heterogeneous graph.

    Only events where ``to_removable_media`` is True are included, matching
    the ``(user, file_copy, file)`` edge semantics in the schema.

    Args:
        file_df:     Raw file DataFrame from ``pipeline.load_raw_logs()``.
                     Must have columns: ``date``, ``user``, ``filename``,
                     ``to_removable_media``.
        user_to_idx: ``{user_id:  row_index}`` from ``build_user_nodes()``.
        file_to_idx: ``{filename: row_index}`` from ``build_file_nodes()``.
        start_date:  Inclusive window start.
        end_date:    Inclusive window end.

    Returns:
        Dict with same keys as ``build_logon_edges()``.

    Edge features (FILE_EDGE_DIM = 4):
        hour_normalized  hour_of_day / 24.0
        is_weekend       1.0 if Saturday or Sunday
        is_after_hours   1.0 if outside business hours
        file_type_code   extension category (doc=1 pdf=2 exe=3 archive=4 other=0)
    """
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)

    df = _ensure_datetime(file_df)
    df = _filter_window(df, t_start, t_end)

    _empty_ei   = torch.zeros((2, 0), dtype=torch.long)
    _empty_attr = torch.zeros((0, FILE_EDGE_DIM), dtype=torch.float32)
    _empty      = {'edge_index': _empty_ei, 'edge_attr': _empty_attr,
                   'rev_edge_index': _empty_ei, 'rev_edge_attr': _empty_attr,
                   'n_skipped': 0}

    if df.empty or 'filename' not in df.columns:
        return _empty

    df = df.copy()

    # Keep only file-copy events
    removable = df.get('to_removable_media', pd.Series(False, index=df.index))
    if removable.dtype == object:
        removable = removable.str.lower() == 'true'
    df = df[removable.fillna(False)].copy()

    if df.empty:
        return _empty

    user_map = pd.Series(user_to_idx)
    file_map = pd.Series(file_to_idx)
    df['_src'] = df['user'].astype(str).map(user_map)
    df['_dst'] = df['filename'].astype(str).map(file_map)

    n_before  = len(df)
    df        = df.dropna(subset=['_src', '_dst'])
    n_skipped = n_before - len(df)

    if n_skipped:
        logger.warning('build_file_edges | %d rows skipped', n_skipped)

    logger.info(
        'build_file_edges | window %s – %s | %d edges  (%d skipped)',
        start_date, end_date, len(df), n_skipped,
    )

    if df.empty:
        return {**_empty, 'n_skipped': n_skipped}

    src  = torch.tensor(df['_src'].astype(int).values, dtype=torch.long)
    dst  = torch.tensor(df['_dst'].astype(int).values, dtype=torch.long)

    type_codes = np.array(
        [float(_file_type_code(f)) for f in df['filename'].astype(str)],
        dtype=np.float32,
    )
    feat = torch.tensor(
        np.column_stack([_temporal3(df), type_codes]),
        dtype=torch.float32,
    )

    return {
        'edge_index':     torch.stack([src, dst]),
        'edge_attr':      feat,
        'rev_edge_index': torch.stack([dst, src]),
        'rev_edge_attr':  feat,
        'n_skipped':      n_skipped,
    }


# ── HeteroData assembly ───────────────────────────────────────────────────────

def _validate_graph(data: HeteroData) -> None:
    """Run sanity checks on a newly assembled HeteroData object.

    Raises AssertionError with a descriptive message on any violation.
    """
    # Node feature dims
    assert data[NODE_USER].x.shape[1] == USER_FEATURE_DIM, (
        f"user feat dim {data[NODE_USER].x.shape[1]} != {USER_FEATURE_DIM}"
    )
    assert data[NODE_PC].x.shape[1] == PC_FEATURE_DIM, (
        f"pc feat dim {data[NODE_PC].x.shape[1]} != {PC_FEATURE_DIM}"
    )
    assert data[NODE_URL].x.shape[1] == URL_FEATURE_DIM, (
        f"url feat dim {data[NODE_URL].x.shape[1]} != {URL_FEATURE_DIM}"
    )
    assert data[NODE_FILE].x.shape[1] == FILE_FEATURE_DIM, (
        f"file feat dim {data[NODE_FILE].x.shape[1]} != {FILE_FEATURE_DIM}"
    )

    n_users = data[NODE_USER].x.shape[0]
    n_pcs   = data[NODE_PC].x.shape[0]
    n_urls  = data[NODE_URL].x.shape[0]
    n_files = data[NODE_FILE].x.shape[0]

    # Edge index bounds and feature dims
    for etype, src_n, dst_n, expected_dim in [
        (EDGE_LOGON,     n_users, n_pcs,   LOGON_EDGE_DIM),
        (EDGE_USB,       n_users, n_pcs,   USB_EDGE_DIM),
        (EDGE_EMAIL,     n_users, n_users, EMAIL_EDGE_DIM),
        (EDGE_HTTP,      n_users, n_urls,  HTTP_EDGE_DIM),
        (EDGE_FILE_COPY, n_users, n_files, FILE_EDGE_DIM),
    ]:
        ei = data[etype].edge_index
        ea = data[etype].edge_attr
        if ei.shape[1] > 0:
            assert ei[0].max().item() < src_n, (
                f"{etype} src index {ei[0].max().item()} out of range ({src_n} nodes)"
            )
            assert ei[1].max().item() < dst_n, (
                f"{etype} dst index {ei[1].max().item()} out of range ({dst_n} nodes)"
            )
        assert ea.shape[1] == expected_dim, (
            f"{etype} edge feat dim {ea.shape[1]} != {expected_dim}"
        )

    # Label vector
    y = data[NODE_USER].y
    assert y.shape[0] == n_users, (
        f"label vector length {y.shape[0]} != n_users {n_users}"
    )
    assert y.dtype == torch.float32, f"label dtype {y.dtype} != float32"
    n_pos = int(y.sum().item())
    n_neg = n_users - n_pos
    logger.info('Graph validated: %d users  %d pos  %d neg  pos_rate=%.4f',
                n_users, n_pos, n_neg, n_pos / max(n_users, 1))


def build_hetero_graph(
    logs: dict,
    labels: set,
    start_date: str,
    end_date: str,
) -> HeteroData:
    """Assemble all node/edge tensors into a single PyG HeteroData object.

    Parameters
    ----------
    logs:
        Dict returned by ``load_raw_logs``: keys are 'logon', 'device',
        'file', 'email', 'http', each holding a raw DataFrame.
    labels:
        Set of positive-class identifiers. Entries may be bare user IDs
        (str) or ``(user_id, date_str)`` tuples — both formats are
        supported so callers can pass ``insiders.csv`` rows directly.
    start_date, end_date:
        Inclusive time window boundaries (e.g. ``'2009-01-01'``).

    Returns
    -------
    HeteroData
        Fully assembled graph with node features, edge indices, edge
        attributes, and user labels.  Two extra attributes are attached
        for downstream convenience:

        * ``data.user_to_idx`` — dict mapping user_id → graph row index
        * ``data.idx_to_user`` — list mapping graph row index → user_id
    """
    # ── Build nodes ───────────────────────────────────────────────────────────
    x_user, user_to_idx = build_user_nodes(logs, start_date, end_date)
    x_pc,   pc_to_idx   = build_pc_nodes(logs, start_date, end_date)
    x_url,  url_to_idx  = build_url_nodes(logs, start_date, end_date)
    x_file, file_to_idx = build_file_nodes(logs, start_date, end_date)

    # ── Build edges ───────────────────────────────────────────────────────────
    logon_e = build_logon_edges(
        logs.get('logon',  pd.DataFrame()),
        user_to_idx, pc_to_idx, start_date, end_date,
    )
    usb_e = build_usb_edges(
        logs.get('device', pd.DataFrame()),
        user_to_idx, pc_to_idx, start_date, end_date,
    )
    email_e = build_email_edges(
        logs.get('email',  pd.DataFrame()),
        user_to_idx, start_date, end_date,
    )
    http_e = build_http_edges(
        logs.get('http',   pd.DataFrame()),
        user_to_idx, url_to_idx, start_date, end_date,
    )
    file_e = build_file_edges(
        logs.get('file',   pd.DataFrame()),
        user_to_idx, file_to_idx, start_date, end_date,
    )

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()

    # nodes
    data[NODE_USER].x = x_user
    data[NODE_PC].x   = x_pc
    data[NODE_URL].x  = x_url
    data[NODE_FILE].x = x_file

    # forward edges
    data[EDGE_LOGON].edge_index     = logon_e['edge_index']
    data[EDGE_LOGON].edge_attr      = logon_e['edge_attr']
    data[EDGE_USB].edge_index       = usb_e['edge_index']
    data[EDGE_USB].edge_attr        = usb_e['edge_attr']
    data[EDGE_EMAIL].edge_index     = email_e['edge_index']
    data[EDGE_EMAIL].edge_attr      = email_e['edge_attr']
    data[EDGE_HTTP].edge_index      = http_e['edge_index']
    data[EDGE_HTTP].edge_attr       = http_e['edge_attr']
    data[EDGE_FILE_COPY].edge_index = file_e['edge_index']
    data[EDGE_FILE_COPY].edge_attr  = file_e['edge_attr']

    # reverse edges
    data[REV_EDGE_LOGON].edge_index     = logon_e['rev_edge_index']
    data[REV_EDGE_LOGON].edge_attr      = logon_e['rev_edge_attr']
    data[REV_EDGE_USB].edge_index       = usb_e['rev_edge_index']
    data[REV_EDGE_USB].edge_attr        = usb_e['rev_edge_attr']
    data[REV_EDGE_EMAIL].edge_index     = email_e['rev_edge_index']
    data[REV_EDGE_EMAIL].edge_attr      = email_e['rev_edge_attr']
    data[REV_EDGE_HTTP].edge_index      = http_e['rev_edge_index']
    data[REV_EDGE_HTTP].edge_attr       = http_e['rev_edge_attr']
    data[REV_EDGE_FILE_COPY].edge_index = file_e['rev_edge_index']
    data[REV_EDGE_FILE_COPY].edge_attr  = file_e['rev_edge_attr']

    # ── Labels ────────────────────────────────────────────────────────────────
    # Accept labels as bare user IDs or (user_id, date) tuples.
    positive_users: set[str] = set()
    for entry in labels:
        uid = entry[0] if isinstance(entry, tuple) else entry
        positive_users.add(uid)

    y = torch.zeros(x_user.shape[0], dtype=torch.float32)
    for uid, idx in user_to_idx.items():
        if uid in positive_users:
            y[idx] = 1.0
    data[NODE_USER].y = y

    # ── Metadata ──────────────────────────────────────────────────────────────
    data.user_to_idx = user_to_idx
    data.idx_to_user = {v: k for k, v in user_to_idx.items()}

    _validate_graph(data)
    return data


# ── Temporal graph cache ──────────────────────────────────────────────────────

_TRAIN_START = '2010-01-02'
_TRAIN_END   = '2010-09-30'
_VAL_END     = '2010-11-30'
_TEST_END    = '2011-05-17'

_SPLITS: list[tuple[str, str, str]] = [
    ('train', _TRAIN_START, _TRAIN_END),
    ('val',   _TRAIN_START, _VAL_END),
    ('test',  _TRAIN_START, _TEST_END),
]


def _graphs_dir() -> str:
    """Return the directory used for cached graph .pt files.

    Reads INNERSIGHT_MODEL_DIR at call time so the env var can be set after
    module import (e.g. in __main__ before calling build_temporal_graphs).
    """
    model_dir = os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints')
    return os.path.join(model_dir, 'graphs')


def build_temporal_graphs(data: dict | None = None) -> dict:
    """Build train / val / test HeteroData graphs and cache them to disk.

    Each graph is *cumulative*: the val graph contains all events from
    ``_TRAIN_START`` through ``_VAL_END``, and the test graph contains all
    events from ``_TRAIN_START`` through ``_TEST_END``.  Labels apply to any
    user active within the respective window.

    Parameters
    ----------
    data:
        Pre-loaded data dict with keys ``'logs'`` (full raw logs dict) and
        ``'labels'`` (set of insider identifiers).  When *None*, raw logs
        and labels are loaded from ``INNERSIGHT_DATA_DIR`` automatically.

    Returns
    -------
    dict
        ``{'train': HeteroData, 'val': HeteroData, 'test': HeteroData}``
    """
    if data is None:
        from innersight.data.pipeline import load_raw_logs, load_labels
        _data_dir = os.environ.get('INNERSIGHT_DATA_DIR')
        if not _data_dir:
            raise EnvironmentError("INNERSIGHT_DATA_DIR environment variable must be set")
        logger.info('Loading raw logs from %s', _data_dir)
        _logs   = load_raw_logs(_data_dir)
        _labels = load_labels(os.path.join(_data_dir, 'answers'))
        data    = {'logs': _logs, 'labels': _labels}

    logs   = data['logs']
    labels = data['labels']

    graphs: dict = {}
    wall_start = _time.perf_counter()

    for split, start, end in _SPLITS:
        logger.info('Building %s graph  (%s – %s) ...', split, start, end)
        t0 = _time.perf_counter()
        graphs[split] = build_hetero_graph(logs, labels, start, end)
        logger.info('%s graph built in %.1fs', split, _time.perf_counter() - t0)

    logger.info('All graphs built in %.1fs', _time.perf_counter() - wall_start)

    gdir = _graphs_dir()
    os.makedirs(gdir, exist_ok=True)
    for split, g in graphs.items():
        path = os.path.join(gdir, f'{split}_graph.pt')
        torch.save(g, path)
        logger.info('Saved %s → %s', split, path)

    return graphs


def load_temporal_graphs() -> dict:
    """Load cached HeteroData graphs from disk; builds them if missing.

    Returns
    -------
    dict
        ``{'train': HeteroData, 'val': HeteroData, 'test': HeteroData}``
    """
    gdir  = _graphs_dir()
    paths = {s: os.path.join(gdir, f'{s}_graph.pt') for s, _, _ in _SPLITS}

    if not all(os.path.exists(p) for p in paths.values()):
        logger.info('Cached graphs not found in %s — building from scratch', gdir)
        return build_temporal_graphs()

    logger.info('Loading cached graphs from %s', gdir)
    graphs = {}
    for split, path in paths.items():
        t0 = _time.perf_counter()
        graphs[split] = torch.load(path, weights_only=False)
        logger.info('Loaded %s in %.2fs', split, _time.perf_counter() - t0)
    return graphs


# ── Windowed graph construction (Phase 5) ─────────────────────────────────────
# build_windowed_graph aggregates raw events over one time window into a single
# HeteroData graph with summary edge features — distinct from the per-event
# builders above, which the legacy temporal-graph pipeline still uses. User node
# features are left as zeros: Module 2's temporal embeddings are injected at
# training time.

def _win_prepare(df):
    """Datetime-normalise and sort a log DataFrame for fast binary-search slicing.

    Sorting by date is a one-time cost per log type per graph build call that
    enables O(log n) window slicing via ``searchsorted`` instead of O(n)
    boolean-mask scans across the full DataFrame on every period slice.
    """
    if df is None or len(df) == 0 or 'date' not in df.columns:
        return None
    df = _ensure_datetime(df)
    # Sort by date so _win_slice can use binary search.
    # If the caller pre-sorted (via _presort_logs), this is a no-op in practice.
    if not (getattr(df, '_date_sorted', False) or df.attrs.get('_date_sorted', False)):
        df = df.sort_values('date').reset_index(drop=True)
        df.attrs['_date_sorted'] = True
    return df


def _win_slice(df, start, end):
    """Rows with start <= date <= end (inclusive), or None if empty.

    Uses binary search (``searchsorted``) for O(log n) slicing, which requires
    the DataFrame to be sorted by ``date`` — guaranteed by ``_win_prepare``.
    This replaces a full O(n) boolean-mask scan that dominated runtime on large
    logs (e.g. 28 M HTTP rows × 3200 periods × 10 slices = 32 000 full scans).
    """
    if df is None:
        return None
    dates = df['date']
    lo = int(dates.searchsorted(pd.Timestamp(start), side='left'))
    hi = int(dates.searchsorted(pd.Timestamp(end), side='right'))
    sub = df.iloc[lo:hi]
    return sub if len(sub) else None


def _win_flags(df):
    """Add per-row _hour / _after / _weekend / _day helper columns."""
    if '_hour' in df.columns and '_after' in df.columns and '_weekend' in df.columns and '_day' in df.columns:
        return df.copy()  # return copy so callers can modify it freely
    df = df.copy()
    h = df['date'].dt.hour
    df['_hour'] = h.astype(float)
    df['_after'] = ((h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)).astype(float)
    df['_weekend'] = (df['date'].dt.dayofweek >= 5).astype(float)
    df['_day'] = df['date'].dt.normalize()
    return df


def _win_max_burst(df, keys):
    """Max single-day event count per group, as a Series indexed by *keys*."""
    daily = df.groupby(keys + ['_day']).size()
    return daily.groupby(level=list(range(len(keys)))).max()


def _win_prior_pairs(df, left, right):
    """Set of (left, right) string pairs present in a prior-window DataFrame."""
    if df is None or len(df) == 0 or left not in df.columns or right not in df.columns:
        return set()
    pairs = df[[left, right]].dropna()
    return set(map(tuple, pairs.astype(str).to_numpy()))


def _win_domain_flag(domain, keywords):
    """1.0 if *domain* matches any keyword/substring, else 0.0."""
    return 1.0 if any(k in domain for k in keywords) else 0.0


def _win_to_removable(df):
    """Boolean Series: did each file event target removable media?

    r5+ logs carry a ``to_removable_media`` flag; r3-4 file logs record only
    removable-media copies, so every row counts as removable there.
    """
    if '_torem' in df.columns:
        return df['_torem'].astype(bool)
    if 'to_removable_media' in df.columns:
        return df['to_removable_media'].astype(str).str.lower().isin(('true', '1'))
    return pd.Series(True, index=df.index)


def _win_empty_edge(dim):
    """Empty (edge_index, edge_attr) pair for an edge type with *dim* features."""
    return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dim, dtype=torch.float32)


def _win_pairs_to_edges(agg, cols, src_map, dst_map):
    """Map an (src_key, dst_key)-indexed aggregate frame to edge tensors.

    Pairs whose src or dst key is absent from the node maps are skipped. This is
    a no-op when every key is present (the default), and is what lets URL/file
    node sets be capped to a top-K subset without crashing the edge builders.
    """
    if agg is None or len(agg) == 0:
        return _win_empty_edge(len(cols))
    src_keys = agg.index.get_level_values(0)
    dst_keys = agg.index.get_level_values(1)
    
    # Vectorised mapping via pandas Series
    src_mapped = pd.Series(src_keys).map(src_map)
    dst_mapped = pd.Series(dst_keys).map(dst_map)
    valid = src_mapped.notna() & dst_mapped.notna()
    
    if not valid.any():
        return _win_empty_edge(len(cols))
    
    if not valid.all():
        src_mapped = src_mapped[valid]
        dst_mapped = dst_mapped[valid]
        agg = agg.iloc[valid.values]
        
    src = src_mapped.astype(np.int64).to_numpy()
    dst = dst_mapped.astype(np.int64).to_numpy()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(agg[cols].to_numpy(dtype='float32'), dtype=torch.float32)
    return edge_index, edge_attr


def _win_email_recipient_ids(email_w):
    """Internal email recipients (local-part == user_id) appearing in the window."""
    ids: set[str] = set()
    suffix = f'@{INTERNAL_DOMAIN.lower()}'
    for col in ('to', 'cc', 'bcc'):
        if col not in email_w.columns:
            continue
        for raw in email_w[col].fillna('').astype(str).str.lower():
            for addr in raw.split(';'):
                addr = addr.strip()
                if addr.endswith(suffix):
                    ids.add(addr.split('@')[0])
    return ids


# ── Windowed node features ────────────────────────────────────────────────────

def _win_pc_features(logon_w, device_w, pc_to_idx):
    """PC node features (8): users, after-hours, shared, logons, usb, weekend, hour stats."""
    n = len(pc_to_idx)
    if n == 0:
        return torch.zeros((0, 8), dtype=torch.float32)
    parts = []
    if logon_w is not None and 'pc' in logon_w.columns:
        d = _win_flags(logon_w)
        d = d.assign(_is_logon=(d['activity'].astype(str).str.lower() == 'logon').astype(float),
                     _is_usb=0.0)
        parts.append(d[['pc', 'user', '_hour', '_after', '_weekend', '_is_logon', '_is_usb']])
    if device_w is not None and 'pc' in device_w.columns:
        d = _win_flags(device_w)
        d = d.assign(_is_logon=0.0,
                     _is_usb=(d['activity'].astype(str).str.lower() == 'connect').astype(float))
        parts.append(d[['pc', 'user', '_hour', '_after', '_weekend', '_is_logon', '_is_usb']])
    feats = np.zeros((n, 8), dtype='float32')
    if not parts:
        return torch.tensor(feats, dtype=torch.float32)

    ev = pd.concat(parts, ignore_index=True)
    ev = ev[ev['pc'].notna()].copy()
    ev['pc'] = ev['pc'].astype(str)
    g = ev.groupby('pc')
    agg = pd.DataFrame({
        'num_users': g['user'].nunique(),
        'after_hours_ratio': g['_after'].mean(),
        'total_logons': g['_is_logon'].sum(),
        'total_usb': g['_is_usb'].sum(),
        'weekend_ratio': g['_weekend'].mean(),
        'mean_hour': g['_hour'].mean(),
        'std_hour': g['_hour'].std().fillna(0.0),
    })
    agg['is_shared'] = (agg['num_users'] > 1).astype(float)
    for pc, i in pc_to_idx.items():
        if pc in agg.index:
            r = agg.loc[pc]
            feats[i] = [r['num_users'], r['after_hours_ratio'], r['is_shared'], r['total_logons'],
                        r['total_usb'], r['weekend_ratio'], r['mean_hour'], r['std_hour']]
    return torch.tensor(feats, dtype=torch.float32)


def _win_url_features(http_w, url_to_idx):
    """URL node features (8): visits, visitors, site flags, after-hours, hour, concentration."""
    n = len(url_to_idx)
    if n == 0:
        return torch.zeros((0, 8), dtype=torch.float32)
    feats = np.zeros((n, 8), dtype='float32')
    if http_w is None:
        return torch.tensor(feats, dtype=torch.float32)
    d = _win_flags(http_w)
    g = d.groupby('_domain')
    agg = pd.DataFrame({
        'visit_count': g.size(),
        'unique_visitors': g['user'].nunique(),
        'after_hours_ratio': g['_after'].mean(),
        'mean_hour': g['_hour'].mean(),
    })
    top_visitor = d.groupby(['_domain', 'user']).size().groupby(level=0).max()
    agg['concentration'] = (top_visitor / agg['visit_count']).fillna(0.0)
    for dom, i in url_to_idx.items():
        if dom in agg.index:
            r = agg.loc[dom]
            feats[i] = [r['visit_count'], r['unique_visitors'],
                        _win_domain_flag(dom, CERT_JOB_DOMAINS),
                        _win_domain_flag(dom, CERT_CLOUD_DOMAINS),
                        _win_domain_flag(dom, CERT_KEYLOGGER_DOMAINS),
                        r['after_hours_ratio'], r['mean_hour'], r['concentration']]
    return torch.tensor(feats, dtype=torch.float32)


def _win_file_features(file_w, file_p, file_to_idx):
    """File node features (6): copies, copiers, is_decoy, removable, after-hours, is_new."""
    n = len(file_to_idx)
    if n == 0:
        return torch.zeros((0, 6), dtype=torch.float32)
    feats = np.zeros((n, 6), dtype='float32')
    if file_w is None or 'filename' not in file_w.columns:
        return torch.tensor(feats, dtype=torch.float32)
    d = _win_flags(file_w)
    d['_torem'] = _win_to_removable(d).astype(float)
    d['filename'] = d['filename'].astype(str)
    g = d.groupby('filename')
    agg = pd.DataFrame({
        'copy_count': g.size(),
        'unique_copiers': g['user'].nunique(),
        'to_removable_count': g['_torem'].sum(),
        'after_hours_ratio': g['_after'].mean(),
    })
    prior_files = set(file_p['filename'].astype(str)) if (
        file_p is not None and 'filename' in file_p.columns) else set()
    for fn, i in file_to_idx.items():
        if fn in agg.index:
            r = agg.loc[fn]
            # is_decoy needs the decoy registry (not in the logs) → left as 0.0.
            feats[i] = [r['copy_count'], r['unique_copiers'], 0.0,
                        r['to_removable_count'], r['after_hours_ratio'],
                        0.0 if fn in prior_files else 1.0]
    return torch.tensor(feats, dtype=torch.float32)


# ── Windowed edge builders ────────────────────────────────────────────────────

def _win_logon_edges(logon_w, user_to_idx, pc_to_idx):
    cols = ['count', 'frac_after_hours', 'frac_weekend', 'mean_hour', 'max_burst_day']
    if logon_w is None or 'activity' not in logon_w.columns:
        return _win_empty_edge(WINDOWED_LOGON_EDGE_DIM)
    d = logon_w[logon_w['activity'].astype(str).str.lower() == 'logon']
    if len(d) == 0:
        return _win_empty_edge(WINDOWED_LOGON_EDGE_DIM)
    d = _win_flags(d)
    d['user'] = d['user'].astype(str)
    d['pc'] = d['pc'].astype(str)
    g = d.groupby(['user', 'pc'])
    agg = pd.DataFrame({
        'count': g.size(),
        'frac_after_hours': g['_after'].mean(),
        'frac_weekend': g['_weekend'].mean(),
        'mean_hour': g['_hour'].mean(),
    })
    agg['max_burst_day'] = _win_max_burst(d, ['user', 'pc'])
    return _win_pairs_to_edges(agg, cols, user_to_idx, pc_to_idx)


def _win_usb_edges(device_w, logon_p, device_p, user_to_idx, pc_to_idx):
    cols = ['count', 'frac_after_hours', 'max_burst_day', 'is_new_pc']
    if device_w is None or 'activity' not in device_w.columns:
        return _win_empty_edge(WINDOWED_USB_EDGE_DIM)
    d = device_w[device_w['activity'].astype(str).str.lower() == 'connect']
    if len(d) == 0:
        return _win_empty_edge(WINDOWED_USB_EDGE_DIM)
    d = _win_flags(d)
    d['user'] = d['user'].astype(str)
    d['pc'] = d['pc'].astype(str)
    g = d.groupby(['user', 'pc'])
    agg = pd.DataFrame({
        'count': g.size(),
        'frac_after_hours': g['_after'].mean(),
    })
    agg['max_burst_day'] = _win_max_burst(d, ['user', 'pc'])
    prior = _win_prior_pairs(logon_p, 'user', 'pc') | _win_prior_pairs(device_p, 'user', 'pc')
    agg['is_new_pc'] = [0.0 if (str(u), str(p)) in prior else 1.0 for u, p in agg.index]
    return _win_pairs_to_edges(agg, cols, user_to_idx, pc_to_idx)


def _win_email_edges(email_w, user_to_idx):
    cols = ['count', 'mean_size', 'max_attachments', 'frac_after_hours', 'is_external']
    if email_w is None:
        return _win_empty_edge(WINDOWED_EMAIL_EDGE_DIM)
    ex = _explode_and_map_recipients(email_w, user_to_idx)
    if ex is None or len(ex) == 0:
        return _win_empty_edge(WINDOWED_EMAIL_EDGE_DIM)
    g = ex.groupby(['_sender_idx', '_recip_idx'])
    agg = pd.DataFrame({
        'count': g.size(),
        'mean_size': g['_size'].mean(),
        'max_attachments': g['_attach'].max(),
        'frac_after_hours': g['_is_after'].mean(),
        'is_external': g['_is_external'].mean(),
    })
    src = [int(s) for s in agg.index.get_level_values(0)]
    dst = [int(s) for s in agg.index.get_level_values(1)]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(agg[cols].to_numpy(dtype='float32'), dtype=torch.float32)
    return edge_index, edge_attr


def _win_http_edges(http_w, http_p, user_to_idx, url_to_idx):
    cols = ['count', 'frac_after_hours', 'is_new_domain', 'visit_concentration']
    if http_w is None:
        return _win_empty_edge(WINDOWED_HTTP_EDGE_DIM)
    d = _win_flags(http_w)
    d['user'] = d['user'].astype(str)
    g = d.groupby(['user', '_domain'])
    agg = pd.DataFrame({
        'count': g.size(),
        'frac_after_hours': g['_after'].mean(),
    })
    burst = _win_max_burst(d, ['user', '_domain'])
    agg['visit_concentration'] = (burst / agg['count']).fillna(0.0)
    prior = _win_prior_pairs(http_p, 'user', '_domain')
    agg['is_new_domain'] = [0.0 if (str(u), str(dom)) in prior else 1.0 for u, dom in agg.index]
    return _win_pairs_to_edges(agg, cols, user_to_idx, url_to_idx)


def _win_file_edges(file_w, file_p, user_to_idx, file_to_idx):
    cols = ['count', 'frac_after_hours', 'frac_to_removable', 'is_new_file']
    if file_w is None or 'filename' not in file_w.columns:
        return _win_empty_edge(WINDOWED_FILE_EDGE_DIM)
    d = _win_flags(file_w)
    d['_torem'] = _win_to_removable(d).astype(float)
    d['user'] = d['user'].astype(str)
    d['filename'] = d['filename'].astype(str)
    g = d.groupby(['user', 'filename'])
    agg = pd.DataFrame({
        'count': g.size(),
        'frac_after_hours': g['_after'].mean(),
        'frac_to_removable': g['_torem'].mean(),
    })
    prior = _win_prior_pairs(file_p, 'user', 'filename')
    agg['is_new_file'] = [0.0 if (str(u), str(f)) in prior else 1.0 for u, f in agg.index]
    return _win_pairs_to_edges(agg, cols, user_to_idx, file_to_idx)


def _win_top_k_set(series, k, *, always_keep=None):
    """Set of the ``k`` most frequent values in ``series`` (plus ``always_keep``).

    ``k`` of ``None``/``<=0`` means "keep everything" (no cap). ``always_keep`` is
    a set of values that are retained even if they fall outside the top-K (used to
    preserve signal-bearing flagged domains). Only values present in ``series`` are
    ever returned.
    """
    if series is None:
        return set()
    vals = series.dropna().astype(str)
    if vals.empty:
        return set()
    counts = vals.value_counts()
    if k is None or k <= 0 or len(counts) <= k:
        return set(counts.index)
    kept = set(counts.index[:k])
    if always_keep:
        kept |= (set(always_keep) & set(counts.index))
    return kept


def build_windowed_graph(logs, window_start, window_end, prior_days=60,
                         max_url_nodes=None, max_file_nodes=None):
    """Build a PyG HeteroData graph for one time window with aggregated edges.

    Only events with ``window_start <= date <= window_end`` are included. For
    each unique (user, entity) pair, all events in the window collapse into a
    single edge whose features summarise the interaction (count, after-hours
    fraction, burst, novelty, …). ``is_new`` flags compare against the
    ``prior_days`` immediately before the window. Raw CERT log DataFrames are
    consumed (not deviation matrices); missing log types are handled gracefully.

    Args:
        logs: Mapping of log name → DataFrame (as in ``CertDataset.logs``).
        window_start: Inclusive window start (timestamp-like).
        window_end: Inclusive window end (timestamp-like).
        prior_days: Days before the window scanned to flag new connections.
        max_url_nodes: If set, cap URL nodes to this many most-frequent domains
            per window (signal-bearing job/cloud/keylogger domains are always
            kept). ``None`` keeps every domain. Bounds memory on large http logs.
        max_file_nodes: If set, cap file nodes to this many most-frequently-copied
            filenames per window. ``None`` keeps every file.

    Returns:
        A ``HeteroData`` with user/pc/url/file nodes, the five forward edge types
        and their reverses, and ``user_to_idx`` / ``pc_to_idx`` / ``url_to_idx``
        / ``file_to_idx`` attributes for aligning injected user embeddings.
    """
    ws = pd.Timestamp(window_start)
    we = pd.Timestamp(window_end)
    prior_start = ws - pd.Timedelta(days=prior_days)
    prior_end = ws - pd.Timedelta(seconds=1)  # strictly before the window

    logon = _win_prepare(logs.get('logon'))
    device = _win_prepare(logs.get('device'))
    email = _win_prepare(logs.get('email'))
    http = _win_prepare(logs.get('http'))
    file = _win_prepare(logs.get('file'))

    logon_w = _win_slice(logon, ws, we)
    device_w = _win_slice(device, ws, we)
    email_w = _win_slice(email, ws, we)
    http_w = _win_slice(http, ws, we)
    file_w = _win_slice(file, ws, we)

    logon_p = _win_slice(logon, prior_start, prior_end)
    device_p = _win_slice(device, prior_start, prior_end)
    http_p = _win_slice(http, prior_start, prior_end)
    file_p = _win_slice(file, prior_start, prior_end)

    # Resolve URLs to domains once (used for both nodes and edges).
    if http_w is not None:
        if '_domain' not in http_w.columns:
            http_w = http_w.copy()
            http_w['_domain'] = http_w['url'].apply(_extract_domain)
    if http_p is not None:
        if '_domain' not in http_p.columns:
            http_p = http_p.copy()
            http_p['_domain'] = http_p['url'].apply(_extract_domain)

    # ── Node sets ──
    users: set[str] = set()
    for df in (logon_w, device_w, email_w, http_w, file_w):
        if df is not None and 'user' in df.columns:
            users.update(df['user'].dropna().astype(str))
    if email_w is not None:
        users.update(_win_email_recipient_ids(email_w))
    pcs: set[str] = set()
    for df in (logon_w, device_w):
        if df is not None and 'pc' in df.columns:
            pcs.update(df['pc'].dropna().astype(str))
    # URL/file node sets, optionally capped to the top-K most frequent per window
    # to bound memory (the long tail of one-off domains/files dominates node count
    # on r4.2-scale logs and carries little signal). Flagged domains are kept.
    if http_w is not None:
        _flagged_domains = {
            d for d in http_w['_domain'].dropna().astype(str).unique()
            if _win_domain_flag(d, CERT_JOB_DOMAINS)
            or _win_domain_flag(d, CERT_CLOUD_DOMAINS)
            or _win_domain_flag(d, CERT_KEYLOGGER_DOMAINS)
        }
        urls = _win_top_k_set(http_w['_domain'], max_url_nodes, always_keep=_flagged_domains)
    else:
        urls = set()
    files = _win_top_k_set(
        file_w['filename'], max_file_nodes,
    ) if (file_w is not None and 'filename' in file_w.columns) else set()

    user_to_idx = {u: i for i, u in enumerate(sorted(users))}
    pc_to_idx = {p: i for i, p in enumerate(sorted(pcs))}
    url_to_idx = {u: i for i, u in enumerate(sorted(urls))}
    file_to_idx = {f: i for i, f in enumerate(sorted(files))}

    data = HeteroData()
    data[NODE_USER].x = torch.zeros(len(user_to_idx), USER_TEMPORAL_DIM, dtype=torch.float32)
    data[NODE_PC].x = _win_pc_features(logon_w, device_w, pc_to_idx)
    data[NODE_URL].x = _win_url_features(http_w, url_to_idx)
    data[NODE_FILE].x = _win_file_features(file_w, file_p, file_to_idx)

    edges = (
        (EDGE_LOGON, REV_EDGE_LOGON, _win_logon_edges(logon_w, user_to_idx, pc_to_idx)),
        (EDGE_USB, REV_EDGE_USB, _win_usb_edges(device_w, logon_p, device_p, user_to_idx, pc_to_idx)),
        (EDGE_EMAIL, REV_EDGE_EMAIL, _win_email_edges(email_w, user_to_idx)),
        (EDGE_HTTP, REV_EDGE_HTTP, _win_http_edges(http_w, http_p, user_to_idx, url_to_idx)),
        (EDGE_FILE_COPY, REV_EDGE_FILE_COPY, _win_file_edges(file_w, file_p, user_to_idx, file_to_idx)),
    )
    for etype, rev, (edge_index, edge_attr) in edges:
        data[etype].edge_index = edge_index
        data[etype].edge_attr = edge_attr
        data[rev].edge_index = edge_index.flip(0)
        data[rev].edge_attr = edge_attr

    # Index maps for aligning injected temporal embeddings with node rows.
    data.user_to_idx = user_to_idx
    data.pc_to_idx = pc_to_idx
    data.url_to_idx = url_to_idx
    data.file_to_idx = file_to_idx
    data.window_start = ws
    data.window_end = we
    return data


def build_graphs_for_dataset(dataset, window_size=28, stride=7):
    """Slide windows across a dataset's date range, one HeteroData graph per window.

    Args:
        dataset: A ``CertDataset`` (uses ``dataset.logs``).
        window_size: Window length in days.
        stride: Days to advance between consecutive windows.

    Returns:
        List of ``HeteroData`` graphs, one per window (empty if no dated logs).
    """
    logs = dataset.logs
    bounds = []
    for df in logs.values():
        if df is not None and len(df) and 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            bounds.append((dates.min(), dates.max()))
    if not bounds:
        return []
    min_date = min(b[0] for b in bounds).normalize()
    max_date = max(b[1] for b in bounds)

    span = pd.Timedelta(days=window_size - 1)
    step = pd.Timedelta(days=stride)
    graphs = []
    start = min_date
    while start + span <= max_date:
        graphs.append(build_windowed_graph(logs, start, start + span))
        start += step
    return graphs


# ── Smoke test / inspection ───────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    ap = argparse.ArgumentParser(description='Build and inspect all node features.')
    ap.add_argument('--data-dir', default=os.environ.get('INNERSIGHT_DATA_DIR', ''),
                    help='Path to CERT data directory (overrides INNERSIGHT_DATA_DIR)')
    ap.add_argument('--model-dir', default=os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints'),
                    help='Directory for cached graph .pt files (overrides INNERSIGHT_MODEL_DIR)')
    args = ap.parse_args()

    data_dir = args.data_dir
    if not data_dir:
        print('ERROR: Set INNERSIGHT_DATA_DIR or pass --data-dir')
        sys.exit(1)

    # Set model dir early so _graphs_dir() picks it up before any function call.
    os.environ['INNERSIGHT_MODEL_DIR'] = args.model_dir

    from innersight.data.pipeline import load_raw_logs

    print(f'\nLoading raw logs from: {data_dir}')
    logs = load_raw_logs(data_dir)
    print('Log sizes:')
    for name, df in logs.items():
        print(f'  {name:<8}: {len(df):>8,} rows')

    TRAIN_START = '2009-01-01'
    TRAIN_END   = '2010-09-30'
    print(f'\nTime window: {TRAIN_START} – {TRAIN_END}')

    # ── helper: print stats table for any node tensor ────────────────────────
    def _print_stats(label: str, x: torch.Tensor, feat_cols: list[str], idx: dict) -> None:
        W = 74
        print(f'\n{"=" * W}')
        print(f'{label}')
        print(f'  tensor shape : {tuple(x.shape)}  (dtype={x.dtype})')
        print(f'  node count   : {x.shape[0]:,}')
        print(f'{"=" * W}')
        if x.shape[0] == 0:
            print('  (no nodes in this window)')
            return

        # One sample entity
        idx_to_id = {v: k for k, v in idx.items()}
        sample_id = idx_to_id[0]
        print(f'\n  Sample entity: {sample_id!r}')
        for col, val in zip(feat_cols, x[0].tolist()):
            print(f'    {col:<32}: {val:>12.4f}')

        # Per-feature stats
        print(f'\n  {"Feature":<32} {"min":>9} {"max":>9} {"mean":>9} {"nonzero%":>9}')
        print(f'  {"-" * 70}')
        for i, col in enumerate(feat_cols):
            v     = x[:, i]
            nzpct = (v != 0).float().mean().item() * 100
            print(f'  {col:<32} {v.min().item():>9.3f} {v.max().item():>9.3f} '
                  f'{v.mean().item():>9.3f} {nzpct:>8.1f}%')

    # ── node builders ─────────────────────────────────────────────────────────
    x_user, user_to_idx = build_user_nodes(logs, TRAIN_START, TRAIN_END)
    _print_stats('USER nodes', x_user, _USER_FEAT_COLS, user_to_idx)

    x_pc, pc_to_idx = build_pc_nodes(logs, TRAIN_START, TRAIN_END)
    _print_stats('PC nodes', x_pc, _PC_FEAT_COLS, pc_to_idx)

    x_url, url_to_idx = build_url_nodes(logs, TRAIN_START, TRAIN_END)
    _print_stats('URL (domain) nodes', x_url, _URL_FEAT_COLS, url_to_idx)

    x_file, file_to_idx = build_file_nodes(logs, TRAIN_START, TRAIN_END)
    _print_stats('FILE nodes', x_file, _FILE_FEAT_COLS, file_to_idx)

    # ── edge helper ───────────────────────────────────────────────────────────
    def _print_edges(
        label: str,
        result: dict,
        src_idx: dict,
        dst_idx: dict,
        edge_cols: list[str],
    ) -> None:
        W = 74
        ei = result['edge_index']
        ea = result['edge_attr']
        print(f'\n{"=" * W}')
        print(f'{label}')
        print(f'  edge_index shape : {tuple(ei.shape)}  (dtype={ei.dtype})')
        print(f'  edge_attr  shape : {tuple(ea.shape)}  (dtype={ea.dtype})')
        print(f'  n_edges          : {ei.shape[1]:,}')
        print(f'  n_skipped        : {result["n_skipped"]:,}')

        if ei.shape[1] == 0:
            print('  (no edges in this window)')
            return

        idx_to_src = {v: k for k, v in src_idx.items()}
        idx_to_dst = {v: k for k, v in dst_idx.items()}

        print(f'\n  Feature columns: {edge_cols}')
        print(f'\n  First 5 forward edges:')
        print(f'  {"src_idx":>7} {"src_name":<12} → {"dst_idx":>7} {"dst_name":<12}  {" | ".join(f"{c}" for c in edge_cols)}')
        print(f'  {"-" * (W - 2)}')
        for i in range(min(5, ei.shape[1])):
            s = ei[0, i].item()
            d = ei[1, i].item()
            feats = '  '.join(f'{v:.3f}' for v in ea[i].tolist())
            print(f'  {s:>7} {idx_to_src.get(s, "?"):<12}   '
                  f'{d:>7} {idx_to_dst.get(d, "?"):<12}  {feats}')

    # ── logon edges ───────────────────────────────────────────────────────────
    logon_edges = build_logon_edges(
        logs.get('logon', pd.DataFrame()),
        user_to_idx, pc_to_idx,
        TRAIN_START, TRAIN_END,
    )
    _print_edges('LOGON edges  (user → pc)', logon_edges,
                 user_to_idx, pc_to_idx, _LOGON_EDGE_COLS)

    # ── USB edges ─────────────────────────────────────────────────────────────
    usb_edges = build_usb_edges(
        logs.get('device', pd.DataFrame()),
        user_to_idx, pc_to_idx,
        TRAIN_START, TRAIN_END,
    )
    _print_edges('USB edges  (user → pc)', usb_edges,
                 user_to_idx, pc_to_idx, _USB_EDGE_COLS)

    # ── email edges ───────────────────────────────────────────────────────────
    email_edges = build_email_edges(
        logs.get('email', pd.DataFrame()),
        user_to_idx,
        TRAIN_START, TRAIN_END,
    )
    _print_edges('EMAIL edges  (user → user)', email_edges,
                 user_to_idx, user_to_idx, _EMAIL_EDGE_COLS)

    # ── HTTP edges ────────────────────────────────────────────────────────────
    http_edges = build_http_edges(
        logs.get('http', pd.DataFrame()),
        user_to_idx, url_to_idx,
        TRAIN_START, TRAIN_END,
    )
    _print_edges('HTTP edges  (user → url)', http_edges,
                 user_to_idx, url_to_idx, _HTTP_EDGE_COLS)

    # ── file edges ────────────────────────────────────────────────────────────
    file_edges = build_file_edges(
        logs.get('file', pd.DataFrame()),
        user_to_idx, file_to_idx,
        TRAIN_START, TRAIN_END,
    )
    _print_edges('FILE edges  (user → file)', file_edges,
                 user_to_idx, file_to_idx, _FILE_EDGE_COLS)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f'\n{"=" * 74}')
    print('Graph summary')
    print(f'\n  Nodes:')
    print(f'  {"type":<8} {"count":>8}  {"feat_dim":>9}')
    print(f'  {"-" * 28}')
    for label, x_t, dim in [
        ('user', x_user, USER_FEATURE_DIM),
        ('pc',   x_pc,   PC_FEATURE_DIM),
        ('url',  x_url,  URL_FEATURE_DIM),
        ('file', x_file, FILE_FEATURE_DIM),
    ]:
        print(f'  {label:<8} {x_t.shape[0]:>8,}  {dim:>9}')

    print(f'\n  Edges (forward only):')
    print(f'  {"type":<26} {"count":>8}  {"feat_dim":>9}  {"skipped":>8}')
    print(f'  {"-" * 56}')
    for label, result, dim in [
        ('(user, logon, pc)',        logon_edges, LOGON_EDGE_DIM),
        ('(user, usb_connect, pc)', usb_edges,   USB_EDGE_DIM),
        ('(user, email_to, user)',  email_edges, EMAIL_EDGE_DIM),
        ('(user, http_request, url)', http_edges, HTTP_EDGE_DIM),
        ('(user, file_copy, file)', file_edges,  FILE_EDGE_DIM),
    ]:
        n  = result['edge_index'].shape[1]
        sk = result['n_skipped']
        print(f'  {label:<26} {n:>8,}  {dim:>9}  {sk:>8,}')
    print(f'{"=" * 74}')

    # ── HeteroData assembly ───────────────────────────────────────────────────
    print('\nAssembling HeteroData...')

    # Load insiders from answers/ if available, otherwise use empty set.
    answers_path = os.path.join(data_dir, 'answers', 'insiders.csv')
    label_set: set = set()
    if os.path.exists(answers_path):
        ans_df = pd.read_csv(answers_path)
        label_set = set(ans_df['user'].tolist())
        print(f'  Loaded {len(label_set)} insider labels from {answers_path}')
    else:
        print('  No insiders.csv found — using empty label set (all negatives)')

    hetero = build_hetero_graph(logs, label_set, TRAIN_START, TRAIN_END)

    print(f'\nHeteroData repr:')
    print(hetero)

    print(f'\nLabel distribution:')
    y = hetero[NODE_USER].y
    n_pos = int(y.sum().item())
    n_neg = y.shape[0] - n_pos
    print(f'  total users : {y.shape[0]:,}')
    print(f'  positives   : {n_pos:,}  ({n_pos / max(y.shape[0], 1):.4%})')
    print(f'  negatives   : {n_neg:,}')

    print(f'\nMetadata:')
    print(f'  user_to_idx entries : {len(hetero.user_to_idx):,}')
    print(f'  idx_to_user entries : {len(hetero.idx_to_user):,}')
    print(f'\nBuild complete.')

    # ── Temporal graphs ───────────────────────────────────────────────────────
    print(f'\n{"=" * 80}')
    print('Building temporal graphs (train / val / test) ...')
    print(f'  Graphs dir: {_graphs_dir()}')
    print()

    # Load answers for label set.
    from innersight.data.pipeline import load_labels as _load_labels
    _answers_dir = os.path.join(data_dir, 'answers')
    _labels = _load_labels(_answers_dir)
    print(f'  Insider labels loaded: {len(_labels)}')

    wall_t0 = _time.perf_counter()
    tg = build_temporal_graphs({'logs': logs, 'labels': _labels})
    wall_elapsed = _time.perf_counter() - wall_t0
    print(f'\n  Total build time: {wall_elapsed:.1f}s')

    # ── Reload and verify ─────────────────────────────────────────────────────
    print('\nReloading from disk ...')
    reload_t0 = _time.perf_counter()
    tg2 = load_temporal_graphs()
    reload_elapsed = _time.perf_counter() - reload_t0
    print(f'  Reload time: {reload_elapsed:.2f}s')

    for split in ('train', 'val', 'test'):
        orig_shape = tg[split][NODE_USER].x.shape
        load_shape = tg2[split][NODE_USER].x.shape
        assert orig_shape == load_shape, \
            f'{split}: user x shape mismatch {orig_shape} vs {load_shape}'
    print('  Reload verification passed.')

    # ── Side-by-side summary ──────────────────────────────────────────────────
    print(f'\n{"=" * 80}')
    print('Temporal graph summary')
    print(f'{"=" * 80}')

    EDGE_LABELS = [
        (EDGE_LOGON,     'logon'),
        (EDGE_USB,       'usb_connect'),
        (EDGE_EMAIL,     'email_to'),
        (EDGE_HTTP,      'http_request'),
        (EDGE_FILE_COPY, 'file_copy'),
    ]

    col_w = 14
    hdr = f"  {'metric':<28}" + ''.join(f'{s:>{col_w}}' for s in ('train', 'val', 'test'))
    sep = '  ' + '-' * (28 + col_w * 3)
    print(hdr)
    print(sep)

    def _row(label: str, vals: list) -> None:
        print(f"  {label:<28}" + ''.join(f'{v:>{col_w},}' for v in vals))

    # Node counts
    for ntype in (NODE_USER, NODE_PC, NODE_URL, NODE_FILE):
        _row(f'nodes/{ntype}', [tg2[s][ntype].x.shape[0] for s in ('train', 'val', 'test')])

    print(sep)

    # Edge counts (forward)
    for etype, elabel in EDGE_LABELS:
        _row(f'edges/{elabel}', [tg2[s][etype].edge_index.shape[1] for s in ('train', 'val', 'test')])

    print(sep)

    # Positives and totals
    _row('positive users', [int(tg2[s][NODE_USER].y.sum().item()) for s in ('train', 'val', 'test')])
    _row('total nodes',    [tg2[s].num_nodes for s in ('train', 'val', 'test')])
    _row('total edges',    [tg2[s].num_edges for s in ('train', 'val', 'test')])

    print(f'{"=" * 80}')
    print(f'\nGraphs saved to: {_graphs_dir()}')
