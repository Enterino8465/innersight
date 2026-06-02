"""Feature engineering for insider-threat detection.

Transforms raw CERT log DataFrames into a per-(user, day) feature matrix.

Public API:
  build_user_day_features(logs_dict, malicious_tuples) -> pd.DataFrame
  build_features_for_split(data)                       -> dict[str, pd.DataFrame]
"""

from __future__ import annotations
import logging
from typing import Any

import pandas as pd

from innersight.config import (
    BUSINESS_HOURS_START   as _AFTER_HOURS_START,
    BUSINESS_HOURS_END     as _AFTER_HOURS_END,
    LARGE_ATTACHMENT_BYTES as _LARGE_ATTACH_BYTES,
    JOB_KEYWORDS           as _JOB_KEYWORDS,
    CLOUD_KEYWORDS         as _CLOUD_KEYWORDS,
    INTERNAL_DOMAIN        as _INTERNAL_DOMAIN,
)

logger = logging.getLogger(__name__)


def _is_after_hours(series: pd.Series) -> pd.Series:
    h = series.dt.hour
    return (h < _AFTER_HOURS_START) | (h >= _AFTER_HOURS_END)


def _is_weekend(series: pd.Series) -> pd.Series:
    return series.dt.dayofweek >= 5  # 5=Saturday, 6=Sunday


def _normalize_date(series: pd.Series) -> pd.Series:
    return series.dt.normalize()


def _add_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['_day'] = _normalize_date(df['date'])
    return df


def _logon_features(logon_df: pd.DataFrame) -> pd.DataFrame:
    if logon_df.empty:
        return pd.DataFrame(columns=[
            'user', '_day', 'logon_count', 'logoff_count',
            'after_hours_logons', 'weekend_logons', 'unique_pcs_used',
        ])
    df = _add_day(logon_df)
    df['_after_hours'] = _is_after_hours(df['date'])
    df['_weekend']     = _is_weekend(df['date'])
    df['_is_logon']    = df['activity'].str.lower() == 'logon'
    df['_is_logoff']   = df['activity'].str.lower() == 'logoff'

    g = df.groupby(['user', '_day'])
    out = pd.DataFrame({
        'logon_count':       g['_is_logon'].sum(),
        'logoff_count':      g['_is_logoff'].sum(),
        'after_hours_logons': g.apply(lambda x: (x['_after_hours'] & x['_is_logon']).sum()),
        'weekend_logons':    g.apply(lambda x: (x['_weekend'] & x['_is_logon']).sum()),
        'unique_pcs_used':   g['pc'].nunique(),
    }).reset_index()
    return out


def _device_features(device_df: pd.DataFrame) -> pd.DataFrame:
    if device_df.empty:
        return pd.DataFrame(columns=[
            'user', '_day', 'usb_connect_count', 'usb_disconnect_count', 'after_hours_usb',
        ])
    df = _add_day(device_df)
    df['_after_hours']    = _is_after_hours(df['date'])
    df['_is_connect']     = df['activity'].str.lower() == 'connect'
    df['_is_disconnect']  = df['activity'].str.lower() == 'disconnect'

    g = df.groupby(['user', '_day'])
    out = pd.DataFrame({
        'usb_connect_count':    g['_is_connect'].sum(),
        'usb_disconnect_count': g['_is_disconnect'].sum(),
        'after_hours_usb':      g['_after_hours'].sum(),
    }).reset_index()
    return out


def _file_features(file_df: pd.DataFrame) -> pd.DataFrame:
    if file_df.empty:
        return pd.DataFrame(columns=[
            'user', '_day', 'file_count', 'file_to_removable_count', 'unique_filenames',
        ])
    df = _add_day(file_df)
    # 'to_removable_media' may be a real bool (r5/r6) or a 'True'/'False' string
    # (r3/r4 adapters tag it). Normalise anything non-boolean via string compare —
    # pandas 3.0 reads string columns as the 'str' dtype, not 'object', so a bare
    # `dtype == object` check silently misses them (summing strings concatenates).
    removable = df.get('to_removable_media', pd.Series(False, index=df.index))
    if not pd.api.types.is_bool_dtype(removable):
        removable = removable.astype(str).str.strip().str.lower() == 'true'
    df['_removable'] = removable.fillna(False)

    g = df.groupby(['user', '_day'])
    out = pd.DataFrame({
        'file_count':              g['filename'].count(),
        'file_to_removable_count': g['_removable'].sum(),
        'unique_filenames':        g['filename'].nunique(),
    }).reset_index()
    return out


def _email_features(email_df: pd.DataFrame) -> pd.DataFrame:
    if email_df.empty:
        return pd.DataFrame(columns=[
            'user', '_day', 'email_sent_count', 'email_to_external_count',
            'large_attachment_count', 'total_email_size',
        ])
    df = _add_day(email_df)

    def _has_external(to_field):
        if pd.isna(to_field):
            return False
        return any(_INTERNAL_DOMAIN not in addr for addr in str(to_field).split(';'))

    def _attachment_count(size):
        return int(size > _LARGE_ATTACH_BYTES) if pd.notna(size) else 0

    df['_to_external']   = df['to'].apply(_has_external)
    df['_large_attach']  = df['size'].apply(_attachment_count)

    g = df.groupby(['user', '_day'])
    out = pd.DataFrame({
        'email_sent_count':       g['id'].count(),
        'email_to_external_count': g['_to_external'].sum(),
        'large_attachment_count': g['_large_attach'].sum(),
        'total_email_size':       g['size'].sum(),
    }).reset_index()
    return out


def _http_features(http_df: pd.DataFrame) -> pd.DataFrame:
    if http_df.empty:
        return pd.DataFrame(columns=[
            'user', '_day', 'http_request_count', 'job_search_visits', 'cloud_upload_visits',
        ])
    df = _add_day(http_df)
    url = df['url'].fillna('').str.lower()
    df['_job_search']    = url.apply(lambda u: any(k in u for k in _JOB_KEYWORDS))
    df['_cloud_upload']  = url.apply(lambda u: any(k in u for k in _CLOUD_KEYWORDS))

    g = df.groupby(['user', '_day'])
    out = pd.DataFrame({
        'http_request_count': g['url'].count(),
        'job_search_visits':  g['_job_search'].sum(),
        'cloud_upload_visits': g['_cloud_upload'].sum(),
    }).reset_index()
    return out


def build_user_day_features(
    logs_dict: dict[str, pd.DataFrame],
    malicious_tuples: set,
) -> pd.DataFrame:
    logon_f  = _logon_features(logs_dict.get('logon',  pd.DataFrame()))
    device_f = _device_features(logs_dict.get('device', pd.DataFrame()))
    file_f   = _file_features(logs_dict.get('file',   pd.DataFrame()))
    email_f  = _email_features(logs_dict.get('email',  pd.DataFrame()))
    http_f   = _http_features(logs_dict.get('http',   pd.DataFrame()))

    # Build the universe of (user, day) pairs from all logs
    non_empty = [
        f[['user', '_day']]
        for f in [logon_f, device_f, file_f, email_f, http_f]
        if not f.empty
    ]
    if not non_empty:
        return pd.DataFrame()
    keys = pd.concat(non_empty).drop_duplicates()

    feature_dfs = [logon_f, device_f, file_f, email_f, http_f]
    merged = keys
    for fdf in feature_dfs:
        if fdf.empty:
            continue
        merged = merged.merge(fdf, on=['user', '_day'], how='left')

    # Fill missing counts with 0
    merged = merged.fillna(0)

    # Cast integer-valued columns
    int_cols = [c for c in merged.columns if c not in ('user', '_day', 'total_email_size')]
    merged[int_cols] = merged[int_cols].astype(int)

    # Label on the normalized day. get_malicious_dates() keys on python date,
    # so compare against _day.dt.date.
    day_dates = merged['_day'].dt.date
    merged['is_malicious'] = [
        int((user, day) in malicious_tuples)
        for user, day in zip(merged['user'], day_dates)
    ]

    # Canonical 'date' column is the normalized day. Rename _day -> date (a single
    # column); a previous version also added a separate date column, producing a
    # duplicate that broke Parquet serialization and str(row['date']) in scoring.
    merged = merged.rename(columns={'_day': 'date'})
    feature_cols = [
        'user', 'date',
        'logon_count', 'logoff_count', 'after_hours_logons', 'weekend_logons', 'unique_pcs_used',
        'usb_connect_count', 'usb_disconnect_count', 'after_hours_usb',
        'file_count', 'file_to_removable_count', 'unique_filenames',
        'email_sent_count', 'email_to_external_count', 'large_attachment_count', 'total_email_size',
        'http_request_count', 'job_search_visits', 'cloud_upload_visits',
        'is_malicious',
    ]
    # Keep only columns that exist (guards against empty sub-DataFrames)
    feature_cols = [c for c in feature_cols if c in merged.columns]
    return merged[feature_cols].reset_index(drop=True)


def build_features_for_split(data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    labels = data['labels']
    result = {}
    for split_name, logs_dict in data['splits'].items():
        df = build_user_day_features(logs_dict, labels)
        ratio = df['is_malicious'].mean() if len(df) else 0.0
        logger.info(
            'build_features_for_split | %s: shape=%s | malicious_ratio=%.4f%%',
            split_name, df.shape, ratio * 100,
        )
        result[split_name] = df
    return result
