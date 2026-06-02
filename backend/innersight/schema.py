"""Shared data contracts for InnerSight UEBA.

Defines the canonical column schemas that every CERT version must be
normalized to. Adapters produce DataFrames matching these contracts;
downstream code (features, models) imports column names from here.

Public API:
    LOGON_SCHEMA      — canonical logon columns and dtypes
    DEVICE_SCHEMA     — canonical device columns and dtypes
    EMAIL_SCHEMA      — canonical email columns and dtypes
    HTTP_SCHEMA       — canonical http columns and dtypes
    FILE_SCHEMA       — canonical file columns and dtypes
    LDAP_SCHEMA       — canonical LDAP columns and dtypes
    PSYCHOMETRIC_SCHEMA — canonical psychometric columns and dtypes
    INSIDER_SCHEMA    — canonical insiders.csv columns and dtypes
    validate_dataframe(df, schema, source) — check df matches schema contract
    FEATURE_NAMES     — the 18 per-user-per-day feature names (canonical ordering)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColumnSchema:
    """Defines expected columns and their pandas dtypes for a log type.

    Attributes:
        name: Human-readable name for error messages (e.g., 'logon').
        required: Columns that MUST be present. Dict of {col_name: expected_dtype_str}.
        optional: Columns that MAY be present. Dict of {col_name: expected_dtype_str}.
    """
    name: str
    required: dict[str, str] = field(default_factory=dict)
    optional: dict[str, str] = field(default_factory=dict)

    @property
    def all_columns(self) -> list[str]:
        return list(self.required.keys()) + list(self.optional.keys())

    @property
    def required_columns(self) -> list[str]:
        return list(self.required.keys())


# ── Canonical schemas ────────────────────────────────────────────────────────
# Every adapter normalizes its version's CSV into these exact column sets.
# 'datetime64[ns]' means pd.to_datetime was applied.
# 'object' means string.
# 'int64' / 'float64' means numeric.
# 'bool' means True/False.

LOGON_SCHEMA = ColumnSchema(
    name='logon',
    required={
        'id': 'object',
        'date': 'datetime64[ns]',
        'user': 'object',
        'pc': 'object',
        'activity': 'object',   # Logon | Logoff (normalized)
    },
)

DEVICE_SCHEMA = ColumnSchema(
    name='device',
    required={
        'id': 'object',
        'date': 'datetime64[ns]',
        'user': 'object',
        'pc': 'object',
        'activity': 'object',   # Connect | Disconnect (normalized from Insert/Remove in r2)
    },
    optional={
        'file_tree': 'object',  # r5+ only: semicolon-delimited directory listing
    },
)

EMAIL_SCHEMA = ColumnSchema(
    name='email',
    required={
        'id': 'object',
        'date': 'datetime64[ns]',
        'user': 'object',
        'pc': 'object',
        'to': 'object',
        'cc': 'object',
        'bcc': 'object',
        'from': 'object',
        'size': 'float64',       # float to handle NaN
        'attachments': 'object', # int-count (r3-r4) or filenames (r5+), stored as string
        'content': 'object',
    },
)

HTTP_SCHEMA = ColumnSchema(
    name='http',
    required={
        'id': 'object',
        'date': 'datetime64[ns]',
        'user': 'object',
        'pc': 'object',
        'url': 'object',
        'content': 'object',
    },
    optional={
        'activity': 'object',   # r6 only: WWW Visit | WWW Upload | WWW Download
    },
)

FILE_SCHEMA = ColumnSchema(
    name='file',
    required={
        'id': 'object',
        'date': 'datetime64[ns]',
        'user': 'object',
        'pc': 'object',
        'filename': 'object',
        'content': 'object',
    },
    optional={
        'activity': 'object',           # r5+: File Open | File Write | File Copy | File Delete
        'to_removable_media': 'object', # r5+: True | False (string)
        'from_removable_media': 'object', # r5+: True | False (string)
    },
)

LDAP_SCHEMA = ColumnSchema(
    name='ldap',
    required={
        'employee_name': 'object',
        'user_id': 'object',
        'email': 'object',
        'role': 'object',
    },
    optional={
        'business_unit': 'object',
        'functional_unit': 'object',
        'department': 'object',
        'team': 'object',
        'supervisor': 'object',
        'projects': 'object',     # r5+ only
    },
)

PSYCHOMETRIC_SCHEMA = ColumnSchema(
    name='psychometric',
    required={
        'employee_name': 'object',
        'user_id': 'object',
        'O': 'int64',
        'C': 'int64',
        'E': 'int64',
        'A': 'int64',
        'N': 'int64',
    },
)

INSIDER_SCHEMA = ColumnSchema(
    name='insiders',
    required={
        'dataset': 'object',
        'scenario': 'int64',
        'details': 'object',
        'user': 'object',
        'start': 'datetime64[ns]',
        'end': 'datetime64[ns]',
    },
)

# ── The 18 per-user-per-day feature names (canonical ordering) ───────────────
FEATURE_NAMES: list[str] = [
    'logon_count', 'logoff_count', 'after_hours_logons', 'weekend_logons', 'unique_pcs_used',
    'usb_connect_count', 'usb_disconnect_count', 'after_hours_usb',
    'file_count', 'file_to_removable_count', 'unique_filenames',
    'email_sent_count', 'email_to_external_count', 'large_attachment_count', 'total_email_size',
    'http_request_count', 'job_search_visits', 'cloud_upload_visits',
]


def validate_dataframe(
    df: pd.DataFrame,
    schema: ColumnSchema,
    source: str = '',
) -> None:
    """Validate that a DataFrame matches its canonical schema contract.

    Checks that all required columns are present. Does NOT enforce dtypes
    (adapters handle type coercion), but logs warnings for missing optional columns.

    Args:
        df: DataFrame to validate.
        schema: The ColumnSchema it should conform to.
        source: Description of where this DataFrame came from (for error messages).

    Raises:
        ValueError: If any required column is missing.
    """
    actual_cols = set(df.columns)
    required = set(schema.required.keys())
    missing = required - actual_cols
    if missing:
        raise ValueError(
            f"DataFrame from {source!r} is missing required {schema.name} columns: "
            f"{sorted(missing)}. Found: {sorted(actual_cols)}"
        )
    optional = set(schema.optional.keys())
    absent_optional = optional - actual_cols
    if absent_optional:
        logger.debug(
            "validate_dataframe | %s from %r: optional columns absent: %s",
            schema.name, source, sorted(absent_optional),
        )
