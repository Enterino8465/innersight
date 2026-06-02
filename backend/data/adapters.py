"""Version-specific adapters for the CERT insider threat dataset.

Each adapter knows how to normalize its version family's CSV files into
the canonical schemas defined in schema.py. Adapters are thin — they
configure the loaders (from loaders.py) with version-specific column
names, then apply any necessary normalization (e.g., stripping DTAA/
prefix from r1 user IDs, mapping Insert→Connect in r2).

Public API:
    CertAdapter         — Protocol defining the adapter interface
    R1Adapter           — r1 family
    R2Adapter           — r2 family
    R3xAdapter          — r3.1, r3.2
    R4xAdapter          — r4.1, r4.2 (primary training version)
    R5xAdapter          — r5.1, r5.2
    R6xAdapter          — r6.1, r6.2 (streaming for large files)
    get_adapter(version) — factory returning the correct adapter
    auto_detect_version(data_dir) — fingerprint CSV headers to identify version
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

from innersight.backend.config import VERSION_FAMILIES
from innersight.backend.data.loaders import (
    load_csv,
    load_csv_chunked,
    load_decoy_files,
    load_ldap_directory,
    load_psychometric,
)
from innersight.backend.schema import (
    DEVICE_SCHEMA,
    EMAIL_SCHEMA,
    FILE_SCHEMA,
    HTTP_SCHEMA,
    LOGON_SCHEMA,
    ColumnSchema,
    validate_dataframe,
)

logger = logging.getLogger(__name__)

# ── Version-specific constants ───────────────────────────────────────────────
_DTAA_PREFIX = 'DTAA/'  # r1 prefixes every user id with this; strip it.
_DEVICE_ACTIVITY_MAP = {'Insert': 'Connect', 'Remove': 'Disconnect'}  # r2 → canonical
_R1_HTTP_COLUMNS = ['id', 'date', 'user', 'pc', 'url']  # r1 http.csv has no header row
_REMOVABLE_MEDIA_TRUE = 'True'  # r4x: every file event is a removable-media copy

# ── CERT file/directory names (relative to data_dir) ─────────────────────────
_LOGON_CSV = 'logon.csv'
_DEVICE_CSV = 'device.csv'
_EMAIL_CSV = 'email.csv'
_HTTP_CSV = 'http.csv'
_FILE_CSV = 'file.csv'
_LDAP_DIR = 'LDAP'
_PSYCHOMETRIC_CSV = 'psychometric.csv'
_DECOY_CSV = 'decoy_file.csv'

# Email column-count threshold below which a file is the stripped r2 schema.
_R2_EMAIL_MAX_COLUMNS = 4


class CertAdapter(Protocol):
    """Protocol for version-specific CERT dataset adapters."""

    def load_logon(self, data_dir: Path) -> pd.DataFrame: ...
    def load_device(self, data_dir: Path) -> pd.DataFrame: ...
    def load_email(self, data_dir: Path) -> pd.DataFrame: ...
    def load_http(self, data_dir: Path) -> pd.DataFrame: ...
    def load_file(self, data_dir: Path) -> pd.DataFrame: ...
    def load_ldap(self, data_dir: Path) -> pd.DataFrame: ...
    def load_psychometric(self, data_dir: Path) -> pd.DataFrame: ...
    def load_decoy_files(self, data_dir: Path) -> pd.DataFrame: ...


def _empty_frame(schema: ColumnSchema) -> pd.DataFrame:
    """Empty DataFrame carrying a schema's required columns (for missing files)."""
    return pd.DataFrame(columns=schema.required_columns)


def _strip_domain_prefix(df: pd.DataFrame, column: str = 'user') -> pd.DataFrame:
    """Strip the 'DTAA/' domain prefix from a user-id column (r1 only)."""
    if not df.empty and column in df.columns:
        df[column] = df[column].astype(str).str.replace(_DTAA_PREFIX, '', regex=False)
    return df


class _BaseAdapter:
    """Shared implementation: the r4x 'standard' schema with headered CSVs.

    Subclasses override only the methods whose version diverges. Missing files
    always yield an empty schema-shaped frame — adapters never raise on absence.
    Loaders preserve any extra columns (e.g. r5x file_tree, r6x http activity),
    so versions that merely *add* columns need no override.
    """

    def _finalize(self, df: pd.DataFrame, schema: ColumnSchema, source: Path) -> pd.DataFrame:
        """Fill any absent required columns with None, then validate the contract."""
        for col in schema.required_columns:
            if col not in df.columns:
                df[col] = None
        validate_dataframe(df, schema, source=str(source))
        return df

    def load_logon(self, data_dir: Path) -> pd.DataFrame:
        path = data_dir / _LOGON_CSV
        if not path.exists():
            return _empty_frame(LOGON_SCHEMA)
        return self._finalize(load_csv(path), LOGON_SCHEMA, path)

    def load_device(self, data_dir: Path) -> pd.DataFrame:
        path = data_dir / _DEVICE_CSV
        if not path.exists():
            return _empty_frame(DEVICE_SCHEMA)
        return self._finalize(load_csv(path), DEVICE_SCHEMA, path)

    def load_email(self, data_dir: Path) -> pd.DataFrame:
        path = data_dir / _EMAIL_CSV
        if not path.exists():
            return _empty_frame(EMAIL_SCHEMA)
        return self._finalize(load_csv(path), EMAIL_SCHEMA, path)

    def load_http(self, data_dir: Path) -> pd.DataFrame:
        path = data_dir / _HTTP_CSV
        if not path.exists():
            return _empty_frame(HTTP_SCHEMA)
        return self._finalize(load_csv(path), HTTP_SCHEMA, path)

    def load_file(self, data_dir: Path) -> pd.DataFrame:
        path = data_dir / _FILE_CSV
        if not path.exists():
            return _empty_frame(FILE_SCHEMA)
        return self._finalize(load_csv(path), FILE_SCHEMA, path)

    def _load_file_implicit_removable(self, data_dir: Path) -> pd.DataFrame:
        """Load a file.csv where every row is an (implicit) removable-media copy.

        r3.x and r4.x file logs have no to_removable_media column, but DATASET_CONTEXT
        §5.6 states every row IS a copy to removable media. Tag them 'True' so the
        file_to_removable_count feature is computed correctly (and uniformly with r5+).
        """
        path = data_dir / _FILE_CSV
        if not path.exists():
            return _empty_frame(FILE_SCHEMA)
        df = load_csv(path)
        df['to_removable_media'] = _REMOVABLE_MEDIA_TRUE
        return self._finalize(df, FILE_SCHEMA, path)

    def load_ldap(self, data_dir: Path) -> pd.DataFrame:
        return load_ldap_directory(data_dir / _LDAP_DIR)

    def load_psychometric(self, data_dir: Path) -> pd.DataFrame:
        return load_psychometric(data_dir / _PSYCHOMETRIC_CSV)

    def load_decoy_files(self, data_dir: Path) -> pd.DataFrame:
        return load_decoy_files(data_dir / _DECOY_CSV)


class R1Adapter(_BaseAdapter):
    """r1: DTAA/-prefixed user ids, headerless http.csv, no email/file/psychometric."""

    def load_logon(self, data_dir: Path) -> pd.DataFrame:
        return _strip_domain_prefix(super().load_logon(data_dir))

    def load_device(self, data_dir: Path) -> pd.DataFrame:
        return _strip_domain_prefix(super().load_device(data_dir))

    def load_email(self, data_dir: Path) -> pd.DataFrame:
        return _empty_frame(EMAIL_SCHEMA)  # r1 has no email.csv

    def load_http(self, data_dir: Path) -> pd.DataFrame:
        path = data_dir / _HTTP_CSV
        if not path.exists():
            return _empty_frame(HTTP_SCHEMA)
        df = load_csv(path, columns=_R1_HTTP_COLUMNS, has_header=False)
        df = _strip_domain_prefix(df)
        df['content'] = None  # r1 http has no content field
        return self._finalize(df, HTTP_SCHEMA, path)


class R2Adapter(_BaseAdapter):
    """r2: device Insert/Remove → Connect/Disconnect; 4-column email (user filled later)."""

    def load_device(self, data_dir: Path) -> pd.DataFrame:
        path = data_dir / _DEVICE_CSV
        if not path.exists():
            return _empty_frame(DEVICE_SCHEMA)
        df = load_csv(path)
        if 'activity' in df.columns:
            df['activity'] = df['activity'].replace(_DEVICE_ACTIVITY_MAP)
        return self._finalize(df, DEVICE_SCHEMA, path)

    # load_email: r2 email is id,date,to,from only. _finalize fills the absent
    # columns (user,pc,cc,bcc,size,attachments,content) with None; the pipeline
    # later infers 'user' via the LDAP email→user_id mapping. Base behaviour suffices.


class R3xAdapter(_BaseAdapter):
    """r3.x: 7-column email (id,date,to,from,size,attachments,content), no user/pc/cc/bcc.

    Base load_email fills the absent user/pc/cc/bcc with None; 'user' is inferred
    downstream from the 'from' address via LDAP. Like r4.x, every file.csv row is
    an implicit removable-media copy (DATASET_CONTEXT §5.6).
    """

    def load_file(self, data_dir: Path) -> pd.DataFrame:
        return self._load_file_implicit_removable(data_dir)


class R4xAdapter(_BaseAdapter):
    """r4.x: primary training version. Every file.csv row is a removable-media copy."""

    def load_file(self, data_dir: Path) -> pd.DataFrame:
        return self._load_file_implicit_removable(data_dir)


class R5xAdapter(_BaseAdapter):
    """r5.x: device gains file_tree; email gains activity; file gains activity +
    to/from_removable_media; LDAP gains projects; decoy_file.csv exists.

    These are all *additive* columns that the loaders preserve verbatim, so the
    standard base behaviour already produces correct, contract-valid frames.
    """


class R6xAdapter(_BaseAdapter):
    """r6.x: http.csv is huge (≈85M rows) and gains an activity column — stream it."""

    def load_http(self, data_dir: Path) -> pd.DataFrame:
        path = data_dir / _HTTP_CSV
        if not path.exists():
            return _empty_frame(HTTP_SCHEMA)
        chunks = list(load_csv_chunked(path))
        df = pd.concat(chunks, ignore_index=True) if chunks else _empty_frame(HTTP_SCHEMA)
        return self._finalize(df, HTTP_SCHEMA, path)


_FAMILY_ADAPTERS: dict[str, type[_BaseAdapter]] = {
    'r1': R1Adapter,
    'r2': R2Adapter,
    'r3x': R3xAdapter,
    'r4x': R4xAdapter,
    'r5x': R5xAdapter,
    'r6x': R6xAdapter,
}


def get_adapter(version: str) -> CertAdapter:
    """Return the adapter instance for a CERT version or family string.

    Accepts either a concrete version (e.g. 'r4.2') or a family string
    (e.g. 'r4x', as returned by auto_detect_version), so the auto-detect →
    adapter chain works without an intermediate version lookup.

    Raises:
        ValueError: If the string is neither a known version nor a known family.
    """
    # A concrete version maps through VERSION_FAMILIES; a family is already a key.
    family = VERSION_FAMILIES.get(version, version if version in _FAMILY_ADAPTERS else None)
    if family is None:
        raise ValueError(
            f"Unknown CERT version: {version!r}. "
            f"Valid versions: {sorted(VERSION_FAMILIES.keys())}; "
            f"valid families: {sorted(_FAMILY_ADAPTERS.keys())}"
        )
    return _FAMILY_ADAPTERS[family]()


def _read_header(path: Path) -> list[str]:
    """Read just the first line of a CSV and return its comma-split, lowercased tokens."""
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        first_line = fh.readline().strip()
    return [token.strip().strip('"').lower() for token in first_line.split(',')]


def auto_detect_version(data_dir: Path) -> str:
    """Fingerprint CSV headers to identify the CERT dataset version family.

    Decision tree (reads only header lines, never whole files):
    - http.csv has no header (no 'url'/'user' tokens) → r1
    - email.csv has ≤4 columns                        → r2
    - email.csv has 'size' but no 'user'              → r3x
    - email.csv has 'user' but no 'activity'          → r4x
    - http.csv header has 'activity'                  → r6x
    - otherwise                                        → r5x

    Returns:
        The version family string (e.g. 'r4x').

    Raises:
        FileNotFoundError: If data_dir does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. Cannot auto-detect CERT version."
        )

    http_path = data_dir / _HTTP_CSV
    email_path = data_dir / _EMAIL_CSV

    if http_path.exists():
        http_header = _read_header(http_path)
        if 'url' not in http_header and 'user' not in http_header:
            return 'r1'  # headerless http.csv is unique to r1

    if email_path.exists():
        email_header = _read_header(email_path)
        if len(email_header) <= _R2_EMAIL_MAX_COLUMNS:
            return 'r2'
        if 'size' in email_header and 'user' not in email_header:
            return 'r3x'
        if 'user' in email_header and 'activity' not in email_header:
            return 'r4x'

    if http_path.exists() and 'activity' in _read_header(http_path):
        return 'r6x'

    return 'r5x'
