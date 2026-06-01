"""Low-level CSV parsers for individual CERT data files.

Each loader reads a single CSV file (or directory for LDAP), parses dates,
and returns a pandas DataFrame. Loaders do NOT handle version-specific
schema normalization — that is the adapter's responsibility. Loaders
receive column names and dtype overrides from the adapter layer.

Public API:
    load_csv(path, columns, date_col, **kwargs) — generic CSV reader with date parsing
    load_csv_chunked(path, columns, date_col, chunk_size) — streaming reader for large files
    load_ldap_directory(ldap_dir) — read all monthly LDAP snapshots, return latest
    load_psychometric(path) — read OCEAN personality scores
    load_decoy_files(path) — read decoy file registry (r5+ only)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from innersight.backend.config import CSV_CHUNK_SIZE
from innersight.backend.schema import (
    LDAP_SCHEMA,
    PSYCHOMETRIC_SCHEMA,
    validate_dataframe,
)

logger = logging.getLogger(__name__)


def load_csv(
    path: Path,
    *,
    columns: list[str] | None = None,
    date_col: str = 'date',
    has_header: bool = True,
    dtype_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read a single CERT CSV file with date parsing.

    Args:
        path: Path to the CSV file.
        columns: Column names to assign (required if has_header=False).
        date_col: Name of the date column to parse as datetime.
        has_header: Whether the CSV has a header row.
        dtype_overrides: Optional pandas dtype overrides for specific columns.

    Returns:
        DataFrame with parsed dates, sorted by date.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If has_header=False and columns is None.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if not has_header and columns is None:
        raise ValueError(
            f"columns must be provided when has_header=False (file: {path})"
        )

    read_kwargs: dict = {
        'dtype': dtype_overrides or {},
        'low_memory': False,
    }
    if has_header:
        read_kwargs['header'] = 0
    else:
        read_kwargs['header'] = None
        read_kwargs['names'] = columns

    df = pd.read_csv(path, **read_kwargs)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=False)
        df = df.sort_values(date_col).reset_index(drop=True)

    logger.info("load_csv | %s: %d rows, %d cols", path.name, len(df), len(df.columns))
    return df


def load_csv_chunked(
    path: Path,
    *,
    columns: list[str] | None = None,
    date_col: str = 'date',
    has_header: bool = True,
    chunk_size: int = CSV_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Streaming CSV reader for very large files (e.g., r6.2 http.csv at 85M rows).

    Yields DataFrames of chunk_size rows with dates parsed.

    Args:
        path: Path to the CSV file.
        columns: Column names (required if has_header=False).
        date_col: Date column to parse.
        has_header: Whether the CSV has a header row.
        chunk_size: Rows per chunk.

    Yields:
        DataFrame chunks with parsed dates.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    read_kwargs: dict = {'chunksize': chunk_size, 'low_memory': False}
    if has_header:
        read_kwargs['header'] = 0
    else:
        if columns is None:
            raise ValueError(f"columns required when has_header=False (file: {path})")
        read_kwargs['header'] = None
        read_kwargs['names'] = columns

    chunk_num = 0
    for chunk in pd.read_csv(path, **read_kwargs):
        if date_col in chunk.columns:
            chunk[date_col] = pd.to_datetime(chunk[date_col], format='mixed', dayfirst=False)
        chunk_num += 1
        yield chunk

    logger.info("load_csv_chunked | %s: %d chunks of %d rows", path.name, chunk_num, chunk_size)


def load_ldap_directory(ldap_dir: Path) -> pd.DataFrame:
    """Read LDAP monthly snapshots and return the latest one.

    CERT LDAP directories contain files named YYYY-MM.csv (e.g., 2010-01.csv).
    We load the latest snapshot by filename sort order.

    Args:
        ldap_dir: Path to the LDAP/ directory.

    Returns:
        DataFrame from the latest monthly snapshot.
        Empty DataFrame with LDAP_SCHEMA columns if directory missing or empty.
    """
    if not ldap_dir.exists() or not ldap_dir.is_dir():
        logger.warning("load_ldap_directory | LDAP directory not found: %s", ldap_dir)
        return pd.DataFrame(columns=LDAP_SCHEMA.required_columns)

    csv_files = sorted(ldap_dir.glob('*.csv'))
    if not csv_files:
        logger.warning("load_ldap_directory | No CSV files in %s", ldap_dir)
        return pd.DataFrame(columns=LDAP_SCHEMA.required_columns)

    latest = csv_files[-1]  # lexicographic sort on YYYY-MM = chronological
    df = pd.read_csv(latest)

    # Normalize column names to lowercase (r1 has Domain, Email, Role)
    df.columns = [c.lower() for c in df.columns]

    # Rename r1's 'domain' column (drop it — always 'dtaa.com', not useful)
    if 'domain' in df.columns:
        df = df.drop(columns=['domain'])

    logger.info(
        "load_ldap_directory | loaded %s: %d employees from %s",
        latest.name, len(df), ldap_dir,
    )
    return df


def load_psychometric(path: Path) -> pd.DataFrame:
    """Read OCEAN personality scores.

    Schema is identical across all versions that have it (r2+).
    r1 has NO psychometric file.

    Args:
        path: Path to psychometric.csv.

    Returns:
        DataFrame with columns [employee_name, user_id, O, C, E, A, N].
        Empty DataFrame if file doesn't exist.
    """
    if not path.exists():
        logger.warning("load_psychometric | file not found: %s", path)
        return pd.DataFrame(columns=PSYCHOMETRIC_SCHEMA.required_columns)

    df = pd.read_csv(path)
    validate_dataframe(df, PSYCHOMETRIC_SCHEMA, source=str(path))
    logger.info("load_psychometric | %d users loaded from %s", len(df), path.name)
    return df


def load_decoy_files(path: Path) -> pd.DataFrame:
    """Read decoy file registry (r5+ only).

    Schema: decoy_filename, pc
    r6 may have quoted fields.

    Args:
        path: Path to decoy_file.csv.

    Returns:
        DataFrame with columns [decoy_filename, pc].
        Empty DataFrame if file doesn't exist.
    """
    if not path.exists():
        logger.warning("load_decoy_files | file not found: %s", path)
        return pd.DataFrame(columns=['decoy_filename', 'pc'])

    df = pd.read_csv(path)
    # Strip quotes that r6 adds
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip('"')

    logger.info("load_decoy_files | %d decoy files from %s", len(df), path.name)
    return df
