"""Universal data loading pipeline for all CERT dataset versions.

Orchestrates adapters, loaders, and answer parsing to produce canonical
DataFrames from any CERT version (r1 through r6.2). Version-specific
schema differences are handled transparently by the adapter layer.

Public API (new — use these going forward):
    load_version(data_dir, version)   — load all data for a specific version
    CertDataset                       — dataclass holding all loaded data for a version

Public API (backward-compatible wrappers — existing callers still work):
    load_raw_logs(data_dir)           — delegates to load_version with auto-detect
    load_labels(answers_dir)          — delegates to answers.get_malicious_dates
    time_split(logs_dict, ...)        — unchanged (pure function, version-agnostic)
    load_data(data_dir)               — delegates to load_version + time_split
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from innersight.config import DATA_DIR, TRAIN_END_DATE, VAL_END_DATE
from innersight.data.adapters import auto_detect_version, get_adapter
from innersight.data.answers import (
    InsiderRecord,
    get_malicious_dates,
    load_insiders,
)

logger = logging.getLogger(__name__)

# Only the first 64KB of each CSV is hashed — a fast fingerprint that detects
# header/schema changes and most edits without reading multi-GB files in full.
_HASH_PREFIX_BYTES = 64 * 1024


def _hash_file_prefix(path: Path, num_bytes: int = _HASH_PREFIX_BYTES) -> str:
    """Return the SHA-256 hex digest of the first num_bytes of a file."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        digest.update(handle.read(num_bytes))
    return digest.hexdigest()


def _compute_file_hashes(data_dir: Path) -> dict[str, str]:
    """Fingerprint every top-level CSV in data_dir (filename → 64KB SHA-256)."""
    hashes: dict[str, str] = {}
    for csv_path in sorted(data_dir.glob('*.csv')):
        try:
            hashes[csv_path.name] = _hash_file_prefix(csv_path)
        except OSError as exc:
            logger.warning("_compute_file_hashes | could not hash %s: %s", csv_path, exc)
    return hashes


@dataclass
class CertDataset:
    """All loaded data for a single CERT dataset version.

    Attributes:
        version: The CERT version string (e.g., 'r4.2').
        data_dir: Path to the dataset directory.
        logs: Dict mapping log type name → canonical DataFrame.
        insiders: List of InsiderRecord objects.
        attack_windows: Dict mapping user_id → InsiderRecord.
        ldap: LDAP employee directory DataFrame.
        psychometric: OCEAN personality scores DataFrame.
        decoy_files: Decoy file registry DataFrame (r5+ only).
        provenance: Manifest of this load (version, timestamps, row counts,
            per-CSV content hashes) for reproducibility and cache validation.
    """
    version: str
    data_dir: Path
    logs: dict[str, pd.DataFrame] = field(default_factory=dict)
    insiders: list[InsiderRecord] = field(default_factory=list)
    attack_windows: dict[str, InsiderRecord] = field(default_factory=dict)
    ldap: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    psychometric: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    decoy_files: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    provenance: dict[str, Any] = field(default_factory=dict)


def load_version(data_dir: str | Path, version: str | None = None) -> CertDataset:
    """Load all data for a CERT dataset version.

    This is the primary entry point for the universal pipeline.

    Args:
        data_dir: Path to the dataset directory (e.g., /data/cert_r4.2/).
        version: CERT version string (e.g., 'r4.2'). If None, auto-detected.

    Returns:
        CertDataset with all logs, labels, LDAP, psychometric loaded.

    Raises:
        FileNotFoundError: If data_dir doesn't exist.
        ValueError: If version can't be detected or is unknown.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}. "
            "Set INNERSIGHT_DATA_DIR environment variable."
        )

    if version is None:
        version = auto_detect_version(data_path)
        logger.info("load_version | auto-detected version: %s", version)

    adapter = get_adapter(version)

    # Load all log types through the adapter
    logs = {
        'logon':  adapter.load_logon(data_path),
        'device': adapter.load_device(data_path),
        'email':  adapter.load_email(data_path),
        'http':   adapter.load_http(data_path),
        'file':   adapter.load_file(data_path),
    }

    # Load supplementary data
    ldap = adapter.load_ldap(data_path)
    psychometric = adapter.load_psychometric(data_path)
    decoy_files = adapter.load_decoy_files(data_path)

    # Load answer labels
    answers_dir = data_path / 'answers'
    insiders = load_insiders(answers_dir, version) if answers_dir.exists() else []
    attack_windows = {r.user_id: r for r in insiders}

    row_counts = {name: len(df) for name, df in logs.items()}
    logger.info(
        "load_version | %s loaded: %s, %d insiders, %d LDAP entries",
        version, row_counts, len(insiders), len(ldap),
    )

    provenance: dict[str, Any] = {
        'version': version,
        'adapter': type(adapter).__name__,
        'data_dir': str(data_path),
        'loaded_at': datetime.now(timezone.utc).isoformat(),
        'row_counts': row_counts,
        'insider_count': len(insiders),
        'ldap_count': len(ldap),
        'psychometric_count': len(psychometric),
        'file_hashes': _compute_file_hashes(data_path),
    }

    return CertDataset(
        version=version,
        data_dir=data_path,
        logs=logs,
        insiders=insiders,
        attack_windows=attack_windows,
        ldap=ldap,
        psychometric=psychometric,
        decoy_files=decoy_files,
        provenance=provenance,
    )


# ── Backward-compatible wrappers ─────────────────────────────────────────────
# These match the OLD pipeline.py signatures so existing callers don't break.

def load_raw_logs(data_dir: str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Backward-compatible wrapper. Loads logs using the universal pipeline."""
    if not data_dir:
        raise FileNotFoundError(
            "Data directory not set. Set INNERSIGHT_DATA_DIR environment variable."
        )
    data_path = Path(data_dir)
    try:
        version = auto_detect_version(data_path)
    except (FileNotFoundError, ValueError):
        version = 'r4.2'  # fallback to r4.2 for backward compat
        logger.warning("load_raw_logs | could not detect version, defaulting to %s", version)

    dataset = load_version(data_path, version)
    return dataset.logs


def load_labels(answers_dir: str) -> set[tuple[str, Any]]:
    """Backward-compatible wrapper. Returns set of (user, date) tuples."""
    answers_path = Path(answers_dir)
    if not answers_path.exists():
        logger.warning("load_labels | answers directory not found: %s", answers_path)
        return set()
    # Try to detect version from parent directory
    try:
        version = auto_detect_version(answers_path.parent)
    except (FileNotFoundError, ValueError):
        version = 'r4.2'
    return get_malicious_dates(answers_path, version)


def time_split(
    logs_dict: dict[str, pd.DataFrame],
    train_end: str,
    val_end: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Temporal train/val/test split. Unchanged from original — version-agnostic."""
    t_train = pd.Timestamp(train_end)
    t_val = pd.Timestamp(val_end)

    splits: dict[str, dict[str, pd.DataFrame]] = {'train': {}, 'val': {}, 'test': {}}
    for name, df in logs_dict.items():
        if 'date' not in df.columns:
            if not df.empty:
                raise ValueError(
                    f"DataFrame '{name}' missing 'date' column for time_split. "
                    f"Found: {sorted(df.columns.tolist())}"
                )
            empty = pd.DataFrame(columns=df.columns)
            splits['train'][name] = splits['val'][name] = splits['test'][name] = empty
            continue
        splits['train'][name] = df[df['date'] <= t_train].reset_index(drop=True)
        splits['val'][name] = df[(df['date'] > t_train) & (df['date'] <= t_val)].reset_index(drop=True)
        splits['test'][name] = df[df['date'] > t_val].reset_index(drop=True)
    return splits


def load_data(data_dir: str = DATA_DIR) -> dict[str, Any]:
    """Backward-compatible wrapper matching original load_data() signature."""
    if not data_dir:
        raise FileNotFoundError(
            "Data directory not set. Set INNERSIGHT_DATA_DIR environment variable."
        )
    data_path = Path(data_dir)
    try:
        version = auto_detect_version(data_path)
    except (FileNotFoundError, ValueError):
        version = 'r4.2'

    dataset = load_version(data_path, version)
    labels = get_malicious_dates(data_path / 'answers', version)
    splits = time_split(dataset.logs, train_end=TRAIN_END_DATE, val_end=VAL_END_DATE)
    return {'splits': splits, 'labels': labels}
