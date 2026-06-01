"""Parser for CERT insider threat answer files.

The answers/ directory contains the ground truth: which users are
insiders, which scenario they belong to, and the exact timestamps
of their attack windows.

Public API:
    InsiderRecord       — frozen dataclass for one insider's metadata
    load_insiders(answers_dir, version) — parse insiders.csv into list of InsiderRecord
    get_attack_windows(answers_dir, version) — dict mapping user_id → InsiderRecord
    get_malicious_dates(answers_dir, version) — set of (user, date) tuples (backward compat)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# insiders.csv 'dataset' values look like "4.2"; version/family inputs look like
# "r4.2" or "r4x". We match on the leading major-version digit because
# auto-detection yields only a family (e.g. 'r4x') and cannot distinguish r4.1
# from r4.2 (their schemas are identical). Each version's answers/ dir contains
# only that version's rows, so major-version matching is unambiguous in practice.
_MAJOR_VERSION_RE = re.compile(r'r?(\d+)', re.IGNORECASE)


def _major_version(version: str) -> str:
    """Extract the major-version digit from a version or family string.

    'r4.2' -> '4', 'r4x' -> '4', '5.2' -> '5'.

    Raises:
        ValueError: If no leading version number can be found.
    """
    match = _MAJOR_VERSION_RE.match(version.strip())
    if match is None:
        raise ValueError(
            f"Cannot extract a major version from {version!r}; "
            "expected a string like 'r4.2', 'r4x', or '5.2'."
        )
    return match.group(1)


def _row_major_version(value: object) -> str | None:
    """Major-version digit of an insiders.csv 'dataset' cell, or None if absent."""
    match = _MAJOR_VERSION_RE.match(str(value).strip())
    return match.group(1) if match is not None else None


@dataclass(frozen=True)
class InsiderRecord:
    """Metadata for a single insider threat actor.

    Attributes:
        user_id: The insider's user ID (e.g., 'AAM0658').
        scenario: Attack scenario number (1-5).
        dataset: CERT dataset version (e.g., '4.2').
        attack_start: Timestamp when the attack begins.
        attack_end: Timestamp when the attack ends.
        details_file: Filename of the per-insider observable events CSV.
    """
    user_id: str
    scenario: int
    dataset: str
    attack_start: pd.Timestamp
    attack_end: pd.Timestamp
    details_file: str


def load_insiders(answers_dir: Path, version: str) -> list[InsiderRecord]:
    """Parse insiders.csv into structured InsiderRecord objects.

    Args:
        answers_dir: Path to the answers/ directory.
        version: CERT version string (e.g., 'r4.2') — used to filter
                 insiders.csv rows to this version only.

    Returns:
        List of InsiderRecord, one per insider in this version.
        Empty list if insiders.csv not found.

    Raises:
        ValueError: If insiders.csv exists but has wrong columns.
    """
    # The insiders.csv master file. version_num is e.g. "4.2" from "r4.2"
    insiders_path = answers_dir / 'insiders.csv'
    if not insiders_path.exists():
        logger.warning("load_insiders | insiders.csv not found in %s", answers_dir)
        return []

    df = pd.read_csv(insiders_path)
    required = {'dataset', 'scenario', 'details', 'user', 'start', 'end'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"insiders.csv missing columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )

    # Filter to the requested version by MAJOR version, so both concrete
    # versions ('r4.2') and auto-detected families ('r4x') select the right rows.
    target_major = _major_version(version)
    df['dataset'] = df['dataset'].astype(str).str.strip()
    version_rows = df[df['dataset'].map(_row_major_version) == target_major]

    records = []
    for _, row in version_rows.iterrows():
        records.append(InsiderRecord(
            user_id=str(row['user']).strip(),
            scenario=int(row['scenario']),
            dataset=str(row['dataset']).strip(),
            attack_start=pd.to_datetime(row['start']),
            attack_end=pd.to_datetime(row['end']),
            details_file=str(row['details']).strip(),
        ))

    logger.info(
        "load_insiders | version %s: %d insiders across %d scenarios",
        version, len(records), len({r.scenario for r in records}),
    )
    return records


def get_attack_windows(
    answers_dir: Path,
    version: str,
) -> dict[str, InsiderRecord]:
    """Get attack windows keyed by user_id.

    This is the primary interface for the overlap-ratio labeling in Phase 2:
    given a 28-day window, check what fraction overlaps with the attack window.

    Args:
        answers_dir: Path to the answers/ directory.
        version: CERT version string.

    Returns:
        Dict mapping user_id → InsiderRecord.
        Users not in this dict are not insiders.
    """
    records = load_insiders(answers_dir, version)
    windows = {r.user_id: r for r in records}

    # Check for duplicates (a user can only be one insider in one version)
    if len(windows) != len(records):
        dupes = [r.user_id for r in records]
        dupes = [u for u in dupes if dupes.count(u) > 1]
        logger.warning(
            "get_attack_windows | duplicate user_ids in insiders: %s",
            sorted(set(dupes)),
        )

    return windows


def get_malicious_dates(
    answers_dir: Path,
    version: str,
) -> set[tuple[str, object]]:
    """Backward-compatible interface: set of (user_id, date) tuples.

    Each date within the attack window (inclusive) gets an entry.
    This matches the old pipeline.load_labels() return format.

    Args:
        answers_dir: Path to the answers/ directory.
        version: CERT version string.

    Returns:
        Set of (user_id, datetime.date) tuples for all malicious days.
    """
    records = load_insiders(answers_dir, version)
    malicious: set[tuple[str, object]] = set()

    for r in records:
        # Generate all dates in the attack window
        dates = pd.date_range(r.attack_start.normalize(), r.attack_end.normalize(), freq='D')
        for d in dates:
            malicious.add((r.user_id, d.date()))

    logger.info("get_malicious_dates | %d malicious (user, date) pairs", len(malicious))
    return malicious
