"""Parquet-based feature persistence for InnerSight UEBA.

Caches computed features and deviation matrices to disk so downstream
modules never recompute. Supports cache invalidation based on source
data modification time.

Public API:
    FeatureStore(store_dir)     — manager for a Parquet-based feature cache
      .save_features(version, df)     — persist daily features
      .load_features(version)         — load cached features (or None)
      .save_deviations(version, df)   — persist z-scored deviation matrices (Phase 2)
      .load_deviations(version)       — load cached deviations (or None)
      .save_metadata(version, meta)   — persist run metadata (provenance)
      .load_metadata(version)         — load run metadata
      .is_stale(version, data_dir)    — check if cache is older than source data
      .clear(version)                 — remove all cached data for a version
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = Path('feature_store')


class FeatureStore:
    """Manages Parquet-based feature caching.

    Directory structure:
        store_dir/
            r4.2/
                features.parquet      — daily (user, date, 18 features, label)
                deviations.parquet    — z-scored deviation matrices (Phase 2)
                metadata.json         — provenance: version, timestamp, row counts, etc.
            r5.2/
                ...
    """

    def __init__(self, store_dir: Path | str = DEFAULT_STORE_DIR) -> None:
        self._store_dir = Path(store_dir)

    def _version_dir(self, version: str) -> Path:
        """Get the cache directory for a specific version."""
        return self._store_dir / version

    def save_features(self, version: str, df: pd.DataFrame) -> Path:
        """Persist daily features as Parquet.

        Args:
            version: CERT version string.
            df: DataFrame from build_user_day_features().

        Returns:
            Path to the written Parquet file.
        """
        vdir = self._version_dir(version)
        vdir.mkdir(parents=True, exist_ok=True)
        path = vdir / 'features.parquet'
        df.to_parquet(path, index=False, engine='pyarrow')
        logger.info("save_features | %s: %d rows → %s", version, len(df), path)
        return path

    def load_features(self, version: str) -> pd.DataFrame | None:
        """Load cached features.

        Returns:
            DataFrame if cache exists, None otherwise.
        """
        path = self._version_dir(version) / 'features.parquet'
        if not path.exists():
            logger.debug("load_features | no cache for %s", version)
            return None
        df = pd.read_parquet(path, engine='pyarrow')
        logger.info("load_features | %s: %d rows from cache", version, len(df))
        return df

    def save_deviations(self, version: str, df: pd.DataFrame) -> Path:
        """Persist z-scored deviation matrices as Parquet (Phase 2)."""
        vdir = self._version_dir(version)
        vdir.mkdir(parents=True, exist_ok=True)
        path = vdir / 'deviations.parquet'
        df.to_parquet(path, index=False, engine='pyarrow')
        logger.info("save_deviations | %s: %d rows → %s", version, len(df), path)
        return path

    def load_deviations(self, version: str) -> pd.DataFrame | None:
        """Load cached deviations. Returns None if not cached."""
        path = self._version_dir(version) / 'deviations.parquet'
        if not path.exists():
            return None
        df = pd.read_parquet(path, engine='pyarrow')
        logger.info("load_deviations | %s: %d rows from cache", version, len(df))
        return df

    def save_metadata(self, version: str, meta: dict) -> Path:
        """Persist run metadata (provenance) as JSON."""
        vdir = self._version_dir(version)
        vdir.mkdir(parents=True, exist_ok=True)
        path = vdir / 'metadata.json'
        # Add timestamp
        meta['cached_at'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(meta, indent=2, default=str))
        logger.info("save_metadata | %s → %s", version, path)
        return path

    def load_metadata(self, version: str) -> dict | None:
        """Load run metadata. Returns None if not found."""
        path = self._version_dir(version) / 'metadata.json'
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def is_stale(self, version: str, data_dir: Path) -> bool:
        """Check if cached features are older than source data.

        Compares the cache write time against the most recent modification
        time of any CSV in data_dir. Returns True if cache should be
        recomputed, or if no cache exists.

        Args:
            version: CERT version string.
            data_dir: Path to the source data directory.

        Returns:
            True if cache is stale or missing, False if fresh.
        """
        meta = self.load_metadata(version)
        if meta is None:
            return True

        cached_at_str = meta.get('cached_at')
        if cached_at_str is None:
            return True

        cached_at = datetime.fromisoformat(cached_at_str)

        # Find most recent CSV modification in data_dir
        csv_files = list(data_dir.glob('*.csv'))
        if not csv_files:
            return True

        latest_source = max(f.stat().st_mtime for f in csv_files)
        latest_source_dt = datetime.fromtimestamp(latest_source, tz=timezone.utc)

        is_stale = latest_source_dt > cached_at
        if is_stale:
            logger.info(
                "is_stale | %s: source modified %s > cache %s",
                version, latest_source_dt.isoformat(), cached_at_str,
            )
        return is_stale

    def clear(self, version: str) -> None:
        """Remove all cached data for a version."""
        vdir = self._version_dir(version)
        if vdir.exists():
            shutil.rmtree(vdir)
            logger.info("clear | removed cache for %s", version)
