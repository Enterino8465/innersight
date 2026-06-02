"""Unit tests for backend/data/feature_store.py Parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from innersight.backend.data.feature_store import FeatureStore


def test_save_load_features_roundtrip(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    df = pd.DataFrame(
        {'user': ['u1', 'u2'], 'date': ['2010-01-01', '2010-01-02'], 'val': [1, 2]}
    )
    store.save_features('r4.2', df)
    loaded = store.load_features('r4.2')
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df)


def test_load_features_returns_none_when_absent(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    assert store.load_features('r4.2') is None


def test_metadata_roundtrip(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    store.save_metadata('r4.2', {'rows': 10})
    meta = store.load_metadata('r4.2')
    assert meta is not None
    assert meta['rows'] == 10
    assert 'cached_at' in meta  # timestamp injected on save


def test_is_stale_true_when_no_cache(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    assert store.is_stale('r4.2', tmp_path) is True


def test_clear_removes_cache(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    store.save_features('r4.2', pd.DataFrame({'a': [1]}))
    store.clear('r4.2')
    assert store.load_features('r4.2') is None
