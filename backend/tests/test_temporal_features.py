"""Unit tests for innersight.features.temporal_features (Phase 3)."""

import numpy as np
import pandas as pd

from innersight.data.answers import InsiderRecord
from innersight.features.temporal_features import (
    N_WINDOW_FEATURES,
    extract_all_window_features,
    extract_window_features,
    get_feature_names,
)
from innersight.models.dataset import DeviationWindowDataset
from innersight.schema import FEATURE_NAMES

N_FEAT = len(FEATURE_NAMES)


def test_extract_window_features_shape():
    rng = np.random.default_rng(0)
    fv = extract_window_features(rng.normal(0, 1, size=(18, 28)))
    assert fv.shape == (N_WINDOW_FEATURES,)
    assert fv.shape == (129,)


def test_all_zeros_window_gives_zero_features():
    fv = extract_window_features(np.zeros((18, 28)))
    assert np.allclose(fv, 0.0)


def test_spike_on_one_feature_yields_large_max_deviation():
    window = np.zeros((18, 28))
    window[3, 10] = 9.0  # spike on feature index 3
    fv = extract_window_features(window)
    # Per-feature block layout: feature i's max_deviation is at index i*7 (max is stat 0).
    max_dev_feature3 = fv[3 * 7 + 0]
    assert max_dev_feature3 == 9.0
    # A feature with no activity has zero max_deviation.
    assert fv[0 * 7 + 0] == 0.0


def test_feature_names_count_matches_vector_length():
    names = get_feature_names()
    fv = extract_window_features(np.random.default_rng(1).normal(size=(18, 28)))
    assert len(names) == fv.shape[0] == N_WINDOW_FEATURES


def test_extract_all_window_features_lengths_match():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2010-01-01", periods=90, freq="D")
    frames = []
    for user in ("a", "b", "ins"):
        mat = rng.normal(0, 1, size=(90, N_FEAT))
        f = pd.DataFrame(mat, columns=FEATURE_NAMES)
        f.insert(0, "date", dates)
        f.insert(0, "user", user)
        frames.append(f)
    deviations = pd.concat(frames, ignore_index=True)
    record = InsiderRecord("ins", 2, "4.2", dates[50], dates[65], "f.csv")
    dataset = DeviationWindowDataset(deviations, {"ins": record}, window_size=28, stride=7)

    X, y, metas = extract_all_window_features(dataset)
    assert X.shape == (len(dataset), N_WINDOW_FEATURES)
    assert y.shape == (len(dataset),)
    assert len(metas) == len(dataset) == X.shape[0]
