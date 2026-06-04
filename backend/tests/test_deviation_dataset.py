"""Unit tests for the sliding-window deviation dataset (Module 1, Phase 2).

Covers innersight.models.dataset.DeviationWindowDataset, its overlap-ratio
labelling, and the custom DataLoader collation.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from innersight.data.answers import InsiderRecord
from innersight.models.dataset import DeviationWindowDataset, _window_collate
from innersight.schema import FEATURE_NAMES

N_FEAT = len(FEATURE_NAMES)
WINDOW_SIZE = 28
N_DAYS = 90
ATTACK_START_IDX = 50
ATTACK_END_IDX = 65


@pytest.fixture()
def synthetic():
    """3 users (2 normal, 1 insider), 90 days, attack on days 50–65, 18 features."""
    dates = pd.date_range("2010-01-01", periods=N_DAYS, freq="D")
    rng = np.random.default_rng(42)
    users = ["normal_a", "normal_b", "insider"]

    frames = []
    for user in users:
        mat = rng.normal(0.0, 1.0, size=(N_DAYS, N_FEAT)).astype(np.float32)
        f = pd.DataFrame(mat, columns=FEATURE_NAMES)
        f.insert(0, "date", dates)
        f.insert(0, "user", user)
        frames.append(f)
    deviations_df = pd.concat(frames, ignore_index=True)

    record = InsiderRecord(
        user_id="insider",
        scenario=2,
        dataset="4.2",
        attack_start=dates[ATTACK_START_IDX],
        attack_end=dates[ATTACK_END_IDX],
        details_file="r4.2-2-insider.csv",
    )
    attack_windows = {"insider": record}
    return deviations_df, attack_windows, dates


def _build(synthetic, **kwargs):
    deviations_df, attack_windows, _ = synthetic
    params = dict(window_size=WINDOW_SIZE, stride=7, overlap_threshold=0.5)
    params.update(kwargs)
    return DeviationWindowDataset(deviations_df, attack_windows, **params)


# ── Shape & content ──────────────────────────────────────────────────────────

def test_window_tensor_shape(synthetic):
    ds = _build(synthetic)
    assert len(ds) > 0
    x, y, meta = ds[0]
    assert tuple(x.shape) == (N_FEAT, WINDOW_SIZE)
    assert tuple(y.shape) == (1,)


def test_metadata_contains_user_id(synthetic):
    ds = _build(synthetic)
    for i in range(len(ds)):
        _, _, meta = ds[i]
        assert "user_id" in meta
        assert meta["user_id"] in {"normal_a", "normal_b", "insider"}


def test_no_nan_in_any_window(synthetic):
    ds = _build(synthetic)
    for i in range(len(ds)):
        x, _, _ = ds[i]
        assert not torch.isnan(x).any()


# ── Labelling by attack overlap ──────────────────────────────────────────────

def test_window_fully_inside_attack_is_positive(synthetic):
    # A window whose dates lie entirely within the attack has overlap 1.0 → label 1.
    _, _, dates = synthetic
    inside = dates[ATTACK_START_IDX:ATTACK_START_IDX + WINDOW_SIZE].to_numpy()
    overlap = DeviationWindowDataset._compute_overlap(
        inside, dates[ATTACK_START_IDX], dates[ATTACK_START_IDX + WINDOW_SIZE - 1]
    )
    assert overlap == 1.0

    # And the dataset assigns label 1 to its high-overlap (≥ threshold) windows.
    ds = _build(synthetic)
    insider_labels = [int(ds[i][1].item()) for i in range(len(ds)) if ds[i][2]["user_id"] == "insider"]
    assert 1 in insider_labels


def test_window_no_overlap_is_negative(synthetic):
    ds = _build(synthetic)
    insider = [(int(ds[i][1].item()), ds[i][2]) for i in range(len(ds)) if ds[i][2]["user_id"] == "insider"]
    zero_overlap = [label for label, meta in insider if meta["overlap_ratio"] == 0.0]
    assert zero_overlap  # at least one
    assert all(label == 0 for label in zero_overlap)


def test_partial_overlap_below_threshold_excluded(synthetic):
    # No retained window may have an ambiguous (0, 0.5) overlap ratio.
    ds = _build(synthetic)
    for i in range(len(ds)):
        _, _, meta = ds[i]
        r = meta["overlap_ratio"]
        assert r == 0.0 or r >= 0.5


def test_normal_users_all_negative(synthetic):
    ds = _build(synthetic)
    for i in range(len(ds)):
        _, y, meta = ds[i]
        if meta["user_id"] != "insider":
            assert int(y.item()) == 0


# ── DataLoader collation ─────────────────────────────────────────────────────

def test_dataloader_collation_batch_shapes(synthetic):
    ds = _build(synthetic)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=_window_collate)
    windows, labels, metas = next(iter(loader))
    b = windows.shape[0]
    assert tuple(windows.shape) == (b, N_FEAT, WINDOW_SIZE)
    assert tuple(labels.shape) == (b, 1)
    assert isinstance(metas, list)
    assert len(metas) == b
