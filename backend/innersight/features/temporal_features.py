"""Handcrafted temporal features from 28-day deviation windows.

Classical classifiers (XGBoost, random forest) can't consume a raw ``(18, 28)``
window the way a Conv1d can, so this module flattens each window into a fixed
129-dim feature vector of interpretable per-feature statistics (magnitude,
spread, escalation slope, burst energy) plus a few cross-feature signals.

Layout (matches :func:`get_feature_names`):
    * 7 stats × 18 features = 126 per-feature values, feature-major order, then
    * 3 cross-feature values.
"""

from __future__ import annotations

import numpy as np

from innersight.schema import FEATURE_NAMES

# Per-feature statistics, in the order they appear in the flat vector.
STAT_NAMES = (
    "max_deviation",
    "mean_deviation",
    "std_deviation",
    "days_above_2sigma",
    "days_above_3sigma",
    "slope",
    "max_burst",
)

# Cross-feature statistics appended after the per-feature block.
CROSS_FEATURE_NAMES = (
    "total_anomalous_days",
    "max_feature_count_per_day",
    "mean_total_deviation",
)

N_FEATURES = len(FEATURE_NAMES)            # 18
N_WINDOW_FEATURES = N_FEATURES * len(STAT_NAMES) + len(CROSS_FEATURE_NAMES)  # 129

_SIGMA_2 = 2.0
_SIGMA_3 = 3.0


def _max_burst(abs_row: np.ndarray) -> float:
    """Largest summed magnitude over a run of consecutive days above 2σ."""
    best = 0.0
    current = 0.0
    for value in abs_row:
        if value > _SIGMA_2:
            current += value
            best = max(best, current)
        else:
            current = 0.0
    return best


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """Flatten one ``(18, 28)`` deviation window into a 129-dim feature vector.

    Args:
        window: Channels-first window of shape ``(n_features, n_days)`` — one
            sample from :class:`~innersight.models.dataset.DeviationWindowDataset`.

    Returns:
        1-D float array of length 129: 7 statistics for each of the 18 features
        (feature-major order) followed by 3 cross-feature statistics.
    """
    window = np.asarray(window, dtype=float)
    if window.ndim != 2:
        raise ValueError(f"window must be 2-D (n_features, n_days), got shape {window.shape}")
    n_features, n_days = window.shape
    abs_window = np.abs(window)            # shape: (n_features, n_days)
    days = np.arange(n_days)

    feats: list[float] = []
    for i in range(n_features):
        row = window[i]                    # shape: (n_days,)
        abs_row = abs_window[i]            # shape: (n_days,)
        slope = float(np.polyfit(days, row, 1)[0]) if n_days > 1 else 0.0
        feats.extend((
            float(abs_row.max()),                       # max_deviation
            float(abs_row.mean()),                      # mean_deviation
            float(row.std()),                           # std_deviation
            float(np.count_nonzero(abs_row > _SIGMA_2)),  # days_above_2sigma
            float(np.count_nonzero(abs_row > _SIGMA_3)),  # days_above_3sigma
            slope,                                      # slope (escalation)
            _max_burst(abs_row),                        # max_burst
        ))

    # ── Cross-feature statistics ─────────────────────────────────────────────
    # Days where ANY feature exceeds 3σ.
    total_anomalous_days = float(np.count_nonzero(np.any(abs_window > _SIGMA_3, axis=0)))
    # Most features simultaneously above 2σ on a single day.
    max_feature_count_per_day = float((abs_window > _SIGMA_2).sum(axis=0).max())
    # Mean of every feature's mean_deviation == overall mean |deviation|.
    mean_total_deviation = float(abs_window.mean())

    feats.extend((total_anomalous_days, max_feature_count_per_day, mean_total_deviation))
    return np.asarray(feats, dtype=float)  # shape: (129,)


def extract_all_window_features(dataset) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Extract handcrafted features for every window in a dataset.

    Args:
        dataset: A :class:`~innersight.models.dataset.DeviationWindowDataset`
            (any sequence yielding ``(window_tensor, label_tensor, meta)``).

    Returns:
        ``(X, y, metas)`` where *X* has shape ``(n_windows, 129)``, *y* has shape
        ``(n_windows,)`` (binary), and *metas* is the list of metadata dicts.
    """
    rows: list[np.ndarray] = []
    labels: list[float] = []
    metas: list[dict] = []
    for i in range(len(dataset)):
        window, label, meta = dataset[i]
        rows.append(extract_window_features(np.asarray(window)))
        labels.append(float(np.asarray(label).reshape(-1)[0]))
        metas.append(meta)

    if not rows:
        return np.empty((0, N_WINDOW_FEATURES), dtype=float), np.empty((0,), dtype=float), metas
    return np.vstack(rows), np.asarray(labels, dtype=float), metas


def get_feature_names() -> list[str]:
    """Return the 129 feature names aligned with :func:`extract_window_features`.

    Per-feature names are ``"{feature_name}_{stat_name}"`` (feature-major order);
    the cross-feature statistics use their bare names.
    """
    names = [f"{feature}_{stat}" for feature in FEATURE_NAMES for stat in STAT_NAMES]
    names.extend(CROSS_FEATURE_NAMES)
    return names
