"""Unit tests for Module 1 — cohort statistics and the per-user EMA baseline.

Covers innersight.models.baseline: compute_global_median_stds,
compute_role_cohorts and the PerUserBaseline class.
"""

import numpy as np
import pandas as pd

from innersight.config import DEFAULT_BASELINE_CONFIG
from innersight.models.baseline import (
    CohortStats,
    PerUserBaseline,
    compute_global_median_stds,
    compute_role_cohorts,
)
from innersight.schema import FEATURE_NAMES

N_FEAT = len(FEATURE_NAMES)
DATES = pd.date_range("2010-06-01", periods=30, freq="D")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _features(rows: dict[str, np.ndarray], n_days: int = 20, seed: int = 0) -> pd.DataFrame:
    """Build a features_df from {user_id: per-day (n_days, 18) array}."""
    dates = pd.date_range("2010-06-01", periods=n_days, freq="D")
    frames = []
    for user, mat in rows.items():
        f = pd.DataFrame(mat, columns=FEATURE_NAMES)
        f.insert(0, "date", dates[: len(mat)])
        f.insert(0, "user", user)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _random_features(users: list[str], n_days: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _features({u: rng.poisson(3, size=(n_days, N_FEAT)).astype(float) for u in users}, n_days)


def _single_user_df(values: np.ndarray) -> dict[str, pd.DataFrame]:
    """One-user input dict for PerUserBaseline.compute_deviations."""
    return {"u": pd.DataFrame(values, columns=FEATURE_NAMES)}


# ── compute_global_median_stds ───────────────────────────────────────────────

def test_global_median_stds_shape():
    df = _random_features([f"u{i}" for i in range(6)], n_days=20)
    out = compute_global_median_stds(df)
    assert isinstance(out, np.ndarray)
    assert out.shape == (N_FEAT,)


def test_global_median_stds_single_user_no_crash():
    df = _random_features(["solo"], n_days=20, seed=1)
    out = compute_global_median_stds(df)
    assert out.shape == (N_FEAT,)
    assert np.isfinite(out).all()


def test_global_median_stds_constant_features_safe_default():
    # Every user has constant features → per-user std 0 → median 0 → replaced 1.0.
    const = np.full((20, N_FEAT), 7.0)
    df = _features({"a": const, "b": const.copy()}, n_days=20)
    out = compute_global_median_stds(df)
    assert (out >= 1.0).all()


# ── compute_role_cohorts ─────────────────────────────────────────────────────

def _ldap(rows: list[tuple], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def test_role_cohorts_keyed_by_user_id():
    users = [f"eng{i}" for i in range(5)]
    df = _random_features(users)
    ldap = _ldap([(u, "Engineer") for u in users], ["user_id", "role"])
    cohorts = compute_role_cohorts(df, ldap)
    assert isinstance(cohorts, dict)
    assert set(cohorts.keys()) == set(users)
    assert all(isinstance(v, CohortStats) for v in cohorts.values())


def test_role_cohorts_same_role_shared_stats():
    users = [f"eng{i}" for i in range(5)]
    df = _random_features(users)
    ldap = _ldap([(u, "Engineer") for u in users], ["user_id", "role"])
    cohorts = compute_role_cohorts(df, ldap)
    assert cohorts["eng0"].cohort_name == "role:Engineer"
    # Cached → all members reference the identical CohortStats object.
    assert cohorts["eng0"] is cohorts["eng1"]
    assert np.array_equal(cohorts["eng0"].mean, cohorts["eng1"].mean)


def test_role_cohorts_small_role_falls_back_to_global():
    # 5 Engineers (valid role cohort) + 1 lone Admin (role too small, no dept).
    users = [f"eng{i}" for i in range(5)] + ["solo"]
    df = _random_features(users)
    rows = [(u, "Engineer") for u in users[:5]] + [("solo", "Admin")]
    ldap = _ldap(rows, ["user_id", "role"])
    cohorts = compute_role_cohorts(df, ldap)
    assert cohorts["solo"].cohort_name == "global"
    assert cohorts["eng0"].cohort_name == "role:Engineer"


def test_role_cohorts_small_role_falls_back_to_department():
    # Lone Admin role, but shares a 6-strong department → department cohort.
    users = [f"eng{i}" for i in range(5)] + ["solo"]
    df = _random_features(users)
    rows = [(u, "Engineer", "Dept1") for u in users[:5]] + [("solo", "Admin", "Dept1")]
    ldap = _ldap(rows, ["user_id", "role", "department"])
    cohorts = compute_role_cohorts(df, ldap)
    assert cohorts["solo"].cohort_name == "department:Dept1"


def test_role_cohorts_empty_ldap_returns_empty_dict():
    df = _random_features([f"u{i}" for i in range(5)])
    empty = pd.DataFrame(columns=["user_id", "role"])
    assert compute_role_cohorts(df, empty) == {}


# ── PerUserBaseline ──────────────────────────────────────────────────────────

def test_baseline_constant_user_near_zero_after_bootstrap():
    values = np.full((30, N_FEAT), 5.0)
    b = PerUserBaseline()
    dev = b.compute_deviations(_single_user_df(values))["u"]
    assert dev.shape == (30, N_FEAT)
    post = dev[b.min_history_days:]  # after the 14-day bootstrap
    assert np.abs(post).max() < 1e-6


def test_baseline_spike_produces_large_deviation():
    values = np.full((30, N_FEAT), 5.0)
    values[29] = 5.0 * 20  # day 30 jumps to 20×
    b = PerUserBaseline()
    dev = b.compute_deviations(_single_user_df(values))["u"]
    assert np.abs(dev[29]).max() > 3.0


def test_baseline_zscore_timing_independent_of_future():
    # The spike-day z-score must be computed BEFORE the EMA absorbs the spike,
    # so appending more days afterwards cannot change it.
    base = np.full((30, N_FEAT), 5.0)
    base[29] = 5.0 * 20
    extended = np.vstack([base, np.full((10, N_FEAT), 5.0)])  # 10 calm days after
    b = PerUserBaseline()
    dev_short = b.compute_deviations(_single_user_df(base))["u"]
    dev_long = b.compute_deviations(_single_user_df(extended))["u"]
    assert np.allclose(dev_short[29], dev_long[29])


def test_baseline_std_floor_caps_zscore_on_zero_variance():
    # Constant during bootstrap (std 0) then a tiny wiggle: the std floor keeps
    # the z-score sane rather than exploding toward infinity.
    values = np.full((16, N_FEAT), 5.0)
    values[15] = 5.0 + 0.01
    b = PerUserBaseline()
    dev = b.compute_deviations(_single_user_df(values))["u"]
    assert np.abs(dev[15]).max() < 100.0


def test_baseline_cold_start_with_cohort_scores_from_day_zero():
    values = np.full((5, N_FEAT), 5.0)  # only 5 days (< min_history)
    cohort = {"u": CohortStats(
        mean=np.zeros(N_FEAT), var=np.ones(N_FEAT), cohort_name="role:Test", user_count=9,
    )}
    b = PerUserBaseline()
    dev = b.compute_deviations(_single_user_df(values), role_cohort_stats=cohort)["u"]
    assert dev.shape == (5, N_FEAT)
    # Day 0 is scored against the cohort prior (x=5 vs mean=0) → non-zero.
    assert np.abs(dev[0]).max() > 0.0


def test_baseline_min_history_without_cohort_all_zero():
    values = np.full((10, N_FEAT), 5.0)  # 10 days < min_history, no cohort
    b = PerUserBaseline()
    dev = b.compute_deviations(_single_user_df(values))["u"]
    assert dev.shape == (10, N_FEAT)
    assert np.abs(dev).max() == 0.0


def test_baseline_from_config_factory():
    b = PerUserBaseline.from_config(DEFAULT_BASELINE_CONFIG)
    assert isinstance(b, PerUserBaseline)
    assert b.ema_alpha == DEFAULT_BASELINE_CONFIG["ema_alpha"]
    assert b.min_history_days == DEFAULT_BASELINE_CONFIG["min_history_days"]
    assert b.std_floor.shape == (N_FEAT,)


def test_baseline_compute_deviations_df_preserves_rows_and_columns():
    df = _random_features([f"u{i}" for i in range(3)], n_days=20)
    b = PerUserBaseline()
    out = b.compute_deviations_df(df)
    assert len(out) == len(df)
    assert list(out.columns) == ["user", "date", *FEATURE_NAMES]
