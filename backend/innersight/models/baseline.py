"""Module 1 — per-user behavioural baseline.

This file holds the per-user baseline used to z-score each user's daily
activity against their own normal behaviour. New users (or users with too
little history) have no personal baseline yet, so they are cold-started from a
*cohort* prior — the average behaviour of similar users.

This first half builds the cohort statistics:
    * :class:`CohortStats`            — frozen mean/variance prior for a cohort.
    * :func:`compute_global_median_stds` — per-feature std floor reference.
    * :func:`compute_role_cohorts`    — assign every user their most specific
      cohort (role → department → global).

The :class:`PerUserBaseline` class (Task 3) consumes these priors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from innersight.schema import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Number of per-user-per-day features (canonical 18; see innersight.schema).
NUM_FEATURES: int = len(FEATURE_NAMES)

# A cohort needs at least this many users to yield a stable prior; smaller
# groups fall back to the next-broader cohort (role → department → global).
MIN_COHORT_SIZE: int = 5


# ── Cohort prior ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CohortStats:
    """Mean/variance prior for a cohort of similar users.

    Used to cold-start a user's baseline before they have enough personal
    history. ``mean`` and ``var`` summarise the daily feature values observed
    across every member of the cohort.

    Attributes:
        mean: Average daily feature values, shape ``(18,)``.
        var: Variance of daily feature values, shape ``(18,)``.
        cohort_name: Human-readable label for logging, e.g. ``'role:SoftwareEngineer'``.
        user_count: Number of users contributing to this cohort.
    """

    mean: np.ndarray
    var: np.ndarray
    cohort_name: str
    user_count: int

    def __post_init__(self) -> None:
        for name, arr in (("mean", self.mean), ("var", self.var)):
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"CohortStats.{name} must be a numpy array, got {type(arr).__name__}")
            if arr.shape != (NUM_FEATURES,):
                raise ValueError(
                    f"CohortStats.{name} must have shape ({NUM_FEATURES},), got {arr.shape}"
                )


# ── Std floor reference ──────────────────────────────────────────────────────
def compute_global_median_stds(features_df: pd.DataFrame) -> np.ndarray:
    """Compute the median per-user standard deviation for each feature.

    For every feature we take each user's own daily std, then the median of
    those stds across all users. This gives a robust, population-level scale for
    each feature that is used to set the baseline std floor::

        std_floor = 0.1 * compute_global_median_stds(features_df)

    The floor prevents tiny per-user variances (e.g. a user who logs on exactly
    twice every day) from blowing up z-scores.

    Args:
        features_df: Daily features with columns ``user``, ``date`` and the 18
            FEATURE_NAMES columns.

    Returns:
        Array of shape ``(18,)``. Features whose median std is NaN (e.g. every
        user has a single day) or zero are replaced with ``1.0`` as a safe
        default scale.
    """
    per_user_std = features_df.groupby("user")[FEATURE_NAMES].std(ddof=1)
    median_std = per_user_std.median(axis=0, skipna=True).to_numpy(dtype=float)
    # NaN (no within-user variation observable) or 0.0 (constant feature) → 1.0.
    median_std = np.where(np.isnan(median_std) | (median_std == 0.0), 1.0, median_std)
    return median_std


# ── Cohort assignment ────────────────────────────────────────────────────────
def _cohort_stats_for(features_df: pd.DataFrame, members: list[str], cohort_name: str) -> CohortStats:
    """Build a :class:`CohortStats` from the daily rows of ``members``."""
    sub = features_df[features_df["user"].isin(members)]
    mean = np.nan_to_num(sub[FEATURE_NAMES].mean().to_numpy(dtype=float), nan=0.0)
    var = np.nan_to_num(sub[FEATURE_NAMES].var(ddof=0).to_numpy(dtype=float), nan=0.0)
    return CohortStats(mean=mean, var=var, cohort_name=cohort_name, user_count=len(members))


def compute_role_cohorts(
    features_df: pd.DataFrame,
    ldap_df: pd.DataFrame,
) -> dict[str, CohortStats]:
    """Assign every user the most specific cohort prior available.

    Resolution order, per user:
        1. ``role`` cohort — if at least ``MIN_COHORT_SIZE`` users share the role.
        2. ``department`` cohort — if the role cohort is too small and a large
           enough department is available (requires the optional ``department``
           LDAP column).
        3. ``global`` cohort — the ultimate fallback (all users with data).

    Cohort membership counts only users that actually appear in ``features_df``,
    since cohort stats are computed from observed daily behaviour. Users present
    in ``features_df`` but absent from ``ldap_df`` fall back to the global cohort.

    Args:
        features_df: Daily features with columns ``user``, ``date`` and the 18
            FEATURE_NAMES columns.
        ldap_df: LDAP records with columns ``user_id``, ``role`` and optionally
            ``department``.

    Returns:
        Mapping of ``user_id`` → the best :class:`CohortStats` for that user, for
        every user that has rows in ``features_df``.
    """
    users_with_data = set(features_df["user"].unique())
    if not users_with_data:
        return {}

    # One LDAP row per user (latest snapshot wins), restricted to users we have
    # behavioural data for — a cohort with no observed days is useless.
    ldap = ldap_df.drop_duplicates(subset="user_id", keep="last")
    ldap = ldap[ldap["user_id"].isin(users_with_data)]
    has_dept = "department" in ldap.columns

    role_of = dict(zip(ldap["user_id"], ldap["role"]))
    dept_of = dict(zip(ldap["user_id"], ldap["department"])) if has_dept else {}

    # Cohort membership (users-with-data) keyed by role and by department.
    role_members = ldap.groupby("role")["user_id"].apply(list).to_dict()
    dept_members: dict[str, list[str]] = {}
    if has_dept:
        dept_members = ldap.dropna(subset=["department"]).groupby("department")["user_id"].apply(list).to_dict()

    global_members = sorted(users_with_data)
    global_stats = _cohort_stats_for(features_df, global_members, "global")

    cache: dict[str, CohortStats] = {"global": global_stats}

    def _resolve(uid: str) -> CohortStats:
        role = role_of.get(uid)
        if role is not None:
            members = role_members.get(role, [])
            if len(members) >= MIN_COHORT_SIZE:
                key = f"role:{role}"
                if key not in cache:
                    cache[key] = _cohort_stats_for(features_df, members, key)
                return cache[key]
        dept = dept_of.get(uid)
        if dept is not None and pd.notna(dept):
            members = dept_members.get(dept, [])
            if len(members) >= MIN_COHORT_SIZE:
                key = f"department:{dept}"
                if key not in cache:
                    cache[key] = _cohort_stats_for(features_df, members, key)
                return cache[key]
        return global_stats

    result = {uid: _resolve(uid) for uid in sorted(users_with_data)}

    n_role = sum(1 for s in result.values() if s.cohort_name.startswith("role:"))
    n_dept = sum(1 for s in result.values() if s.cohort_name.startswith("department:"))
    n_global = sum(1 for s in result.values() if s.cohort_name == "global")
    logger.info(
        "compute_role_cohorts | %d users → role:%d department:%d global:%d (%d distinct cohorts)",
        len(result), n_role, n_dept, n_global, len(cache),
    )
    return result
