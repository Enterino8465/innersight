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


# ── Per-user EMA baseline ────────────────────────────────────────────────────
class PerUserBaseline:
    """EMA baseline that z-scores each user's day against their own normal.

    For each user we maintain an exponentially-weighted mean and variance of
    daily behaviour. A day's deviation is its z-score *against the baseline as it
    stood before that day* — so a sudden change shows up as a large deviation
    before the baseline has had a chance to absorb it.

    Cold-start: a brand-new user has no personal history, so their baseline is
    seeded from a cohort prior (see :func:`compute_role_cohorts`). Without a
    cohort, a user with enough history bootstraps from their own first
    ``min_history_days``; a user with too little history is treated as invisible
    (all-zero deviations).

    Attributes:
        ema_alpha: EMA smoothing factor in ``(0, 1]``; larger adapts faster.
        min_history_days: Days required to bootstrap a baseline without a cohort.
        std_floor: Per-feature lower bound on the z-score denominator, shape ``(18,)``.
        variance_eps: Small constant added to the EMA variance before its square
            root to avoid a zero-variance blow-up.
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        min_history_days: int = 14,
        std_floor: np.ndarray | None = None,
        variance_eps: float = 1e-6,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.min_history_days = int(min_history_days)
        self.variance_eps = float(variance_eps)

        if std_floor is None:
            std_floor = np.full(NUM_FEATURES, 0.1, dtype=float)  # shape: (18,)
        else:
            std_floor = np.asarray(std_floor, dtype=float)
            if std_floor.shape != (NUM_FEATURES,):
                raise ValueError(f"std_floor must have shape ({NUM_FEATURES},), got {std_floor.shape}")
        self.std_floor = std_floor  # shape: (18,)

    @classmethod
    def from_config(
        cls,
        config: dict,
        global_median_stds: np.ndarray | None = None,
    ) -> PerUserBaseline:
        """Build a baseline from a config dict (e.g. ``DEFAULT_BASELINE_CONFIG``).

        Args:
            config: Dict with ``ema_alpha``, ``min_history_days``,
                ``std_floor_ratio`` and ``variance_eps`` keys.
            global_median_stds: Optional per-feature median std, shape ``(18,)``
                (see :func:`compute_global_median_stds`). When given, the std
                floor is ``global_median_stds * config['std_floor_ratio']``;
                otherwise the constructor default (all ``0.1``) is used.

        Returns:
            A configured :class:`PerUserBaseline`.
        """
        std_floor: np.ndarray | None = None
        if global_median_stds is not None:
            # shape: (18,)
            std_floor = np.asarray(global_median_stds, dtype=float) * config["std_floor_ratio"]
        return cls(
            ema_alpha=config["ema_alpha"],
            min_history_days=config["min_history_days"],
            std_floor=std_floor,
            variance_eps=config["variance_eps"],
        )

    def compute_deviations(
        self,
        user_features: dict[str, pd.DataFrame],
        role_cohort_stats: dict[str, CohortStats] | None = None,
    ) -> dict[str, np.ndarray]:
        """Z-score every user's daily features against their evolving baseline.

        Args:
            user_features: Mapping ``user_id`` → DataFrame whose columns are the
                18 FEATURE_NAMES, with rows sorted by date ascending.
            role_cohort_stats: Optional mapping ``user_id`` → :class:`CohortStats`
                used to seed cold-start users. Users present here are scored from
                day 0 using the cohort prior.

        Returns:
            Mapping ``user_id`` → deviation array of shape ``(n_days, 18)``.
            Rows before a user's start day (and every row for invisible users)
            are zero.
        """
        role_cohort_stats = role_cohort_stats or {}
        out: dict[str, np.ndarray] = {}

        for uid, df in user_features.items():
            x = df[FEATURE_NAMES].to_numpy(dtype=float)        # shape: (n_days, 18)
            n_days = x.shape[0]
            dev = np.zeros((n_days, NUM_FEATURES), dtype=float)  # shape: (n_days, 18)

            cohort = role_cohort_stats.get(uid)
            if cohort is not None:
                ema_mean = cohort.mean.astype(float).copy()    # shape: (18,)
                ema_var = cohort.var.astype(float).copy()      # shape: (18,)
                start_day = 0
            elif n_days >= self.min_history_days:
                boot = x[: self.min_history_days]              # shape: (min_history_days, 18)
                ema_mean = boot.mean(axis=0)                   # shape: (18,)
                ema_var = boot.var(axis=0)                     # shape: (18,) (population var)
                start_day = self.min_history_days
            else:
                out[uid] = dev  # too little history, no cohort → invisible (all zeros)
                continue

            alpha = self.ema_alpha
            for t in range(start_day, n_days):
                x_t = x[t]                                     # shape: (18,)
                # (a) Score against the PAST baseline — before this day is absorbed.
                std = np.sqrt(ema_var + self.variance_eps)     # shape: (18,)
                denom = np.maximum(std, self.std_floor)        # shape: (18,)
                dev[t] = (x_t - ema_mean) / denom              # shape: (18,)
                # (b) Update the EMA mean, then (c) the EMA variance about it.
                ema_mean = (1.0 - alpha) * ema_mean + alpha * x_t
                ema_var = (1.0 - alpha) * ema_var + alpha * (x_t - ema_mean) ** 2

            out[uid] = dev  # shape: (n_days, 18)

        return out

    def compute_deviations_df(
        self,
        features_df: pd.DataFrame,
        role_cohort_stats: dict[str, CohortStats] | None = None,
    ) -> pd.DataFrame:
        """Z-score a full features DataFrame, returning the same shape of frame.

        Groups ``features_df`` by user, sorts each user's rows by date, runs
        :meth:`compute_deviations`, and reassembles the per-user deviation
        arrays back into a single DataFrame. This is the primary interface for
        the compute_baselines script (Task 4).

        Args:
            features_df: Daily features with columns ``user``, ``date`` and the
                18 FEATURE_NAMES columns (raw counts).
            role_cohort_stats: Optional cohort priors for cold-start users.

        Returns:
            DataFrame with columns ``user``, ``date`` and the 18 FEATURE_NAMES,
            where the feature columns now hold z-scored deviations instead of
            raw counts.
        """
        user_features: dict[str, pd.DataFrame] = {}
        user_dates: dict[str, np.ndarray] = {}
        for uid, grp in features_df.groupby("user", sort=False):
            ordered = grp.sort_values("date")
            user_features[uid] = ordered[FEATURE_NAMES].reset_index(drop=True)
            user_dates[uid] = ordered["date"].to_numpy()

        deviations = self.compute_deviations(user_features, role_cohort_stats)

        frames: list[pd.DataFrame] = []
        for uid, dev in deviations.items():
            frame = pd.DataFrame(dev, columns=FEATURE_NAMES)  # shape: (n_days, 18)
            frame.insert(0, "date", user_dates[uid])
            frame.insert(0, "user", uid)
            frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=["user", "date", *FEATURE_NAMES])
        return pd.concat(frames, ignore_index=True)
