"""Model-agnostic evaluation harness for InnerSight (Phase 3 onward).

Every model from Phase 3 through Phase 6 (XGBoost, MLP, Conv1d, GNN, …) is
scored through this module, so it works on plain numpy probability/label arrays
rather than any framework's tensors.

Primary metric is **AUPRC** (area under the precision-recall curve), which is the
honest metric under the extreme class imbalance of insider-threat data; AUROC is
kept only for comparison with the literature. Beyond scalar metrics this module
also provides operational measures: Precision@k, per-scenario breakdowns,
detection latency, and a leakage-safe user-level cross-validation driver.
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Scalar metrics aggregated across folds/seeds by run_cross_validation.
_RANK_METRICS = ("auprc", "auroc", "f1_best", "p_at_10", "p_at_20", "p_at_50")


# ── Core scalar metrics ──────────────────────────────────────────────────────

def _precision_at_k(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of the top-``k`` highest-scoring samples that are positives."""
    n = probs.shape[0]
    if n == 0:
        return 0.0
    k = min(k, n)
    top_idx = np.argsort(probs)[::-1][:k]  # k highest-scoring indices
    return float(labels[top_idx].sum() / k)


def compute_metrics(probs, labels) -> dict:
    """Compute the standard evaluation metrics for one set of predictions.

    Args:
        probs: Predicted positive-class probabilities, shape ``(n,)``.
        labels: True binary labels, shape ``(n,)``.

    Returns:
        Dict with ``auprc`` (primary), ``auroc``, ``f1_best``, ``threshold_best``
        and ``p_at_10`` / ``p_at_20`` / ``p_at_50``. If ``labels`` is single-class
        (all 0 or all 1) every metric is ``0.0`` (threshold ``0.5``) rather than
        crashing — the rank-based metrics are undefined without both classes.
    """
    probs = np.asarray(probs, dtype=float).reshape(-1)    # shape: (n,)
    labels = np.asarray(labels, dtype=float).reshape(-1)  # shape: (n,)

    n_pos = int(labels.sum())
    single_class = n_pos == 0 or n_pos == labels.shape[0]
    if single_class:
        return {
            "auprc": 0.0, "auroc": 0.0, "f1_best": 0.0, "threshold_best": 0.5,
            "p_at_10": 0.0, "p_at_20": 0.0, "p_at_50": 0.0,
        }

    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    auprc = float(average_precision_score(labels, probs))
    auroc = float(roc_auc_score(labels, probs))

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1.0), 0.0)
    if thresholds.size:
        # precision/recall carry one extra trailing point with no threshold.
        best = int(np.argmax(f1[: thresholds.size]))
        f1_best = float(f1[best])
        threshold_best = float(thresholds[best])
    else:
        f1_best, threshold_best = 0.0, 0.5

    return {
        "auprc": auprc,
        "auroc": auroc,
        "f1_best": f1_best,
        "threshold_best": threshold_best,
        "p_at_10": _precision_at_k(probs, labels, 10),
        "p_at_20": _precision_at_k(probs, labels, 20),
        "p_at_50": _precision_at_k(probs, labels, 50),
    }


def per_scenario_metrics(probs, labels, scenarios) -> dict[int, dict]:
    """Compute metrics separately for each attack scenario vs. all benign samples.

    Args:
        probs: Predicted probabilities, shape ``(n,)``.
        labels: True binary labels, shape ``(n,)`` (0 = benign).
        scenarios: Scenario id per sample, shape ``(n,)`` (0 = benign, 1-5 = attack).

    Returns:
        Dict mapping each present attack scenario id → its :func:`compute_metrics`
        result, where positives are that scenario's samples and negatives are all
        benign samples (other scenarios are excluded from that comparison).
    """
    probs = np.asarray(probs, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    scenarios = np.asarray(scenarios).reshape(-1)

    benign = labels == 0
    out: dict[int, dict] = {}
    for scenario in sorted({int(s) for s in scenarios.tolist()}):
        if scenario == 0:
            continue
        positive = scenarios == scenario
        mask = positive | benign
        out[scenario] = compute_metrics(probs[mask], positive[mask].astype(float))
    return out


# ── Detection latency ────────────────────────────────────────────────────────

def detection_latency(window_probs, window_metas, threshold, attack_windows) -> dict:
    """Measure how quickly flagged insiders are detected after their attack starts.

    An insider is "detected" if any of their windows scores above ``threshold``;
    latency is measured from the attack start to the first such window.

    Args:
        window_probs: Per-window probabilities, shape ``(n_windows,)``, aligned
            with ``window_metas``.
        window_metas: Per-window metadata dicts with ``user_id`` and
            ``window_start`` (from :class:`DeviationWindowDataset`).
        threshold: Decision threshold; a window with prob > threshold is a flag.
        attack_windows: Mapping ``user_id`` → ``InsiderRecord`` (has ``attack_start``).

    Returns:
        Dict with ``median_days``, ``p90_days`` (NaN if nothing detected),
        ``detected_count``, ``total_insiders`` and ``per_insider`` — a list of
        ``{user_id, days_to_detect, first_flagged_window}`` for detected insiders.
    """
    probs = np.asarray(window_probs, dtype=float).reshape(-1)

    # Earliest flagged window start per user.
    first_flag: dict[str, pd.Timestamp] = {}
    for i, meta in enumerate(window_metas):
        if probs[i] > threshold:
            uid = meta["user_id"]
            start = pd.Timestamp(meta["window_start"])
            if uid not in first_flag or start < first_flag[uid]:
                first_flag[uid] = start

    per_insider: list[dict] = []
    latencies: list[float] = []
    for uid, record in attack_windows.items():
        flagged = first_flag.get(uid)
        if flagged is None:
            continue  # never flagged → not detected
        days = float((flagged - pd.Timestamp(record.attack_start)).days)
        per_insider.append({
            "user_id": uid,
            "days_to_detect": days,
            "first_flagged_window": flagged,
        })
        latencies.append(days)

    if latencies:
        median_days = float(np.median(latencies))
        p90_days = float(np.percentile(latencies, 90))
    else:
        median_days = math.nan
        p90_days = math.nan

    return {
        "median_days": median_days,
        "p90_days": p90_days,
        "detected_count": len(per_insider),
        "total_insiders": len(attack_windows),
        "per_insider": per_insider,
    }


# ── Cross-validation ─────────────────────────────────────────────────────────

def temporal_stratified_kfold(user_ids, labels, n_folds: int = 5, seed: int = 42):
    """User-level stratified K-fold split, expanded to window-level indices.

    Splitting on users (not windows) prevents a user's windows from leaking
    across the train/val boundary. Folds are stratified so each holds a
    proportional share of insider users.

    Args:
        user_ids: User id per sample, shape ``(n,)``.
        labels: Binary label per sample, shape ``(n,)`` (a user is an insider if
            any of their samples is positive).
        n_folds: Number of folds.
        seed: Shuffle seed for reproducibility.

    Returns:
        List of ``n_folds`` ``(train_idx, val_idx)`` tuples of integer indices
        into the original arrays, with disjoint user sets per fold.
    """
    from sklearn.model_selection import StratifiedKFold

    user_ids = np.asarray(user_ids)
    labels = np.asarray(labels, dtype=float).reshape(-1)

    unique_users = np.unique(user_ids)  # sorted, unique
    user_is_insider = np.array(
        [1 if labels[user_ids == u].max() > 0 else 0 for u in unique_users]
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_u, val_u in skf.split(unique_users, user_is_insider):
        train_users = unique_users[train_u]
        val_users = unique_users[val_u]
        train_idx = np.where(np.isin(user_ids, train_users))[0]
        val_idx = np.where(np.isin(user_ids, val_users))[0]
        folds.append((train_idx, val_idx))
    return folds


def _mean_std(dicts: list[dict], keys) -> tuple[dict, dict]:
    """Per-key mean and population std across a list of metric dicts."""
    mean = {k: float(np.mean([d[k] for d in dicts])) for k in keys}
    std = {k: float(np.std([d[k] for d in dicts])) for k in keys}
    return mean, std


def run_cross_validation(
    model_fn: Callable,
    X,
    y,
    metas,
    attack_windows,
    n_folds: int = 5,
    seeds=(42, 123, 456),
) -> dict:
    """Run repeated user-level stratified cross-validation and aggregate metrics.

    For every ``seed`` and fold, the data is split with
    :func:`temporal_stratified_kfold`, ``model_fn`` trains and returns validation
    probabilities, and :func:`compute_metrics` scores them. Results are averaged
    within each seed and then across seeds.

    Args:
        model_fn: Callable ``(X_train, y_train, X_val, y_val, seed) -> val_probs``.
        X: Feature array, shape ``(n, ...)``.
        y: Binary labels, shape ``(n,)``.
        metas: Per-sample metadata dicts (must contain ``user_id``; ``window_start``
            enables detection-latency reporting).
        attack_windows: Mapping ``user_id`` → ``InsiderRecord`` for latency.
        n_folds: Folds per seed.
        seeds: Seeds to repeat the whole CV over.

    Returns:
        Dict with ``per_seed`` (per-seed mean/std), ``mean`` and ``std`` (across
        seeds, including ``median_days`` detection latency), and ``per_fold``
        (every individual fold result).
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float).reshape(-1)
    user_ids = np.array([m["user_id"] for m in metas])

    per_fold: list[dict] = []
    per_seed: list[dict] = []

    for seed in seeds:
        folds = temporal_stratified_kfold(user_ids, y, n_folds=n_folds, seed=seed)
        seed_metrics: list[dict] = []
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            val_probs = model_fn(X[train_idx], y[train_idx], X[val_idx], y[val_idx], seed)
            val_probs = np.asarray(val_probs, dtype=float).reshape(-1)

            metrics = compute_metrics(val_probs, y[val_idx])

            # Detection latency on this fold's validation insiders only.
            val_users = set(user_ids[val_idx].tolist())
            fold_attacks = {u: r for u, r in attack_windows.items() if u in val_users}
            val_metas = [metas[i] for i in val_idx]
            latency = detection_latency(val_probs, val_metas, metrics["threshold_best"], fold_attacks)
            metrics["median_days"] = latency["median_days"]

            per_fold.append({"seed": seed, "fold": fold_i, **metrics})
            seed_metrics.append(metrics)

        seed_mean, seed_std = _mean_std(seed_metrics, _RANK_METRICS)
        seed_mean["median_days"] = _nanmean([m["median_days"] for m in seed_metrics])
        per_seed.append({"seed": seed, "mean": seed_mean, "std": seed_std})

    seed_means = [ps["mean"] for ps in per_seed]
    overall_mean, overall_std = _mean_std(seed_means, _RANK_METRICS)
    overall_mean["median_days"] = _nanmean([sm["median_days"] for sm in seed_means])
    overall_std["median_days"] = _nanstd([sm["median_days"] for sm in seed_means])

    return {"per_seed": per_seed, "mean": overall_mean, "std": overall_std, "per_fold": per_fold}


def _nanmean(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else math.nan


def _nanstd(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr)) if not np.all(np.isnan(arr)) else math.nan


# ── Reporting ────────────────────────────────────────────────────────────────

def format_results_table(results_dict: dict) -> str:
    """Render a comparison table of cross-validation results as a string.

    Args:
        results_dict: Mapping ``model_name`` → result dict from
            :func:`run_cross_validation`.

    Returns:
        A fixed-width text table with columns
        ``Model | AUPRC ± std | P@10 | P@20 | F1 | Detection Latency``.
    """
    header = (
        f"{'Model':<20} {'AUPRC ± std':<18} {'P@10':>6} {'P@20':>6} "
        f"{'F1':>6} {'Latency':>10}"
    )
    lines = [header, "-" * len(header)]
    for name, res in results_dict.items():
        mean, std = res["mean"], res["std"]
        auprc = f"{mean['auprc']:.3f} ± {std['auprc']:.3f}"
        latency = mean.get("median_days", math.nan)
        latency_str = "n/a" if latency is None or math.isnan(latency) else f"{latency:.1f}d"
        lines.append(
            f"{name:<20} {auprc:<18} {mean['p_at_10']:>6.3f} {mean['p_at_20']:>6.3f} "
            f"{mean['f1_best']:>6.3f} {latency_str:>10}"
        )
    return "\n".join(lines)
