#!/usr/bin/env python
"""Temporal CNN trainer (Module 2, Phase 4).

Trains the :class:`TemporalPatternEncoder` (dilated causal CNN + attention) plus
a linear classification head on the **raw** ``(18, 28)`` deviation windows —
unlike the MLP baseline, the time structure is preserved rather than flattened.
Training uses focal loss, AdamW, cosine annealing with linear warmup, gradient
clipping, and early stopping on validation AUPRC, all under the shared
user-level cross-validation harness.

Usage:
    python -m innersight.scripts.train_temporal \
        --version r4.2 --store-dir feature_store --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from innersight.config import setup_logging
from innersight.data.answers import load_insiders
from innersight.data.feature_store import FeatureStore
from innersight.models.dataset import DeviationWindowDataset
from innersight.models.losses import FocalLoss
from innersight.models.temporal_encoder import TemporalPatternEncoder
from innersight.scripts import compute_baselines
from innersight.training.evaluation import (
    compute_metrics,
    detection_latency,
    format_results_table,
    per_scenario_metrics,
    run_cross_validation,
    temporal_stratified_kfold,
)
from innersight.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cpu")  # CPU keeps cross-validation reproducible

# Defaults mirror configs/train_temporal.yaml (used when a key is absent).
_DEFAULT_MODEL = {"in_channels": 18, "hidden": 64, "out_dim": 128, "dropout": 0.3, "kernel_size": 3}
_DEFAULT_TRAINING = {
    "focal_alpha": 0.75, "focal_gamma": 2.0, "lr": 1e-3, "weight_decay": 1e-4,
    "grad_clip": 1.0, "warmup_epochs": 5, "eta_min": 1e-5, "max_epochs": 100,
    "patience": 10, "batch_size": 64,
}
_DEFAULT_EVAL = {"n_folds": 5, "seeds": [42, 123, 456]}


class _TemporalClassifier(nn.Module):
    """TemporalPatternEncoder + linear head → per-window logit."""

    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.encoder = TemporalPatternEncoder(
            in_channels=model_cfg["in_channels"],
            hidden=model_cfg["hidden"],
            out_dim=model_cfg["out_dim"],
            dropout=model_cfg["dropout"],
            kernel_size=model_cfg["kernel_size"],
        )
        self.head = nn.Linear(model_cfg["out_dim"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 18, T)
        return self.head(self.encoder(x))  # shape: (batch, 1)


def _build_scheduler(optimizer, warmup_epochs: int, max_epochs: int, eta_min: float):
    """Linear warmup (0→lr over warmup_epochs) then cosine annealing to eta_min."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=max(1, warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=eta_min)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[max(1, warmup_epochs)])


def _inner_holdout(y_train, frac: float = 0.2, seed: int = 0):
    """Stratified inner train/val index split used only for early stopping.

    Carves a held-out monitoring set from the *training* fold so checkpoint
    selection never touches the outer validation fold (which would leak the
    reported metric). Positives and negatives are split proportionally so the
    monitor set keeps some of the rare positives when possible.

    Returns:
        ``(inner_train_idx, inner_val_idx)`` integer arrays into ``y_train``.
    """
    y = np.asarray(y_train).reshape(-1)
    rng = np.random.default_rng(seed)
    pos = np.where(y > 0)[0]
    neg = np.where(y <= 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    # Need ≥2 of a class to spare one for the monitor set.
    n_pos_val = int(round(len(pos) * frac)) if len(pos) >= 2 else 0
    n_neg_val = int(round(len(neg) * frac)) if len(neg) >= 2 else 0
    val_idx = np.concatenate([pos[:n_pos_val], neg[:n_neg_val]]).astype(int)
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    return np.where(train_mask)[0], val_idx


def _fit(X_train, y_train, X_val, y_val, model_cfg: dict, train_cfg: dict, seed: int) -> dict:
    """Train one temporal-CNN model and return val probs + checkpoint material.

    Leakage-safe model selection: early stopping / best-checkpoint selection uses
    an INNER validation split carved from the training fold only. The outer
    ``X_val`` fold is scored exactly once, at the end, and is never used to choose
    the epoch — so the reported cross-validation metric is not optimistically
    biased by tuning on the data it is measured on.
    """
    seed_everything(seed)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)

    inner_tr, inner_va = _inner_holdout(y_train, frac=0.2, seed=seed)
    # Early stopping is only meaningful if the monitor set holds a positive.
    early_stop = inner_va.size > 0 and float(y_train[inner_va].sum()) > 0
    if not early_stop:
        inner_tr = np.arange(len(y_train))  # too few positives → train on the full fold

    Xtr = torch.tensor(X_train[inner_tr], dtype=torch.float32, device=_DEVICE)   # shape: (n_tr, 18, T)
    ytr = torch.tensor(y_train[inner_tr], dtype=torch.float32, device=_DEVICE).reshape(-1, 1)
    Xiv = (torch.tensor(X_train[inner_va], dtype=torch.float32, device=_DEVICE)
           if early_stop else None)                                              # shape: (n_iv, 18, T)
    yiv = y_train[inner_va] if early_stop else None
    Xva = torch.tensor(np.asarray(X_val), dtype=torch.float32, device=_DEVICE)    # shape: (n_va, 18, T)

    model = _TemporalClassifier(model_cfg).to(_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = _build_scheduler(optimizer, train_cfg["warmup_epochs"],
                                 train_cfg["max_epochs"], train_cfg["eta_min"])
    criterion = FocalLoss(alpha=train_cfg["focal_alpha"], gamma=train_cfg["focal_gamma"])

    n = Xtr.shape[0]
    batch_size = train_cfg["batch_size"]
    grad_clip = train_cfg["grad_clip"]
    generator = torch.Generator().manual_seed(seed)

    best_metric = -1.0
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_since_improvement = 0

    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=generator)
        grad_norms: list[float] = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xtr[idx])               # shape: (batch, 1)
            loss = criterion(logits, ytr[idx])
            if torch.isnan(loss):
                logger.warning("train_temporal | NaN loss; skipping batch.")
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norms.append(float(grad_norm))
            optimizer.step()
        scheduler.step()

        if not early_stop:
            continue  # no inner monitor set → just train the full epoch budget

        model.eval()
        with torch.no_grad():
            iv_probs = torch.sigmoid(model(Xiv)).reshape(-1).cpu().numpy()  # shape: (n_iv,)
        metric = compute_metrics(iv_probs, yiv)["auprc"]
        mean_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
        logger.debug("train_temporal | epoch %d | inner_val_auprc=%.4f | lr=%.2e | grad_norm=%.3f",
                     epoch, metric, scheduler.get_last_lr()[0], mean_grad)

        if metric > best_metric:
            best_metric, best_epoch = metric, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= train_cfg["patience"]:
                break

    if early_stop:
        model.load_state_dict(best_state)
    else:
        # No monitoring happened: keep the final-epoch weights.
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_epoch = train_cfg["max_epochs"] - 1
        best_metric = float("nan")

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(Xva)).reshape(-1).cpu().numpy()  # shape: (n_va,)

    return {
        "val_probs": val_probs,
        "model_state_dict": best_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": best_epoch,
        "best_metric": best_metric,
    }


def load_checkpoint(path: str | Path) -> dict:
    """Load a temporal-CNN checkpoint (weights only, on CPU)."""
    return torch.load(path, weights_only=True, map_location="cpu")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train the temporal CNN (Module 2) on raw deviation windows with user-level CV.',
    )
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', metavar='PATH',
                   help='Dataset directory (answers/ labels, and to compute deviations if uncached)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--config', metavar='PATH', help='Path to a train_temporal.yaml config')
    p.add_argument('--output', default='temporal_results.json', metavar='PATH',
                   help='Where to write the results JSON (default: temporal_results.json)')
    p.add_argument('--checkpoint-dir', default='checkpoints', metavar='PATH',
                   help="Directory for the model checkpoint (default: 'checkpoints')")
    return p.parse_args(argv)


def _load_config(path: str | None) -> tuple[dict, dict, dict]:
    """Load model/training/evaluation config, falling back to defaults."""
    raw = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    model_cfg = {**_DEFAULT_MODEL, **raw.get("model", {})}
    train_cfg = {**_DEFAULT_TRAINING, **raw.get("training", {})}
    eval_cfg = {**_DEFAULT_EVAL, **raw.get("evaluation", {})}
    return model_cfg, train_cfg, eval_cfg


def _build_window_tensors(dataset) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Stack raw (18, T) windows into X (n, 18, T) without flattening."""
    windows: list[np.ndarray] = []
    labels: list[float] = []
    metas: list[dict] = []
    for i in range(len(dataset)):
        window, label, meta = dataset[i]
        windows.append(np.asarray(window, dtype=np.float32))
        labels.append(float(np.asarray(label).reshape(-1)[0]))
        metas.append(meta)
    if not windows:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=float), metas
    return np.stack(windows), np.asarray(labels, dtype=float), metas


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str | None):
    """Return cached deviations, computing them from data_dir if absent."""
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    if not data_dir:
        logger.error("train_temporal | no deviations cached for %s and no --data-dir. "
                     "Run compute_baselines first.", version)
        return None
    logger.info("train_temporal | deviations not cached; computing from %s …", data_dir)
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        logger.error("train_temporal | compute_baselines failed (rc=%d).", rc)
        return None
    return store.load_deviations(version)


def _log_baseline_comparison(output_path: Path, temporal_cv: dict) -> None:
    """Append Phase 3 baseline results to the comparison table when available."""
    comparison = {"TemporalCNN": temporal_cv}
    for name, fname in (("XGBoost", "xgboost_results.json"), ("MLP", "mlp_results.json")):
        for candidate in (output_path.parent / fname, Path(fname)):
            if candidate.exists():
                try:
                    comparison[name] = json.loads(candidate.read_text())["cross_validation"]
                except (KeyError, ValueError) as exc:
                    logger.warning("train_temporal | could not read %s: %s", candidate, exc)
                break
    logger.info("train_temporal | comparison:\n%s", format_results_table(comparison))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    model_cfg, train_cfg, eval_cfg = _load_config(args.config)
    seeds = [int(s) for s in eval_cfg["seeds"]]
    n_folds = int(eval_cfg["n_folds"])
    seed_everything(seeds[0] if seeds else 42)

    store = FeatureStore(args.store_dir)
    deviations = _load_deviations(store, args.version, args.store_dir, args.data_dir)
    if deviations is None:
        return 1

    if not args.data_dir:
        logger.error("train_temporal | --data-dir is required to load attack windows (answers/).")
        return 1
    answers_dir = Path(args.data_dir) / 'answers'
    if not answers_dir.exists():
        logger.error("train_temporal | no answers/ directory at %s.", answers_dir)
        return 1
    attack_windows = {r.user_id: r for r in load_insiders(answers_dir, args.version)}
    logger.info("train_temporal | %d insiders for %s.", len(attack_windows), args.version)

    dataset = DeviationWindowDataset(deviations, attack_windows)
    X, y, metas = _build_window_tensors(dataset)
    n_pos = int(y.sum())
    logger.info("train_temporal | %d windows of shape %s (%d positive).",
                X.shape[0], tuple(X.shape[1:]), n_pos)
    if X.shape[0] == 0 or n_pos == 0:
        logger.error("train_temporal | need labelled positive windows to train; aborting.")
        return 1

    user_ids = np.array([m['user_id'] for m in metas])
    scenarios = np.array([m.get('scenario', 0) for m in metas])

    def model_fn(X_train, y_train, X_val, y_val, seed):
        return _fit(X_train, y_train, X_val, y_val, model_cfg, train_cfg, seed)["val_probs"]

    cv = run_cross_validation(model_fn, X, y, metas, attack_windows, n_folds=n_folds, seeds=seeds)
    logger.info("train_temporal | CV AUPRC=%.4f ± %.4f | P@10=%.3f | F1=%.3f",
                cv['mean']['auprc'], cv['std']['auprc'], cv['mean']['p_at_10'], cv['mean']['f1_best'])

    # Out-of-fold predictions for per-scenario metrics + detection latency.
    oof = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in temporal_stratified_kfold(user_ids, y, n_folds=n_folds, seed=seeds[0]):
        oof[val_idx] = _fit(X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                            model_cfg, train_cfg, seeds[0])["val_probs"]
    scenario_metrics = per_scenario_metrics(oof, y, scenarios)
    threshold = compute_metrics(oof, y)['threshold_best']
    latency = detection_latency(oof, metas, threshold, attack_windows)
    logger.info("train_temporal | detection: %d/%d insiders flagged, median latency=%s days.",
                latency['detected_count'], latency['total_insiders'], latency['median_days'])

    # Final model trained on all data → checkpoint.
    final = _fit(X, y, X, y, model_cfg, train_cfg, seeds[0])
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"temporal_cnn_{args.version}.pt"
    torch.save({
        "model_state_dict": final["model_state_dict"],
        "optimizer_state_dict": final["optimizer_state_dict"],
        "epoch": final["epoch"],
        "best_metric": final["best_metric"],
        "model_config": model_cfg,
    }, ckpt_path)
    logger.info("train_temporal | checkpoint saved to %s (best_metric=%.4f).",
                ckpt_path, final["best_metric"])

    results = {
        "version": args.version,
        "model": "temporal_cnn",
        "n_windows": int(X.shape[0]),
        "n_positive": n_pos,
        "n_folds": n_folds,
        "seeds": seeds,
        "config": {"model": model_cfg, "training": train_cfg},
        "cross_validation": cv,
        "per_scenario": scenario_metrics,
        "detection_latency": latency,
        "oof_threshold": threshold,
        "checkpoint": str(ckpt_path),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("train_temporal | results written to %s.", output_path)

    _log_baseline_comparison(output_path, cv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
