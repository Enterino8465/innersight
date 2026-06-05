#!/usr/bin/env python
"""Visualise the temporal encoder's per-day attention (Phase 4).

Loads a trained :class:`TemporalPatternEncoder` and, for each insider's windows
(or one chosen user), plots the softmax attention weights over the 28 days with
the known attack period shaded. Peaks reveal which days the model focused on —
an interpretability check that the encoder attends to the actual attack.

Usage:
    python -m innersight.scripts.visualize_attention \
        --checkpoint checkpoints/temporal_cnn_r4.2.pt \
        --version r4.2 --data-dir /path/to/data --output-dir attention_plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from innersight.config import setup_logging
from innersight.data.answers import load_insiders
from innersight.data.feature_store import FeatureStore
from innersight.models.dataset import DeviationWindowDataset
from innersight.models.temporal_encoder import TemporalPatternEncoder
from innersight.scripts import compute_baselines
from innersight.utils.reproducibility import seed_everything

plt.switch_backend("Agg")  # headless — no display required

logger = logging.getLogger(__name__)

# Cap on subplots in the summary grid (logged when exceeded — never silent).
_MAX_GRID = 24


def _load_encoder(checkpoint_path: str) -> TemporalPatternEncoder:
    """Reconstruct a TemporalPatternEncoder from a Task 2 checkpoint."""
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    cfg = ckpt.get("model_config", {})
    encoder = TemporalPatternEncoder(
        in_channels=cfg.get("in_channels", 18),
        hidden=cfg.get("hidden", 64),
        out_dim=cfg.get("out_dim", 128),
        dropout=cfg.get("dropout", 0.3),
        kernel_size=cfg.get("kernel_size", 3),
    )
    state = ckpt.get("model_state_dict", ckpt)
    # The checkpoint stores the full classifier (encoder.* + head.*); keep encoder.*.
    encoder_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    if encoder_state:
        encoder.load_state_dict(encoder_state)
    else:
        encoder.load_state_dict(state, strict=False)
    encoder.eval()
    return encoder


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str | None):
    """Return cached deviations, computing them from data_dir if absent."""
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    if not data_dir:
        logger.error("visualize_attention | no deviations cached for %s and no --data-dir.", version)
        return None
    logger.info("visualize_attention | deviations not cached; computing from %s …", data_dir)
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        logger.error("visualize_attention | compute_baselines failed (rc=%d).", rc)
        return None
    return store.load_deviations(version)


def _attack_overlap_days(meta: dict, record, n_days: int) -> list[int]:
    """Day indices (0..n_days-1) of the window that fall inside the attack period."""
    if record is None:
        return []
    window_start = pd.Timestamp(meta["window_start"]).normalize()
    attack_start = pd.Timestamp(record.attack_start).normalize()
    attack_end = pd.Timestamp(record.attack_end).normalize()
    days = []
    for i in range(n_days):
        day = window_start + pd.Timedelta(days=i)
        if attack_start <= day <= attack_end:
            days.append(i)
    return days


def _plot_window(ax, attn: np.ndarray, meta: dict, record) -> None:
    """Draw one window's attention bar chart with the attack region shaded."""
    n_days = attn.shape[0]
    ax.bar(range(n_days), attn, color="steelblue")

    attack_days = _attack_overlap_days(meta, record, n_days)
    if attack_days:
        ax.axvspan(min(attack_days) - 0.5, max(attack_days) + 0.5,
                   color="crimson", alpha=0.2, label="attack window")
        ax.legend(loc="upper right", fontsize=6)

    start = pd.Timestamp(meta["window_start"]).date()
    end = pd.Timestamp(meta["window_end"]).date()
    ax.set_title(
        f"{meta['user_id']} | scenario {meta.get('scenario', 0)} | "
        f"{start}–{end} | overlap {meta['overlap_ratio']:.2f}",
        fontsize=8,
    )
    ax.set_xlabel("day index", fontsize=7)
    ax.set_ylabel("attention", fontsize=7)
    ax.tick_params(labelsize=6)


def _collect_windows(dataset, encoder, target_users: set[str]):
    """Yield (meta, attention_weights) for every dataset window of the target users."""
    results = []
    for i in range(len(dataset)):
        window, _label, meta = dataset[i]
        if meta["user_id"] not in target_users:
            continue
        with torch.no_grad():
            attn = encoder.get_attention_weights(window.unsqueeze(0))[0].cpu().numpy()  # shape: (T,)
        results.append((meta, attn))
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Visualise temporal-encoder attention over insider windows.',
    )
    p.add_argument('--checkpoint', required=True, metavar='PATH', help='Path to a trained .pt checkpoint')
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', required=True, metavar='PATH',
                   help='Dataset directory (answers/ labels; computes deviations if uncached)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--output-dir', default='attention_plots', metavar='PATH',
                   help="Where to save PNGs (default: 'attention_plots')")
    p.add_argument('--user', metavar='USER_ID', help='Plot only this user (default: all insiders)')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    seed_everything(42)

    encoder = _load_encoder(args.checkpoint)

    store = FeatureStore(args.store_dir)
    deviations = _load_deviations(store, args.version, args.store_dir, args.data_dir)
    if deviations is None:
        return 1

    answers_dir = Path(args.data_dir) / 'answers'
    if not answers_dir.exists():
        logger.error("visualize_attention | no answers/ directory at %s.", answers_dir)
        return 1
    attack_windows = {r.user_id: r for r in load_insiders(answers_dir, args.version)}

    if args.user:
        target_users = {args.user}
    else:
        target_users = set(attack_windows.keys())
    if not target_users:
        logger.warning("visualize_attention | no target users (no insiders / no --user); nothing to plot.")
        return 0

    dataset = DeviationWindowDataset(deviations, attack_windows)
    windows = _collect_windows(dataset, encoder, target_users)
    if not windows:
        logger.warning("visualize_attention | no windows found for %s.", sorted(target_users))
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-window figures.
    for meta, attn in windows:
        fig, ax = plt.subplots(figsize=(8, 3))
        _plot_window(ax, attn, meta, attack_windows.get(meta["user_id"]))
        fig.tight_layout()
        start = pd.Timestamp(meta["window_start"]).date()
        fig.savefig(output_dir / f"attention_{meta['user_id']}_{start}.png", dpi=150)
        plt.close(fig)
    logger.info("visualize_attention | saved %d per-window plots to %s.", len(windows), output_dir)

    # Summary grid.
    shown = windows[:_MAX_GRID]
    if len(windows) > _MAX_GRID:
        logger.info("visualize_attention | summary grid capped at %d of %d windows.",
                    _MAX_GRID, len(windows))
    ncols = min(4, len(shown))
    nrows = ceil(len(shown) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for (meta, attn), ax in zip(shown, axes.flat):
        ax.axis("on")
        _plot_window(ax, attn, meta, attack_windows.get(meta["user_id"]))
    fig.suptitle(f"Temporal attention — {args.version}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    grid_path = output_dir / f"attention_summary_{args.version}.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    logger.info("visualize_attention | saved summary grid to %s.", grid_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
