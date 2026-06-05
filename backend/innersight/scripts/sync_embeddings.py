#!/usr/bin/env python
"""Extract per-user embeddings from a checkpoint and sync them to Qdrant (Phase 5).

Runs the trained temporal encoder over every deviation window, averages the
per-window embeddings into one vector per user, and upserts them into Qdrant via
:class:`SuspectFinder` for similarity-based suspect discovery.

Usage:
    python -m innersight.scripts.sync_embeddings \
        --checkpoint checkpoints/temporal_cnn_r4.2.pt --version r4.2 --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from innersight.config import setup_logging
from innersight.data.answers import load_insiders
from innersight.data.feature_store import FeatureStore
from innersight.models.dataset import DeviationWindowDataset
from innersight.models.temporal_encoder import TemporalPatternEncoder
from innersight.scoring.suspect_discovery import SuspectFinder
from innersight.scripts import compute_baselines
from innersight.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cpu")


def _load_temporal_encoder(checkpoint_path: str) -> TemporalPatternEncoder:
    """Reconstruct the temporal encoder from a temporal_cnn or temporal_graph checkpoint."""
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

    cfg = ckpt.get("model_config")
    if cfg is None:
        cfg = ckpt.get("config", {}).get("temporal", {})
    encoder = TemporalPatternEncoder(
        in_channels=cfg.get("in_channels", 18), hidden=cfg.get("hidden", 64),
        out_dim=cfg.get("out_dim", 128), dropout=cfg.get("dropout", 0.3),
        kernel_size=cfg.get("kernel_size", 3),
    )
    state = ckpt.get("model_state_dict", ckpt)
    encoder_state: dict = {}
    for prefix in ("temporal.", "encoder."):  # temporal_graph uses temporal.*, temporal_cnn encoder.*
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if sub:
            encoder_state = sub
            break
    if encoder_state:
        encoder.load_state_dict(encoder_state)
    else:
        encoder.load_state_dict(state, strict=False)
    encoder.eval()
    return encoder


def _per_user_embeddings(encoder, dataset, attack_windows):
    """Average each user's window embeddings into one (user_ids, embeddings, metadata)."""
    if len(dataset) == 0:
        return [], np.empty((0, 0)), []
    windows = torch.stack([dataset[i][0] for i in range(len(dataset))])     # (n, 18, 28)
    user_ids = [str(dataset[i][2]["user_id"]) for i in range(len(dataset))]

    chunks = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], 512):
            chunks.append(encoder(windows[start:start + 512].to(_DEVICE)).cpu().numpy())
    embs = np.concatenate(chunks, axis=0)                                   # (n, 128)

    by_user: dict[str, list[np.ndarray]] = defaultdict(list)
    for emb, uid in zip(embs, user_ids):
        by_user[uid].append(emb)

    ordered = sorted(by_user)
    mean_embeddings = np.stack([np.mean(by_user[u], axis=0) for u in ordered])  # (U, 128)
    metadata = [
        {"scenario": attack_windows[u].scenario if u in attack_windows else 0,
         "score": 0.0, "department": ""}
        for u in ordered
    ]
    return ordered, mean_embeddings, metadata


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str | None):
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    if not data_dir:
        logger.error("sync_embeddings | no deviations cached for %s and no --data-dir.", version)
        return None
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        return None
    return store.load_deviations(version)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Extract per-user embeddings from a checkpoint and sync them to Qdrant.',
    )
    p.add_argument('--checkpoint', required=True, metavar='PATH', help='Trained .pt checkpoint')
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', metavar='PATH',
                   help='Dataset directory (answers/ for scenarios; computes deviations if uncached)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--qdrant-url', default='http://localhost:6333', metavar='URL',
                   help="Qdrant server URL (default: 'http://localhost:6333')")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    seed_everything(42)

    finder = SuspectFinder(qdrant_url=args.qdrant_url)
    if not finder.health_check():
        logger.error("sync_embeddings | Qdrant unreachable at %s; aborting.", args.qdrant_url)
        return 1

    store = FeatureStore(args.store_dir)
    deviations = _load_deviations(store, args.version, args.store_dir, args.data_dir)
    if deviations is None:
        return 1

    attack_windows = {}
    if args.data_dir:
        answers_dir = Path(args.data_dir) / 'answers'
        if answers_dir.exists():
            attack_windows = {r.user_id: r for r in load_insiders(answers_dir, args.version)}

    encoder = _load_temporal_encoder(args.checkpoint)
    dataset = DeviationWindowDataset(deviations, attack_windows)
    user_ids, embeddings, metadata = _per_user_embeddings(encoder, dataset, attack_windows)
    if not user_ids:
        logger.error("sync_embeddings | no windows produced; nothing to sync.")
        return 1

    n_synced = finder.sync_embeddings(embeddings, user_ids, metadata, args.version)
    logger.info("sync_embeddings | synced %d/%d user embeddings to Qdrant.", n_synced, len(user_ids))
    return 0 if n_synced > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
