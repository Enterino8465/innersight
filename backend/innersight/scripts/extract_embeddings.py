#!/usr/bin/env python
"""Extract per-user fusion embeddings from an InsiderThreatDetector checkpoint (Phase 6).

Runs the full fusion model's ``get_embeddings`` over every deviation window,
averaging each user's windows into one 272-d embedding (temporal 128 + graph 128
+ static 16). These embeddings are saved as a ``.pt`` file and can optionally be
synced straight to Qdrant for k-NN suspect discovery.

(Replaces the legacy GraphSAGE embedding extractor.)

Usage:
    python -m innersight.scripts.extract_embeddings \
        --checkpoint checkpoints/fusion_r4.2.pt \
        --version r4.2 --data-dir /data/cert_r4.2 \
        --output embeddings_r4.2.pt --sync
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
from innersight.data.pipeline import load_version
from innersight.models.dataset import DeviationWindowDataset
from innersight.models.fusion_model import InsiderThreatDetector
from innersight.scoring.suspect_discovery import SuspectFinder
from innersight.scripts import compute_baselines
from innersight.scripts.train_fusion import _build_id_vocab, _build_ocean_map, _static_arrays
from innersight.scripts.train_temporal_graph import (
    _build_period_graphs,
    _build_window_registry,
    _group_positions_by_period,
)
from innersight.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cpu")
_EMBED_DIM = 272


def _fused_vector(model, windows, sample_user_ids, graph, ocean, roles, depts) -> torch.Tensor:
    """The fusion model's ``get_embeddings`` (272-d) with per-sample/graph alignment.

    Mirrors ``train_fusion._fusion_logits`` but returns the fused vector before
    the classification head (what gets stored in Qdrant).
    """
    temporal_emb = model.temporal(windows)                       # shape: (n, 128)
    user_map = getattr(graph, "user_to_idx", {})
    n_user_nodes = graph["user"].x.shape[0]

    graph_emb = torch.zeros_like(temporal_emb)                   # shape: (n, 128)
    in_graph = [i for i, u in enumerate(sample_user_ids) if u in user_map]
    if n_user_nodes > 0 and in_graph:
        rows = torch.tensor([user_map[sample_user_ids[i]] for i in in_graph], dtype=torch.long)
        user_x = torch.zeros(n_user_nodes, temporal_emb.shape[1],
                             dtype=temporal_emb.dtype, device=temporal_emb.device)
        user_x = user_x.index_copy(0, rows, temporal_emb[in_graph])
        enriched = model.graph({**graph.x_dict, "user": user_x},
                               graph.edge_index_dict, graph.edge_attr_dict)["user"]
        graph_emb = graph_emb.index_copy(0, torch.tensor(in_graph, dtype=torch.long), enriched[rows])

    static = torch.cat([ocean, model.role_emb(roles), model.dept_emb(depts)], dim=1)  # (n, 16)
    return torch.cat([temporal_emb, graph_emb, static], dim=1)   # shape: (n, 272)


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str):
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        return None
    return store.load_deviations(version)


def _per_user_embeddings(model, registry, period_graphs):
    """Average each user's per-window fused embeddings into one (272,) vector."""
    windows_t, user_ids, period_keys = registry["windows"], registry["user_ids"], registry["periods"]
    ocean, roles, depts = registry["ocean"], registry["roles"], registry["depts"]

    by_user: dict[str, list[torch.Tensor]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for period, pos in _group_positions_by_period(np.arange(len(user_ids)), period_keys).items():
            graph = period_graphs.get(period)
            if graph is None:
                continue
            fused = _fused_vector(
                model, windows_t[pos], [user_ids[i] for i in pos], graph,
                torch.tensor(ocean[pos], dtype=torch.float32),
                torch.tensor(roles[pos], dtype=torch.long),
                torch.tensor(depts[pos], dtype=torch.long),
            )
            for k, i in enumerate(pos):
                by_user[user_ids[i]].append(fused[k])

    ordered = sorted(by_user)
    embeddings = torch.stack([torch.stack(by_user[u]).mean(dim=0) for u in ordered])  # (N, 272)
    return ordered, embeddings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Extract per-user fusion embeddings and optionally sync them to Qdrant.',
    )
    p.add_argument('--checkpoint', required=True, metavar='PATH', help='Fusion model checkpoint (.pt)')
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', required=True, metavar='PATH',
                   help='Dataset directory (raw logs, LDAP, psychometric, answers/, deviations)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--output', default='embeddings.pt', metavar='PATH',
                   help='Where to write the embeddings .pt (default: embeddings.pt)')
    p.add_argument('--sync', action='store_true', help='Also sync embeddings to Qdrant')
    p.add_argument('--qdrant-url', default='http://localhost:6333', metavar='URL',
                   help="Qdrant server URL (default: 'http://localhost:6333')")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    seed_everything(42)

    if not Path(args.checkpoint).exists():
        logger.error("extract_embeddings | checkpoint not found: %s", args.checkpoint)
        return 1

    # Reconstruct the fusion model from the checkpoint config.
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    config = ckpt.get("config", {})
    num_roles = int(config.get("num_roles", 50))
    num_depts = int(config.get("num_depts", 15))

    store = FeatureStore(args.store_dir)
    deviations = _load_deviations(store, args.version, args.store_dir, args.data_dir)
    if deviations is None:
        logger.error("extract_embeddings | could not load/compute deviations for %s.", args.version)
        return 1

    answers_dir = Path(args.data_dir) / 'answers'
    attack_windows = (
        {r.user_id: r for r in load_insiders(answers_dir, args.version)} if answers_dir.exists() else {}
    )
    logger.info("extract_embeddings | loading raw logs, LDAP and psychometric …")
    dataset_obj = load_version(args.data_dir, args.version)
    logs, ldap, psych = dataset_obj.logs, dataset_obj.ldap, dataset_obj.psychometric

    win_dataset = DeviationWindowDataset(deviations, attack_windows)
    windows_t, user_ids, period_keys, _y, _metas = _build_window_registry(win_dataset)
    if len(user_ids) == 0:
        logger.error("extract_embeddings | no windows produced; nothing to extract.")
        return 1

    # Static context (same-version vocab → ids align with the trained tables).
    role_uid, _nr = _build_id_vocab(ldap, "role")
    dept_uid, _nd = _build_id_vocab(ldap, "department")
    ocean_map = _build_ocean_map(psych)
    ocean, roles, depts = _static_arrays(user_ids, ocean_map, role_uid, dept_uid)
    roles = np.clip(roles, 0, num_roles - 1)  # guard against vocab drift
    depts = np.clip(depts, 0, num_depts - 1)

    period_graphs = _build_period_graphs(logs, period_keys, exclude_users=None)
    metadata = next(iter(period_graphs.values())).metadata()

    model = InsiderThreatDetector(metadata, num_roles=num_roles, num_depts=num_depts,
                                  temporal_config=config.get("temporal") or None,
                                  graph_config=config.get("graph") or None)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(_DEVICE)

    registry = {"windows": windows_t, "user_ids": user_ids, "periods": period_keys,
                "ocean": ocean, "roles": roles, "depts": depts}
    ordered_users, embeddings = _per_user_embeddings(model, registry, period_graphs)
    logger.info("extract_embeddings | %d users → embeddings %s.", len(ordered_users), tuple(embeddings.shape))

    # ── Save .pt in the standard checkpoint format ──────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings,
        "user_ids": ordered_users,
        "version": args.version,
        "metadata": {
            "source_checkpoint": str(args.checkpoint),
            "model": "fusion",
            "embedding_dim": _EMBED_DIM,
            "n_users": len(ordered_users),
        },
    }, output_path)
    logger.info("extract_embeddings | saved embeddings to %s.", output_path)

    # ── Optional Qdrant sync ────────────────────────────────────────────────
    if args.sync:
        finder = SuspectFinder(qdrant_url=args.qdrant_url)
        if not finder.health_check():
            logger.warning("extract_embeddings | Qdrant unreachable at %s; skipped sync.", args.qdrant_url)
        else:
            dept_name = ({str(r["user_id"]): str(r["department"]) for _, r in ldap.iterrows()}
                         if ldap is not None and not ldap.empty and "department" in ldap.columns else {})
            payloads = [
                {"scenario": attack_windows[u].scenario if u in attack_windows else 0,
                 "department": dept_name.get(u, ""), "score": 0.0}
                for u in ordered_users
            ]
            n = finder.sync_embeddings(embeddings.numpy(), ordered_users, payloads, args.version)
            logger.info("extract_embeddings | synced %d/%d embeddings to Qdrant.", n, len(ordered_users))

    return 0


if __name__ == '__main__':
    sys.exit(main())
