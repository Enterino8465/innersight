#!/usr/bin/env python
"""Temporal + Graph chained model — step 3 of the progressive ladder (Phase 5).

Chains Module 2 and Module 3: for each 28-day window, the TemporalPatternEncoder
turns a user's deviation window into a 128-d embedding, that embedding is placed
into the windowed graph's user node feature matrix (via the graph's
``user_to_idx``), the GraphContextEncoder enriches it with neighbourhood context
(PCs, URLs, files, other users), and a linear head scores each user.

Alignment: deviation windows and graphs share the same ``[window_start,
window_end]`` period; samples are grouped by period so one graph serves all of a
period's user windows.

Inductive split: validation users are filtered OUT of the *training* graphs (no
node, no edges) so the model can't pass messages through held-out users; at eval
the full-log graph (with those users restored) is used.

Usage:
    python -m innersight.scripts.train_temporal_graph \
        --version r4.2 --store-dir feature_store --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from innersight.config import setup_logging, BUSINESS_HOURS_START, BUSINESS_HOURS_END
from innersight.data.answers import load_insiders
from innersight.data.feature_store import FeatureStore
from innersight.data.pipeline import load_version
from innersight.models.dataset import DeviationWindowDataset
from innersight.models.graph_builder import build_windowed_graph
from innersight.models.graph_encoder import GraphContextEncoder
from innersight.models.losses import FocalLoss
from innersight.models.temporal_encoder import TemporalPatternEncoder
from innersight.scripts import compute_baselines
from innersight.scripts.train_temporal import _DEFAULT_TRAINING, _build_scheduler, _resolve_device
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

_DEVICE = torch.device("cpu")  # default; overridden by --device in main()

_DEFAULT_TEMPORAL = {"in_channels": 18, "hidden": 64, "out_dim": 128, "dropout": 0.3, "kernel_size": 3}
_DEFAULT_GRAPH = {"hidden_dim": 128, "num_layers": 2, "heads": 4, "dropout": 0.3}
_DEFAULT_EVAL = {"n_folds": 5, "seeds": [42, 123, 456]}


class ChainedTemporalGraph(nn.Module):
    """TemporalPatternEncoder → inject into graph → GraphContextEncoder → head."""

    def __init__(self, temporal_cfg: dict, graph_cfg: dict, metadata) -> None:
        super().__init__()
        self.temporal = TemporalPatternEncoder(
            in_channels=temporal_cfg["in_channels"], hidden=temporal_cfg["hidden"],
            out_dim=temporal_cfg["out_dim"], dropout=temporal_cfg["dropout"],
            kernel_size=temporal_cfg["kernel_size"],
        )
        self.graph = GraphContextEncoder(
            metadata, hidden_dim=graph_cfg["hidden_dim"], num_layers=graph_cfg["num_layers"],
            heads=graph_cfg["heads"], dropout=graph_cfg["dropout"],
        )
        self.head = nn.Linear(graph_cfg["hidden_dim"], 1)


def _chained_logits(model: ChainedTemporalGraph, windows: torch.Tensor,
                    sample_user_ids, graph) -> torch.Tensor:
    """Score a period's window samples through the temporal→graph→head chain.

    Args:
        model: The chained model.
        windows: ``(n_samples, 18, T)`` deviation windows for this period.
        sample_user_ids: Length-``n_samples`` user ids aligned with ``windows``.
        graph: The period's :class:`HeteroData` (carries ``user_to_idx``).

    Returns:
        ``(n_samples, 1)`` logits in the order of ``windows``.
    """
    windows = windows.to(_DEVICE)
    graph = graph.to(_DEVICE)  # HeteroData → move node/edge tensors to the model's device
    temporal_emb = model.temporal(windows)                       # shape: (n_samples, 128)
    user_map = getattr(graph, "user_to_idx", {})
    n_user_nodes = graph["user"].x.shape[0]

    in_graph = [i for i, u in enumerate(sample_user_ids) if u in user_map]
    if n_user_nodes == 0 or not in_graph:
        # No graph context available — fall back to the temporal embedding.
        return model.head(temporal_emb)                          # shape: (n_samples, 1)

    rows = torch.tensor([user_map[sample_user_ids[i]] for i in in_graph],
                        dtype=torch.long, device=_DEVICE)
    # Inject the temporal embeddings into the user node matrix (grad flows to src).
    user_x = torch.zeros(n_user_nodes, temporal_emb.shape[1],
                         dtype=temporal_emb.dtype, device=temporal_emb.device)
    user_x = user_x.index_copy(0, rows, temporal_emb[in_graph])  # shape: (n_user_nodes, 128)

    x_dict = dict(graph.x_dict)
    x_dict["user"] = user_x
    enriched = model.graph(x_dict, graph.edge_index_dict, graph.edge_attr_dict)["user"]  # (N, 128)

    # Per-sample embedding: enriched for in-graph users, temporal-only otherwise.
    emb = temporal_emb.clone()
    emb = emb.index_copy(0, torch.tensor(in_graph, dtype=torch.long, device=_DEVICE), enriched[rows])
    return model.head(emb)                                        # shape: (n_samples, 1)


# ── Data assembly ─────────────────────────────────────────────────────────────

def _build_window_registry(dataset):
    """Return (windows (n,18,T) tensor, user_ids, period_keys, y, metas)."""
    windows, user_ids, periods, labels, metas = [], [], [], [], []
    for i in range(len(dataset)):
        window, label, meta = dataset[i]
        windows.append(window)
        user_ids.append(str(meta["user_id"]))
        periods.append((meta["window_start"], meta["window_end"]))
        labels.append(float(label.reshape(-1)[0]))
        metas.append(meta)
    if not windows:
        return torch.empty(0), np.array([]), [], np.empty((0,)), metas
    return (torch.stack(windows), np.array(user_ids, dtype=object), periods,
            np.asarray(labels, dtype=float), metas)


def _filter_logs(logs: dict, exclude_users: set[str]) -> dict:
    """Drop rows authored by excluded users (inductive removal from train graph)."""
    if not exclude_users:
        return logs
    out = {}
    for name, df in logs.items():
        if df is not None and 'user' in df.columns:
            out[name] = df[~df['user'].astype(str).isin(exclude_users)]
        else:
            out[name] = df
    return out


def _presort_logs(logs: dict) -> dict:
    """Sort every log DataFrame by date once so _win_slice can use searchsorted.

    This is a one-time O(n log n) sort per log type that replaces 32 000+
    O(n) boolean-mask scans (one per period × slice call inside
    build_windowed_graph). On a 28 M-row HTTP log the speedup is ~100-1000×
    for the graph-building phase.
    """
    import pandas as _pd
    from innersight.models.graph_builder import _extract_domain
    out = {}
    for name, df in logs.items():
        if df is not None and 'date' in df.columns:
            df = df.copy()
            if not _pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = _pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df._date_sorted = True  # signal to _win_prepare that sort is done

            # Precompute datetime flags to avoid dt accessors in the loop
            h = df['date'].dt.hour
            df['_hour'] = h.astype(float)
            df['_after'] = ((h < BUSINESS_HOURS_START) | (h >= BUSINESS_HOURS_END)).astype(float)
            df['_weekend'] = (df['date'].dt.dayofweek >= 5).astype(float)
            df['_day'] = df['date'].dt.normalize()

            # Precompute domain for http
            if name == 'http' and 'url' in df.columns:
                df['_domain'] = df['url'].apply(_extract_domain)

            # Precompute removable flag for file
            if name == 'file':
                if 'to_removable_media' in df.columns:
                    rem = df['to_removable_media']
                    if rem.dtype == object:
                        rem = rem.astype(str).str.lower().isin(('true', '1'))
                    else:
                        rem = rem.astype(bool)
                    df['_torem'] = rem.astype(float)
                else:
                    df['_torem'] = 1.0

            out[name] = df
        else:
            out[name] = df
    logger.info('train_temporal_graph | logs pre-sorted and flags precomputed.')
    return out


def _build_period_graphs(logs: dict, periods, exclude_users: set[str] | None = None,
                         max_url_nodes: int | None = None,
                         max_file_nodes: int | None = None) -> dict:
    """Build one windowed HeteroData per unique period (optionally excluding users) in parallel.

    ``max_url_nodes`` / ``max_file_nodes`` cap the URL/file node count per window
    to bound peak memory (see :func:`build_windowed_graph`).
    """
    used_logs = _filter_logs(logs, exclude_users) if exclude_users else logs
    # Pre-sort once by date and precompute flags
    used_logs = _presort_logs(used_logs)

    unique_periods = sorted(list(set(periods)))
    logger.info('train_temporal_graph | building %d period graphs%s …',
                len(unique_periods),
                f' (excluding {len(exclude_users)} val users)' if exclude_users else '')

    from concurrent.futures import ProcessPoolExecutor
    import os

    # Use parallel processing since the server has many CPU cores.
    # Cap at 32 workers to be friendly to other system processes.
    num_workers = min(32, os.cpu_count() or 1)
    logger.info('train_temporal_graph | using %d parallel workers for graph building.', num_workers)

    graphs = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                build_windowed_graph,
                used_logs, ws, we,
                max_url_nodes=max_url_nodes,
                max_file_nodes=max_file_nodes
            ): (ws, we)
            for (ws, we) in unique_periods
        }

        count = 0
        for future in futures:
            period = futures[future]
            graphs[period] = future.result()
            count += 1
            if count % 100 == 0:
                logger.info('train_temporal_graph | built %d/%d period graphs.',
                            count, len(unique_periods))
    return graphs


def _group_positions_by_period(positions, period_keys) -> dict:
    grouped = defaultdict(list)
    for pos in positions:
        grouped[period_keys[pos]].append(int(pos))
    return grouped


# ── Per-fold train / predict ──────────────────────────────────────────────────

def _predict(model, positions, registry, graphs) -> np.ndarray:
    """Probabilities for *positions*, returned in that exact order."""
    windows_t, user_ids, period_keys = registry["windows"], registry["user_ids"], registry["periods"]
    probs = np.zeros(len(positions), dtype=float)
    local = {int(p): k for k, p in enumerate(positions)}
    model.eval()
    with torch.no_grad():
        for period, pos in _group_positions_by_period(positions, period_keys).items():
            graph = graphs.get(period)
            if graph is None:
                continue
            logits = _chained_logits(model, windows_t[pos], list(user_ids[pos]), graph)
            batch_probs = torch.sigmoid(logits).reshape(-1).cpu().numpy()
            for k, pos_i in enumerate(pos):
                probs[local[pos_i]] = batch_probs[k]
    return probs


def _inner_user_split(train_pos, user_ids, y, frac: float = 0.2, seed: int = 0):
    """Hold out ~``frac`` of the TRAIN users (stratified by insider status) as an
    inner monitor set for early stopping.

    Selecting the epoch on this inner holdout — never on the outer validation
    fold — keeps the reported CV metric free of model-selection leakage. The
    split is user-level so the same user's windows never straddle the inner
    train/monitor boundary.

    Returns:
        ``(inner_train_pos, inner_val_pos)`` integer arrays (subsets of ``train_pos``).
    """
    train_pos = np.asarray(train_pos, dtype=int)
    y = np.asarray(y)
    users = np.array([user_ids[p] for p in train_pos], dtype=object)
    uniq = list(dict.fromkeys(users.tolist()))  # stable unique order
    pos_u = [u for u in uniq if y[train_pos[users == u]].max() > 0]
    pos_set = set(pos_u)
    neg_u = [u for u in uniq if u not in pos_set]
    rng = np.random.default_rng(seed)
    rng.shuffle(pos_u)
    rng.shuffle(neg_u)
    n_pos_val = int(round(len(pos_u) * frac)) if len(pos_u) >= 2 else 0
    n_neg_val = int(round(len(neg_u) * frac)) if len(neg_u) >= 2 else 0
    val_users = set(pos_u[:n_pos_val]) | set(neg_u[:n_neg_val])
    in_val = np.array([u in val_users for u in users], dtype=bool)
    return train_pos[~in_val], train_pos[in_val]


def _fit_fold(train_pos, val_pos, seed, *, registry, logs, full_graphs,
              temporal_cfg, graph_cfg, train_cfg, metadata,
              max_url_nodes=None, max_file_nodes=None):
    """Train the chained model on one fold; return ``(val_probs, model)``.

    Leakage-safe model selection: early stopping uses an INNER user-level holdout
    carved from the training fold and scored on the (val-user-excluded) training
    graphs. The outer ``val_pos`` fold is scored exactly once, at the end, on the
    full-log graphs, and never influences which epoch is chosen.
    """
    seed_everything(seed)
    windows_t, user_ids, period_keys, y = (
        registry["windows"], registry["user_ids"], registry["periods"], registry["y"])

    val_users = set(user_ids[val_pos].tolist())
    train_graphs = _build_period_graphs(
        logs, period_keys, exclude_users=val_users,
        max_url_nodes=max_url_nodes, max_file_nodes=max_file_nodes,
    )

    inner_train_pos, inner_val_pos = _inner_user_split(train_pos, user_ids, y, frac=0.2, seed=seed)
    early_stop = inner_val_pos.size > 0 and float(y[inner_val_pos].sum()) > 0
    if not early_stop:
        inner_train_pos = np.asarray(train_pos, dtype=int)  # too few positives to monitor

    model = ChainedTemporalGraph(temporal_cfg, graph_cfg, metadata).to(_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = _build_scheduler(optimizer, train_cfg["warmup_epochs"],
                                 train_cfg["max_epochs"], train_cfg["eta_min"])
    criterion = FocalLoss(alpha=train_cfg["focal_alpha"], gamma=train_cfg["focal_gamma"])

    train_by_period = _group_positions_by_period(inner_train_pos, period_keys)
    period_order = list(train_by_period.keys())

    best_metric, since_improve = -1.0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _epoch in range(train_cfg["max_epochs"]):
        model.train()
        generator = torch.Generator().manual_seed(seed + _epoch)
        for j in torch.randperm(len(period_order), generator=generator).tolist():
            period = period_order[j]
            pos = train_by_period[period]
            graph = train_graphs.get(period)
            if graph is None or graph["user"].x.shape[0] == 0:
                continue
            targets = torch.tensor(y[pos], dtype=torch.float32, device=_DEVICE).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = _chained_logits(model, windows_t[pos], list(user_ids[pos]), graph)
            loss = criterion(logits, targets)
            if torch.isnan(loss):
                logger.warning("train_temporal_graph | NaN loss; skipping period batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
        scheduler.step()

        if not early_stop:
            continue  # no inner monitor → train the full epoch budget, keep final weights

        # Monitor the inner holdout on the training graphs — never the outer fold.
        iv_probs = _predict(model, inner_val_pos, registry, train_graphs)
        metric = compute_metrics(iv_probs, y[inner_val_pos])["auprc"]
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= train_cfg["patience"]:
                break

    if early_stop:
        model.load_state_dict(best_state)  # else keep final-epoch weights

    # Free this fold's training graphs before scoring (keeps peak RAM bounded
    # across the 15 fold×seed fits — the per-fold graphs are no longer needed).
    del train_graphs
    gc.collect()

    val_probs = _predict(model, val_pos, registry, full_graphs)
    return val_probs, model


# ── Config / deviations ───────────────────────────────────────────────────────

def _load_config(path: str | None) -> tuple[dict, dict, dict, dict]:
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    raw = raw or {}
    model = raw.get("model", {})
    temporal_cfg = {**_DEFAULT_TEMPORAL, **model.get("temporal", {})}
    graph_cfg = {**_DEFAULT_GRAPH, **model.get("graph", {})}
    train_cfg = {**_DEFAULT_TRAINING, **raw.get("training", {})}
    eval_cfg = {**_DEFAULT_EVAL, **raw.get("evaluation", {})}
    return temporal_cfg, graph_cfg, train_cfg, eval_cfg


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str | None):
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    if not data_dir:
        logger.error("train_temporal_graph | no deviations cached for %s and no --data-dir.", version)
        return None
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        return None
    return store.load_deviations(version)


def _log_baseline_comparison(output_path: Path, chained_cv: dict) -> None:
    comparison = {"TemporalGraph": chained_cv}
    for name, fname in (("XGBoost", "xgboost_results.json"), ("MLP", "mlp_results.json"),
                        ("TemporalCNN", "temporal_results.json")):
        for candidate in (output_path.parent / fname, Path(fname)):
            if candidate.exists():
                try:
                    comparison[name] = json.loads(candidate.read_text())["cross_validation"]
                except (KeyError, ValueError):
                    pass
                break
    logger.info("train_temporal_graph | comparison:\n%s", format_results_table(comparison))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train the chained Temporal+Graph model with inductive user-level CV.',
    )
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', metavar='PATH',
                   help='Dataset directory (raw logs for graphs, answers/ labels, deviations)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--config', metavar='PATH', help='Path to a train_temporal_graph.yaml config')
    p.add_argument('--output', default='temporal_graph_results.json', metavar='PATH',
                   help='Where to write the results JSON (default: temporal_graph_results.json)')
    p.add_argument('--checkpoint-dir', default='checkpoints', metavar='PATH',
                   help="Directory for the model checkpoint (default: 'checkpoints')")
    p.add_argument('--max-url-nodes', type=int, default=2000, metavar='K',
                   help="Cap URL nodes to the K most frequent domains per window "
                        "(0 = no cap). Bounds memory on large http logs. Default: 2000.")
    p.add_argument('--max-file-nodes', type=int, default=2000, metavar='K',
                   help="Cap file nodes to the K most-copied filenames per window "
                        "(0 = no cap). Default: 2000.")
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                   help="Compute device for the model: 'auto' = cuda>mps>cpu (default). "
                        "Graph construction stays on CPU regardless.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    global _DEVICE
    _DEVICE = _resolve_device(args.device)
    logger.info("train_temporal_graph | device=%s", _DEVICE)
    temporal_cfg, graph_cfg, train_cfg, eval_cfg = _load_config(args.config)
    seeds = [int(s) for s in eval_cfg["seeds"]]
    n_folds = int(eval_cfg["n_folds"])
    seed_everything(seeds[0] if seeds else 42)

    if not args.data_dir:
        logger.error("train_temporal_graph | --data-dir is required (raw logs build the graphs).")
        return 1

    store = FeatureStore(args.store_dir)
    deviations = _load_deviations(store, args.version, args.store_dir, args.data_dir)
    if deviations is None:
        return 1

    answers_dir = Path(args.data_dir) / 'answers'
    if not answers_dir.exists():
        logger.error("train_temporal_graph | no answers/ directory at %s.", answers_dir)
        return 1
    attack_windows = {r.user_id: r for r in load_insiders(answers_dir, args.version)}

    logger.info("train_temporal_graph | loading raw logs for graph construction …")
    dataset_obj = load_version(args.data_dir, args.version)
    logs = dataset_obj.logs

    win_dataset = DeviationWindowDataset(deviations, attack_windows)
    windows_t, user_ids, period_keys, y, metas = _build_window_registry(win_dataset)
    n_pos = int(y.sum())
    logger.info("train_temporal_graph | %d windows over %d periods (%d positive).",
                len(y), len(set(period_keys)), n_pos)
    if len(y) == 0 or n_pos == 0:
        logger.error("train_temporal_graph | need labelled positive windows; aborting.")
        return 1

    max_url_nodes = args.max_url_nodes if args.max_url_nodes and args.max_url_nodes > 0 else None
    max_file_nodes = args.max_file_nodes if args.max_file_nodes and args.max_file_nodes > 0 else None
    logger.info("train_temporal_graph | node caps: max_url_nodes=%s max_file_nodes=%s",
                max_url_nodes, max_file_nodes)

    # Full-log graphs (used for evaluation and to derive a consistent metadata).
    full_graphs = _build_period_graphs(logs, period_keys, exclude_users=None,
                                       max_url_nodes=max_url_nodes, max_file_nodes=max_file_nodes)
    metadata = next(iter(full_graphs.values())).metadata()

    registry = {"windows": windows_t, "user_ids": user_ids, "periods": period_keys, "y": y}

    def fit(train_pos, val_pos, seed):
        val_probs, _ = _fit_fold(train_pos, val_pos, seed, registry=registry, logs=logs,
                                 full_graphs=full_graphs, temporal_cfg=temporal_cfg,
                                 graph_cfg=graph_cfg, train_cfg=train_cfg, metadata=metadata,
                                 max_url_nodes=max_url_nodes, max_file_nodes=max_file_nodes)
        return val_probs

    # Index registry trick: X carries original positions so the harness's slicing
    # gives the model_fn the original sample indices it needs for graph lookup.
    positions = np.arange(len(y)).reshape(-1, 1)

    def model_fn(X_train, _y_train, X_val, _y_val, seed):
        return fit(X_train.ravel().astype(int), X_val.ravel().astype(int), seed)

    cv = run_cross_validation(model_fn, positions, y, metas, attack_windows,
                              n_folds=n_folds, seeds=seeds)
    logger.info("train_temporal_graph | CV AUPRC=%.4f ± %.4f | P@10=%.3f | F1=%.3f",
                cv['mean']['auprc'], cv['std']['auprc'], cv['mean']['p_at_10'], cv['mean']['f1_best'])

    # Out-of-fold predictions for per-scenario metrics + detection latency.
    scenarios = np.array([m.get('scenario', 0) for m in metas])
    oof = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in temporal_stratified_kfold(user_ids, y, n_folds=n_folds, seed=seeds[0]):
        oof[val_idx] = fit(train_idx, val_idx, seeds[0])
    scenario_metrics = per_scenario_metrics(oof, y, scenarios)
    threshold = compute_metrics(oof, y)['threshold_best']
    latency = detection_latency(oof, metas, threshold, attack_windows)
    logger.info("train_temporal_graph | detection: %d/%d insiders flagged, median latency=%s days.",
                latency['detected_count'], latency['total_insiders'], latency['median_days'])

    # Final model on all data → checkpoint. (_fit_fold returns the trained model;
    # the previous version saved a freshly-initialised, untrained model by mistake.)
    all_pos = np.arange(len(y))
    _, final = _fit_fold(all_pos, all_pos, seeds[0], registry=registry, logs=logs,
                         full_graphs=full_graphs, temporal_cfg=temporal_cfg, graph_cfg=graph_cfg,
                         train_cfg=train_cfg, metadata=metadata,
                         max_url_nodes=max_url_nodes, max_file_nodes=max_file_nodes)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"temporal_graph_{args.version}.pt"
    torch.save({"model_state_dict": final.state_dict(),
                "config": {"temporal": temporal_cfg, "graph": graph_cfg}}, ckpt_path)

    results = {
        "version": args.version,
        "model": "temporal_graph",
        "n_windows": int(len(y)),
        "n_positive": n_pos,
        "n_folds": n_folds,
        "seeds": seeds,
        "config": {"temporal": temporal_cfg, "graph": graph_cfg, "training": train_cfg},
        "cross_validation": cv,
        "per_scenario": scenario_metrics,
        "detection_latency": latency,
        "oof_threshold": threshold,
        "checkpoint": str(ckpt_path),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("train_temporal_graph | results written to %s.", output_path)
    _log_baseline_comparison(output_path, cv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
