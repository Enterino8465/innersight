#!/usr/bin/env python
"""Full fusion model trainer — step 4 / final rung of the ladder (Phase 6).

Extends the Temporal+Graph chained trainer with the static context channel:
OCEAN personality scores plus learned role and department embeddings. Per window
the temporal embedding is injected into the windowed graph, the graph refines it,
and the fused ``[temporal | graph | ocean | role | dept]`` vector is classified.

Reuses the period-aligned, inductive, user-level CV protocol from
``train_temporal_graph`` and adds the static lookups.

Usage:
    python -m innersight.scripts.train_fusion \
        --version r4.2 --store-dir feature_store --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from innersight.config import setup_logging
from innersight.data.answers import load_insiders
from innersight.data.feature_store import FeatureStore
from innersight.data.pipeline import load_version
from innersight.models.dataset import DeviationWindowDataset
from innersight.models.fusion_model import InsiderThreatDetector
from innersight.models.losses import FocalLoss
from innersight.scripts import compute_baselines
from innersight.scripts.train_temporal import _DEFAULT_TRAINING, _build_scheduler, _resolve_device
from innersight.scripts.train_temporal_graph import (
    _DEFAULT_EVAL,
    _DEFAULT_GRAPH,
    _DEFAULT_TEMPORAL,
    _build_period_graphs,
    _build_window_registry,
    _group_positions_by_period,
)
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
_OCEAN_COLS = ("O", "C", "E", "A", "N")
_UNKNOWN = 0  # reserved embedding index for unknown role/department


# ── Static-context vocabularies ───────────────────────────────────────────────

def _build_id_vocab(ldap: pd.DataFrame, field: str) -> tuple[dict[str, int], int]:
    """Map user_id → integer id for an LDAP categorical field; index 0 = unknown."""
    value_to_id = {"<unknown>": _UNKNOWN}
    user_to_id: dict[str, int] = {}
    if ldap is not None and not ldap.empty and field in ldap.columns and "user_id" in ldap.columns:
        for _, row in ldap.iterrows():
            value = row.get(field)
            if pd.isna(value):
                continue
            value = str(value)
            if value not in value_to_id:
                value_to_id[value] = len(value_to_id)
            user_to_id[str(row["user_id"])] = value_to_id[value]
    return user_to_id, len(value_to_id)


def _build_ocean_map(psychometric: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map user_id → z-scored OCEAN vector (5,); empty if psychometric absent."""
    if psychometric is None or psychometric.empty or not all(c in psychometric.columns for c in _OCEAN_COLS):
        return {}
    raw = {str(row["user_id"]): np.array([row[c] for c in _OCEAN_COLS], dtype=float)
           for _, row in psychometric.iterrows()}
    if not raw:
        return {}
    stacked = np.stack(list(raw.values()))
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-8
    return {uid: (vec - mean) / std for uid, vec in raw.items()}


def _static_arrays(user_ids, ocean_map, role_uid, dept_uid):
    """Per-sample OCEAN (n,5), role ids (n,) and dept ids (n,), with safe fallbacks."""
    ocean = np.stack([ocean_map.get(u, np.zeros(len(_OCEAN_COLS))) for u in user_ids])
    roles = np.array([role_uid.get(u, _UNKNOWN) for u in user_ids], dtype=np.int64)
    depts = np.array([dept_uid.get(u, _UNKNOWN) for u in user_ids], dtype=np.int64)
    return ocean.astype(np.float32), roles, depts


# ── Fusion forward with per-sample alignment ──────────────────────────────────

def _fusion_logits(model, windows, sample_user_ids, graph, ocean, roles, depts) -> torch.Tensor:
    """Score a period's samples through the fusion model with sample/graph alignment.

    Mirrors ``train_temporal_graph._chained_logits`` (temporal → inject → graph)
    and adds the static context, using the model's submodules so users absent
    from the graph get a temporal-only (zero graph part) embedding.
    """
    windows = windows.to(_DEVICE)
    graph = graph.to(_DEVICE)  # HeteroData → model's device
    temporal_emb = model.temporal(windows)                       # shape: (n, 128)
    user_map = getattr(graph, "user_to_idx", {})
    n_user_nodes = graph["user"].x.shape[0]

    graph_emb = torch.zeros_like(temporal_emb)                   # shape: (n, 128)
    in_graph = [i for i, u in enumerate(sample_user_ids) if u in user_map]
    if n_user_nodes > 0 and in_graph:
        rows = torch.tensor([user_map[sample_user_ids[i]] for i in in_graph],
                            dtype=torch.long, device=_DEVICE)
        user_x = torch.zeros(n_user_nodes, temporal_emb.shape[1],
                             dtype=temporal_emb.dtype, device=temporal_emb.device)
        user_x = user_x.index_copy(0, rows, temporal_emb[in_graph])
        enriched = model.graph({**graph.x_dict, "user": user_x},
                               graph.edge_index_dict, graph.edge_attr_dict)["user"]
        graph_emb = graph_emb.index_copy(
            0, torch.tensor(in_graph, dtype=torch.long, device=_DEVICE), enriched[rows])

    static = torch.cat([ocean, model.role_emb(roles), model.dept_emb(depts)], dim=1)  # (n, 16)
    fused = torch.cat([temporal_emb, graph_emb, static], dim=1)  # shape: (n, 272)
    return model.head(fused)                                     # shape: (n, 1)


def _slice_static(registry, positions):
    return (
        torch.tensor(registry["ocean"][positions], dtype=torch.float32, device=_DEVICE),
        torch.tensor(registry["roles"][positions], dtype=torch.long, device=_DEVICE),
        torch.tensor(registry["depts"][positions], dtype=torch.long, device=_DEVICE),
    )


def _predict(model, positions, registry, graphs) -> np.ndarray:
    """Probabilities for *positions* in that order."""
    windows_t, user_ids, period_keys = registry["windows"], registry["user_ids"], registry["periods"]
    probs = np.zeros(len(positions), dtype=float)
    local = {int(p): k for k, p in enumerate(positions)}
    model.eval()
    with torch.no_grad():
        for period, pos in _group_positions_by_period(positions, period_keys).items():
            graph = graphs.get(period)
            if graph is None:
                continue
            ocean, roles, depts = _slice_static(registry, pos)
            logits = _fusion_logits(model, windows_t[pos], list(user_ids[pos]), graph, ocean, roles, depts)
            batch = torch.sigmoid(logits).reshape(-1).cpu().numpy()
            for k, pos_i in enumerate(pos):
                probs[local[pos_i]] = batch[k]
    return probs


def _build_model(metadata, num_roles, num_depts, temporal_cfg, graph_cfg, head_dropout):
    return InsiderThreatDetector(
        metadata, num_roles=num_roles, num_depts=num_depts,
        temporal_config=temporal_cfg, graph_config=graph_cfg, head_dropout=head_dropout,
    ).to(_DEVICE)


def _fit_fold(train_pos, val_pos, seed, *, registry, logs, full_graphs,
              temporal_cfg, graph_cfg, head_dropout, train_cfg, metadata, num_roles, num_depts,
              max_url_nodes=None, max_file_nodes=None, cache_dir=None):
    """Train the fusion model on one fold; return val probabilities (val_pos order)."""
    seed_everything(seed)
    windows_t, user_ids, period_keys, y = (
        registry["windows"], registry["user_ids"], registry["periods"], registry["y"])

    val_users = set(user_ids[val_pos].tolist())
    train_graphs = _build_period_graphs(
        logs, period_keys, exclude_users=val_users,
        max_url_nodes=max_url_nodes, max_file_nodes=max_file_nodes,
        cache_dir=cache_dir,
    )

    model = _build_model(metadata, num_roles, num_depts, temporal_cfg, graph_cfg, head_dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = _build_scheduler(optimizer, train_cfg["warmup_epochs"],
                                 train_cfg["max_epochs"], train_cfg["eta_min"])
    criterion = FocalLoss(alpha=train_cfg["focal_alpha"], gamma=train_cfg["focal_gamma"])

    train_by_period = _group_positions_by_period(train_pos, period_keys)
    period_order = list(train_by_period.keys())

    best_auprc, best_state, since_improve = -1.0, None, 0
    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        for j in torch.randperm(len(period_order), generator=generator).tolist():
            period = period_order[j]
            pos = train_by_period[period]
            graph = train_graphs.get(period)
            if graph is None or graph["user"].x.shape[0] == 0:
                continue
            ocean, roles, depts = _slice_static(registry, pos)
            targets = torch.tensor(y[pos], dtype=torch.float32, device=_DEVICE).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = _fusion_logits(model, windows_t[pos], list(user_ids[pos]), graph, ocean, roles, depts)
            loss = criterion(logits, targets)
            if torch.isnan(loss):
                logger.warning("train_fusion | NaN loss; skipping period batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
        scheduler.step()

        val_probs = _predict(model, val_pos, registry, full_graphs)
        auprc = compute_metrics(val_probs, y[val_pos])["auprc"]
        if auprc > best_auprc:
            best_auprc = auprc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= train_cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return _predict(model, val_pos, registry, full_graphs), model


# ── Config / IO ───────────────────────────────────────────────────────────────

def _load_config(path: str | None):
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    raw = raw or {}
    model = raw.get("model", {})
    temporal_cfg = {**_DEFAULT_TEMPORAL, **model.get("temporal", {})}
    graph_cfg = {**_DEFAULT_GRAPH, **model.get("graph", {})}
    head_dropout = model.get("head", {}).get("dropout", 0.4)
    train_cfg = {**_DEFAULT_TRAINING, **raw.get("training", {})}
    eval_cfg = {**_DEFAULT_EVAL, **raw.get("evaluation", {})}
    return temporal_cfg, graph_cfg, head_dropout, train_cfg, eval_cfg


def _load_deviations(store: FeatureStore, version: str, store_dir: str, data_dir: str | None):
    deviations = store.load_deviations(version)
    if deviations is not None:
        return deviations
    if not data_dir:
        logger.error("train_fusion | no deviations cached for %s and no --data-dir.", version)
        return None
    rc = compute_baselines.main(['--version', version, '--data-dir', data_dir, '--store-dir', store_dir])
    if rc != 0:
        return None
    return store.load_deviations(version)


def _load_baseline(results_dir: str, fname: str) -> dict | None:
    path = Path(results_dir) / fname
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["cross_validation"]
    except (KeyError, ValueError):
        return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train the full fusion model (temporal + graph + static context) with user-level CV.',
    )
    p.add_argument('--version', required=True, help="CERT version string, e.g. 'r4.2'")
    p.add_argument('--data-dir', metavar='PATH',
                   help='Dataset directory (raw logs, LDAP, psychometric, answers/, deviations)')
    p.add_argument('--store-dir', default='feature_store', metavar='PATH',
                   help="Feature store directory (default: 'feature_store')")
    p.add_argument('--config', metavar='PATH', help='Path to a train_fusion.yaml config')
    p.add_argument('--output', default='fusion_results.json', metavar='PATH',
                   help='Where to write the results JSON (default: fusion_results.json)')
    p.add_argument('--checkpoint-dir', default='checkpoints', metavar='PATH',
                   help="Directory for the model checkpoint (default: 'checkpoints')")
    p.add_argument('--baseline-results-dir', default='.', metavar='PATH',
                   help="Directory holding prior-phase result JSONs (default: '.')")
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                   help="Compute device for the model: 'auto' = cuda>mps>cpu (default). "
                        "Graph construction stays on CPU regardless.")
    p.add_argument('--max-url-nodes', type=int, default=2000, metavar='K',
                   help="Cap URL nodes to the K most frequent domains per window "
                        "(0 = no cap). Default: 2000.")
    p.add_argument('--max-file-nodes', type=int, default=2000, metavar='K',
                   help="Cap file nodes to the K most-copied filenames per window "
                        "(0 = no cap). Default: 2000.")
    p.add_argument('--graph-cache-dir', default=None, metavar='PATH',
                   help='Directory to cache built period graphs to disk.')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    global _DEVICE
    _DEVICE = _resolve_device(args.device)
    logger.info("train_fusion | device=%s", _DEVICE)
    temporal_cfg, graph_cfg, head_dropout, train_cfg, eval_cfg = _load_config(args.config)
    seeds = [int(s) for s in eval_cfg["seeds"]]
    n_folds = int(eval_cfg["n_folds"])
    seed_everything(seeds[0] if seeds else 42)

    if not args.data_dir:
        logger.error("train_fusion | --data-dir is required (raw logs, LDAP, psychometric).")
        return 1

    store = FeatureStore(args.store_dir)
    deviations = _load_deviations(store, args.version, args.store_dir, args.data_dir)
    if deviations is None:
        return 1

    answers_dir = Path(args.data_dir) / 'answers'
    if not answers_dir.exists():
        logger.error("train_fusion | no answers/ directory at %s.", answers_dir)
        return 1
    attack_windows = {r.user_id: r for r in load_insiders(answers_dir, args.version)}

    logger.info("train_fusion | loading raw logs, LDAP and psychometric …")
    dataset_obj = load_version(args.data_dir, args.version)
    logs = dataset_obj.logs

    win_dataset = DeviationWindowDataset(deviations, attack_windows)
    windows_t, user_ids, period_keys, y, metas = _build_window_registry(win_dataset)
    n_pos = int(y.sum())
    if len(y) == 0 or n_pos == 0:
        logger.error("train_fusion | need labelled positive windows; aborting.")
        return 1

    # Static context vocabularies and per-sample arrays.
    role_uid, num_roles = _build_id_vocab(dataset_obj.ldap, "role")
    dept_uid, num_depts = _build_id_vocab(dataset_obj.ldap, "department")
    ocean_map = _build_ocean_map(dataset_obj.psychometric)
    ocean, roles, depts = _static_arrays(user_ids, ocean_map, role_uid, dept_uid)
    logger.info("train_fusion | %d windows, %d positive | %d roles, %d depts, OCEAN for %d users.",
                len(y), n_pos, num_roles, num_depts, len(ocean_map))

    full_graphs = _build_period_graphs(
        logs, period_keys, exclude_users=None,
        max_url_nodes=max_url_nodes, max_file_nodes=max_file_nodes,
        cache_dir=graph_cache_dir,
    )
    metadata = next(iter(full_graphs.values())).metadata()
    registry = {"windows": windows_t, "user_ids": user_ids, "periods": period_keys, "y": y,
                "ocean": ocean, "roles": roles, "depts": depts}

    def fit(train_pos, val_pos, seed):
        val_probs, _model = _fit_fold(train_pos, val_pos, seed, registry=registry, logs=logs,
                                      full_graphs=full_graphs, temporal_cfg=temporal_cfg,
                                      graph_cfg=graph_cfg, head_dropout=head_dropout,
                                      train_cfg=train_cfg, metadata=metadata,
                                      num_roles=num_roles, num_depts=num_depts,
                                      max_url_nodes=max_url_nodes, max_file_nodes=max_file_nodes,
                                      cache_dir=graph_cache_dir)
        return val_probs

    positions = np.arange(len(y)).reshape(-1, 1)

    def model_fn(X_train, _y_train, X_val, _y_val, seed):
        return fit(X_train.ravel().astype(int), X_val.ravel().astype(int), seed)

    cv = run_cross_validation(model_fn, positions, y, metas, attack_windows,
                              n_folds=n_folds, seeds=seeds)
    logger.info("train_fusion | CV AUPRC=%.4f ± %.4f | P@10=%.3f | F1=%.3f",
                cv['mean']['auprc'], cv['std']['auprc'], cv['mean']['p_at_10'], cv['mean']['f1_best'])

    # Out-of-fold predictions for per-scenario metrics + detection latency.
    scenarios = np.array([m.get('scenario', 0) for m in metas])
    oof = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in temporal_stratified_kfold(user_ids, y, n_folds=n_folds, seed=seeds[0]):
        oof[val_idx] = fit(train_idx, val_idx, seeds[0])
    scenario_metrics = per_scenario_metrics(oof, y, scenarios)
    threshold = compute_metrics(oof, y)['threshold_best']
    latency = detection_latency(oof, metas, threshold, attack_windows)

    # Final model trained on all data → checkpoint.
    all_pos = np.arange(len(y))
    _, final = _fit_fold(all_pos, np.array([], dtype=int), seeds[0], registry=registry, logs=logs,
                         full_graphs=full_graphs, temporal_cfg=temporal_cfg, graph_cfg=graph_cfg,
                         head_dropout=head_dropout, train_cfg=train_cfg, metadata=metadata,
                         num_roles=num_roles, num_depts=num_depts,
                         max_url_nodes=max_url_nodes, max_file_nodes=max_file_nodes,
                         cache_dir=graph_cache_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fusion_{args.version}.pt"
    torch.save({"model_state_dict": final.state_dict(),
                "config": {"temporal": temporal_cfg, "graph": graph_cfg,
                           "num_roles": num_roles, "num_depts": num_depts}}, ckpt_path)

    # ── Full progressive ladder (5 rungs) ───────────────────────────────────
    ladder = {}
    for label, fname in (("XGBoost", "xgboost_results.json"), ("MLP", "mlp_results.json"),
                         ("TemporalCNN", "temporal_results.json"),
                         ("TemporalGraph", "temporal_graph_results.json")):
        prior = _load_baseline(args.baseline_results_dir, fname)
        if prior is not None:
            ladder[label] = prior
    ladder["Fusion"] = cv
    table = format_results_table(ladder)
    logger.info("train_fusion | progressive ladder:\n%s", table)

    results = {
        "version": args.version,
        "model": "fusion",
        "n_windows": int(len(y)),
        "n_positive": n_pos,
        "n_folds": n_folds,
        "seeds": seeds,
        "num_roles": num_roles,
        "num_depts": num_depts,
        "cross_validation": cv,
        "ladder_table": table,
        "per_scenario": scenario_metrics,
        "detection_latency": latency,
        "oof_threshold": threshold,
        "checkpoint": str(ckpt_path),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("train_fusion | results written to %s.", output_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
