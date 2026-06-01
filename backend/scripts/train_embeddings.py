#!/usr/bin/env python
"""CLI script: train graph embeddings (Node2Vec or MetaPath2Vec) on the CERT graph.

Reads hyperparameters from the node2vec section of a YAML config, builds (or
loads cached) temporal graphs, trains embeddings, and saves the resulting user
embeddings so that train.py can pick them up without re-running this step.

Usage
-----
    python scripts/train_embeddings.py --config configs/train_node2vec.yaml
    python scripts/train_embeddings.py --config configs/train_metapath2vec.yaml
    python scripts/train_embeddings.py --config configs/train_node2vec.yaml --device cpu
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import yaml


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train Node2Vec on the CERT graph and save user embeddings.'
    )
    p.add_argument('--config', required=True, metavar='PATH',
                   help='Path to YAML config (must contain a node2vec section)')
    p.add_argument('--device', metavar='DEVICE',
                   help='Override device: cpu | cuda | cuda:N')
    p.add_argument('--force', action='store_true',
                   help='Re-train even if embeddings file already exists')
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Validate config ───────────────────────────────────────────────────────
    if 'node2vec' not in cfg:
        print(f'ERROR: no node2vec section found in {args.config}')
        return 1

    n2v_cfg = cfg['node2vec']

    # ── Env vars (must be set before backend imports read config.py) ──────────
    data_dir = (cfg.get('data') or {}).get('dir')
    if data_dir:
        os.environ['INNERSIGHT_DATA_DIR'] = data_dir

    out_cfg   = cfg.get('output') or {}
    model_dir = os.path.dirname(out_cfg.get('model_path', 'checkpoints/model.pt')) or 'checkpoints'
    os.environ['INNERSIGHT_MODEL_DIR'] = model_dir
    os.makedirs(model_dir, exist_ok=True)

    log_file = out_cfg.get('log_file', 'logs/training_node2vec.log')
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )
    logger = logging.getLogger(__name__)

    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # ── Embeddings path ───────────────────────────────────────────────────────
    emb_path = n2v_cfg.get('embeddings_path', os.path.join(model_dir, 'node2vec_embeddings.pt'))
    if os.path.exists(emb_path) and not args.force:
        print(f'Embeddings already exist at {emb_path}')
        print('Use --force to retrain.  Exiting.')
        return 0

    # ── Dispatch on method ────────────────────────────────────────────────────
    method = n2v_cfg.get('method', 'node2vec').lower()
    if method not in ('node2vec', 'metapath2vec'):
        print(f'ERROR: unknown node2vec.method "{method}" — expected node2vec or metapath2vec')
        return 1

    # ── Imports (after env vars are locked in) ────────────────────────────────
    from innersight.backend.models.graph_builder import load_temporal_graphs
    from innersight.backend.models.node2vec_trainer import (
        save_embeddings,
        train_node2vec,
        train_metapath2vec,
    )

    # ── Load graph ────────────────────────────────────────────────────────────
    print(f'\nLoading temporal graphs  (model-dir={model_dir}) ...')
    t0 = time.perf_counter()
    graphs = load_temporal_graphs()
    graph  = graphs['train']
    print(f'  loaded in {time.perf_counter() - t0:.1f}s')
    print(f'  user : {graph["user"].x.shape[0]:,} nodes')
    print(f'  pc   : {graph["pc"].x.shape[0]:,} nodes')
    print(f'  url  : {graph["url"].x.shape[0]:,} nodes')
    print(f'  file : {graph["file"].x.shape[0]:,} nodes')
    for etype in graph.edge_types:
        n_e = graph[etype].edge_index.shape[1]
        if n_e > 0:
            print(f'  {str(etype):<45} {n_e:>9,} edges')

    # ── Build embedding config from YAML and dispatch ─────────────────────────
    emb_config = {
        'embedding_dim':        n2v_cfg.get('embedding_dim',        128),
        'walk_length':          n2v_cfg.get('walk_length',          20),
        'context_size':         n2v_cfg.get('context_size',         10),
        'walks_per_node':       n2v_cfg.get('walks_per_node',       20),
        'num_negative_samples': n2v_cfg.get('num_negative_samples', 1),
        'batch_size':           n2v_cfg.get('batch_size',           128),
        'lr':                   n2v_cfg.get('lr',                   0.01),
        'epochs':               n2v_cfg.get('epochs',               50),
        # Node2Vec-only
        'p': n2v_cfg.get('p', 1.0),
        'q': n2v_cfg.get('q', 0.8),
    }

    if method == 'node2vec':
        print(f'\nTraining Node2Vec')
        print(f'  embedding_dim   : {emb_config["embedding_dim"]}')
        print(f'  walk_length     : {emb_config["walk_length"]}')
        print(f'  walks_per_node  : {emb_config["walks_per_node"]}')
        print(f'  p={emb_config["p"]}  q={emb_config["q"]}')
        print(f'  epochs          : {emb_config["epochs"]}')
        print()
        t_train = time.perf_counter()
        embeddings = train_node2vec(graph, emb_config)
    else:
        print(f'\nTraining MetaPath2Vec')
        print(f'  embedding_dim   : {emb_config["embedding_dim"]}  (per metapath)')
        print(f'  walk_length     : {emb_config["walk_length"]}')
        print(f'  walks_per_node  : {emb_config["walks_per_node"]}')
        print(f'  epochs          : {emb_config["epochs"]}')
        print()
        t_train = time.perf_counter()
        embeddings = train_metapath2vec(graph, emb_config)

    train_secs = time.perf_counter() - t_train

    # ── Summary ───────────────────────────────────────────────────────────────
    e = embeddings
    print(f'\nEmbedding stats:')
    print(f'  shape : {tuple(e.shape)}')
    print(f'  min   : {e.min().item():.4f}')
    print(f'  max   : {e.max().item():.4f}')
    print(f'  mean  : {e.mean().item():.4f}')
    print(f'  std   : {e.std().item():.4f}')
    print(f'  training time : {train_secs:.1f}s')

    # ── Save ──────────────────────────────────────────────────────────────────
    save_embeddings(embeddings, graph.user_to_idx, emb_path)
    print(f'\nEmbeddings saved → {emb_path}')
    logger.info('%s training complete. Embeddings saved to %s', method, emb_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
