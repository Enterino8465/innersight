#!/usr/bin/env python
"""CLI training script for InnerSight models (MLP, Node2Vec-enriched MLP, GraphSAGE).

Remote GPU usage:
  1. git clone <repo> on the GPU machine
  2. Upload CERT dataset, set INNERSIGHT_DATA_DIR
  3. pip install -r requirements.txt
  4. python scripts/train.py --config configs/train_mlp.yaml
  5. Download checkpoints/ back to your Mac
"""

import argparse
import logging
import os
import sys

import yaml


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train an InnerSight model from a YAML config.')
    p.add_argument('--config', required=True, metavar='PATH', help='Path to YAML config file')
    p.add_argument('--device', metavar='DEVICE',
                   help='Override device auto-detect: cpu | cuda | cuda:N')
    p.add_argument('--quick', action='store_true',
                   help='Override epochs=3, patience=1 for rapid pipeline verification')
    return p.parse_args()


def _setup_logging(log_file: str) -> None:
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def _console_callback(event: dict) -> None:
    """Print formatted training progress; skip per-batch noise."""
    if 'class_imbalance' in event:
        ratio = event['class_imbalance']['ratio']
        print(f"  class imbalance — positive rate: {ratio:.4%}")
    elif 'epoch' in event and 'val_loss' in event:
        e = event['epoch']
        train_part = (f" train_loss={event['train_loss']:.4f} |"
                      if 'train_loss' in event else '')
        print(
            f"  epoch {e:>3} |{train_part} val_loss={event['val_loss']:.4f} | "
            f"P={event['val_precision']:.3f} "
            f"R={event['val_recall']:.3f} "
            f"F1={event['val_f1']:.4f}"
        )
    elif event.get('status') == 'done':
        cm = event['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        print(f"\n  confusion matrix (val): TN={tn}  FP={fp}  FN={fn}  TP={tp}")


def main() -> int:
    args = _parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Quick mode: minimal epochs for pipeline verification ─────────────────
    if args.quick:
        print("QUICK MODE: 3 epochs for pipeline verification only\n")
        cfg.setdefault('training', {})
        cfg['training']['epochs']  = 3
        cfg['training']['patience'] = 1
        if cfg.get('node2vec'):
            cfg['node2vec']['walks_per_node'] = 2

    # ── Set env vars BEFORE any innersight.backend import (config.py reads them) ─
    data_dir = (cfg.get('data') or {}).get('dir')
    if data_dir:
        os.environ['INNERSIGHT_DATA_DIR'] = data_dir

    out_cfg   = cfg.get('output') or {}
    model_path = out_cfg.get('model_path', 'checkpoints/model.pt')
    model_dir  = os.path.dirname(model_path) or 'checkpoints'
    os.environ['INNERSIGHT_MODEL_DIR'] = model_dir
    os.makedirs(model_dir, exist_ok=True)

    # --device cpu hides all CUDA devices; cuda/cuda:N lets auto-detect pick GPU
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    log_file = out_cfg.get('log_file', 'logs/training.log')
    _setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info('Config   : %s', args.config)
    logger.info('Model dir: %s', model_dir)
    logger.info('Log file : %s', log_file)

    model_type = (cfg.get('model') or {}).get('type', 'mlp')

    # ── GraphSAGE path ────────────────────────────────────────────────────────
    if model_type == 'graphsage':
        from innersight.backend.models.gnn_trainer import train_gnn

        out_cfg      = cfg.get('output', {})
        training_sec = cfg.get('training', {})
        model_sec    = cfg.get('model', {})

        # graph_cache is the directory holding pre-built HeteroData .pt files
        # (e.g. "checkpoints/graphs/").  INNERSIGHT_MODEL_DIR must point to its
        # parent ("checkpoints") so load_temporal_graphs() and gnn_trainer find
        # the files at <model_dir>/graphs/train_graph.pt.
        data_sec    = cfg.get('data', {}) or {}
        graph_cache = data_sec.get('graph_cache', 'checkpoints/graphs').rstrip('/')
        graphs_dir  = os.path.dirname(graph_cache) or 'checkpoints'
        os.environ['INNERSIGHT_MODEL_DIR'] = graphs_dir

        print(f"\nTraining InsiderThreatGNN (GraphSAGE)  |  config: {args.config}")
        print(f"  hidden_dim   : {model_sec.get('hidden_dim', 128)}")
        print(f"  num_layers   : {model_sec.get('num_layers', 2)}")
        print(f"  head_layers  : {model_sec.get('head_layers', [128, 64])}")
        print(f"  epochs       : {training_sec.get('epochs', 20)}"
              f"  (patience {training_sec.get('patience', 5)})")
        print(f"  batch_size   : {training_sec.get('batch_size', 128)}"
              f"  lr {training_sec.get('lr', 0.001)}")
        print(f"  num_neighbors: {training_sec.get('num_neighbors', [10, 5])}")
        print(f"  pos_weight   : {training_sec.get('pos_weight', 50.0)}")
        print()

        try:
            results = train_gnn(cfg, event_callback=_console_callback)
        except Exception as exc:
            logger.error('GNN training failed: %s', exc, exc_info=True)
            return 1

        model_path = out_cfg.get('model_path', 'checkpoints/graphsage/best_model.pt')

        print(f"\nFinal results:")
        print(f"  best val F1  : {results['best_val_f1']:.4f}")
        print(f"  test loss    : {results['test_loss']:.4f}")
        print(f"  test P/R/F1  : {results['test_precision']:.3f} / "
              f"{results['test_recall']:.3f} / {results['test_f1']:.4f}")
        print(f"\nCheckpoints saved:")
        print(f"  model        : {model_path}")
        print(f"  log          : {log_file}")
        return 0

    # ── MLP / Node2Vec path ───────────────────────────────────────────────────
    from innersight.backend.training.trainer import train
    from innersight.backend.config import BEST_MODEL_PT_FILE, STANDARDIZER_FILE

    training  = cfg.get('training', {})
    model_sec = cfg.get('model', {})
    flat_config = {
        'epochs':      training.get('epochs',     50),
        'batch_size':  training.get('batch_size', 64),
        'lr':          training.get('lr',          0.001),
        'pos_weight':  training.get('pos_weight',  50.0),
        'patience':    training.get('patience',    5),
        'layer_sizes': model_sec.get('layer_sizes', [18, 64, 32, 1]),
    }

    # ── Node2Vec embeddings (optional) ────────────────────────────────────────
    embedding_manager = None
    n2v_cfg = cfg.get('node2vec')
    if n2v_cfg:
        from innersight.backend.models.embeddings import EmbeddingManager

        emb_path = n2v_cfg.get('embeddings_path', os.path.join(model_dir, 'node2vec_embeddings.pt'))
        if not os.path.exists(emb_path):
            logger.error(
                'Node2Vec embeddings not found at %s. '
                'Run scripts/train_embeddings.py --config %s first.',
                emb_path, args.config,
            )
            return 1

        embedding_manager = EmbeddingManager(emb_path)
        if not embedding_manager.available:
            logger.error('Failed to load embeddings from %s', emb_path)
            return 1

        logger.info(
            'Node2Vec embeddings loaded: %d users × %d dims',
            len(embedding_manager.user_to_idx), embedding_manager.embedding_dim,
        )
        print(f'  embeddings   : {emb_path}')
        print(f'                 {len(embedding_manager.user_to_idx):,} users × '
              f'{embedding_manager.embedding_dim} dims')
    else:
        print(f'  embeddings   : none (flat features only)')

    print(f"\nTraining InsiderThreatMLP  |  config: {args.config}")
    print(f"  architecture : {flat_config['layer_sizes']}")
    print(f"  epochs       : {flat_config['epochs']}  (patience {flat_config['patience']})")
    print(f"  batch_size   : {flat_config['batch_size']}  lr {flat_config['lr']}")
    print(f"  pos_weight   : {flat_config['pos_weight']}")
    print()

    try:
        results = train(flat_config, event_callback=_console_callback,
                        embedding_manager=embedding_manager)
    except Exception as exc:
        logger.error('Training failed: %s', exc, exc_info=True)
        return 1

    print(f"\nFinal results:")
    print(f"  best val F1  : {results['best_val_f1']:.4f}")
    print(f"  test loss    : {results['test_loss']:.4f}")
    print(f"  test P/R/F1  : {results['test_precision']:.3f} / "
          f"{results['test_recall']:.3f} / {results['test_f1']:.4f}")
    print(f"\nCheckpoints saved:")
    print(f"  model        : {BEST_MODEL_PT_FILE}")
    print(f"  standardizer : {STANDARDIZER_FILE}")
    print(f"  log          : {log_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
