"""GNN training loop for InsiderThreatGNN (HeteroGraphSAGE + classification head).

Public API — same contract as training.trainer.train() so the CLI, API, and
SSE streaming require no changes:

  train_gnn(config, event_callback=None) -> dict

Event stream (identical schema to MLP trainer):
  {'class_imbalance': {'ratio': float}}
  {'epoch': int, 'batch': int, 'loss': float}        — per mini-batch
  {'epoch': int, 'train_loss': float,
   'val_loss': float, 'val_f1': float,
   'val_recall': float, 'val_precision': float}       — end of epoch
  {'status': 'done', 'confusion_matrix': [[TN,FP],[FN,TP]]}

Config shape (nested, mirrors the YAML):
  model:
    type: graphsage
    hidden_dim: 128
    num_layers: 2
    dropout: 0.3
    head_layers: [128, 64]
  training:
    epochs: 20
    batch_size: 128
    lr: 0.001
    pos_weight: 50.0
    patience: 5
    num_neighbors: [10, 5]
  output:
    model_path: checkpoints/graphsage/best_model.pt
    log_file: logs/training_graphsage.log
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from innersight.models.graph_builder import load_temporal_graphs
from innersight.models.graph_loader import build_graph_dataloaders
from innersight.models.factory import build_model

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _emit(callback: Callable[[dict], None] | None, event: dict) -> None:
    if callback is not None:
        callback(event)


def _seed_size(batch) -> int:
    """Number of target (seed) user nodes in a NeighborLoader batch.

    NeighborLoader sets ``batch['user'].batch_size`` to the seed count.
    The full-graph fallback (``full_graph_loader``) does not set this
    attribute, so we fall back to the total node count.
    """
    return getattr(batch['user'], 'batch_size', batch['user'].x.shape[0])


def _metrics(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    preds = (probs > 0.5).astype(int)
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def _confusion_matrix(probs: np.ndarray, labels: np.ndarray) -> list[list[int]]:
    preds = (probs > 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return [[tn, fp], [fn, tp]]


def _evaluate_gnn(
    model: nn.Module,
    loader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Eval loop over *loader*; returns avg_loss, P, R, F1, probs, labels."""
    model.eval()
    total_loss = 0.0
    all_probs:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x_dict, batch.edge_index_dict)

            n = _seed_size(batch)
            seed_logits = logits[:n]                        # (n, 1)
            seed_labels = batch['user'].y[:n].float()       # (n,)

            loss = criterion(seed_logits, seed_labels.unsqueeze(1))
            total_loss += loss.item()

            probs = torch.sigmoid(seed_logits).squeeze(1).cpu()
            all_probs.append(probs)
            all_labels.append(seed_labels.cpu())

    avg_loss = total_loss / max(len(loader), 1)
    probs_np  = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)
    p, r, f1  = _metrics(probs_np, labels_np)
    return avg_loss, p, r, f1, probs_np, labels_np


# ── Public API ────────────────────────────────────────────────────────────────

def train_gnn(
    config: dict[str, Any],
    event_callback: Callable[[dict], None] | None = None,
    graphs: dict | None = None,
) -> dict[str, float]:
    """Train InsiderThreatGNN and save the best checkpoint by val F1.

    Parameters
    ----------
    config:
        Nested config dict (see module docstring for shape).
    event_callback:
        Optional callable invoked with each progress event dict.
    graphs:
        Optional pre-loaded ``{'train': HeteroData, 'val': HeteroData,
        'test': HeteroData}``.  When provided, ``load_temporal_graphs()``
        is skipped.  Useful for testing / synthetic data.

    Returns
    -------
    dict
        ``{'best_val_f1', 'test_loss', 'test_precision', 'test_recall', 'test_f1'}``
    """
    training_cfg = config.get('training', {})
    epochs       = int(training_cfg.get('epochs',       10))
    batch_size   = int(training_cfg.get('batch_size',  128))
    lr           = float(training_cfg.get('lr',        0.001))
    pos_weight   = float(training_cfg.get('pos_weight', 50.0))
    patience     = int(training_cfg.get('patience',      5))
    num_neighbors = training_cfg.get('num_neighbors', [10, 5])

    output_cfg = config.get('output', {})
    model_path = output_cfg.get('model_path', 'checkpoints/graphsage/best_model.pt')

    # ── 1. Load graphs ────────────────────────────────────────────────────────
    if graphs is None:
        logger.info('Loading temporal graphs …')
        graphs = load_temporal_graphs()
    else:
        logger.info('Using pre-loaded graphs (skipping load_temporal_graphs)')

    train_g = graphs['train']
    y_train = train_g['user'].y.float()
    pos_rate = float(y_train.mean().item())
    _emit(event_callback, {'class_imbalance': {'ratio': round(pos_rate, 6)}})
    logger.info(
        'Train graph: %d users  %.2f%% positive',
        train_g['user'].x.shape[0], pos_rate * 100,
    )

    # ── 2. DataLoaders ────────────────────────────────────────────────────────
    logger.info('Building NeighborLoader DataLoaders …')
    loaders = build_graph_dataloaders(
        graphs,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
    )
    train_loader = loaders['train_loader']
    val_loader   = loaders['val_loader']
    test_loader  = loaders['test_loader']

    # ── 3. Model / loss / optimiser ───────────────────────────────────────────
    model, device = build_model(config, metadata=train_g.metadata())
    logger.info('Model on device: %s', device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── 4. Training loop ──────────────────────────────────────────────────────
    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0
    n_train_batches = len(train_loader) if hasattr(train_loader, '__len__') else '?'

    for epoch in range(epochs):
        model.train()
        total_loss   = 0.0
        n_batches    = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)

            n = _seed_size(batch)
            seed_logits = model(batch.x_dict, batch.edge_index_dict)[:n]   # (n, 1)
            seed_labels = batch['user'].y[:n].float().unsqueeze(1)          # (n, 1)

            optimizer.zero_grad()
            loss = criterion(seed_logits, seed_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            _emit(event_callback, {
                'epoch': epoch + 1,
                'batch': batch_idx,
                'loss':  round(float(loss.item()), 6),
            })

        avg_train_loss = total_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, p, r, f1, val_probs, val_labels = _evaluate_gnn(
            model, val_loader, criterion, device
        )

        logger.info(
            'epoch %d/%d | train_loss=%.4f | val_loss=%.4f | P=%.3f R=%.3f F1=%.3f',
            epoch + 1, epochs, avg_train_loss, val_loss, p, r, f1,
        )
        _emit(event_callback, {
            'epoch':         epoch + 1,
            'train_loss':    round(avg_train_loss, 4),
            'val_loss':      round(val_loss, 4),
            'val_f1':        round(f1, 4),
            'val_recall':    round(r, 4),
            'val_precision': round(p, 4),
        })

        # ── Checkpoint / early stopping ───────────────────────────────────────
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
            logger.info('  ✓ new best val F1: %.4f — checkpoint saved', f1)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    'Early stopping at epoch %d (no F1 improvement for %d epochs)',
                    epoch + 1, patience,
                )
                break

    # ── 5. Restore best checkpoint ────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    _, _, _, _, val_probs, val_labels = _evaluate_gnn(
        model, val_loader, criterion, device
    )
    cm = _confusion_matrix(val_probs, val_labels)
    _emit(event_callback, {'status': 'done', 'confusion_matrix': cm})

    # ── 6. Test evaluation ────────────────────────────────────────────────────
    test_loss, test_p, test_r, test_f1, _, _ = _evaluate_gnn(
        model, test_loader, criterion, device
    )
    logger.info(
        'Test | loss=%.4f P=%.3f R=%.3f F1=%.3f',
        test_loss, test_p, test_r, test_f1,
    )

    # ── 7. Persist ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    torch.save(
        {
            'state_dict':  model.state_dict(),
            'model_type':  config.get('model', {}).get('type', 'graphsage'),
            'metadata':    train_g.metadata(),
            'config':      config.get('model', {}),
            # Scoring needs to load graphs from this directory to map idx→user_id.
            'graphs_dir':  os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints'),
        },
        model_path,
    )
    logger.info('Model saved to %s', model_path)

    # ── 8. Extract and save user embeddings ───────────────────────────────────
    graphs_dir   = os.environ.get('INNERSIGHT_MODEL_DIR', 'checkpoints')
    emb_out_path = os.path.join(graphs_dir, 'gnn_embeddings.pt')
    graph_path   = os.path.join(graphs_dir, 'graphs', 'train_graph.pt')
    try:
        from innersight.scripts.extract_embeddings import extract_gnn_embeddings
        extract_gnn_embeddings(
            model_path=model_path,
            graph_path=graph_path if os.path.exists(graph_path) else None,
            output_path=emb_out_path,
        )
        logger.info('GNN embeddings saved to %s', emb_out_path)
    except Exception as exc:
        logger.warning('Embedding extraction skipped: %s', exc)

    return {
        'best_val_f1':    round(best_val_f1, 4),
        'test_loss':      round(test_loss, 4),
        'test_precision': round(test_p, 4),
        'test_recall':    round(test_r, 4),
        'test_f1':        round(test_f1, 4),
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Point at the cached graphs before any config.py import resolves the path.
    os.environ.setdefault('INNERSIGHT_MODEL_DIR', 'checkpoints')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    cfg = {
        'model': {
            'type':        'graphsage',
            'hidden_dim':  64,
            'num_layers':  2,
            'dropout':     0.3,
            'head_layers': [64, 32],
        },
        'training': {
            'epochs':        3,
            'batch_size':    128,
            'lr':            0.001,
            'pos_weight':    50.0,
            'patience':      5,
            'num_neighbors': [10, 5],
        },
        'output': {
            'model_path': 'checkpoints/graphsage/best_model.pt',
            'log_file':   'logs/training_graphsage.log',
        },
    }

    def _cb(event: dict) -> None:
        if 'class_imbalance' in event:
            print(f"  positive rate: {event['class_imbalance']['ratio']:.4%}")
        elif 'epoch' in event and 'val_loss' in event:
            e = event['epoch']
            print(
                f"  epoch {e:>2} | train_loss={event['train_loss']:.4f}"
                f" | val_loss={event['val_loss']:.4f}"
                f" | P={event['val_precision']:.3f}"
                f" R={event['val_recall']:.3f}"
                f" F1={event['val_f1']:.4f}"
            )
        elif event.get('status') == 'done':
            cm = event['confusion_matrix']
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            print(f"\n  confusion matrix (val): TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    print('\n' + '=' * 60)
    print('GNN Trainer — 3-epoch smoke test')
    print('=' * 60 + '\n')

    # The cached graphs have all-zero labels (built without raw CERT CSVs).
    # Inject synthetic 5% positives so the trainer has signal to learn from
    # and we can verify that loss decreases and F1 > 0.
    import copy as _copy
    from innersight.models.graph_builder import load_temporal_graphs as _load

    _raw = _load()
    _graphs = {}
    _rng = torch.Generator().manual_seed(0)
    for split, g in _raw.items():
        g2 = _copy.deepcopy(g)
        n  = g2['user'].x.shape[0]
        # 5% positives
        y  = torch.zeros(n, dtype=torch.float32)
        y[torch.randperm(n, generator=_rng)[:max(1, n // 20)]] = 1.0
        g2['user'].y = y
        _graphs[split] = g2
        pos = int(y.sum())
        print(f'  {split:<6}: {n} users  {pos} synthetic positives ({pos/n:.1%})')
    print()

    results = train_gnn(cfg, event_callback=_cb, graphs=_graphs)

    print(f'\nFinal results:')
    print(f"  best val F1  : {results['best_val_f1']:.4f}")
    print(f"  test loss    : {results['test_loss']:.4f}")
    print(f"  test P/R/F1  : {results['test_precision']:.3f} / "
          f"{results['test_recall']:.3f} / {results['test_f1']:.4f}")
