"""PyTorch training loop for InsiderThreatMLP.

Public API (consumed by api.py, scoring.py, feedback.py, main.py):
  train(config, event_callback)  — full training run
  load_best_model()              — returns numpy Network (compat shim)
  announce()                     — module descriptor for main.py
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from innersight.backend.b2_data.pipeline import load_data
from innersight.backend.b3_network.network import Network
from innersight.backend.config import (
    PREPROCESSOR_FILE, BEST_MODEL_FILE, BEST_MODEL_PT_FILE,
    STANDARDIZER_FILE, DEFAULT_TRAINING_CONFIG,
)
from innersight.backend.models.mlp import InsiderThreatMLP, build_mlp
from innersight.backend.models.dataset import build_dataloaders

_PREPROCESSOR_PATH  = PREPROCESSOR_FILE   # .npz — compat with api.py / feedback.py
_BEST_MODEL_PATH    = BEST_MODEL_FILE      # .npz — compat with load_best_model()
_BEST_MODEL_PT_PATH = BEST_MODEL_PT_FILE   # .pt  — canonical PyTorch checkpoint
_STANDARDIZER_PATH  = STANDARDIZER_FILE    # .pt  — PyTorch standardizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emit(event_callback, event: dict) -> None:
    if event_callback is not None:
        event_callback(event)


def _evaluate(
    model: InsiderThreatMLP,
    loader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Run model in eval mode over *loader*, return loss + P/R/F1 + raw arrays."""
    model.eval()
    total_loss = 0.0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            total_loss += criterion(logits, y_b).item()
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(y_b.cpu())

    avg_loss = total_loss / len(loader)
    probs  = torch.cat(all_probs).numpy().flatten()
    labels = torch.cat(all_labels).numpy().flatten().astype(int)
    preds  = (probs > 0.5).astype(int)

    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, precision, recall, f1, preds, labels


def _confusion_matrix(preds: np.ndarray, labels: np.ndarray) -> list[list[int]]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return [[tn, fp], [fn, tp]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(config: dict, event_callback=None, embedding_manager=None):
    """Train InsiderThreatMLP and persist the best checkpoint.

    Keeps the same event-callback contract as the legacy numpy trainer so
    api.py requires no changes:

      {'class_imbalance': {'ratio': float}}
      {'epoch': int, 'batch': int, 'loss': float}   (per mini-batch)
      {'epoch': int, 'val_loss': float, 'val_f1': float,
       'val_recall': float, 'val_precision': float}  (end of epoch)
      {'status': 'done', 'confusion_matrix': [[TN,FP],[FN,TP]]}

    Args:
        config: Training hyper-parameters; missing keys fall back to
            ``DEFAULT_TRAINING_CONFIG``.
        event_callback: Optional callable invoked with each progress dict.
        embedding_manager: Optional EmbeddingManager. When provided and active,
            Node2Vec embeddings are appended to each split's feature tensor
            before the DataLoader is built, increasing the input width from 18
            to ``18 + embedding_dim``. The layer_sizes in *config* must reflect
            this wider input (e.g. ``[146, 128, 64, 1]``).

    Returns:
        Dict with final val and test metrics.
    """
    epochs     = config.get('epochs',     DEFAULT_TRAINING_CONFIG['epochs'])
    batch_size = config.get('batch_size', DEFAULT_TRAINING_CONFIG['batch_size'])
    lr         = config.get('lr',         DEFAULT_TRAINING_CONFIG['lr'])
    pos_weight = config.get('pos_weight', DEFAULT_TRAINING_CONFIG['pos_weight'])
    patience   = config.get('patience',   DEFAULT_TRAINING_CONFIG['patience'])

    # ── 1. Data ───────────────────────────────────────────────────────────────
    data    = load_data()
    loaders = build_dataloaders(data, batch_size=batch_size, embedding_manager=embedding_manager)

    train_loader = loaders['train_loader']
    val_loader   = loaders['val_loader']
    test_loader  = loaders['test_loader']
    standardizer = loaders['standardizer']

    # Class imbalance from training labels tensor
    y_train_all = train_loader.dataset.tensors[1]  # type: ignore[union-attr]
    imbalance_ratio = float(y_train_all.mean().item())
    _emit(event_callback, {'class_imbalance': {'ratio': round(imbalance_ratio, 6)}})

    # ── 2. Model / loss / optimiser ───────────────────────────────────────────
    model, device = build_mlp(config)
    criterion     = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── 3. Training loop ──────────────────────────────────────────────────────
    best_val_f1   = -1.0
    best_state    = None
    no_improve    = 0

    for epoch in range(epochs):
        model.train()
        for batch_idx, (X_b, y_b) in enumerate(train_loader):
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            _emit(event_callback, {
                'epoch': epoch + 1,
                'batch': batch_idx,
                'loss':  round(float(loss.item()), 6),
            })

        # ── end-of-epoch validation ───────────────────────────────────────────
        val_loss, precision, recall, f1, val_preds, val_labels = _evaluate(
            model, val_loader, criterion, device
        )

        logger.info(
            'epoch %d/%d | val_loss=%.4f | P=%.3f R=%.3f F1=%.3f',
            epoch + 1, epochs, val_loss, precision, recall, f1,
        )
        _emit(event_callback, {
            'epoch':         epoch + 1,
            'val_loss':      round(val_loss, 4),
            'val_f1':        round(f1, 4),
            'val_recall':    round(recall, 4),
            'val_precision': round(precision, 4),
        })

        # ── checkpoint / early stopping ───────────────────────────────────────
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    'Early stopping at epoch %d (no F1 improvement for %d epochs)',
                    epoch + 1, patience,
                )
                break

    # ── 4. Restore best and emit final confusion matrix ───────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    _, _, _, _, val_preds, val_labels = _evaluate(model, val_loader, criterion, device)
    cm = _confusion_matrix(val_preds, val_labels)
    _emit(event_callback, {'status': 'done', 'confusion_matrix': cm})

    # ── 5. Test evaluation ────────────────────────────────────────────────────
    test_loss, test_p, test_r, test_f1, _, _ = _evaluate(
        model, test_loader, criterion, device
    )
    logger.info('Test | loss=%.4f P=%.3f R=%.3f F1=%.3f', test_loss, test_p, test_r, test_f1)

    # ── 6. Persist ────────────────────────────────────────────────────────────
    _save_model(model)

    # Preprocessor: save in both formats.
    # .npz — backward compat with api.py / feedback.py (np.load)
    # .pt  — for scoring.py PyTorch path (Standardizer.load)
    mean_np = standardizer.mean.cpu().numpy()  # type: ignore[union-attr]
    std_np  = standardizer.std.cpu().numpy()   # type: ignore[union-attr]
    os.makedirs(os.path.dirname(_PREPROCESSOR_PATH), exist_ok=True)
    np.savez(_PREPROCESSOR_PATH, mean=mean_np, std=std_np)
    standardizer.save(_STANDARDIZER_PATH)

    return {
        'best_val_f1':  round(best_val_f1, 4),
        'test_loss':    round(test_loss, 4),
        'test_precision': round(test_p, 4),
        'test_recall':  round(test_r, 4),
        'test_f1':      round(test_f1, 4),
    }


def _save_model(model: InsiderThreatMLP) -> None:
    """Save model in PyTorch format (.pt) and legacy numpy format (.npz).

    The .npz file keeps scoring.py / feedback.py / load_best_model() working
    without changes while the .pt file is the canonical new checkpoint.
    """
    os.makedirs(os.path.dirname(_BEST_MODEL_PT_PATH), exist_ok=True)

    # ── PyTorch checkpoint (includes layer_sizes so loaders don't need to guess) ─
    torch.save(
        {"state_dict": model.state_dict(), "layer_sizes": model.layer_sizes, "model_type": "mlp"},
        _BEST_MODEL_PT_PATH,
    )

    # ── Legacy numpy checkpoint (in, out) weights ─────────────────────────────
    # nn.Linear stores weight as (out, in); Network expects (in, out).
    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    arrays: dict = {}
    for i, layer in enumerate(linear_layers):
        arrays[f'W_{i}'] = layer.weight.detach().cpu().numpy().T          # (in, out)
        arrays[f'b_{i}'] = layer.bias.detach().cpu().numpy().reshape(1, -1)  # (1, out)
    arrays['n_layers'] = np.array(len(linear_layers))
    np.savez(_BEST_MODEL_PATH, **arrays)


def load_best_model() -> Network:
    """Load the best checkpoint as a numpy Network.

    Returns a ``Network`` instance compatible with scoring.py and feedback.py
    (which use the hand-rolled numpy forward pass, AdamOptimizer, etc.).
    """
    data     = np.load(_BEST_MODEL_PATH)
    n_layers = int(data['n_layers'])

    weights = [data[f'W_{i}'] for i in range(n_layers)]
    biases  = [data[f'b_{i}'] for i in range(n_layers)]

    layer_sizes = [weights[0].shape[0]] + [W.shape[1] for W in weights]
    net         = Network(layer_sizes)
    net.weights = weights
    net.biases  = biases
    return net


def announce() -> None:
    """Print a one-line module descriptor (consumed by main.py)."""
    print('b7_training.trainer  — PyTorch training loop (InsiderThreatMLP)')
