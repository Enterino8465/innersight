"""Loss functions and decision-threshold calibration for imbalanced training.

Insider-threat labels are extremely imbalanced (positives are rare), so plain
BCE is dominated by easy negatives. :class:`FocalLoss` down-weights easy,
well-classified examples and up-weights the rare positive class, and
:func:`calibrate_threshold` picks the probability cut-off that maximises F1 on a
validation set rather than defaulting to 0.5.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss on raw logits (Lin et al., 2017).

    Scales BCE by ``(1 - p_t) ** gamma`` so confident-correct ("easy") examples
    contribute little, and by a class weight ``alpha`` that favours positives.

    Args:
        alpha: Weight on the positive class in ``[0, 1]``; ``0.75`` up-weights
            the rare positives.
        gamma: Focusing parameter; higher values down-weight easy examples more.
            ``0.0`` recovers (alpha-weighted) BCE.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the mean focal loss.

        Args:
            logits: Raw (pre-sigmoid) logits, shape ``(batch,)`` or ``(batch, 1)``.
            targets: Binary 0/1 targets, same shape as ``logits``.

        Returns:
            Scalar mean focal loss.
        """
        logits = logits.reshape(-1)              # shape: (batch,)
        targets = targets.reshape(-1).float()    # shape: (batch,)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # shape: (batch,)
        p = torch.sigmoid(logits)                # shape: (batch,)
        # Probability assigned to the true class for each example.
        p_t = p * targets + (1 - p) * (1 - targets)            # shape: (batch,)
        focal_weight = (1 - p_t) ** self.gamma                 # shape: (batch,)
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # shape: (batch,)
        return (alpha_weight * focal_weight * bce).mean()


def calibrate_threshold(val_probs, val_labels) -> float:
    """Find the probability threshold that maximises F1 on a validation set.

    Sweeps every candidate threshold from the precision-recall curve and returns
    the one with the highest F1 score. This replaces the naive 0.5 cut-off, which
    is rarely optimal under heavy class imbalance.

    Args:
        val_probs: Predicted positive-class probabilities, shape ``(batch,)`` or
            ``(batch, 1)``.
        val_labels: True binary labels, same shape as ``val_probs``.

    Returns:
        The F1-optimal threshold in ``[0, 1]`` (defaults to ``0.5`` when no
        valid threshold can be derived, e.g. a single-class validation set).
    """
    from sklearn.metrics import precision_recall_curve

    probs = np.asarray(val_probs, dtype=float).reshape(-1)    # shape: (batch,)
    labels = np.asarray(val_labels, dtype=float).reshape(-1)  # shape: (batch,)

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if thresholds.size == 0:
        return 0.5

    # F1 per point; precision/recall have one extra trailing entry (no threshold).
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1.0), 0.0)
    # Only the first len(thresholds) points correspond to an actual threshold.
    best_idx = int(np.argmax(f1[: thresholds.size]))
    return float(thresholds[best_idx])
