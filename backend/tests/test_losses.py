"""Unit tests for innersight.models.losses (Phase 3)."""

import numpy as np
import torch

from innersight.models.losses import FocalLoss, calibrate_threshold


# ── FocalLoss ────────────────────────────────────────────────────────────────

def test_focal_loss_returns_scalar():
    fl = FocalLoss()
    loss = fl(torch.randn(16), (torch.rand(16) > 0.5).float())
    assert loss.shape == torch.Size([])  # scalar
    assert torch.isfinite(loss)


def test_focal_loss_accepts_both_shapes():
    fl = FocalLoss()
    logits = torch.tensor([2.0, -1.0, 3.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    flat = fl(logits, targets)
    col = fl(logits.reshape(-1, 1), targets.reshape(-1, 1))
    assert torch.allclose(flat, col)


def test_focal_gamma0_alpha_half_approximates_bce():
    # gamma=0 removes focusing; alpha=0.5 makes the class weight a flat 0.5,
    # so focal loss == 0.5 * mean BCE.
    fl = FocalLoss(alpha=0.5, gamma=0.0)
    torch.manual_seed(0)
    logits = torch.randn(200)
    targets = (torch.rand(200) > 0.5).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(fl(logits, targets), 0.5 * bce, atol=1e-6)


def test_focal_gradient_smaller_for_easy_than_hard():
    fl = FocalLoss()
    # Easy: confident & correct (high p_t). Hard: confident & wrong (low p_t).
    easy = torch.tensor([6.0], requires_grad=True)
    fl(easy, torch.tensor([1.0])).backward()
    hard = torch.tensor([-6.0], requires_grad=True)
    fl(hard, torch.tensor([1.0])).backward()
    assert abs(easy.grad.item()) < abs(hard.grad.item())


# ── calibrate_threshold ──────────────────────────────────────────────────────

def test_calibrate_threshold_returns_float_in_unit_interval():
    rng = np.random.default_rng(0)
    labels = (rng.random(100) < 0.2).astype(float)
    probs = np.clip(0.3 * labels + 0.4 * rng.random(100), 0, 1)
    thr = calibrate_threshold(probs, labels)
    assert isinstance(thr, float)
    assert 0.0 <= thr <= 1.0


def test_calibrate_threshold_on_perfect_predictions():
    # Negatives in [0, 0.3], positives in [0.7, 1.0]: the F1-optimal threshold
    # sits between the clusters and yields perfect classification.
    rng = np.random.default_rng(1)
    probs = np.concatenate([rng.uniform(0.0, 0.3, 80), rng.uniform(0.7, 1.0, 20)])
    labels = np.concatenate([np.zeros(80), np.ones(20)])
    thr = calibrate_threshold(probs, labels)
    assert 0.3 < thr <= 1.0
    preds = (probs >= thr).astype(int)
    assert (preds == labels).all()
