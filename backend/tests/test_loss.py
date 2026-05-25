import pytest
import torch
import torch.nn as nn
from innersight.backend.models.mlp import InsiderThreatMLP


BATCH = 32


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return InsiderThreatMLP([18, 8, 1])


@pytest.fixture()
def criterion():
    return nn.BCEWithLogitsLoss()


@pytest.fixture()
def weighted_criterion():
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))


# ── basic properties ──────────────────────────────────────────────────────────

def test_loss_is_scalar(model, criterion):
    x      = torch.randn(BATCH, 18)
    logits = model(x)
    target = torch.zeros(BATCH, 1)
    loss   = criterion(logits, target)
    assert loss.ndim == 0


def test_loss_is_positive(model, criterion):
    logits = model(torch.randn(BATCH, 18))
    target = torch.zeros(BATCH, 1)
    assert criterion(logits, target).item() > 0


# ── pos_weight effect ─────────────────────────────────────────────────────────

def test_bce_with_logits_pos_weight_increases_positive_loss():
    """pos_weight=50 must penalise false negatives more than false positives."""
    logits = torch.zeros(10, 1)   # 50 % probability prediction
    ones   = torch.ones(10, 1)    # all malicious

    loss_unweighted = nn.BCEWithLogitsLoss()(logits, ones)
    loss_weighted   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))(logits, ones)

    assert loss_weighted.item() > loss_unweighted.item()


def test_pos_weight_does_not_affect_negative_class():
    """pos_weight only changes loss for positive labels; negatives unchanged."""
    logits = torch.zeros(10, 1)
    zeros  = torch.zeros(10, 1)

    loss1 = nn.BCEWithLogitsLoss()(logits, zeros)
    loss2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))(logits, zeros)

    assert abs(loss1.item() - loss2.item()) < 1e-6


def test_false_negative_loss_higher_than_false_positive():
    """With pos_weight=50, missing a positive costs more than a false alarm."""
    logits      = torch.zeros(10, 1)
    fn_target   = torch.ones(10, 1)    # true positive we predict as 0.5 → FN
    fp_target   = torch.zeros(10, 1)   # true negative we predict as 0.5 → FP

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))
    fn_loss   = criterion(logits, fn_target)
    fp_loss   = criterion(logits, fp_target)

    assert fn_loss.item() > fp_loss.item()


# ── gradient flow through loss ────────────────────────────────────────────────

def test_loss_backward_populates_gradients(model):
    criterion = nn.BCEWithLogitsLoss()
    x         = torch.randn(BATCH, 18)
    logits    = model(x)
    target    = torch.zeros(BATCH, 1)
    loss      = criterion(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"gradient missing for {name}"


def test_loss_backward_no_nan_gradients(model):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))
    logits    = model(torch.randn(BATCH, 18))
    target    = (torch.rand(BATCH, 1) > 0.98).float()
    criterion(logits, target).backward()

    for name, param in model.named_parameters():
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


# ── perfect predictions have low loss ─────────────────────────────────────────

def test_perfect_negative_predictions_low_loss():
    criterion = nn.BCEWithLogitsLoss()
    logits    = torch.full((BATCH, 1), -10.0)  # sigmoid ≈ 0
    target    = torch.zeros(BATCH, 1)
    assert criterion(logits, target).item() < 0.01


def test_perfect_positive_predictions_low_loss():
    criterion = nn.BCEWithLogitsLoss()
    logits    = torch.full((BATCH, 1), 10.0)   # sigmoid ≈ 1
    target    = torch.ones(BATCH, 1)
    assert criterion(logits, target).item() < 0.01
