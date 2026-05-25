import pytest
import torch
import torch.nn as nn
from innersight.backend.models.mlp import InsiderThreatMLP


BATCH = 16


@pytest.fixture()
def model():
    torch.manual_seed(42)
    return InsiderThreatMLP([18, 32, 1])


@pytest.fixture()
def loss_and_backward(model):
    """Run a forward + backward pass, return (model, loss)."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))
    x         = torch.randn(BATCH, 18)
    target    = (torch.rand(BATCH, 1) > 0.95).float()
    loss      = criterion(model(x), target)
    loss.backward()
    return model, loss


# ── gradient flow ─────────────────────────────────────────────────────────────

def test_gradients_flow_to_all_parameters(loss_and_backward):
    model, _ = loss_and_backward
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_all_gradients_are_finite(loss_and_backward):
    model, _ = loss_and_backward
    for name, param in model.named_parameters():
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


# ── gradient magnitude ────────────────────────────────────────────────────────

def test_gradient_magnitude_non_zero(loss_and_backward):
    model, _ = loss_and_backward
    for name, param in model.named_parameters():
        assert param.grad.abs().max().item() > 0, f"Zero gradient in {name}"


def test_gradient_magnitude_not_exploding(loss_and_backward):
    """Gradients should stay within a reasonable range for a randomly initialised net."""
    model, _ = loss_and_backward
    for name, param in model.named_parameters():
        max_grad = param.grad.abs().max().item()
        assert max_grad < 1e4, f"Gradient exploding in {name}: {max_grad}"


# ── optimizer step reduces loss ───────────────────────────────────────────────

def test_optimizer_step_reduces_loss():
    """A single Adam step should reduce training loss on a fixed batch."""
    torch.manual_seed(0)
    m         = InsiderThreatMLP([18, 16, 1])
    criterion = nn.BCEWithLogitsLoss()
    opt       = torch.optim.Adam(m.parameters(), lr=0.01)
    x         = torch.randn(32, 18)
    target    = (torch.rand(32, 1) > 0.5).float()

    m.train()
    loss_before = criterion(m(x), target).item()

    for _ in range(5):
        opt.zero_grad()
        criterion(m(x), target).backward()
        opt.step()

    loss_after = criterion(m(x), target).item()
    assert loss_after < loss_before


def test_zero_grad_clears_gradients(model):
    criterion = nn.BCEWithLogitsLoss()
    opt       = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion(model(torch.randn(8, 18)), torch.zeros(8, 1)).backward()
    opt.zero_grad()
    for name, param in model.named_parameters():
        assert param.grad is None or param.grad.abs().max().item() == 0, \
            f"Gradient not cleared for {name}"


# ── no_grad mode ─────────────────────────────────────────────────────────────

def test_no_grad_does_not_populate_gradients():
    m = InsiderThreatMLP([18, 8, 1])
    with torch.no_grad():
        out = m(torch.randn(4, 18))
    assert out.requires_grad is False
