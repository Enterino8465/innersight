"""Tests for reproducibility utilities."""

import torch
from innersight.utils.reproducibility import seed_everything


def test_seed_produces_identical_tensors() -> None:
    """Same seed must produce identical random tensors."""
    seed_everything(123)
    a = torch.randn(10)

    seed_everything(123)
    b = torch.randn(10)

    assert torch.equal(a, b)


def test_different_seeds_produce_different_tensors() -> None:
    """Different seeds must produce different random tensors."""
    seed_everything(1)
    a = torch.randn(10)

    seed_everything(2)
    b = torch.randn(10)

    assert not torch.equal(a, b)
