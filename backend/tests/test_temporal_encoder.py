"""Unit tests for innersight.models.temporal_encoder (Phase 4)."""

import torch

from innersight.models.temporal_encoder import TemporalPatternEncoder
from innersight.utils.reproducibility import seed_everything


def _encoder():
    enc = TemporalPatternEncoder()
    enc.eval()  # disable dropout for deterministic, full-path tests
    return enc


# ── Shape ────────────────────────────────────────────────────────────────────

def test_forward_shape_batch():
    out = _encoder()(torch.randn(4, 18, 28))
    assert out.shape == (4, 128)


def test_forward_shape_single_sample():
    out = _encoder()(torch.randn(1, 18, 28))
    assert out.shape == (1, 128)


def test_attention_weights_shape():
    attn = _encoder().get_attention_weights(torch.randn(4, 18, 28))
    assert attn.shape == (4, 28)


# ── Attention ────────────────────────────────────────────────────────────────

def test_attention_weights_sum_to_one():
    attn = _encoder().get_attention_weights(torch.randn(5, 18, 28))
    assert torch.allclose(attn.sum(dim=1), torch.ones(5), atol=1e-5)


def test_attention_weights_non_negative():
    attn = _encoder().get_attention_weights(torch.randn(5, 18, 28))
    assert (attn >= 0).all()


# ── Causality (most important) ───────────────────────────────────────────────

def test_causal_padding_no_future_leakage():
    enc = _encoder()
    x = torch.randn(1, 18, 28)
    x_mod = x.clone()
    x_mod[0, :, 27] = 50.0  # perturb ONLY the last day (the future)

    # Run the conv stack manually (before attention pooling).
    with torch.no_grad():
        h = x
        h_mod = x_mod
        for block in enc.blocks:
            h = block(h)            # shape: (1, out_dim, 28)
            h_mod = block(h_mod)

    # Causal: output at time t depends only on inputs <= t, so perturbing day 27
    # cannot change any output before day 27.
    before = (h[..., :27] - h_mod[..., :27]).abs().max().item()
    at_27 = (h[..., 27] - h_mod[..., 27]).abs().max().item()
    assert before < 1e-6, f"future leaked into the past (max diff {before})"
    assert at_27 > 0.0, "perturbing day 27 should change the day-27 output"


# ── Gradient flow ────────────────────────────────────────────────────────────

def test_gradient_flows_to_all_parameters():
    enc = _encoder()
    out = enc(torch.randn(4, 18, 28))
    out.sum().backward()
    for name, param in enc.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum().item() > 0.0, f"{name} has a zero gradient"


# ── Parameter count ──────────────────────────────────────────────────────────

def test_parameter_count_in_expected_range():
    n_params = sum(p.numel() for p in TemporalPatternEncoder().parameters())
    assert 30_000 <= n_params <= 80_000, f"unexpected parameter count: {n_params}"


# ── Determinism ──────────────────────────────────────────────────────────────

def test_determinism_with_seed():
    x = torch.randn(3, 18, 28)
    seed_everything(42)
    enc1 = TemporalPatternEncoder()
    enc1.eval()
    out1 = enc1(x)
    seed_everything(42)
    enc2 = TemporalPatternEncoder()
    enc2.eval()
    out2 = enc2(x)
    assert torch.equal(out1, out2)
