import pytest
import torch
from innersight.backend.models.mlp import InsiderThreatMLP, build_mlp, get_device


LAYER_SIZES = [18, 64, 32, 1]
BATCH       = 16


@pytest.fixture()
def model():
    return InsiderThreatMLP(LAYER_SIZES)


# ── creation ──────────────────────────────────────────────────────────────────

def test_mlp_creation():
    m = InsiderThreatMLP(LAYER_SIZES)
    assert m.layer_sizes == LAYER_SIZES


def test_mlp_stores_layer_sizes():
    sizes = [4, 8, 4, 1]
    m = InsiderThreatMLP(sizes)
    assert m.layer_sizes == sizes


# ── forward shape ─────────────────────────────────────────────────────────────

def test_mlp_forward_shape_standard_batch(model):
    x   = torch.randn(BATCH, 18)
    out = model(x)
    assert out.shape == (BATCH, 1)


@pytest.mark.parametrize("batch", [1, 4, 32])
def test_mlp_forward_shape_various_batches(batch):
    m   = InsiderThreatMLP(LAYER_SIZES)
    out = m(torch.randn(batch, 18))
    assert out.shape == (batch, 1)


# ── logits, not probabilities ─────────────────────────────────────────────────

def test_mlp_output_is_logits(model):
    """Raw logits must be able to go negative (no sigmoid on the output)."""
    torch.manual_seed(0)
    outputs = model(torch.randn(256, 18))
    assert outputs.min().item() < 0, "output should include negative logits"


def test_mlp_output_not_clamped_to_unit_interval(model):
    outputs = model(torch.randn(256, 18))
    # At least some logits should be outside (0,1) for a random-init network
    outside = ((outputs < 0) | (outputs > 1)).any()
    assert outside.item()


# ── parameter count ───────────────────────────────────────────────────────────

def test_mlp_parameter_count(model):
    # [18→64]: 18*64 + 64 = 1216
    # [64→32]: 64*32 + 32 = 2080
    # [32→1 ]: 32*1  +  1 =   33
    expected = 1216 + 2080 + 33
    actual   = sum(p.numel() for p in model.parameters())
    assert actual == expected


# ── architecture depth ────────────────────────────────────────────────────────

def test_mlp_relu_on_hidden_only(model):
    """ReLU should appear between hidden layers but NOT after the output layer."""
    import torch.nn as nn
    layers = list(model.net)
    assert isinstance(layers[-1], nn.Linear), "last module should be Linear (no activation)"
    relu_count = sum(1 for m in layers if isinstance(m, nn.ReLU))
    # [18→64→32→1]: two hidden transitions → two ReLUs
    assert relu_count == len(LAYER_SIZES) - 2


# ── validation / error handling ───────────────────────────────────────────────

def test_mlp_invalid_single_element():
    with pytest.raises(ValueError, match="at least 2"):
        InsiderThreatMLP([18])


def test_mlp_invalid_empty():
    with pytest.raises(ValueError):
        InsiderThreatMLP([])


def test_mlp_invalid_zero_size():
    with pytest.raises(ValueError):
        InsiderThreatMLP([18, 0, 1])


def test_mlp_invalid_negative_size():
    with pytest.raises(ValueError):
        InsiderThreatMLP([18, -4, 1])


def test_mlp_invalid_non_integer():
    with pytest.raises(ValueError):
        InsiderThreatMLP([18, 32.5, 1])  # type: ignore[list-item]


# ── get_device / build_mlp factory ────────────────────────────────────────────

def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_build_mlp_returns_model_and_device():
    from innersight.backend.config import DEFAULT_TRAINING_CONFIG
    model, device = build_mlp(DEFAULT_TRAINING_CONFIG)
    assert isinstance(model, InsiderThreatMLP)
    assert isinstance(device, torch.device)


def test_build_mlp_model_on_correct_device():
    from innersight.backend.config import DEFAULT_TRAINING_CONFIG
    model, device = build_mlp(DEFAULT_TRAINING_CONFIG)
    param_device = next(model.parameters()).device
    assert param_device.type == device.type


def test_build_mlp_uses_config_layer_sizes():
    config = {'layer_sizes': [18, 4, 1]}
    model, _ = build_mlp(config)
    assert model.layer_sizes == [18, 4, 1]
