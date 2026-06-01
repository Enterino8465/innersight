"""Production PyTorch MLP for insider-threat scoring."""

from __future__ import annotations

import torch
import torch.nn as nn


def _validate_layer_sizes(layer_sizes: list[int]) -> None:
    """Raise ValueError if layer_sizes is not a valid MLP architecture spec."""
    if not isinstance(layer_sizes, (list, tuple)) or len(layer_sizes) < 2:
        raise ValueError(
            f"layer_sizes must have at least 2 elements (input + output), got {layer_sizes!r}"
        )
    for i, size in enumerate(layer_sizes):
        if not isinstance(size, int) or size <= 0:
            raise ValueError(
                f"layer_sizes[{i}] must be a positive integer, got {size!r}"
            )


class InsiderThreatMLP(nn.Module):
    """Feedforward MLP for binary insider-threat classification.

    Produces raw logits; pair with ``nn.BCEWithLogitsLoss`` for training.

    Args:
        layer_sizes: Sizes of each layer, e.g. ``[18, 64, 32, 1]``.
            Must have at least two elements. All values must be positive ints.

    Raises:
        ValueError: If ``layer_sizes`` is invalid.
    """

    def __init__(self, layer_sizes: list[int]) -> None:
        super().__init__()
        _validate_layer_sizes(layer_sizes)
        self.layer_sizes = list(layer_sizes)

        layers: list[nn.Module] = []
        pairs = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        for i, (in_features, out_features) in enumerate(pairs):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(pairs) - 1:   # hidden layers only
                layers.append(nn.ReLU())
        # no activation on the output layer — BCEWithLogitsLoss fuses sigmoid

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return raw logits.

        Args:
            x: Input tensor of shape ``(batch, layer_sizes[0])``.

        Returns:
            Logit tensor of shape ``(batch, layer_sizes[-1])``.
        """
        return self.net(x)


def get_device() -> torch.device:
    """Return the best available compute device: CUDA > MPS > CPU.

    Returns:
        A ``torch.device`` instance ready for ``.to(device)`` calls.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_mlp(config: dict) -> tuple[InsiderThreatMLP, torch.device]:
    """Construct an ``InsiderThreatMLP`` and move it to the best device.

    Args:
        config: Training configuration dict.  ``layer_sizes`` is read from it;
            if absent, the default from ``config.DEFAULT_TRAINING_CONFIG`` is
            used.

    Returns:
        A ``(model, device)`` tuple.  The model is already on ``device``.

    Raises:
        ValueError: If the resolved ``layer_sizes`` is invalid.
    """
    from config import DEFAULT_TRAINING_CONFIG  # local import to avoid circular deps

    layer_sizes: list[int] = config.get(
        "layer_sizes", DEFAULT_TRAINING_CONFIG["layer_sizes"]
    )
    device = get_device()
    model = InsiderThreatMLP(layer_sizes).to(device)
    return model, device


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import DEFAULT_TRAINING_CONFIG

    model, device = build_mlp(DEFAULT_TRAINING_CONFIG)

    print("=" * 60)
    print("Architecture")
    print("=" * 60)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Device:           {device}")

    dummy = torch.randn(4, DEFAULT_TRAINING_CONFIG["layer_sizes"][0], device=device)
    out = model(dummy)
    print(f"\nDummy input  shape: {tuple(dummy.shape)}")
    print(f"Logit output shape: {tuple(out.shape)}")
    print(f"Output sample:      {out.detach().cpu()}")

    # Validate error handling
    try:
        InsiderThreatMLP([18])
    except ValueError as exc:
        print(f"\nError-handling check (expected): {exc}")
