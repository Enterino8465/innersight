"""Model factory for InnerSight UEBA.

Provides a single entry point that reads a config dict and returns the
correct model + device pair.  All training scripts should go through
``build_model`` rather than constructing models directly so that swapping
architectures only requires a config change.

Supported model types
---------------------
``mlp``        — flat feature MLP (original baseline)
``graphsage``  — InsiderThreatGNN: HeteroGraphSAGE backbone + MLP head

Config shape (YAML → dict)
--------------------------
model:
  type: graphsage         # or mlp
  # MLP-specific
  layer_sizes: [18, 64, 32, 1]
  # GraphSAGE-specific
  hidden_dim: 128
  num_layers: 2
  dropout: 0.3
  head_layers: [128, 64]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import InsiderThreatMLP, get_device
from .graphsage import InsiderThreatGNN


def build_model(
    config: dict,
    metadata: tuple | None = None,
) -> tuple[nn.Module, torch.device]:
    """Construct a model from a config dict and move it to the best device.

    Parameters
    ----------
    config:
        Training configuration.  The ``model`` sub-dict controls which
        architecture is built and with which hyper-parameters.
    metadata:
        ``HeteroData.metadata()`` tuple — required when
        ``config['model']['type'] == 'graphsage'``, ignored otherwise.

    Returns
    -------
    tuple[nn.Module, torch.device]
        ``(model, device)`` where ``model`` is already on ``device``.

    Raises
    ------
    ValueError
        If ``model_type`` is unrecognised.
    AssertionError
        If ``model_type == 'graphsage'`` but ``metadata`` is ``None``.
    """
    model_cfg  = config.get('model', {})
    model_type = model_cfg.get('type', 'mlp')
    device     = get_device()

    if model_type == 'mlp':
        model: nn.Module = InsiderThreatMLP(model_cfg['layer_sizes'])

    elif model_type == 'graphsage':
        assert metadata is not None, \
            "build_model: 'graphsage' model type requires graph metadata"
        model = InsiderThreatGNN(
            metadata=metadata,
            hidden_dim=model_cfg.get('hidden_dim', 128),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.3),
            head_layers=model_cfg.get('head_layers', [128, 64]),
        )

    else:
        raise ValueError(
            f"build_model: unknown model type '{model_type}'. "
            f"Valid options: 'mlp', 'graphsage'."
        )

    model = model.to(device)
    return model, device
