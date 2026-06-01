"""Reproducibility utilities for deterministic training.

Public API:
    seed_everything(seed) - Fix all random seeds for full reproducibility.
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Fixes seeds for: Python stdlib random, NumPy, PyTorch (CPU + CUDA).
    Also sets deterministic algorithm flags for CUDA reproducibility.

    Args:
        seed: Integer seed value. Default 42.

    Note:
        Deterministic mode may reduce performance by 10-20% on GPU.
        This is acceptable for research; disable for production inference.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (reproducibility > speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.0+ deterministic mode
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # Not all ops have deterministic implementations

    logger.info("Seeded all RNGs with seed=%d", seed)
