"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.

    Args:
        seed: Master seed reused for every RNG.
        deterministic_torch: If ``True``, also disable cuDNN benchmark mode and set
            ``torch.use_deterministic_algorithms(True)`` (when available). Has no
            visible cost on CPU; toggle off for GPU runs that benefit from non-
            deterministic kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
