"""Device selection for training/evaluation.

Centralises the "which device do we run on?" decision so every training
loop can share it. The rules, in priority order:

1. An explicit ``prefer`` argument (e.g. from a ``--device`` CLI flag).
2. The ``FL_DEVICE`` environment variable (handy on Kaggle where each
   worker process can be pinned to a different GPU via
   ``FL_DEVICE=cuda:0`` / ``cuda:1``).
3. ``cuda`` if a GPU is visible, else ``cpu``.

Everything downstream is written to *infer* the device from the model
(``next(model.parameters()).device``) rather than threading a device
argument through every function, so moving a model with ``.to(device)``
is the single switch that turns GPU on. On a machine with no GPU this
module always returns ``cpu`` and behaviour is bit-identical to before.
"""
from __future__ import annotations

import os

import torch

__all__ = ["get_device", "resolve_device"]


def resolve_device(prefer: str | torch.device | None = None) -> torch.device:
    """Return the torch device to use, honouring ``prefer`` then ``FL_DEVICE``.

    Args:
        prefer: Explicit device string/object (e.g. ``"cuda"``, ``"cuda:1"``,
            ``"cpu"``). Takes precedence over everything. ``None`` falls back
            to the ``FL_DEVICE`` env var, then to auto-detection.

    Returns:
        A concrete :class:`torch.device`. Never raises for an unavailable
        CUDA device here — construction is cheap; the failure (if any)
        surfaces when a tensor is actually moved.
    """
    if prefer is not None:
        return torch.device(prefer)
    env = os.environ.get("FL_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Back-compat / conventional alias.
get_device = resolve_device
