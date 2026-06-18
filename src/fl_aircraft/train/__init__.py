"""Training entrypoints (centralized, local-only, federated).

Public API::

    from fl_aircraft.train import (
        train_centralized, train_one_epoch, evaluate,
        EpochRecord, TrainingHistory, history_as_rows,
    )
"""
from __future__ import annotations

from .centralized import (
    EpochRecord,
    TrainingHistory,
    evaluate,
    history_as_rows,
    iter_state_dict_floats,
    train_centralized,
    train_one_epoch,
)

__all__ = [
    "EpochRecord",
    "TrainingHistory",
    "evaluate",
    "history_as_rows",
    "iter_state_dict_floats",
    "train_centralized",
    "train_one_epoch",
]
