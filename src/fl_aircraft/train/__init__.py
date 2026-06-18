"""Training entrypoints (centralized, local-only, federated).

Public API::

    from fl_aircraft.train import (
        train_centralized, train_one_epoch, evaluate,
        EpochRecord, TrainingHistory, history_as_rows,
        train_local_only_clients, ClientRun, LocalOnlyResults,
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
from .local_only import ClientRun, LocalOnlyResults, train_local_only_clients

__all__ = [
    "ClientRun",
    "EpochRecord",
    "LocalOnlyResults",
    "TrainingHistory",
    "evaluate",
    "history_as_rows",
    "iter_state_dict_floats",
    "train_centralized",
    "train_local_only_clients",
    "train_one_epoch",
]
