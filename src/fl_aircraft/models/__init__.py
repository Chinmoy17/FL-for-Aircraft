"""Neural network models for joint RUL regression and fault detection.

Public API::

    from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss
"""
from __future__ import annotations

from .losses import LossOutputs, MultiTaskLoss
from .multitask_cnn import MultiTaskCNN, MultiTaskCNNConfig, RULPrediction

__all__ = [
    "LossOutputs",
    "MultiTaskCNN",
    "MultiTaskCNNConfig",
    "MultiTaskLoss",
    "RULPrediction",
]
