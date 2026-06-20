"""Evaluation metrics (RMSE, NASA score, AUPRC, F1) and plotting utilities.

Public API::

    from fl_aircraft.eval import (
        rmse, mae, nasa_score, compute_regression_metrics, RegressionMetrics,
        auprc, compute_classification_metrics, ClassificationMetrics,
    )
"""
from __future__ import annotations

from .metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    auprc,
    compute_classification_metrics,
    compute_regression_metrics,
    mae,
    nasa_score,
    rmse,
)

__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
    "auprc",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "mae",
    "nasa_score",
    "rmse",
]
