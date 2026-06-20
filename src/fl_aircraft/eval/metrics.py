"""Evaluation metrics for RUL regression and fault detection.

Why these specific metrics?

- **RMSE / MAE** — the two universally reported regression metrics in CMAPSS
  literature; required for cross-paper comparison.
- **NASA CMAPSS score** — the official PHM '08 challenge metric. Asymmetric
  exponential penalty: late predictions (overestimating RUL) are penalised
  much more heavily than early ones, because in an aviation safety context an
  overdue engine failure is far worse than premature maintenance. Formal
  definition (Saxena, Goebel, Simon & Eklund, PHM 2008):

      d = rul_pred - rul_true
      s_i = exp(-d_i / 13) - 1     if d_i <  0   (early)
            exp( d_i / 10) - 1     if d_i >= 0   (late)
      Score = sum_i s_i

  Lower is better; zero is perfect. The asymmetric constants (13 vs 10) come
  straight from the challenge specification.
- **AUPRC** — Area under the precision-recall curve. The right discrimination
  metric for the imbalanced fault head (vs ROC-AUC which inflates scores under
  imbalance). Established as the imbalance-friendly choice by Davis & Goadrich
  (ICML 2006) and reinforced by the FedCCFA paper this project's RQ4 cites.
- **F1 / precision / recall @ 0.5** — operational metrics for a ground
  engineer deciding whether to act on a fault alert.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# RUL regression metrics
# ---------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error in cycles."""
    _check_shapes(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in cycles."""
    _check_shapes(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official CMAPSS / PHM-2008 asymmetric score (lower is better, 0 = perfect)."""
    _check_shapes(y_true, y_pred)
    d = y_pred.astype(np.float64) - y_true.astype(np.float64)
    early = d < 0
    s = np.empty_like(d, dtype=np.float64)
    s[early] = np.exp(-d[early] / 13.0) - 1.0
    s[~early] = np.exp(d[~early] / 10.0) - 1.0
    return float(np.sum(s))


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    nasa_score: float

    def as_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "nasa_score": self.nasa_score}


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> RegressionMetrics:
    """All RUL metrics in one pass."""
    return RegressionMetrics(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        nasa_score=nasa_score(y_true, y_pred),
    )


# ---------------------------------------------------------------------------
# Fault detection metrics
# ---------------------------------------------------------------------------
def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (a.k.a. average precision).

    ``y_score`` must be a probability or any monotonic confidence score;
    ``y_true`` must be ``{0, 1}``.
    """
    _check_shapes(y_true, y_score)
    y_true = y_true.astype(np.int8)
    if y_true.min() < 0 or y_true.max() > 1:
        raise ValueError("y_true must contain only 0 / 1 labels for AUPRC.")
    if y_true.sum() == 0:
        # AUPRC is undefined with zero positives; return 0.0 by convention so
        # downstream aggregations remain numerical.
        return 0.0
    return float(average_precision_score(y_true, y_score))


@dataclass(frozen=True)
class ClassificationMetrics:
    auprc: float
    f1: float
    precision: float
    recall: float
    threshold: float
    positive_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "auprc": self.auprc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "threshold": self.threshold,
            "positive_rate": self.positive_rate,
        }


def compute_classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> ClassificationMetrics:
    """Discrimination (AUPRC) + thresholded operational metrics in one pass."""
    _check_shapes(y_true, y_score)
    y_true = y_true.astype(np.int8)
    y_pred = (y_score >= threshold).astype(np.int8)
    return ClassificationMetrics(
        auprc=auprc(y_true, y_score),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        threshold=float(threshold),
        positive_rate=float(y_true.mean()),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D arrays, got shape {a.shape}.")
    if a.size == 0:
        raise ValueError("Empty array; cannot compute metric.")
