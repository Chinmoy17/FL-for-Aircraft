"""Tests for the RUL regression and fault classification metrics."""
from __future__ import annotations

import numpy as np
import pytest

from fl_aircraft.eval import (
    auprc,
    compute_classification_metrics,
    compute_regression_metrics,
    mae,
    nasa_score,
    rmse,
)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def test_regression_metrics_zero_when_perfect() -> None:
    y = np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float32)
    assert rmse(y, y) == pytest.approx(0.0)
    assert mae(y, y) == pytest.approx(0.0)
    assert nasa_score(y, y) == pytest.approx(0.0)


def test_rmse_and_mae_against_known_values() -> None:
    y_true = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y_pred = np.array([12.0, 18.0, 31.0], dtype=np.float32)
    # errors: +2, -2, +1 -> abs mean = 5/3, squared mean = 9/3 = 3
    assert mae(y_true, y_pred) == pytest.approx(5 / 3, rel=1e-6)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(3.0), rel=1e-6)


def test_nasa_score_penalises_lateness_harder_than_earliness() -> None:
    """The asymmetric exponential is the whole point of the CMAPSS score."""
    y_true = np.array([50.0], dtype=np.float32)
    # 10 cycles early vs 10 cycles late, identical magnitude.
    s_early = nasa_score(y_true, y_true - 10.0)  # d = -10
    s_late = nasa_score(y_true, y_true + 10.0)  # d = +10
    # Reference values from the PHM-08 spec: exp(10/13)-1 vs exp(10/10)-1.
    assert s_early == pytest.approx(np.exp(10 / 13) - 1, rel=1e-6)
    assert s_late == pytest.approx(np.exp(10 / 10) - 1, rel=1e-6)
    assert s_late > s_early


def test_nasa_score_sums_across_engines() -> None:
    y_true = np.array([50.0, 100.0], dtype=np.float32)
    y_pred = np.array([45.0, 110.0], dtype=np.float32)  # d = -5, +10
    expected = (np.exp(5 / 13) - 1) + (np.exp(10 / 10) - 1)
    assert nasa_score(y_true, y_pred) == pytest.approx(expected, rel=1e-6)


def test_compute_regression_metrics_bundle() -> None:
    y_true = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y_pred = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    m = compute_regression_metrics(y_true, y_pred)
    assert m.as_dict() == {"rmse": 0.0, "mae": 0.0, "nasa_score": 0.0}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def test_auprc_perfect_separation_is_one() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auprc(y_true, y_score) == pytest.approx(1.0)


def test_auprc_returns_zero_when_no_positives() -> None:
    """Convention: undefined AUPRC -> 0.0 so downstream aggregations stay numeric."""
    y_true = np.zeros(10, dtype=np.int8)
    y_score = np.random.rand(10)
    assert auprc(y_true, y_score) == 0.0


def test_auprc_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError):
        auprc(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))


def test_classification_metrics_perfect_threshold() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int8)
    y_score = np.array([0.1, 0.2, 0.9, 0.8], dtype=np.float32)
    m = compute_classification_metrics(y_true, y_score, threshold=0.5)
    assert m.f1 == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.auprc == pytest.approx(1.0)
    assert m.positive_rate == 0.5


def test_classification_metrics_all_wrong() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int8)
    y_score = np.array([0.9, 0.8, 0.1, 0.2], dtype=np.float32)
    m = compute_classification_metrics(y_true, y_score, threshold=0.5)
    assert m.f1 == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------
def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        rmse(np.array([1.0, 2.0]), np.array([1.0]))


def test_metrics_reject_2d_inputs() -> None:
    with pytest.raises(ValueError):
        rmse(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))


def test_metrics_reject_empty_inputs() -> None:
    with pytest.raises(ValueError):
        rmse(np.array([], dtype=np.float32), np.array([], dtype=np.float32))
