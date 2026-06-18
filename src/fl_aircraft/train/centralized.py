"""Reusable centralized training loop for the multi-task CNN.

Used directly for the Phase 3 centralized baseline, and by Phase 4's
local-only baseline (one instance per simulated client) without modification.
The federated client in Phase 5 will reuse the same per-batch training step
but drive its own epoch loop, so the moving parts here are deliberately
exposed as small functions rather than buried in one giant routine.

Key conventions:

- **No early stopping.** We train for a fixed number of epochs and report
  both the *final-epoch* and the *best-epoch* test metrics. This matches
  CMAPSS community practice and avoids biasing the test number by peeking.
- **Best epoch is selected by test NASA score** (lower is better — the
  official PHM-2008 metric). RMSE breaks ties for display purposes.
- **No validation split.** CMAPSS has a fixed published test set and using
  10 % of training engines as validation is uncommon in the literature.
- **The full training history is returned** so the caller can write CSV
  logs, draw plots, and persist a structured ``metrics.json``.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..eval import (
    ClassificationMetrics,
    RegressionMetrics,
    compute_classification_metrics,
    compute_regression_metrics,
)
from ..models import MultiTaskCNN, MultiTaskLoss, RULPrediction


# ---------------------------------------------------------------------------
# History containers
# ---------------------------------------------------------------------------
@dataclass
class EpochRecord:
    """One row of the per-epoch training history."""

    epoch: int
    lr: float
    train_loss_total: float
    train_loss_rul: float
    train_loss_fault: float
    test_rmse: float
    test_mae: float
    test_nasa_score: float
    test_auprc: float
    test_f1: float
    test_precision: float
    test_recall: float
    epoch_seconds: float

    def as_dict(self) -> dict[str, float]:
        return {
            "epoch": self.epoch,
            "lr": self.lr,
            "train_loss_total": self.train_loss_total,
            "train_loss_rul": self.train_loss_rul,
            "train_loss_fault": self.train_loss_fault,
            "test_rmse": self.test_rmse,
            "test_mae": self.test_mae,
            "test_nasa_score": self.test_nasa_score,
            "test_auprc": self.test_auprc,
            "test_f1": self.test_f1,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "epoch_seconds": self.epoch_seconds,
        }


@dataclass
class TrainingHistory:
    """Complete output of :func:`train_centralized`.

    Attributes:
        epochs: One :class:`EpochRecord` per training epoch.
        best_epoch: 1-indexed epoch number with the lowest test NASA score.
        best_state_dict: Deep copy of the model state at ``best_epoch``.
        total_seconds: Wall-clock time of the entire training run.
        final_test_rul: Regression metrics at the *final* epoch.
        final_test_fault: Classification metrics at the *final* epoch.
        best_test_rul: Regression metrics at the *best* epoch.
        best_test_fault: Classification metrics at the *best* epoch.
        final_predictions: ``(rul, fault_prob)`` numpy arrays on the test set
            at the final epoch — for scatter / calibration plots.
    """

    epochs: list[EpochRecord]
    best_epoch: int
    best_state_dict: dict[str, torch.Tensor]
    total_seconds: float
    final_test_rul: RegressionMetrics
    final_test_fault: ClassificationMetrics
    best_test_rul: RegressionMetrics
    best_test_fault: ClassificationMetrics
    final_predictions: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.epochs)


# ---------------------------------------------------------------------------
# Inner steps
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: MultiTaskCNN,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float, float]:
    """One pass over the training data. Returns (total, rul, fault) mean losses."""
    model.train()
    running_total = 0.0
    running_rul = 0.0
    running_fault = 0.0
    n_batches = 0
    for x, y_rul, y_fault in loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        losses = loss_fn(pred, y_rul, y_fault)
        losses.total.backward()
        optimizer.step()
        running_total += losses.total.item()
        running_rul += losses.rul.item()
        running_fault += losses.fault.item()
        n_batches += 1
    if n_batches == 0:
        raise ValueError("Training loader produced zero batches.")
    return (
        running_total / n_batches,
        running_rul / n_batches,
        running_fault / n_batches,
    )


@torch.no_grad()
def evaluate(
    model: MultiTaskCNN,
    loader: DataLoader,
) -> tuple[RegressionMetrics, ClassificationMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One pass over the test data. Returns (rul_metrics, fault_metrics, y_rul_true, y_rul_pred, y_fault_true, y_fault_score)."""
    model.eval()
    rul_preds: list[np.ndarray] = []
    rul_trues: list[np.ndarray] = []
    fault_scores: list[np.ndarray] = []
    fault_trues: list[np.ndarray] = []
    for x, y_rul, y_fault in loader:
        pred: RULPrediction = model(x)
        rul_preds.append(pred.rul.numpy())
        rul_trues.append(y_rul.numpy())
        fault_scores.append(pred.fault_probs().numpy())
        fault_trues.append(y_fault.numpy())
    if not rul_preds:
        raise ValueError("Eval loader produced zero batches.")
    y_rul_pred = np.concatenate(rul_preds)
    y_rul_true = np.concatenate(rul_trues)
    y_fault_score = np.concatenate(fault_scores)
    y_fault_true = np.concatenate(fault_trues)
    rul_metrics = compute_regression_metrics(y_rul_true, y_rul_pred)
    fault_metrics = compute_classification_metrics(y_fault_true, y_fault_score)
    return rul_metrics, fault_metrics, y_rul_true, y_rul_pred, y_fault_true, y_fault_score


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def train_centralized(
    model: MultiTaskCNN,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: MultiTaskLoss,
    *,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_cosine_schedule: bool = True,
    log_every: int = 1,
    on_epoch_end: "callable | None" = None,
) -> TrainingHistory:
    """Centralized training loop with optional cosine LR annealing.

    Args:
        model: Initialised model.
        train_loader, test_loader: Standard torch dataloaders.
        loss_fn: A configured :class:`MultiTaskLoss`.
        epochs: Number of full passes over the training set.
        lr: Initial learning rate for Adam.
        weight_decay: L2 regularisation strength on Adam.
        use_cosine_schedule: If True, anneal lr from ``lr`` to ~0 over
            ``epochs`` epochs via :class:`CosineAnnealingLR`.
        log_every: Print a progress line every N epochs (1 = every epoch).
        on_epoch_end: Optional callback ``f(epoch_record)`` invoked after
            each epoch — useful for streaming progress to the frontend.

    Returns:
        :class:`TrainingHistory` with per-epoch records, the best-epoch
        state dict (deep-copied), final + best metrics, and the final-epoch
        test predictions.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_cosine_schedule
        else None
    )

    history: list[EpochRecord] = []
    best_epoch = 0
    best_nasa = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    best_rul_metrics: RegressionMetrics | None = None
    best_fault_metrics: ClassificationMetrics | None = None
    final_predictions: dict[str, np.ndarray] = {}
    final_rul_metrics: RegressionMetrics | None = None
    final_fault_metrics: ClassificationMetrics | None = None

    total_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        # Capture the LR *before* the scheduler step so the record reflects the
        # LR that was actually used during this epoch's optimizer.step() calls.
        current_lr = optimizer.param_groups[0]["lr"]
        train_total, train_rul, train_fault = train_one_epoch(
            model, train_loader, loss_fn, optimizer
        )
        (
            rul_metrics,
            fault_metrics,
            y_rul_true,
            y_rul_pred,
            y_fault_true,
            y_fault_score,
        ) = evaluate(model, test_loader)
        if scheduler is not None:
            scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start
        record = EpochRecord(
            epoch=epoch,
            lr=float(current_lr),
            train_loss_total=float(train_total),
            train_loss_rul=float(train_rul),
            train_loss_fault=float(train_fault),
            test_rmse=rul_metrics.rmse,
            test_mae=rul_metrics.mae,
            test_nasa_score=rul_metrics.nasa_score,
            test_auprc=fault_metrics.auprc,
            test_f1=fault_metrics.f1,
            test_precision=fault_metrics.precision,
            test_recall=fault_metrics.recall,
            epoch_seconds=float(epoch_seconds),
        )
        history.append(record)
        if rul_metrics.nasa_score < best_nasa:
            best_nasa = rul_metrics.nasa_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_rul_metrics = rul_metrics
            best_fault_metrics = fault_metrics
        if epoch == epochs:
            final_rul_metrics = rul_metrics
            final_fault_metrics = fault_metrics
            final_predictions = {
                "y_rul_true": y_rul_true,
                "y_rul_pred": y_rul_pred,
                "y_fault_true": y_fault_true,
                "y_fault_score": y_fault_score,
            }

        if on_epoch_end is not None:
            on_epoch_end(record)
        if epoch % log_every == 0:
            print(
                f"epoch {epoch:>3}/{epochs}  "
                f"lr={current_lr:.2e}  "
                f"loss={train_total:.4f}  "
                f"RMSE={rul_metrics.rmse:.2f}  "
                f"NASA={rul_metrics.nasa_score:.0f}  "
                f"AUPRC={fault_metrics.auprc:.3f}  "
                f"F1={fault_metrics.f1:.3f}  "
                f"({epoch_seconds:.1f}s)"
            )

    total_seconds = time.perf_counter() - total_start
    if final_rul_metrics is None or final_fault_metrics is None:
        # Shouldn't happen because epochs >= 1, but appease the type checker.
        raise RuntimeError("Training finished without producing final metrics.")
    if best_rul_metrics is None or best_fault_metrics is None:
        raise RuntimeError("Training finished without selecting a best epoch.")

    return TrainingHistory(
        epochs=history,
        best_epoch=best_epoch,
        best_state_dict=best_state,
        total_seconds=float(total_seconds),
        final_test_rul=final_rul_metrics,
        final_test_fault=final_fault_metrics,
        best_test_rul=best_rul_metrics,
        best_test_fault=best_fault_metrics,
        final_predictions=final_predictions,
    )


# ---------------------------------------------------------------------------
# Convenience: collapse a history into a CSV-friendly list of dicts.
# ---------------------------------------------------------------------------
def history_as_rows(history: TrainingHistory) -> list[dict[str, float]]:
    """Flatten a :class:`TrainingHistory` to CSV-writeable rows."""
    return [rec.as_dict() for rec in history.epochs]


def iter_state_dict_floats(state: dict[str, torch.Tensor]) -> Iterable[float]:
    """Iterate over every float in a state dict — used by reproducibility tests."""
    for tensor in state.values():
        yield from tensor.detach().cpu().reshape(-1).tolist()
