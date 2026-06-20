"""Combined RUL + fault loss for the multi-task CNN.

Design choices
--------------
- **Huber** for RUL regression, not MSE. Huber is quadratic for small errors and
  linear for large ones, which keeps a few outlier engines (very long lifetimes)
  from dominating the gradient.
- **BCEWithLogitsLoss** for fault classification — numerically stable and
  accepts an optional ``pos_weight`` for the class-imbalance fix that
  :ref:`RQ2` will lean on.
- **Tunable mixing weight** ``lambda_fault``. We default to ``0.5`` so the two
  losses have comparable magnitudes during early training (RUL Huber sits
  around 10–30 with capped RUL=125; BCE around 0.4–0.7 before training kicks
  in, so a factor of ~30 on the BCE side would otherwise dominate; ``0.5``
  keeps RUL primary, which matches the project brief's emphasis on
  prognostics).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .multitask_cnn import RULPrediction


@dataclass(frozen=True)
class LossOutputs:
    """Container for one mini-batch loss evaluation.

    All tensors are 0-dimensional scalars; ``total`` is what you call
    ``.backward()`` on. The components are exposed for per-task logging.
    """

    total: torch.Tensor
    rul: torch.Tensor
    fault: torch.Tensor


class MultiTaskLoss(nn.Module):
    """``L = L_huber(rul) + lambda_fault * L_bce(fault_logits)``.

    Args:
        lambda_fault: Weight on the BCE term. Larger values prioritise the
            classification head; smaller values prioritise RUL accuracy.
        huber_delta: Huber transition point in label units (cycles). Default
            ``10.0`` cycles ≈ 8 % of the 125-cycle cap — small enough to keep
            most predictions in the quadratic regime, large enough that early
            outlier gradients are bounded.
        fault_pos_weight: Multiplies the positive-class BCE term. Pass
            ``num_negatives / num_positives`` from the training set to
            compensate for fault-label imbalance. ``None`` => disabled.
    """

    def __init__(
        self,
        lambda_fault: float = 0.5,
        huber_delta: float = 10.0,
        fault_pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        if lambda_fault < 0:
            raise ValueError(f"lambda_fault must be >= 0, got {lambda_fault}.")
        if huber_delta <= 0:
            raise ValueError(f"huber_delta must be > 0, got {huber_delta}.")
        self.lambda_fault = float(lambda_fault)
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="mean")
        pw = (
            torch.tensor(float(fault_pos_weight), dtype=torch.float32)
            if fault_pos_weight is not None
            else None
        )
        # Registered as a buffer so it moves with .to(device) and is captured
        # by state_dict (matters for FL when the server inspects clients).
        self.register_buffer("fault_pos_weight", pw)
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pw if pw is not None else None, reduction="mean"
        )

    def forward(
        self,
        pred: RULPrediction,
        y_rul: torch.Tensor,
        y_fault: torch.Tensor,
    ) -> LossOutputs:
        if y_rul.shape != pred.rul.shape:
            raise ValueError(
                f"y_rul shape {tuple(y_rul.shape)} != pred.rul shape "
                f"{tuple(pred.rul.shape)}."
            )
        if y_fault.shape != pred.fault_logits.shape:
            raise ValueError(
                f"y_fault shape {tuple(y_fault.shape)} != pred.fault_logits "
                f"shape {tuple(pred.fault_logits.shape)}."
            )
        if y_fault.dtype != torch.float32:
            # BCEWithLogitsLoss requires float targets.
            y_fault = y_fault.float()
        l_rul = self.huber(pred.rul, y_rul)
        l_fault = self.bce(pred.fault_logits, y_fault)
        total = l_rul + self.lambda_fault * l_fault
        return LossOutputs(total=total, rul=l_rul.detach(), fault=l_fault.detach())
