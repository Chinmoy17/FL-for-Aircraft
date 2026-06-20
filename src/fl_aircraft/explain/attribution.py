"""Integrated-Gradients attribution wrapper for the multi-task CNN.

Why Integrated Gradients (IG)?

- **Axiomatic.** IG satisfies the *completeness* axiom: the sum of the
  attribution scores equals the difference in model output between the
  baseline and the actual input (Sundararajan, Taly & Yan, ICML 2017).
  That makes the per-cell contribution scores additively meaningful in
  cycle units.
- **Cheap.** For our ~30k-param model on a ``(30, 17)`` window, IG runs
  in milliseconds on CPU — ~50 forward+backward passes via 50-step
  Riemann approximation. No need for SHAP's combinatorial sampling.
- **Tested in PyTorch.** ``captum.attr.IntegratedGradients`` is the
  canonical implementation.

What "attribution" means here:

For one test engine's window ``X`` of shape ``(window_size, n_features)``,
IG produces a same-shape tensor ``A`` where ``A[t, f]`` says: how much of
the predicted RUL (in cycles) is due to feature ``f`` at cycle ``t``.

Positive values **raise** RUL ("looks healthy"); negative values **lower**
RUL ("looks like a fault"). Sum over all cells ≈ ``predicted_rul -
baseline_rul``.

This module is purely numeric. The ontology lookup and English narrative
live in :mod:`fl_aircraft.explain.narrative`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch import nn

from ..models import MultiTaskCNN


TargetHead = Literal["rul", "fault"]


@dataclass(frozen=True)
class AttributionResult:
    """Numeric output of :func:`attribute_window`.

    Attributes:
        window: The input window, shape ``(window_size, n_features)``.
        feature_cols: Feature column names in the order matching ``window``.
        attribution: Same shape as ``window``. ``attribution[t, f]`` is the
            contribution (in cycles for the RUL head, or in logit units for
            the fault head) of feature ``f`` at cycle ``t``.
        predicted_value: The model's output for this window.
        baseline_value: The model's output for the baseline window (typically
            the all-zeros window in normalized space, which corresponds to
            "average" sensor readings).
        target_head: Which head was attributed (``"rul"`` or ``"fault"``).
        convergence_delta: How well IG's completeness axiom is satisfied —
            should be close to zero. Larger absolute values mean the
            attribution map is less trustworthy.

    The :py:meth:`per_sensor_score` helper returns a 1-D array summing each
    sensor's contribution across the time axis.
    """

    window: np.ndarray
    feature_cols: tuple[str, ...]
    attribution: np.ndarray
    predicted_value: float
    baseline_value: float
    target_head: TargetHead
    convergence_delta: float

    def __post_init__(self) -> None:
        if self.window.shape != self.attribution.shape:
            raise ValueError(
                f"window {self.window.shape} != attribution {self.attribution.shape}"
            )
        if self.window.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"feature_cols length {len(self.feature_cols)} != window cols "
                f"{self.window.shape[1]}"
            )

    @property
    def window_size(self) -> int:
        return int(self.window.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.window.shape[1])

    def per_sensor_score(self) -> np.ndarray:
        """Sum attribution across the time axis. Shape: ``(n_features,)``."""
        return self.attribution.sum(axis=0)

    def top_sensors(self, k: int = 5) -> list[tuple[str, float]]:
        """Top-k feature names by **absolute** contribution magnitude.

        Returns ``[(column, signed_score), ...]`` ordered by ``|score|`` desc.
        """
        scores = self.per_sensor_score()
        order = np.argsort(-np.abs(scores))[:k]
        return [(self.feature_cols[int(i)], float(scores[int(i)])) for i in order]

    def total_attribution(self) -> float:
        """Sum across the entire attribution map (≈ predicted − baseline)."""
        return float(self.attribution.sum())


# ---------------------------------------------------------------------------
# Single-window attribution
# ---------------------------------------------------------------------------
def _wrap_head(model: MultiTaskCNN, target_head: TargetHead) -> nn.Module:
    """Return a callable that yields the scalar output of one head, per sample.

    captum's IntegratedGradients expects the model to produce a (batch,)
    tensor of scalars. Our model returns a ``RULPrediction`` dataclass, so
    we wrap it.
    """

    class _Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pred = self.inner(x)
            if target_head == "rul":
                return pred.rul
            return pred.fault_logits

    return _Wrapper()


def attribute_window(
    model: MultiTaskCNN,
    window: np.ndarray,
    feature_cols: list[str],
    *,
    target_head: TargetHead = "rul",
    n_steps: int = 50,
    baseline: np.ndarray | None = None,
) -> AttributionResult:
    """Compute Integrated-Gradients attribution for one window.

    Args:
        model: Trained :class:`MultiTaskCNN`. Will be set to ``eval()`` mode
            for the duration of the call; restored to its prior mode after.
        window: Shape ``(window_size, n_features)``. Already z-score
            normalized (same preprocessing as training).
        feature_cols: Column names in the order matching ``window``.
        target_head: ``"rul"`` (default) or ``"fault"``. RUL attributions are
            in cycle units; fault attributions are in pre-sigmoid logit units.
        n_steps: Riemann-approximation steps for the path integral. 50 is the
            captum default; 20 is sufficient for our model with negligible
            quality loss.
        baseline: Reference input. Defaults to the all-zeros window, which
            in z-score space corresponds to a sensor reading at the
            training-set mean for every sensor. Must have the same shape as
            ``window`` if provided.
    """
    if window.ndim != 2:
        raise ValueError(f"window must be 2-D, got shape {window.shape}")
    if window.shape[1] != len(feature_cols):
        raise ValueError(
            f"window cols {window.shape[1]} != len(feature_cols) {len(feature_cols)}"
        )
    if baseline is not None and baseline.shape != window.shape:
        raise ValueError(
            f"baseline {baseline.shape} must match window {window.shape}"
        )
    if target_head not in ("rul", "fault"):
        raise ValueError(f"target_head must be 'rul' or 'fault', got {target_head!r}")

    was_training = model.training
    model.eval()
    try:
        wrapped = _wrap_head(model, target_head)
        ig = IntegratedGradients(wrapped)

        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)  # (1, T, F)
        if baseline is None:
            b = torch.zeros_like(x)
        else:
            b = torch.from_numpy(baseline.astype(np.float32)).unsqueeze(0)

        # Reference values BEFORE attribution call so we know the predicted output.
        with torch.no_grad():
            predicted = float(wrapped(x).item())
            baseline_v = float(wrapped(b).item())

        # Captum's IG accepts both `target` and direct scalar output; since
        # `wrapped` already returns a scalar per sample we don't need a target.
        attributions, delta = ig.attribute(
            x, baselines=b, n_steps=n_steps, return_convergence_delta=True,
        )
    finally:
        if was_training:
            model.train()

    a = attributions.detach().cpu().numpy().squeeze(0)  # (T, F)
    return AttributionResult(
        window=window.astype(np.float32),
        feature_cols=tuple(feature_cols),
        attribution=a.astype(np.float32),
        predicted_value=predicted,
        baseline_value=baseline_v,
        target_head=target_head,
        convergence_delta=float(delta.detach().item()),
    )


# ---------------------------------------------------------------------------
# Batch attribution
# ---------------------------------------------------------------------------
def attribute_dataset(
    model: MultiTaskCNN,
    windows: np.ndarray,
    feature_cols: list[str],
    *,
    target_head: TargetHead = "rul",
    n_steps: int = 50,
    baseline: np.ndarray | None = None,
) -> list[AttributionResult]:
    """Run :func:`attribute_window` over a batch of windows.

    Convenience wrapper for explaining several engines in one call.
    """
    if windows.ndim != 3:
        raise ValueError(f"windows must be 3-D (N, T, F), got shape {windows.shape}")
    return [
        attribute_window(
            model, windows[i], feature_cols,
            target_head=target_head, n_steps=n_steps, baseline=baseline,
        )
        for i in range(windows.shape[0])
    ]
