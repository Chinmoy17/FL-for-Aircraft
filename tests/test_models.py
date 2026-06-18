"""Tests for the multi-task CNN and its combined loss."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from fl_aircraft.models import (
    MultiTaskCNN,
    MultiTaskCNNConfig,
    MultiTaskLoss,
    RULPrediction,
)
from fl_aircraft.utils import seed_everything


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_config_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        MultiTaskCNNConfig(n_features=0)
    with pytest.raises(ValueError):
        MultiTaskCNNConfig(n_features=17, window_size=0)
    with pytest.raises(ValueError):
        # 17 is not divisible by gn_groups=8 — caught at config time.
        MultiTaskCNNConfig(n_features=17, conv_channels=(17, 32))
    with pytest.raises(ValueError):
        MultiTaskCNNConfig(n_features=17, dropout=1.0)
    with pytest.raises(ValueError):
        MultiTaskCNNConfig(n_features=17, conv_channels=(32,), kernel_sizes=(3, 5))


# ---------------------------------------------------------------------------
# Forward / backward shapes
# ---------------------------------------------------------------------------
def test_forward_returns_correct_shapes() -> None:
    seed_everything(0)
    cfg = MultiTaskCNNConfig(n_features=17, window_size=30)
    model = MultiTaskCNN(cfg).eval()
    x = torch.randn(4, 30, 17)
    out = model(x)
    assert isinstance(out, RULPrediction)
    assert out.rul.shape == (4,)
    assert out.fault_logits.shape == (4,)
    # RUL head uses softplus so predictions must be non-negative.
    assert (out.rul >= 0).all()


def test_forward_rejects_wrong_input_dims() -> None:
    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=17))
    with pytest.raises(ValueError):
        model(torch.randn(30, 17))  # missing batch dim


def test_forward_is_window_size_agnostic() -> None:
    """AdaptiveAvgPool means the same model accepts variable window lengths."""
    cfg = MultiTaskCNNConfig(n_features=17, window_size=30)
    model = MultiTaskCNN(cfg).eval()
    for T in (20, 30, 50):
        out = model(torch.randn(2, T, 17))
        assert out.rul.shape == (2,)


def test_backward_updates_every_trainable_parameter() -> None:
    seed_everything(0)
    cfg = MultiTaskCNNConfig(n_features=17, window_size=30)
    model = MultiTaskCNN(cfg)
    loss_fn = MultiTaskLoss()
    x = torch.randn(8, 30, 17)
    y_rul = torch.rand(8) * 125
    y_fault = (torch.rand(8) > 0.5).float()
    out = model(x)
    losses = loss_fn(out, y_rul, y_fault)
    losses.total.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient."
        # Allow zero gradients (rare for ReLU dead inputs) but disallow NaN/Inf.
        assert torch.isfinite(p.grad).all(), f"{name} grad has non-finite values."


# ---------------------------------------------------------------------------
# Parameter budget
# ---------------------------------------------------------------------------
def test_parameter_count_under_budget() -> None:
    """Stay under 50k params so training stays fast on CPU."""
    cfg = MultiTaskCNNConfig(n_features=17, window_size=30)
    model = MultiTaskCNN(cfg)
    n = model.count_parameters()
    assert 20_000 < n < 50_000, f"Unexpected param count {n}."


# ---------------------------------------------------------------------------
# FL-safety: no BatchNorm running stats to aggregate
# ---------------------------------------------------------------------------
def test_no_batchnorm_layers_present() -> None:
    """BatchNorm running stats would break naive FedAvg; the model must use GroupNorm."""
    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=17))
    for module in model.modules():
        assert not isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        ), f"Found BatchNorm in model: {type(module).__name__} — incompatible with FedAvg."


def test_train_and_eval_modes_produce_identical_outputs() -> None:
    """GroupNorm has no running stats, so train/eval must agree given fixed dropout."""
    seed_everything(0)
    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=17, dropout=0.0))
    x = torch.randn(4, 30, 17)
    model.train()
    out_train = model(x)
    model.eval()
    out_eval = model(x)
    assert torch.allclose(out_train.rul, out_eval.rul, atol=1e-6)
    assert torch.allclose(out_train.fault_logits, out_eval.fault_logits, atol=1e-6)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_seeded_initialisation_is_reproducible() -> None:
    seed_everything(42)
    a = MultiTaskCNN(MultiTaskCNNConfig(n_features=17))
    seed_everything(42)
    b = MultiTaskCNN(MultiTaskCNNConfig(n_features=17))
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Param {n1} differs between identically-seeded models."


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def test_loss_combines_components_with_lambda() -> None:
    """L_total must equal L_rul + lambda * L_fault to the float precision."""
    cfg = MultiTaskCNNConfig(n_features=17)
    model = MultiTaskCNN(cfg).eval()
    loss_fn = MultiTaskLoss(lambda_fault=0.3)
    x = torch.randn(16, 30, 17)
    y_rul = torch.rand(16) * 125
    y_fault = (torch.rand(16) > 0.5).float()
    pred = model(x)
    losses = loss_fn(pred, y_rul, y_fault)
    expected = losses.rul + 0.3 * losses.fault
    assert torch.allclose(losses.total, expected, atol=1e-6)


def test_loss_rejects_shape_mismatch() -> None:
    cfg = MultiTaskCNNConfig(n_features=17)
    model = MultiTaskCNN(cfg).eval()
    loss_fn = MultiTaskLoss()
    pred = model(torch.randn(4, 30, 17))
    with pytest.raises(ValueError):
        loss_fn(pred, torch.rand(8), torch.zeros(4))
    with pytest.raises(ValueError):
        loss_fn(pred, torch.rand(4), torch.zeros(8))


def test_loss_accepts_int_fault_targets() -> None:
    """CMAPSSWindowDataset emits float fault labels, but int8 inputs must also work."""
    cfg = MultiTaskCNNConfig(n_features=17)
    model = MultiTaskCNN(cfg).eval()
    loss_fn = MultiTaskLoss()
    pred = model(torch.randn(4, 30, 17))
    y_rul = torch.rand(4) * 125
    y_fault_int = torch.tensor([0, 1, 1, 0], dtype=torch.int8)
    losses = loss_fn(pred, y_rul, y_fault_int)
    assert torch.isfinite(losses.total)


def test_loss_pos_weight_increases_loss_on_misclassified_positives() -> None:
    cfg = MultiTaskCNNConfig(n_features=17)
    model = MultiTaskCNN(cfg).eval()
    # Force the model to confidently predict negative for all-positive labels.
    pred = RULPrediction(
        rul=torch.zeros(8),
        fault_logits=torch.full((8,), -5.0),  # logit -> ~0.0067 fault probability
    )
    y_rul = torch.zeros(8)
    y_fault = torch.ones(8)
    base = MultiTaskLoss()(pred, y_rul, y_fault).fault
    weighted = MultiTaskLoss(fault_pos_weight=5.0)(pred, y_rul, y_fault).fault
    assert weighted > base


def test_loss_rejects_invalid_kwargs() -> None:
    with pytest.raises(ValueError):
        MultiTaskLoss(lambda_fault=-0.1)
    with pytest.raises(ValueError):
        MultiTaskLoss(huber_delta=0.0)
