"""Tests for the reusable centralized training loop."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fl_aircraft.data import (
    CMAPSSConfig,
    CMAPSSWindowDataset,
    Normalizer,
    load_and_label_train,
    load_raw,
    load_test_rul,
    make_test_windows,
    make_training_windows,
)
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss
from fl_aircraft.train import (
    TrainingHistory,
    history_as_rows,
    iter_state_dict_floats,
    train_centralized,
)
from fl_aircraft.utils import seed_everything


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fd001_loaders(data_dir: Path) -> tuple[DataLoader, DataLoader, CMAPSSConfig, float]:
    """Tiny FD001 train / test loaders shared by all tests in this module."""
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir, window_size=30)
    train_df = load_and_label_train(cfg)
    normalizer = Normalizer.fit(train_df, cfg.feature_cols)
    train_arrays = make_training_windows(
        normalizer.transform(train_df), cfg.feature_cols, cfg.window_size, cfg.stride
    )
    test_arrays = make_test_windows(
        normalizer.transform(load_raw(cfg.subset, "test", data_dir)),
        load_test_rul(cfg.subset, data_dir),
        cfg.feature_cols,
        cfg.window_size,
        cfg.rul_cap,
        cfg.fault_threshold,
    )
    train_ds = CMAPSSWindowDataset(train_arrays)
    test_ds = CMAPSSWindowDataset(test_arrays)
    # Drop shuffle for reproducibility tests.
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    pos_weight = float(train_arrays.y_fault.shape[0] - train_arrays.y_fault.sum()) / float(
        max(train_arrays.y_fault.sum(), 1)
    )
    return train_loader, test_loader, cfg, pos_weight


def _make_model_and_loss(cfg: CMAPSSConfig, pos_weight: float) -> tuple[MultiTaskCNN, MultiTaskLoss]:
    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=cfg.n_features, window_size=cfg.window_size))
    loss_fn = MultiTaskLoss(lambda_fault=0.5, fault_pos_weight=pos_weight)
    return model, loss_fn


# ---------------------------------------------------------------------------
# Basic shape + correctness
# ---------------------------------------------------------------------------
def test_train_centralized_2epoch_smoke(fd001_loaders) -> None:
    train_loader, test_loader, cfg, pw = fd001_loaders
    seed_everything(0)
    model, loss_fn = _make_model_and_loss(cfg, pw)
    history = train_centralized(
        model, train_loader, test_loader, loss_fn,
        epochs=2, lr=1e-3, log_every=99,
    )
    assert isinstance(history, TrainingHistory)
    assert len(history) == 2
    assert history.epochs[0].epoch == 1 and history.epochs[1].epoch == 2
    for rec in history.epochs:
        assert rec.train_loss_total > 0
        assert np.isfinite(rec.train_loss_total)
        assert np.isfinite(rec.test_rmse)
        assert np.isfinite(rec.test_nasa_score)
        assert 0 <= rec.test_auprc <= 1
        assert 0 <= rec.test_f1 <= 1
        assert rec.lr > 0


def test_train_centralized_rejects_zero_epochs(fd001_loaders) -> None:
    train_loader, test_loader, cfg, pw = fd001_loaders
    model, loss_fn = _make_model_and_loss(cfg, pw)
    with pytest.raises(ValueError):
        train_centralized(model, train_loader, test_loader, loss_fn, epochs=0)


# ---------------------------------------------------------------------------
# Best-epoch tracking
# ---------------------------------------------------------------------------
def test_best_epoch_has_lowest_nasa_score(fd001_loaders) -> None:
    train_loader, test_loader, cfg, pw = fd001_loaders
    seed_everything(0)
    model, loss_fn = _make_model_and_loss(cfg, pw)
    history = train_centralized(
        model, train_loader, test_loader, loss_fn,
        epochs=3, lr=1e-3, log_every=99,
    )
    nasa_scores = [rec.test_nasa_score for rec in history.epochs]
    assert history.best_test_rul.nasa_score == min(nasa_scores)
    assert history.epochs[history.best_epoch - 1].test_nasa_score == history.best_test_rul.nasa_score


def test_best_state_dict_is_a_deep_copy(fd001_loaders) -> None:
    """Mutating the model after training must not affect history.best_state_dict."""
    train_loader, test_loader, cfg, pw = fd001_loaders
    seed_everything(0)
    model, loss_fn = _make_model_and_loss(cfg, pw)
    history = train_centralized(
        model, train_loader, test_loader, loss_fn,
        epochs=2, lr=1e-3, log_every=99,
    )
    pre_floats = list(iter_state_dict_floats(history.best_state_dict))
    # Brutalise the live model.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    post_floats = list(iter_state_dict_floats(history.best_state_dict))
    assert pre_floats == post_floats


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_same_seed_produces_same_final_metrics(fd001_loaders) -> None:
    train_loader, test_loader, cfg, pw = fd001_loaders

    def _run() -> tuple[float, float, float]:
        seed_everything(42)
        model, loss_fn = _make_model_and_loss(cfg, pw)
        history = train_centralized(
            model, train_loader, test_loader, loss_fn,
            epochs=2, lr=1e-3, log_every=99,
        )
        return (
            history.final_test_rul.rmse,
            history.final_test_rul.nasa_score,
            history.final_test_fault.auprc,
        )

    a = _run()
    b = _run()
    # Deterministic mode + identical seed = bit-exact metrics on CPU.
    assert a == b


# ---------------------------------------------------------------------------
# Cosine schedule actually anneals the LR
# ---------------------------------------------------------------------------
def test_cosine_schedule_anneals_lr(fd001_loaders) -> None:
    train_loader, test_loader, cfg, pw = fd001_loaders
    seed_everything(0)
    model, loss_fn = _make_model_and_loss(cfg, pw)
    history = train_centralized(
        model, train_loader, test_loader, loss_fn,
        epochs=3, lr=1e-3, use_cosine_schedule=True, log_every=99,
    )
    lrs = [rec.lr for rec in history.epochs]
    # Strictly decreasing under cosine.
    assert lrs[0] > lrs[1] > lrs[2]


def test_no_schedule_keeps_lr_constant(fd001_loaders) -> None:
    train_loader, test_loader, cfg, pw = fd001_loaders
    seed_everything(0)
    model, loss_fn = _make_model_and_loss(cfg, pw)
    history = train_centralized(
        model, train_loader, test_loader, loss_fn,
        epochs=3, lr=1e-3, use_cosine_schedule=False, log_every=99,
    )
    lrs = [rec.lr for rec in history.epochs]
    assert lrs[0] == lrs[1] == lrs[2] == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------
def test_history_as_rows_round_trips_through_csv(fd001_loaders, tmp_path: Path) -> None:
    import csv as _csv

    train_loader, test_loader, cfg, pw = fd001_loaders
    seed_everything(0)
    model, loss_fn = _make_model_and_loss(cfg, pw)
    history = train_centralized(
        model, train_loader, test_loader, loss_fn,
        epochs=2, lr=1e-3, log_every=99,
    )
    rows = history_as_rows(history)
    assert len(rows) == 2
    expected_keys = {
        "epoch", "lr",
        "train_loss_total", "train_loss_rul", "train_loss_fault",
        "test_rmse", "test_mae", "test_nasa_score",
        "test_auprc", "test_f1", "test_precision", "test_recall",
        "epoch_seconds",
    }
    assert set(rows[0].keys()) == expected_keys

    out = tmp_path / "history.csv"
    with out.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# On-epoch-end callback
# ---------------------------------------------------------------------------
def test_on_epoch_end_callback_fires_once_per_epoch(fd001_loaders) -> None:
    train_loader, test_loader, cfg, pw = fd001_loaders
    seed_everything(0)
    model, loss_fn = _make_model_and_loss(cfg, pw)
    received: list[int] = []
    train_centralized(
        model, train_loader, test_loader, loss_fn,
        epochs=3, lr=1e-3, log_every=99,
        on_epoch_end=lambda rec: received.append(rec.epoch),
    )
    assert received == [1, 2, 3]
