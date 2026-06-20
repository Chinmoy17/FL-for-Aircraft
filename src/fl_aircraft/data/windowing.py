"""Sliding-window construction and a thin PyTorch ``Dataset`` wrapper.

A "window" is the model's input: ``window_size`` consecutive cycles of one
engine. Following standard CMAPSS practice, each window's labels (RUL and
fault) are taken from its **last** cycle.

Training windows: stride over every engine, yielding many overlapping samples.
Test windows: exactly one window per test engine — the trailing ``window_size``
cycles of its truncated trajectory — paired with the ground-truth RUL from
``RUL_FD00X.txt``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

from .constants import CYCLE_COL, UNIT_ID_COL


@dataclass(frozen=True)
class WindowedArrays:
    """Numpy arrays produced by the windowing functions.

    Shapes:
        X        — ``(N, window_size, n_features)``  float32
        y_rul    — ``(N,)``                          float32  (capped RUL at window end)
        y_fault  — ``(N,)``                          int8     (binary fault label at window end)
        unit_ids — ``(N,)``                          int64    (origin engine id per window)
    """

    X: np.ndarray
    y_rul: np.ndarray
    y_fault: np.ndarray
    unit_ids: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def window_size(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[2])

    def fault_positive_rate(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return float(self.y_fault.mean())


def _slide_one_engine(
    features: np.ndarray, window_size: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(windows, end_indices)`` for one engine's feature matrix.

    ``windows`` shape: ``(k, window_size, n_features)`` where ``k`` is the
    number of valid windows. ``end_indices`` are the indices into ``features``
    of each window's final cycle.
    """
    n_cycles = features.shape[0]
    if n_cycles < window_size:
        empty_w = np.empty((0, window_size, features.shape[1]), dtype=features.dtype)
        return empty_w, np.empty((0,), dtype=np.int64)
    # sliding_window_view -> (n_windows, n_features, window_size); transpose for clarity.
    raw = sliding_window_view(features, window_shape=window_size, axis=0)
    raw = raw.transpose(0, 2, 1)  # (n_windows, window_size, n_features)
    if stride > 1:
        raw = raw[::stride]
    raw = np.ascontiguousarray(raw)
    end_indices = np.arange(window_size - 1, n_cycles, stride, dtype=np.int64)
    return raw, end_indices


def make_training_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    stride: int = 1,
) -> WindowedArrays:
    """Build sliding-window training samples from a labeled, normalized frame."""
    if "RUL_capped" not in df.columns or "fault" not in df.columns:
        raise ValueError(
            "DataFrame must already have RUL_capped and fault columns "
            "(call compute_rul_labels / compute_fault_labels first)."
        )
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}.")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    X_chunks: list[np.ndarray] = []
    y_rul_chunks: list[np.ndarray] = []
    y_fault_chunks: list[np.ndarray] = []
    unit_chunks: list[np.ndarray] = []

    for unit_id, group in df.groupby(UNIT_ID_COL, sort=True):
        group = group.sort_values(CYCLE_COL)
        features = group[feature_cols].to_numpy(dtype=np.float32, copy=False)
        windows, end_idx = _slide_one_engine(features, window_size, stride)
        if windows.shape[0] == 0:
            continue
        rul = group["RUL_capped"].to_numpy(dtype=np.float32, copy=False)
        fault = group["fault"].to_numpy(dtype=np.int8, copy=False)
        X_chunks.append(windows)
        y_rul_chunks.append(rul[end_idx])
        y_fault_chunks.append(fault[end_idx])
        unit_chunks.append(np.full(end_idx.shape[0], int(unit_id), dtype=np.int64))

    if not X_chunks:
        raise ValueError(
            f"No engine in the input frame is long enough to form a window of size {window_size}."
        )

    return WindowedArrays(
        X=np.concatenate(X_chunks, axis=0),
        y_rul=np.concatenate(y_rul_chunks, axis=0),
        y_fault=np.concatenate(y_fault_chunks, axis=0),
        unit_ids=np.concatenate(unit_chunks, axis=0),
    )


def make_test_windows(
    test_df: pd.DataFrame,
    test_rul: np.ndarray,
    feature_cols: list[str],
    window_size: int,
    rul_cap: int,
    fault_threshold: int,
) -> WindowedArrays:
    """One window per test engine — the trailing ``window_size`` cycles.

    Engines shorter than ``window_size`` are left-padded by repeating their
    first cycle so every test engine yields exactly one fixed-shape window.
    """
    engines = sorted(test_df[UNIT_ID_COL].unique())
    if len(engines) != len(test_rul):
        raise ValueError(
            f"# test engines ({len(engines)}) != len(test_rul) ({len(test_rul)}). "
            "These must match — check that test_df and RUL_FD00X.txt are paired correctly."
        )

    X_list: list[np.ndarray] = []
    rul_list: list[float] = []
    fault_list: list[int] = []
    unit_list: list[int] = []

    for unit_id, true_rul in zip(engines, test_rul):
        group = test_df.loc[test_df[UNIT_ID_COL] == unit_id].sort_values(CYCLE_COL)
        features = group[feature_cols].to_numpy(dtype=np.float32, copy=False)
        n_cycles = features.shape[0]
        if n_cycles < window_size:
            pad = np.repeat(features[:1], window_size - n_cycles, axis=0)
            window = np.concatenate([pad, features], axis=0)
        else:
            window = features[-window_size:]
        X_list.append(np.ascontiguousarray(window)[None, ...])
        rul_list.append(min(float(true_rul), float(rul_cap)))
        fault_list.append(1 if float(true_rul) <= float(fault_threshold) else 0)
        unit_list.append(int(unit_id))

    return WindowedArrays(
        X=np.concatenate(X_list, axis=0),
        y_rul=np.asarray(rul_list, dtype=np.float32),
        y_fault=np.asarray(fault_list, dtype=np.int8),
        unit_ids=np.asarray(unit_list, dtype=np.int64),
    )


class CMAPSSWindowDataset(Dataset):
    """Torch ``Dataset`` wrapping :class:`WindowedArrays`.

    Each sample is ``(X, y_rul, y_fault)``: a window tensor of shape
    ``(window_size, n_features)`` plus two scalar float targets.
    """

    def __init__(self, arrays: WindowedArrays) -> None:
        self._X = torch.from_numpy(arrays.X)  # (N, T, F) float32
        self._y_rul = torch.from_numpy(arrays.y_rul)  # (N,) float32
        # BCEWithLogitsLoss expects float targets even for binary classification.
        self._y_fault = torch.from_numpy(arrays.y_fault.astype(np.float32))
        self._unit_ids = arrays.unit_ids

    def __len__(self) -> int:
        return int(self._X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._X[idx], self._y_rul[idx], self._y_fault[idx]

    @property
    def unit_ids(self) -> np.ndarray:
        return self._unit_ids

    @property
    def n_features(self) -> int:
        return int(self._X.shape[2])

    @property
    def window_size(self) -> int:
        return int(self._X.shape[1])
