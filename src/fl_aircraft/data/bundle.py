"""``TrainTestBundle`` — the data contract shared by every training entry point.

Background
----------
Phase 6 introduces a Non-IID baseline that combines FD001 and FD003 into one
training set. Up to P5, every training routine in this project took a
:class:`CMAPSSConfig` (which encodes a single subset) and loaded its data
internally. That coupling does not extend to multi-subset experiments.

A :class:`TrainTestBundle` decouples *what data we are working with* from *how
we are training it*. The same dataclass shape works for:

- single-subset baselines (P3 / P4 / P5)            — built via :func:`bundle_from_config`
- combined-subset baselines (P6)                     — built via :func:`load_multi_subset_bundle`
- future RQ experiments with custom client splits    — built however needed

All ``train_*_from_bundle`` / ``run_*_from_bundle`` functions in
``fl_aircraft.train`` and ``fl_aircraft.fl`` accept this dataclass. The
single-subset legacy entry points are thin wrappers around the bundle-based
versions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .cmapss import (
    CMAPSSConfig,
    load_and_label_train,
    load_raw,
    load_test_rul,
)


@dataclass(frozen=True)
class TrainTestBundle:
    """Everything needed to train and evaluate a model, minus the partition recipe.

    Attributes:
        train_df: Training rows with ``RUL_raw``, ``RUL_capped``, and ``fault``
            columns already attached, but **not** normalized.
        test_raw_df: Raw test rows (no labels, **not** normalized).
        test_rul: Ground-truth final-cycle RUL per test engine, in the same
            engine order as ``test_raw_df.unit_id.unique()`` sorted.
        feature_cols: Column names fed into the model (op_settings + sensors).
        window_size: Sliding-window length in cycles.
        stride: Sliding-window stride in cycles.
        rul_cap: Piecewise-linear cap applied to RUL labels.
        fault_threshold: Binary fault label is 1 iff raw RUL <= this.
        subsets: Tuple of CMAPSS subset names contributing rows; one element
            for the single-subset baselines, multiple for combined-subset
            experiments.
        name: Optional friendly name (e.g. ``"FD001_FD003"``) used in plots
            and logs. Defaults to ``"_".join(subsets)`` if not given.
    """

    train_df: pd.DataFrame
    test_raw_df: pd.DataFrame
    test_rul: np.ndarray
    feature_cols: list[str]
    window_size: int
    stride: int
    rul_cap: int
    fault_threshold: int
    subsets: tuple[str, ...]
    name: Optional[str] = None

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def display_name(self) -> str:
        return self.name or "_".join(self.subsets)


def bundle_from_config(config: CMAPSSConfig) -> TrainTestBundle:
    """Build a :class:`TrainTestBundle` from a single-subset :class:`CMAPSSConfig`."""
    train_df = load_and_label_train(config)
    test_raw_df = load_raw(config.subset, "test", config.data_dir)
    test_rul = load_test_rul(config.subset, config.data_dir)
    return TrainTestBundle(
        train_df=train_df,
        test_raw_df=test_raw_df,
        test_rul=test_rul,
        feature_cols=list(config.feature_cols),
        window_size=config.window_size,
        stride=config.stride,
        rul_cap=config.rul_cap,
        fault_threshold=config.fault_threshold,
        subsets=(config.subset,),
    )
