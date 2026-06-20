"""Combine multiple C-MAPSS subsets into a single :class:`TrainTestBundle`.

Phase 6 of this project uses FD001 + FD003 together to create a structurally
Non-IID federation: clients 1–2 receive only FD001 engines (HPC fault mode)
and clients 3–4 receive only FD003 engines (HPC + Fan fault modes). Sharing
the same processing path with single-subset experiments requires:

1. **Disjoint engine identifiers.** FD001 has engines 1–100 and FD003 *also*
   has engines 1–100. We offset FD003's ids so the combined frame has
   engines 1–100 (FD001) and 101–200 (FD003), with a ``source_subset`` column
   for downstream partitioning.
2. **A consistent informative-sensor list.** All combined subsets must agree
   on which sensors to drop, otherwise the model's input width changes
   across engines. FD001 and FD003 both drop ``{1, 5, 6, 10, 16, 18, 19}``,
   so this works. The helper raises if anyone tries to combine incompatible
   subsets (e.g. FD001 + FD002).
3. **Label recomputation** on the combined frame. RUL labels are per-engine
   (``max(cycle) − current_cycle``); because the offsets keep ids disjoint,
   the ``groupby('unit_id')`` in :func:`compute_rul_labels` produces correct
   per-engine RULs even after concatenation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import TrainTestBundle
from .cmapss import (
    compute_fault_labels,
    compute_rul_labels,
    load_raw,
    load_test_rul,
)
from .constants import (
    CYCLE_COL,
    DEFAULT_FAULT_THRESHOLD,
    DEFAULT_RUL_CAP,
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    OP_SETTING_COLS,
    SUBSETS,
    UNIT_ID_COL,
    informative_sensors,
)


SUBSET_COL: str = "source_subset"
"""Column name added by :func:`load_multi_subset_bundle` to tag each row with its origin subset."""


@dataclass(frozen=True)
class MultiSubsetConfig:
    """Hyperparameters for a multi-subset bundle.

    Attributes:
        subsets: Ordered tuple of CMAPSS subset names to combine. The order
            determines the unit-id offset assignment.
        data_dir: Directory containing the raw CMAPSS text files.
        window_size / stride / rul_cap / fault_threshold / include_op_settings:
            Same meaning as in :class:`CMAPSSConfig`.
    """

    subsets: tuple[str, ...]
    data_dir: Path
    window_size: int = DEFAULT_WINDOW_SIZE
    stride: int = DEFAULT_STRIDE
    rul_cap: int = DEFAULT_RUL_CAP
    fault_threshold: int = DEFAULT_FAULT_THRESHOLD
    include_op_settings: bool = True

    def __post_init__(self) -> None:
        if len(self.subsets) < 1:
            raise ValueError("subsets must contain at least one subset name.")
        for s in self.subsets:
            if s not in SUBSETS:
                raise ValueError(f"Unknown subset {s!r}; expected one of {SUBSETS}.")
        # All subsets must share the same informative sensor set.
        sensor_sets = {frozenset(informative_sensors(s)) for s in self.subsets}
        if len(sensor_sets) != 1:
            raise ValueError(
                f"Subsets {self.subsets} have different informative sensor sets — "
                "cannot combine into a single bundle. "
                "FD001/FD003 are compatible; FD002/FD004 are compatible with each other; "
                "mixing across those groups requires regime-aware preprocessing."
            )
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}.")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}.")

    @property
    def feature_cols(self) -> list[str]:
        sensors = informative_sensors(self.subsets[0])
        return (OP_SETTING_COLS if self.include_op_settings else []) + sensors

    @property
    def display_name(self) -> str:
        return "+".join(self.subsets)


def load_multi_subset_bundle(config: MultiSubsetConfig) -> TrainTestBundle:
    """Load every subset, offset unit_ids to keep them disjoint, then label and bundle.

    The returned bundle's ``train_df`` and ``test_raw_df`` carry an extra
    ``source_subset`` column so downstream code (e.g. :func:`partition_by_subset_halves`)
    can split by origin.
    """
    train_dfs: list[pd.DataFrame] = []
    test_dfs: list[pd.DataFrame] = []
    test_ruls: list[np.ndarray] = []

    offset = 0
    for subset in config.subsets:
        raw_train = load_raw(subset, "train", config.data_dir).copy()
        raw_test = load_raw(subset, "test", config.data_dir).copy()
        max_uid = int(max(raw_train[UNIT_ID_COL].max(), raw_test[UNIT_ID_COL].max()))

        raw_train[UNIT_ID_COL] = raw_train[UNIT_ID_COL] + offset
        raw_train[SUBSET_COL] = subset
        train_dfs.append(raw_train)

        raw_test[UNIT_ID_COL] = raw_test[UNIT_ID_COL] + offset
        raw_test[SUBSET_COL] = subset
        test_dfs.append(raw_test)

        test_ruls.append(load_test_rul(subset, config.data_dir))
        offset += max_uid

    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = compute_rul_labels(train_df, config.rul_cap)
    train_df = compute_fault_labels(train_df, config.fault_threshold)

    test_raw_df = pd.concat(test_dfs, ignore_index=True)
    test_rul = np.concatenate(test_ruls).astype(np.float32)

    return TrainTestBundle(
        train_df=train_df,
        test_raw_df=test_raw_df,
        test_rul=test_rul,
        feature_cols=config.feature_cols,
        window_size=config.window_size,
        stride=config.stride,
        rul_cap=config.rul_cap,
        fault_threshold=config.fault_threshold,
        subsets=tuple(config.subsets),
        name=config.display_name,
    )


def engine_ids_by_subset(df: pd.DataFrame) -> dict[str, list[int]]:
    """Return ``{subset_name: sorted_unit_ids}`` for a DataFrame with a ``source_subset`` column."""
    if SUBSET_COL not in df.columns:
        raise ValueError(
            f"DataFrame is missing the {SUBSET_COL!r} column — was it built by "
            "load_multi_subset_bundle()?"
        )
    out: dict[str, list[int]] = {}
    for subset, group in df.groupby(SUBSET_COL):
        out[str(subset)] = sorted(int(u) for u in group[UNIT_ID_COL].unique())
    return out
