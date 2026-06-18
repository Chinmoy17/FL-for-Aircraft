"""Data loading, preprocessing, windowing, and client partitioning for C-MAPSS.

Public API — import these directly from ``fl_aircraft.data``::

    from fl_aircraft.data import (
        CMAPSSConfig, load_and_label_train, Normalizer,
        make_training_windows, make_test_windows, CMAPSSWindowDataset,
        partition_by_lifetime, slice_for_client,
    )
"""
from __future__ import annotations

from .cmapss import (
    CMAPSSConfig,
    Normalizer,
    compute_fault_labels,
    compute_rul_labels,
    load_and_label_train,
    load_raw,
    load_test_rul,
)
from .constants import (
    CONSTANT_SENSORS_PER_SUBSET,
    CYCLE_COL,
    DEFAULT_FAULT_THRESHOLD,
    DEFAULT_RUL_CAP,
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    OP_SETTING_COLS,
    SENSOR_COLS,
    SUBSETS,
    UNIT_ID_COL,
    informative_sensors,
)
from .partition import ClientShard, partition_by_lifetime, slice_for_client
from .windowing import (
    CMAPSSWindowDataset,
    WindowedArrays,
    make_test_windows,
    make_training_windows,
)

__all__ = [
    # constants
    "CONSTANT_SENSORS_PER_SUBSET",
    "CYCLE_COL",
    "DEFAULT_FAULT_THRESHOLD",
    "DEFAULT_RUL_CAP",
    "DEFAULT_STRIDE",
    "DEFAULT_WINDOW_SIZE",
    "OP_SETTING_COLS",
    "SENSOR_COLS",
    "SUBSETS",
    "UNIT_ID_COL",
    "informative_sensors",
    # cmapss
    "CMAPSSConfig",
    "Normalizer",
    "compute_fault_labels",
    "compute_rul_labels",
    "load_and_label_train",
    "load_raw",
    "load_test_rul",
    # partition
    "ClientShard",
    "partition_by_lifetime",
    "slice_for_client",
    # windowing
    "CMAPSSWindowDataset",
    "WindowedArrays",
    "make_test_windows",
    "make_training_windows",
]
