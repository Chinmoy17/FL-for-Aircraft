"""Data loading, preprocessing, windowing, and client partitioning for C-MAPSS.

Public API — import these directly from ``fl_aircraft.data``::

    from fl_aircraft.data import (
        CMAPSSConfig, load_and_label_train, Normalizer,
        make_training_windows, make_test_windows, CMAPSSWindowDataset,
        partition_by_lifetime, partition_by_subset_halves, slice_for_client,
        TrainTestBundle, bundle_from_config,
        MultiSubsetConfig, load_multi_subset_bundle, engine_ids_by_subset,
    )
"""
from __future__ import annotations

from .bundle import TrainTestBundle, bundle_from_config
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
from .multi_subset import (
    SUBSET_COL,
    MultiSubsetConfig,
    engine_ids_by_subset,
    load_multi_subset_bundle,
)
from .partition import (
    ClientShard,
    partition_by_lifetime,
    partition_by_subset_halves,
    slice_for_client,
)
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
    "SUBSET_COL",
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
    # bundle
    "TrainTestBundle",
    "bundle_from_config",
    # multi_subset
    "MultiSubsetConfig",
    "engine_ids_by_subset",
    "load_multi_subset_bundle",
    # partition
    "ClientShard",
    "partition_by_lifetime",
    "partition_by_subset_halves",
    "slice_for_client",
    # windowing
    "CMAPSSWindowDataset",
    "WindowedArrays",
    "make_test_windows",
    "make_training_windows",
]
