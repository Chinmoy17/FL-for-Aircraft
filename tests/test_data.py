"""Unit + integration tests for the data pipeline.

These tests exercise the real C-MAPSS files (FD001) — they are skipped if the
dataset is missing. They protect every preprocessing invariant the rest of the
project relies on: schema, label correctness, normalization, windowing shapes,
and partitioning balance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fl_aircraft.data import (
    CMAPSSConfig,
    CMAPSSWindowDataset,
    Normalizer,
    OP_SETTING_COLS,
    UNIT_ID_COL,
    compute_fault_labels,
    compute_rul_labels,
    informative_sensors,
    load_and_label_train,
    load_raw,
    load_test_rul,
    make_test_windows,
    make_training_windows,
    partition_by_lifetime,
    slice_for_client,
)


# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------
def test_informative_sensors_drops_expected_count() -> None:
    assert len(informative_sensors("FD001")) == 14
    assert len(informative_sensors("FD003")) == 14
    assert len(informative_sensors("FD002")) == 16
    assert len(informative_sensors("FD004")) == 16


def test_informative_sensors_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError):
        informative_sensors("FD999")


def test_cmapss_config_features(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    # 3 op settings + 14 informative sensors
    assert cfg.n_features == 17
    assert cfg.feature_cols[:3] == OP_SETTING_COLS
    assert all(c.startswith("s_") for c in cfg.feature_cols[3:])


def test_cmapss_config_rejects_invalid_inputs(data_dir: Path) -> None:
    with pytest.raises(ValueError):
        CMAPSSConfig(subset="FD999", data_dir=data_dir)
    with pytest.raises(ValueError):
        CMAPSSConfig(subset="FD001", data_dir=data_dir, window_size=0)
    with pytest.raises(ValueError):
        CMAPSSConfig(subset="FD001", data_dir=data_dir, stride=0)


# ---------------------------------------------------------------------------
# Raw I/O
# ---------------------------------------------------------------------------
def test_load_raw_fd001_shape(data_dir: Path) -> None:
    df = load_raw("FD001", "train", data_dir)
    assert df.shape == (20631, 26)  # cross-checked against the EDA notebook
    assert df.isna().sum().sum() == 0
    assert df["unit_id"].nunique() == 100


def test_load_raw_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw("FD001", "train", tmp_path)


def test_load_test_rul_fd001(data_dir: Path) -> None:
    rul = load_test_rul("FD001", data_dir)
    assert rul.shape == (100,)
    assert rul.dtype == np.float32
    assert (rul > 0).all()


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
def test_rul_labels_last_cycle_per_engine_is_zero(data_dir: Path) -> None:
    df = load_raw("FD001", "train", data_dir)
    df = compute_rul_labels(df, cap=125)
    last_rows = df.sort_values("cycle").groupby(UNIT_ID_COL).tail(1)
    # By construction, the last cycle of each engine has RUL = 0.
    assert (last_rows["RUL_raw"] == 0).all()
    # The cap must hold across the whole frame.
    assert df["RUL_capped"].max() == 125.0


def test_fault_labels_match_eda_positive_rate(data_dir: Path) -> None:
    df = load_raw("FD001", "train", data_dir)
    df = compute_rul_labels(df, cap=125)
    df = compute_fault_labels(df, threshold=30)
    pos_rate = df["fault"].mean()
    # From the EDA notebook: FD001 positive rate = 15.03%.
    assert 0.149 <= pos_rate <= 0.152


def test_compute_fault_requires_rul_first() -> None:
    import pandas as pd

    bad = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        compute_fault_labels(bad)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------
def test_normalizer_zeros_mean_and_unit_std_on_training_data(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    df = load_and_label_train(cfg)
    normalizer = Normalizer.fit(df, cfg.feature_cols)
    out = normalizer.transform(df)
    arr = out[cfg.feature_cols].to_numpy()
    # Tolerances reflect float32 storage of the transformed values: with some
    # sensor std values ~ 4e-2, even a 1e-5 residual raw-mean gets amplified to
    # ~3e-4 in z-score space. atol=1e-3 sits well below any precision the
    # downstream torch model cares about (float32).
    assert np.allclose(arr.mean(axis=0), 0.0, atol=1e-3)
    # Every column's std must either be ~ 1 (originally variable feature) or
    # exactly 0 (originally constant feature: std was clipped to 1, so every
    # row becomes (constant - constant) / 1 = 0).
    for col, s in zip(cfg.feature_cols, arr.std(axis=0)):
        assert abs(s - 1.0) < 1e-3 or s == 0.0, (
            f"Column {col!r}: transformed std={s} (expected ~1 or 0)."
        )


def test_normalizer_handles_constant_columns_without_div_by_zero(data_dir: Path) -> None:
    """Constant op_settings on FD001 must not produce NaN/inf after normalization."""
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    df = load_and_label_train(cfg)
    normalizer = Normalizer.fit(df, cfg.feature_cols)
    out = normalizer.transform(df)
    arr = out[cfg.feature_cols].to_numpy()
    assert np.isfinite(arr).all()


def test_normalizer_rejects_missing_columns(data_dir: Path) -> None:
    import pandas as pd

    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    df = load_and_label_train(cfg)
    normalizer = Normalizer.fit(df, cfg.feature_cols)
    with pytest.raises(KeyError):
        normalizer.transform(pd.DataFrame({"unrelated": [1.0]}))


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def test_training_windows_shapes_and_invariants(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir, window_size=30)
    df = load_and_label_train(cfg)
    normalizer = Normalizer.fit(df, cfg.feature_cols)
    df = normalizer.transform(df)
    arrays = make_training_windows(df, cfg.feature_cols, cfg.window_size, cfg.stride)

    assert arrays.X.ndim == 3
    assert arrays.X.shape[1] == cfg.window_size
    assert arrays.X.shape[2] == cfg.n_features
    assert arrays.X.dtype == np.float32
    assert arrays.y_rul.shape == (arrays.n_samples,)
    assert arrays.y_fault.shape == (arrays.n_samples,)
    assert arrays.unit_ids.shape == (arrays.n_samples,)
    assert np.isfinite(arrays.X).all()
    # Sliding window math: sum over engines of (life - window_size + 1) windows.
    raw_df = load_raw("FD001", "train", data_dir)
    expected = int(
        sum(
            max(0, n - cfg.window_size + 1)
            for n in raw_df.groupby(UNIT_ID_COL).size()
        )
    )
    assert arrays.n_samples == expected
    # Label sanity: capped RUL stays within [0, cap].
    assert (arrays.y_rul >= 0).all() and (arrays.y_rul <= cfg.rul_cap).all()


def test_training_windows_last_cycle_labels(data_dir: Path) -> None:
    """The label of the final window of any engine must equal that engine's last-cycle label."""
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir, window_size=30)
    df = load_and_label_train(cfg)
    norm = Normalizer.fit(df, cfg.feature_cols)
    df_norm = norm.transform(df)
    arrays = make_training_windows(df_norm, cfg.feature_cols, cfg.window_size)

    # For each engine the LAST training window's label must be 0 (engine has just failed).
    for unit_id in np.unique(arrays.unit_ids):
        mask = arrays.unit_ids == unit_id
        last_label = arrays.y_rul[mask][-1]
        assert last_label == 0.0, f"Engine {unit_id} last-window RUL != 0 ({last_label})"


def test_training_windows_rejects_unlabeled_frame(data_dir: Path) -> None:
    df = load_raw("FD001", "train", data_dir)  # not labeled
    with pytest.raises(ValueError):
        make_training_windows(df, ["s_2", "s_3"], window_size=30)


def test_test_windows_one_per_engine(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir, window_size=30)
    df_train = load_and_label_train(cfg)
    normalizer = Normalizer.fit(df_train, cfg.feature_cols)

    test_df = load_raw("FD001", "test", data_dir)
    test_df = normalizer.transform(test_df)
    test_rul = load_test_rul("FD001", data_dir)
    arrays = make_test_windows(
        test_df,
        test_rul,
        cfg.feature_cols,
        cfg.window_size,
        cfg.rul_cap,
        cfg.fault_threshold,
    )
    assert arrays.n_samples == len(test_rul) == 100
    assert arrays.X.shape == (100, cfg.window_size, cfg.n_features)
    # Each window must be paired back to a unique engine.
    assert len(np.unique(arrays.unit_ids)) == 100


def test_test_windows_rul_capped_to_config(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir, window_size=30)
    df_train = load_and_label_train(cfg)
    normalizer = Normalizer.fit(df_train, cfg.feature_cols)
    test_df = normalizer.transform(load_raw("FD001", "test", data_dir))
    arrays = make_test_windows(
        test_df,
        load_test_rul("FD001", data_dir),
        cfg.feature_cols,
        cfg.window_size,
        cfg.rul_cap,
        cfg.fault_threshold,
    )
    assert arrays.y_rul.max() <= cfg.rul_cap


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------
def test_partition_covers_all_engines_exactly_once(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    df = load_and_label_train(cfg)
    shards = partition_by_lifetime(df, n_clients=4, seed=42)
    all_units = sorted([u for s in shards for u in s.unit_ids])
    expected = sorted(df[UNIT_ID_COL].unique().tolist())
    assert all_units == expected
    # No client owns any engine that any other client also owns.
    for i, a in enumerate(shards):
        for b in shards[i + 1 :]:
            assert not set(a.unit_ids) & set(b.unit_ids)


def test_partition_is_balanced(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    df = load_and_label_train(cfg)
    shards = partition_by_lifetime(df, n_clients=4, seed=42)
    counts = [s.n_engines for s in shards]
    assert max(counts) - min(counts) <= 1  # at most 1-engine imbalance


def test_partition_is_reproducible(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    df = load_and_label_train(cfg)
    a = partition_by_lifetime(df, n_clients=4, seed=42)
    b = partition_by_lifetime(df, n_clients=4, seed=42)
    assert [s.unit_ids for s in a] == [s.unit_ids for s in b]
    c = partition_by_lifetime(df, n_clients=4, seed=7)
    assert [s.unit_ids for s in a] != [s.unit_ids for s in c]


def test_partition_rejects_too_many_clients(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    df = load_and_label_train(cfg)
    with pytest.raises(ValueError):
        partition_by_lifetime(df, n_clients=10_000)


def test_slice_for_client_matches_engine_ids(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    df = load_and_label_train(cfg)
    shards = partition_by_lifetime(df, n_clients=4, seed=42)
    for shard in shards:
        sub = slice_for_client(df, shard)
        assert set(sub[UNIT_ID_COL].unique()) == set(shard.unit_ids)


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------
def test_cmapss_window_dataset_yields_correct_tensors(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir, window_size=30)
    df = load_and_label_train(cfg)
    df = Normalizer.fit(df, cfg.feature_cols).transform(df)
    arrays = make_training_windows(df, cfg.feature_cols, cfg.window_size)
    ds = CMAPSSWindowDataset(arrays)
    assert len(ds) == arrays.n_samples
    x, y_rul, y_fault = ds[0]
    assert x.shape == (cfg.window_size, cfg.n_features)
    assert x.dtype == torch.float32
    assert y_rul.dtype == torch.float32
    assert y_fault.dtype == torch.float32  # BCEWithLogitsLoss expects float targets
    assert torch.isfinite(x).all()


# ---------------------------------------------------------------------------
# End-to-end FL-style pipeline
# ---------------------------------------------------------------------------
def test_end_to_end_per_client_pipeline(data_dir: Path) -> None:
    """Walk one client end-to-end: raw load -> per-client normalizer -> windows -> Dataset."""
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir, window_size=30)
    df = load_and_label_train(cfg)
    shards = partition_by_lifetime(df, n_clients=4, seed=42)
    shard = shards[0]

    client_df = slice_for_client(df, shard)
    client_norm = Normalizer.fit(client_df, cfg.feature_cols)
    client_df = client_norm.transform(client_df)
    client_arrays = make_training_windows(client_df, cfg.feature_cols, cfg.window_size)

    ds = CMAPSSWindowDataset(client_arrays)
    assert len(ds) > 0
    # The client's data must contain only engines from its shard.
    assert set(client_arrays.unit_ids.tolist()).issubset(set(shard.unit_ids))
