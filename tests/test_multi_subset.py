"""Tests for the TrainTestBundle + multi-subset loader + partition_by_subset_halves."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fl_aircraft.data import (
    CMAPSSConfig,
    ClientShard,
    MultiSubsetConfig,
    SUBSET_COL,
    TrainTestBundle,
    bundle_from_config,
    engine_ids_by_subset,
    informative_sensors,
    load_multi_subset_bundle,
    partition_by_subset_halves,
)


# ---------------------------------------------------------------------------
# MultiSubsetConfig validation
# ---------------------------------------------------------------------------
def test_multi_subset_config_requires_at_least_one_subset(data_dir: Path) -> None:
    with pytest.raises(ValueError):
        MultiSubsetConfig(subsets=(), data_dir=data_dir)


def test_multi_subset_config_rejects_unknown_subset(data_dir: Path) -> None:
    with pytest.raises(ValueError):
        MultiSubsetConfig(subsets=("FD001", "FD999"), data_dir=data_dir)


def test_multi_subset_config_rejects_incompatible_sensor_sets(data_dir: Path) -> None:
    """FD001 (14 sensors) + FD002 (16 sensors) cannot share one feature vector."""
    with pytest.raises(ValueError):
        MultiSubsetConfig(subsets=("FD001", "FD002"), data_dir=data_dir)


def test_multi_subset_config_accepts_fd001_fd003_pair(data_dir: Path) -> None:
    cfg = MultiSubsetConfig(subsets=("FD001", "FD003"), data_dir=data_dir)
    assert cfg.feature_cols == ["os_1", "os_2", "os_3"] + informative_sensors("FD001")
    assert cfg.display_name == "FD001_FD003"


# ---------------------------------------------------------------------------
# bundle_from_config (single-subset)
# ---------------------------------------------------------------------------
def test_bundle_from_config_fd001_shape(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    bundle = bundle_from_config(cfg)
    assert isinstance(bundle, TrainTestBundle)
    assert bundle.subsets == ("FD001",)
    assert bundle.n_features == 17  # 3 op_settings + 14 sensors
    assert len(bundle.train_df) == 20_631
    assert "RUL_capped" in bundle.train_df.columns
    assert "fault" in bundle.train_df.columns
    assert bundle.test_rul.shape == (100,)


def test_bundle_display_name_is_subset_join(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD003", data_dir=data_dir)
    bundle = bundle_from_config(cfg)
    assert bundle.display_name == "FD003"


# ---------------------------------------------------------------------------
# load_multi_subset_bundle: FD001 + FD003
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fd001_fd003_bundle(data_dir: Path) -> TrainTestBundle:
    cfg = MultiSubsetConfig(subsets=("FD001", "FD003"), data_dir=data_dir)
    return load_multi_subset_bundle(cfg)


def test_combined_train_row_count_equals_sum_of_subsets(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    # FD001 train = 20,631 rows, FD003 train = 24,720 rows
    assert len(fd001_fd003_bundle.train_df) == 20_631 + 24_720


def test_combined_unit_ids_are_disjoint_and_offset(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    """FD001 engines should keep 1..100, FD003 engines should be offset to 101..200."""
    by_subset = engine_ids_by_subset(fd001_fd003_bundle.train_df)
    assert sorted(by_subset.keys()) == ["FD001", "FD003"]
    assert by_subset["FD001"] == list(range(1, 101))
    assert by_subset["FD003"] == list(range(101, 201))
    # Disjointness:
    assert set(by_subset["FD001"]).isdisjoint(set(by_subset["FD003"]))


def test_combined_test_rul_concatenates_in_order(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    """FD001 contributes 100 RULs, FD003 contributes 100 RULs → 200 total."""
    assert fd001_fd003_bundle.test_rul.shape == (200,)
    assert fd001_fd003_bundle.test_rul.dtype == np.float32


def test_combined_test_engines_have_correct_count(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    n_test_engines = fd001_fd003_bundle.test_raw_df["unit_id"].nunique()
    assert n_test_engines == 200


def test_combined_train_rul_labels_within_cap(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    assert fd001_fd003_bundle.train_df["RUL_capped"].max() == 125.0
    assert (fd001_fd003_bundle.train_df["RUL_capped"] >= 0).all()


def test_combined_train_has_subset_column(fd001_fd003_bundle: TrainTestBundle) -> None:
    assert SUBSET_COL in fd001_fd003_bundle.train_df.columns
    counts = fd001_fd003_bundle.train_df[SUBSET_COL].value_counts()
    assert int(counts.loc["FD001"]) == 20_631
    assert int(counts.loc["FD003"]) == 24_720


def test_combined_no_nans(fd001_fd003_bundle: TrainTestBundle) -> None:
    assert fd001_fd003_bundle.train_df.isna().sum().sum() == 0
    assert fd001_fd003_bundle.test_raw_df.isna().sum().sum() == 0


def test_engine_ids_by_subset_rejects_missing_column() -> None:
    bad = pd.DataFrame({"unit_id": [1, 2]})
    with pytest.raises(ValueError):
        engine_ids_by_subset(bad)


# ---------------------------------------------------------------------------
# partition_by_subset_halves
# ---------------------------------------------------------------------------
def test_partition_by_subset_halves_two_subsets_two_halves(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    shards = partition_by_subset_halves(
        fd001_fd003_bundle.train_df,
        subsets=("FD001", "FD003"),
        n_clients_per_subset=2,
        seed=42,
    )
    assert len(shards) == 4
    assert [s.client_id for s in shards] == [
        "client_1", "client_2", "client_3", "client_4"
    ]


def test_partition_by_subset_halves_covers_every_engine_exactly_once(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    shards = partition_by_subset_halves(
        fd001_fd003_bundle.train_df, subsets=("FD001", "FD003"), seed=42,
    )
    all_engines = sorted(u for s in shards for u in s.unit_ids)
    expected = sorted(fd001_fd003_bundle.train_df["unit_id"].unique().tolist())
    assert all_engines == expected
    for i, a in enumerate(shards):
        for b in shards[i + 1:]:
            assert not set(a.unit_ids) & set(b.unit_ids)


def test_partition_by_subset_halves_first_two_are_fd001(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    """Clients 1 & 2 must own only FD001 engines (ids 1..100)."""
    shards = partition_by_subset_halves(
        fd001_fd003_bundle.train_df, subsets=("FD001", "FD003"), seed=42,
    )
    for shard in shards[:2]:
        assert all(1 <= u <= 100 for u in shard.unit_ids), (
            f"{shard.client_id} contains a non-FD001 engine"
        )
    for shard in shards[2:]:
        assert all(101 <= u <= 200 for u in shard.unit_ids), (
            f"{shard.client_id} contains a non-FD003 engine"
        )


def test_partition_by_subset_halves_balanced_within_subset(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    shards = partition_by_subset_halves(
        fd001_fd003_bundle.train_df, subsets=("FD001", "FD003"), seed=42,
    )
    sizes = [s.n_engines for s in shards]
    # FD001: 100 engines / 2 clients = 50 each. FD003: same.
    assert sizes == [50, 50, 50, 50]


def test_partition_by_subset_halves_reproducible_with_seed(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    a = partition_by_subset_halves(
        fd001_fd003_bundle.train_df, subsets=("FD001", "FD003"), seed=42,
    )
    b = partition_by_subset_halves(
        fd001_fd003_bundle.train_df, subsets=("FD001", "FD003"), seed=42,
    )
    assert [s.unit_ids for s in a] == [s.unit_ids for s in b]
    c = partition_by_subset_halves(
        fd001_fd003_bundle.train_df, subsets=("FD001", "FD003"), seed=7,
    )
    assert [s.unit_ids for s in a] != [s.unit_ids for s in c]


def test_partition_by_subset_halves_rejects_empty_subsets_list(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    with pytest.raises(ValueError):
        partition_by_subset_halves(fd001_fd003_bundle.train_df, subsets=())


def test_partition_by_subset_halves_rejects_missing_subset_column() -> None:
    bad = pd.DataFrame({"unit_id": [1, 2, 3]})
    with pytest.raises(ValueError):
        partition_by_subset_halves(bad, subsets=("FD001",))


# ---------------------------------------------------------------------------
# End-to-end: bundle works with the legacy train_centralized via DataLoader
# ---------------------------------------------------------------------------
def test_combined_bundle_works_with_existing_training_primitives(
    fd001_fd003_bundle: TrainTestBundle,
) -> None:
    """Smoke test that a combined bundle plugs into the existing pipeline."""
    from torch.utils.data import DataLoader

    from fl_aircraft.data import (
        CMAPSSWindowDataset,
        Normalizer,
        make_test_windows,
        make_training_windows,
    )

    bundle = fd001_fd003_bundle
    norm = Normalizer.fit(bundle.train_df, bundle.feature_cols)
    train_arrays = make_training_windows(
        norm.transform(bundle.train_df),
        bundle.feature_cols, bundle.window_size, bundle.stride,
    )
    test_arrays = make_test_windows(
        norm.transform(bundle.test_raw_df), bundle.test_rul,
        bundle.feature_cols, bundle.window_size, bundle.rul_cap, bundle.fault_threshold,
    )
    assert train_arrays.n_features == 17
    assert test_arrays.n_samples == 200
    loader = DataLoader(CMAPSSWindowDataset(train_arrays), batch_size=256, shuffle=False)
    batch = next(iter(loader))
    x, y_rul, y_fault = batch
    assert x.shape[1:] == (bundle.window_size, bundle.n_features)
