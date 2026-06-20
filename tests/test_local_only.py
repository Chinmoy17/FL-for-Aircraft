"""Tests for the local-only training loop."""
from __future__ import annotations

from pathlib import Path

import pytest

from fl_aircraft.data import CMAPSSConfig
from fl_aircraft.train import (
    ClientRun,
    LocalOnlyResults,
    train_local_only_clients,
)


# ---------------------------------------------------------------------------
# Fixture: a tiny 2-epoch / 4-client run reused across tests.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_run(data_dir: Path) -> LocalOnlyResults:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    return train_local_only_clients(
        cfg, n_clients=4, epochs=2, lr=1e-3, seed=42, log_every=99, client_log=False,
    )


# ---------------------------------------------------------------------------
# Basic shape + invariants
# ---------------------------------------------------------------------------
def test_train_local_only_produces_one_run_per_client(tiny_run: LocalOnlyResults) -> None:
    assert len(tiny_run.clients) == 4
    assert all(isinstance(c, ClientRun) for c in tiny_run.clients)
    # Client IDs must be unique.
    ids = [c.client_id for c in tiny_run.clients]
    assert len(set(ids)) == 4


def test_every_client_has_finite_metrics(tiny_run: LocalOnlyResults) -> None:
    import numpy as np
    for c in tiny_run.clients:
        assert np.isfinite(c.best_rul.rmse)
        assert np.isfinite(c.best_rul.nasa_score)
        assert 0 <= c.best_fault.auprc <= 1


def test_engines_are_partitioned_disjointly(tiny_run: LocalOnlyResults) -> None:
    seen: set[int] = set()
    for c in tiny_run.clients:
        client_engines = set(c.shard.unit_ids)
        assert seen.isdisjoint(client_engines), (
            f"Engine appears in two clients: {seen & client_engines}"
        )
        seen.update(client_engines)
    # FD001 has 100 training engines.
    assert len(seen) == 100


def test_train_window_count_matches_centralized_total(tiny_run: LocalOnlyResults) -> None:
    """Sum of per-client window counts must equal the centralized training set size."""
    # P1 sanity script reports 17,731 centralized windows on FD001 / window=30.
    total = sum(c.n_train_windows for c in tiny_run.clients)
    assert total == 17_731


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def test_per_client_rows_have_expected_columns(tiny_run: LocalOnlyResults) -> None:
    rows = tiny_run.per_client_rows("best")
    assert len(rows) == 4
    expected = {
        "client_id", "n_engines", "n_train_windows", "pos_weight", "best_epoch",
        "rmse", "mae", "nasa_score", "auprc", "f1", "precision", "recall",
        "train_seconds",
    }
    assert set(rows[0].keys()) == expected


def test_per_client_rows_which_validation(tiny_run: LocalOnlyResults) -> None:
    with pytest.raises(ValueError):
        tiny_run.per_client_rows("middle")


def test_aggregate_mean_matches_manual_computation(tiny_run: LocalOnlyResults) -> None:
    import numpy as np
    rows = tiny_run.per_client_rows("best")
    expected = float(np.mean([r["rmse"] for r in rows]))
    agg = tiny_run.aggregate("best")
    assert agg["rmse_mean"] == pytest.approx(expected, rel=1e-9)
    assert agg["n_clients"] == 4


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_rejects_zero_clients(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    with pytest.raises(ValueError):
        train_local_only_clients(cfg, n_clients=0, epochs=1)


def test_rejects_zero_epochs(data_dir: Path) -> None:
    cfg = CMAPSSConfig(subset="FD001", data_dir=data_dir)
    with pytest.raises(ValueError):
        train_local_only_clients(cfg, n_clients=4, epochs=0)
