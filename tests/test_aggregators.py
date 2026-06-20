"""Tests for the RQ2 imbalance-aware aggregators + simulation loop."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fl_aircraft.data import (
    MultiSubsetConfig,
    load_multi_subset_bundle,
    partition_by_subset_halves,
)
from fl_aircraft.fl import (
    ClientUpdate,
    FedAvgServer,
    ImbalanceAwareHistory,
    fedavg_aggregate,
    make_fault_count_aggregator,
    make_inverse_loss_aggregator,
    make_validation_signal_aggregator,
    run_fedavg_imbalance_aware,
)


# ---------------------------------------------------------------------------
# fault_count aggregator
# ---------------------------------------------------------------------------
def _two_simple_updates():
    a = {"w": torch.tensor([10.0])}
    b = {"w": torch.tensor([20.0])}
    return [
        ClientUpdate("a", a, n_samples=100),
        ClientUpdate("b", b, n_samples=100),
    ]


def test_fault_count_aggregator_weights_by_fault_count_not_sample_count() -> None:
    fault = {"a": 30, "b": 10}  # 75/25 split
    agg = make_fault_count_aggregator(fault)
    out = agg(_two_simple_updates())
    # 0.75 * 10 + 0.25 * 20 = 12.5
    assert torch.allclose(out["w"], torch.tensor([12.5]))


def test_fault_count_aggregator_falls_back_to_zero_neg_pos_safety() -> None:
    """Negative counts and zero totals must be rejected at construction time."""
    with pytest.raises(ValueError):
        make_fault_count_aggregator({})
    with pytest.raises(ValueError):
        make_fault_count_aggregator({"a": -1, "b": 1})
    with pytest.raises(ValueError):
        make_fault_count_aggregator({"a": 0, "b": 0})


def test_fault_count_aggregator_dtype_preservation() -> None:
    agg = make_fault_count_aggregator({"a": 1, "b": 1})
    out = agg([
        ClientUpdate("a", {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)}, n_samples=1),
        ClientUpdate("b", {"w": torch.tensor([3.0, 4.0], dtype=torch.float32)}, n_samples=1),
    ])
    assert out["w"].dtype == torch.float32


# ---------------------------------------------------------------------------
# validation_signal aggregator
# ---------------------------------------------------------------------------
def test_validation_signal_aggregator_low_temperature_picks_winner() -> None:
    # Client b has the much higher F1; with T=0.01 it should get nearly all the weight.
    signals = {"a": 0.5, "b": 0.95}
    agg = make_validation_signal_aggregator(lambda: signals, temperature=0.01, floor=0.0)
    out = agg(_two_simple_updates())
    # Should be very close to 20.0 (b's tensor).
    assert 19.0 < float(out["w"]) <= 20.0


def test_validation_signal_aggregator_high_temperature_approaches_uniform() -> None:
    signals = {"a": 0.5, "b": 0.95}
    agg = make_validation_signal_aggregator(lambda: signals, temperature=100.0, floor=0.0)
    out = agg(_two_simple_updates())
    # T=100 makes softmax close to uniform => 0.5 * 10 + 0.5 * 20 = 15
    assert abs(float(out["w"]) - 15.0) < 0.5


def test_validation_signal_aggregator_floor_prevents_collapse() -> None:
    """With floor=0.4, no client should ever get less than 40% weight."""
    signals = {"a": 0.0, "b": 1.0}
    agg = make_validation_signal_aggregator(
        lambda: signals, temperature=0.001, floor=0.4,
    )
    out = agg(_two_simple_updates())
    # 0.4 * 10 + 0.6 * 20 = 16 — but with floor=0.4 the minimum stake of a
    # is 0.4 of the (after-softmax) reweight, so the result is 16 not >19.
    val = float(out["w"])
    # Both ends bounded.
    assert 14.0 <= val <= 16.0


def test_validation_signal_aggregator_invert_for_loss_minimisation() -> None:
    """invert=True: lower signal => higher weight (use this for losses, where lower is better)."""
    signals = {"a": 5.0, "b": 0.5}  # b has much lower "loss"
    agg = make_validation_signal_aggregator(
        lambda: signals, temperature=0.01, floor=0.0, invert=True,
    )
    out = agg(_two_simple_updates())
    assert 19.0 < float(out["w"]) <= 20.0


def test_validation_signal_aggregator_rejects_missing_client_signal() -> None:
    agg = make_validation_signal_aggregator(lambda: {"a": 0.5}, temperature=1.0)
    with pytest.raises(ValueError):
        agg(_two_simple_updates())


def test_validation_signal_aggregator_rejects_non_dict_provider() -> None:
    agg = make_validation_signal_aggregator(lambda: 0.5, temperature=1.0)
    with pytest.raises(TypeError):
        agg(_two_simple_updates())


def test_validation_signal_aggregator_validates_constructor_args() -> None:
    with pytest.raises(ValueError):
        make_validation_signal_aggregator(lambda: {}, temperature=0.0)
    with pytest.raises(ValueError):
        make_validation_signal_aggregator(lambda: {}, floor=1.0)


# ---------------------------------------------------------------------------
# inverse_loss aggregator
# ---------------------------------------------------------------------------
def test_inverse_loss_aggregator_low_loss_gets_high_weight() -> None:
    losses = {"a": 100.0, "b": 1.0}
    agg = make_inverse_loss_aggregator(lambda: losses)
    out = agg(_two_simple_updates())
    # 1/100 + 1/1 ≈ 1.01; a's weight ≈ 0.01, b's ≈ 0.99
    # Result ≈ 0.01 * 10 + 0.99 * 20 ≈ 19.9
    assert 19.0 < float(out["w"]) <= 20.0


def test_inverse_loss_aggregator_rejects_negative_loss() -> None:
    agg = make_inverse_loss_aggregator(lambda: {"a": -1.0, "b": 1.0})
    with pytest.raises(ValueError):
        agg(_two_simple_updates())


def test_inverse_loss_aggregator_rejects_invalid_epsilon() -> None:
    with pytest.raises(ValueError):
        make_inverse_loss_aggregator(lambda: {}, epsilon=0.0)


# ---------------------------------------------------------------------------
# Pluggability: FedAvgServer accepts custom aggregators
# ---------------------------------------------------------------------------
def test_fedavg_server_accepts_custom_aggregator() -> None:
    init = {"w": torch.tensor([0.0])}
    server = FedAvgServer(init, aggregator=make_fault_count_aggregator({"a": 3, "b": 1}))
    new = server.aggregate([
        ClientUpdate("a", {"w": torch.tensor([8.0])}, n_samples=999),  # n_samples ignored
        ClientUpdate("b", {"w": torch.tensor([0.0])}, n_samples=1),
    ])
    # 0.75 * 8 + 0.25 * 0 = 6
    assert torch.allclose(new["w"], torch.tensor([6.0]))


def test_canonical_fedavg_aggregator_still_works_as_a_callable() -> None:
    """Smoke test that the legacy aggregator hasn't regressed."""
    out = fedavg_aggregate(_two_simple_updates())
    # Equal n_samples => arithmetic mean = 15.
    assert torch.allclose(out["w"], torch.tensor([15.0]))


# ---------------------------------------------------------------------------
# End-to-end smoke: run_fedavg_imbalance_aware on a tiny CMAPSS instance
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_p6_bundle(data_dir: Path):
    """Same partition as P6: 4 clients across FD001 + FD003."""
    cfg = MultiSubsetConfig(subsets=("FD001", "FD003"), data_dir=data_dir)
    bundle = load_multi_subset_bundle(cfg)
    shards = partition_by_subset_halves(
        bundle.train_df, subsets=("FD001", "FD003"), n_clients_per_subset=2, seed=42,
    )
    return bundle, shards


@pytest.mark.parametrize("aggregator", ["fedavg", "fault_count", "validation_f1", "inverse_loss"])
def test_run_fedavg_imbalance_aware_smoke(tiny_p6_bundle, aggregator) -> None:
    """Each aggregator must produce a valid 2-round history with finite metrics."""
    bundle, shards = tiny_p6_bundle
    history = run_fedavg_imbalance_aware(
        bundle, shards,
        aggregator=aggregator,
        val_fraction=0.2,
        n_rounds=2, local_epochs=1,
        lr=1e-3, seed=42, log_every=99,
    )
    assert isinstance(history, ImbalanceAwareHistory)
    assert len(history) == 2
    assert history.aggregator_name == aggregator
    for rec in history.rounds:
        assert np.isfinite(rec.global_test_rmse)
        assert 0 <= rec.global_test_auprc <= 1
    # Each client should have a 2-element list of aggregation weights summing to 1.
    for cid in history.client_ids:
        w = history.aggregation_weights[cid]
        assert len(w) == 2
        assert all(0 <= x <= 1 for x in w)
    weight_sums = [
        sum(history.aggregation_weights[cid][r] for cid in history.client_ids)
        for r in range(2)
    ]
    for s in weight_sums:
        assert abs(s - 1.0) < 1e-5


def test_validation_f1_records_per_client_val_signals(tiny_p6_bundle) -> None:
    bundle, shards = tiny_p6_bundle
    history = run_fedavg_imbalance_aware(
        bundle, shards,
        aggregator="validation_f1",
        val_fraction=0.2,
        n_rounds=2, local_epochs=1,
        lr=1e-3, seed=42, log_every=99,
    )
    for cid in history.client_ids:
        assert len(history.per_round_client_val_f1[cid]) == 2
        assert all(0 <= v <= 1 for v in history.per_round_client_val_f1[cid])


def test_non_validation_aggregators_do_not_compute_val_signals(tiny_p6_bundle) -> None:
    """Schemes A and C should leave val signal slots as NaN (they don't use them)."""
    bundle, shards = tiny_p6_bundle
    history = run_fedavg_imbalance_aware(
        bundle, shards,
        aggregator="fault_count",
        n_rounds=2, local_epochs=1,
        lr=1e-3, seed=42, log_every=99,
    )
    for cid in history.client_ids:
        for v in history.per_round_client_val_f1[cid]:
            assert np.isnan(v)


def test_run_fedavg_imbalance_aware_seed_reproducibility(tiny_p6_bundle) -> None:
    bundle, shards = tiny_p6_bundle
    h1 = run_fedavg_imbalance_aware(
        bundle, shards, aggregator="fault_count",
        n_rounds=2, local_epochs=1, lr=1e-3, seed=42, log_every=99,
    )
    h2 = run_fedavg_imbalance_aware(
        bundle, shards, aggregator="fault_count",
        n_rounds=2, local_epochs=1, lr=1e-3, seed=42, log_every=99,
    )
    assert h1.final_test_rul.rmse == h2.final_test_rul.rmse


def test_run_fedavg_imbalance_aware_rejects_invalid_aggregator(tiny_p6_bundle) -> None:
    bundle, shards = tiny_p6_bundle
    with pytest.raises(ValueError):
        run_fedavg_imbalance_aware(
            bundle, shards, aggregator="not_a_real_scheme",  # type: ignore[arg-type]
            n_rounds=1, local_epochs=1,
        )
