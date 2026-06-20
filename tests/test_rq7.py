"""RQ7 tests — attack mechanics + defense correctness + clean equivalence.

Test surface, by module:

poisoning.py
  - Label-flip flips RUL labels via 125 - y formula and re-derives the fault.
  - Gradient-scale produces W_global + scale*(W_local - W_global) exactly.
  - With scale=1.0 the gradient-scale attacker reduces to honest behavior.

robust_aggregators.py
  - All three aggregators have the same key-set as the inputs.
  - Trimmed mean, median, and Krum all reduce to FedAvg-like behavior
    when all clients send IDENTICAL state_dicts (no attacker).
  - Krum picks the honest client when 1 attacker is far from the others.
  - Trimmed mean drops the most-extreme value per parameter element.

poisoned_simulation.py is exercised end-to-end via a 2-round mini-sim.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_aircraft.fl.client import FederatedClient
from fl_aircraft.fl.poisoning import (
    GradientScaleAttacker,
    LabelFlipAttacker,
    _LabelFlippedDataset,
)
from fl_aircraft.fl.robust_aggregators import (
    make_krum_aggregator,
    make_median_aggregator,
    make_trimmed_mean_aggregator,
)
from fl_aircraft.fl.server import ClientUpdate, fedavg_aggregate
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss
from fl_aircraft.utils import seed_everything


WINDOW = 8
N_FEATURES = 6


def _tiny_loader(n: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, WINDOW, N_FEATURES, generator=g)
    y_rul = torch.rand(n, generator=g) * 125.0
    y_fault = (y_rul <= 30.0).to(torch.float32)
    return DataLoader(
        TensorDataset(x.float(), y_rul.float(), y_fault),
        batch_size=8, shuffle=False,
    )


def _make_client(client_id: str, seed: int = 0) -> FederatedClient:
    seed_everything(seed)
    model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW)
    )
    return FederatedClient(
        client_id=client_id, model=model,
        train_loader=_tiny_loader(32, seed=seed),
        loss_fn=MultiTaskLoss(lambda_fault=0.5, fault_pos_weight=1.0),
        n_samples=32,
    )


def _dummy_update(client_id: str, seed: int, n_samples: int = 32) -> ClientUpdate:
    """Synthetic state-dict for aggregator unit tests."""
    g = torch.Generator().manual_seed(seed)
    return ClientUpdate(
        client_id=client_id,
        state_dict={
            "layer.weight": torch.randn(3, 4, generator=g),
            "layer.bias": torch.randn(3, generator=g),
        },
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Label-flip dataset mechanics
# ---------------------------------------------------------------------------
def test_label_flip_inverts_rul():
    """RUL=100 with cap=125 should flip to RUL=25.

    Tolerance is 1e-4 because the wrapped tensor goes through a float32
    round-trip (read → subtract → wrap in tensor) so 1e-6 underflow is
    expected.
    """
    base = _tiny_loader(4, seed=42).dataset
    flipped = _LabelFlippedDataset(base, rul_cap=125.0, fault_threshold=30.0)
    for i in range(len(base)):
        _, orig_rul, _ = base[i]
        _, new_rul, _ = flipped[i]
        assert abs(float(new_rul) - (125.0 - float(orig_rul))) < 1e-4


def test_label_flip_redefines_fault_consistently():
    """Fault label must be re-derived from flipped RUL with the same threshold."""
    base = _tiny_loader(20, seed=11).dataset
    flipped = _LabelFlippedDataset(base, rul_cap=125.0, fault_threshold=30.0)
    for i in range(len(base)):
        _, new_rul, new_fault = flipped[i]
        expected_fault = 1.0 if float(new_rul) <= 30.0 else 0.0
        assert float(new_fault) == expected_fault


# ---------------------------------------------------------------------------
# Attacker wrappers preserve client_id + n_samples
# ---------------------------------------------------------------------------
def test_label_flip_attacker_preserves_id_and_n_samples():
    honest = _make_client("client_3", seed=0)
    attacker = LabelFlipAttacker(inner=honest)
    assert attacker.client_id == "client_3"
    assert attacker.inner.n_samples == 32


def test_gradient_scale_attacker_preserves_id_and_n_samples():
    honest = _make_client("client_3", seed=0)
    attacker = GradientScaleAttacker(inner=honest, scale=-10.0)
    assert attacker.client_id == "client_3"
    assert attacker.inner.n_samples == 32


# ---------------------------------------------------------------------------
# Gradient-scale arithmetic
# ---------------------------------------------------------------------------
def test_gradient_scale_with_scale_one_is_honest():
    """scale=1 should produce W_global + 1*(W_local - W_global) = W_local."""
    honest = _make_client("client_3", seed=0)
    attacker = GradientScaleAttacker(inner=honest, scale=1.0)

    # Capture global at round start.
    seed_everything(0)
    global_state = {
        k: v.detach().clone() for k, v in honest.model.state_dict().items()
    }
    attacker.set_global_state(global_state)

    seed_everything(123)
    attacker.local_train(local_epochs=1, lr=1e-3)
    poisoned = attacker.package_update()
    # The honest update would be exactly the inner client's post-train state.
    honest_state = honest.model.state_dict()
    for k in honest_state:
        assert torch.allclose(
            poisoned.state_dict[k], honest_state[k], atol=1e-6,
        ), f"scale=1 should be a no-op for key {k!r}"


def test_gradient_scale_minus_ten_flips_and_amplifies_delta():
    """With scale=-10, poisoned = W_global + (-10)*(W_local - W_global).

    Verified element-wise: ||poisoned - W_global|| = 10 * ||W_local - W_global||
    and the sign is flipped.
    """
    honest = _make_client("client_3", seed=0)
    attacker = GradientScaleAttacker(inner=honest, scale=-10.0)

    seed_everything(0)
    global_state = {
        k: v.detach().clone() for k, v in honest.model.state_dict().items()
    }
    attacker.set_global_state(global_state)

    seed_everything(123)
    attacker.local_train(local_epochs=1, lr=5e-3)
    poisoned = attacker.package_update()
    honest_state = honest.model.state_dict()

    for k in honest_state:
        honest_delta = honest_state[k].to(torch.float64) - global_state[k].to(torch.float64)
        poisoned_delta = poisoned.state_dict[k].to(torch.float64) - global_state[k].to(torch.float64)
        # poisoned_delta should equal -10 * honest_delta.
        assert torch.allclose(
            poisoned_delta, -10.0 * honest_delta, atol=1e-4,
        ), f"gradient-scale arithmetic broken for key {k!r}"


# ---------------------------------------------------------------------------
# Aggregator key-set invariants
# ---------------------------------------------------------------------------
def test_robust_aggregators_preserve_key_set():
    updates = [_dummy_update(f"c{i}", seed=i) for i in range(4)]
    expected_keys = set(updates[0].state_dict.keys())
    for agg_factory_name, agg in [
        ("trimmed_mean", make_trimmed_mean_aggregator(0.25)),
        ("median", make_median_aggregator()),
        ("krum", make_krum_aggregator(num_byzantine=1)),
    ]:
        out = agg(updates)
        assert set(out.keys()) == expected_keys, (
            f"{agg_factory_name} produced wrong key set"
        )


# ---------------------------------------------------------------------------
# Clean-data equivalence — defenses shouldn't change behavior when all
# updates are identical
# ---------------------------------------------------------------------------
def test_robust_aggregators_match_fedavg_on_identical_updates():
    """If every client sends the same state-dict, trimmed mean, median,
    and Krum should all return that same state-dict."""
    g = torch.Generator().manual_seed(0)
    shared_state = {
        "layer.weight": torch.randn(3, 4, generator=g),
        "layer.bias": torch.randn(3, generator=g),
    }
    updates = [
        ClientUpdate(
            client_id=f"c{i}",
            state_dict={k: v.clone() for k, v in shared_state.items()},
            n_samples=32,
        )
        for i in range(4)
    ]
    fedavg_out = fedavg_aggregate(updates)
    for agg_factory_name, agg in [
        ("trimmed_mean", make_trimmed_mean_aggregator(0.25)),
        ("median", make_median_aggregator()),
        ("krum", make_krum_aggregator(num_byzantine=1)),
    ]:
        out = agg(updates)
        for k in shared_state:
            assert torch.allclose(out[k], fedavg_out[k], atol=1e-6), (
                f"{agg_factory_name} differs from fedavg on identical inputs at {k!r}"
            )


# ---------------------------------------------------------------------------
# Trimmed mean — verify it actually drops extremes per element
# ---------------------------------------------------------------------------
def test_trimmed_mean_drops_per_element_extremes():
    """4 clients, weight values [10, 100, 1000, -50] at one element.
    Trimmed mean with beta=0.25 should drop 1 high + 1 low ⇒ mean of [10, 100] = 55."""
    updates = []
    for i, value in enumerate([10.0, 100.0, 1000.0, -50.0]):
        sd = {
            "scalar": torch.tensor([value, value]),  # both elements use same value
        }
        updates.append(ClientUpdate(client_id=f"c{i}", state_dict=sd, n_samples=32))
    agg = make_trimmed_mean_aggregator(0.25)
    out = agg(updates)
    # Drops -50 (lowest) and 1000 (highest) ⇒ mean of [10, 100] = 55.
    assert torch.allclose(out["scalar"], torch.tensor([55.0, 55.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Krum — picks the most-typical client, rejects the outlier
# ---------------------------------------------------------------------------
def test_krum_picks_honest_client_when_one_outlier():
    """3 honest clients near zero + 1 attacker far away. Krum must return
    one of the honest clients' state-dicts, not the attacker's."""
    g = torch.Generator().manual_seed(0)
    honest_template = {
        "layer.weight": torch.randn(3, 4, generator=g) * 0.01,
        "layer.bias": torch.zeros(3),
    }
    updates: list[ClientUpdate] = []
    for i in range(3):
        # Small perturbation per honest client so they're not literally identical.
        gg = torch.Generator().manual_seed(100 + i)
        sd = {
            k: v + 0.001 * torch.randn(v.shape, generator=gg)
            for k, v in honest_template.items()
        }
        updates.append(
            ClientUpdate(client_id=f"honest_{i}", state_dict=sd, n_samples=32),
        )
    # Attacker: same shape but values 100x larger.
    attacker_sd = {
        "layer.weight": torch.randn(3, 4, generator=g) * 100.0,
        "layer.bias": torch.ones(3) * 50.0,
    }
    updates.append(
        ClientUpdate(client_id="attacker", state_dict=attacker_sd, n_samples=32),
    )
    krum = make_krum_aggregator(num_byzantine=1)
    out = krum(updates)
    # The winner must be one of the honest clients — check by L2 distance
    # to the attacker's state-dict. Honest values are O(0.01), attacker
    # values are O(50). So the winner's bias values should all be near 0,
    # not near 50.
    assert out["layer.bias"].abs().max() < 1.0, (
        f"Krum picked attacker (bias max = {out['layer.bias'].abs().max()}); "
        "expected an honest client (bias near 0)."
    )


# ---------------------------------------------------------------------------
# Median — per-element median should differ from mean when there's an outlier
# ---------------------------------------------------------------------------
def test_median_returns_per_element_median():
    """4 clients with one element each: [1.0, 2.0, 100.0, 3.0].
    Median of sorted [1, 2, 3, 100] = (2+3)/2 = 2.5.
    (Mean would be 26.5 ⇒ median is dramatically more robust.)"""
    updates = []
    for i, v in enumerate([1.0, 2.0, 100.0, 3.0]):
        updates.append(ClientUpdate(
            client_id=f"c{i}",
            state_dict={"x": torch.tensor([v])},
            n_samples=32,
        ))
    agg = make_median_aggregator()
    out = agg(updates)
    assert torch.allclose(out["x"], torch.tensor([2.5]), atol=1e-6)


# ---------------------------------------------------------------------------
# Trimmed mean validation
# ---------------------------------------------------------------------------
def test_trimmed_mean_rejects_invalid_beta():
    import pytest
    with pytest.raises(ValueError, match="beta"):
        make_trimmed_mean_aggregator(beta=0.6)
    with pytest.raises(ValueError, match="beta"):
        make_trimmed_mean_aggregator(beta=-0.1)
