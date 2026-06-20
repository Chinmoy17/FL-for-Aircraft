"""FedProx tests — proximal-term correctness + backward compatibility.

The single most important guarantee is that ``mu = 0.0`` is bit-exact
equivalent to the vanilla FedAvg behaviour we shipped in P5 / P6 / RQ2.
The remaining tests pin down the math and the drift-reduction property
that makes FedProx the right answer to RQ2's negative finding.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_aircraft.fl.client import FederatedClient
from fl_aircraft.fl.server import fedavg_aggregate
from fl_aircraft.models import (
    MultiTaskCNN,
    MultiTaskCNNConfig,
    MultiTaskLoss,
)
from fl_aircraft.utils import seed_everything


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
WINDOW = 8
N_FEATURES = 6


def _tiny_loader(n_samples: int, seed: int) -> DataLoader:
    """Tiny synthetic loader so the tests run in <1 s without touching disk."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, WINDOW, N_FEATURES, generator=g)
    y_rul = torch.rand(n_samples, generator=g) * 125.0
    y_fault = (y_rul <= 30.0).to(torch.float32)
    return DataLoader(
        TensorDataset(x.float(), y_rul.float(), y_fault),
        batch_size=8, shuffle=False,
    )


def _make_client(client_id: str, n_samples: int = 32, seed: int = 0) -> FederatedClient:
    seed_everything(seed)
    model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW)
    )
    return FederatedClient(
        client_id=client_id,
        model=model,
        train_loader=_tiny_loader(n_samples, seed=seed),
        loss_fn=MultiTaskLoss(lambda_fault=0.5, fault_pos_weight=1.0),
        n_samples=n_samples,
    )


def _params_as_vec(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _state_dicts_close(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    atol: float = 0.0,
) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    return all(torch.allclose(a[k], b[k], atol=atol) for k in a)


# ---------------------------------------------------------------------------
# Backward compatibility — the must-pass test
# ---------------------------------------------------------------------------
def test_mu_zero_is_bit_exact_vanilla_fedavg():
    """mu=0.0 must produce the same state-dict as omitting the kwarg.

    This is the test that protects every prior phase's reproducibility.
    """
    client_explicit = _make_client("c", seed=7)
    client_implicit = _make_client("c", seed=7)

    seed_everything(7)
    client_explicit.local_train(local_epochs=2, lr=1e-3, mu=0.0)

    seed_everything(7)
    client_implicit.local_train(local_epochs=2, lr=1e-3)

    assert _state_dicts_close(
        client_explicit.model.state_dict(),
        client_implicit.model.state_dict(),
        atol=0.0,
    ), "mu=0.0 produced different weights than omitting the kwarg"


def test_mu_zero_no_extra_compute_path():
    """When mu=0.0 the proximal-term branch must not contribute gradients.

    Verified indirectly by checking the post-training weights match a
    fresh client trained without any FedProx-related machinery.
    """
    a = _make_client("c", seed=11)
    b = _make_client("c", seed=11)
    seed_everything(11)
    a.local_train(local_epochs=1, lr=5e-3, mu=0.0)
    seed_everything(11)
    b.local_train(local_epochs=1, lr=5e-3)
    assert _state_dicts_close(a.model.state_dict(), b.model.state_dict(), atol=0.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_negative_mu_raises():
    client = _make_client("c")
    with pytest.raises(ValueError, match="mu must be >= 0"):
        client.local_train(local_epochs=1, lr=1e-3, mu=-0.01)


# ---------------------------------------------------------------------------
# Drift-reduction — the core mechanistic claim
# ---------------------------------------------------------------------------
def test_larger_mu_keeps_weights_closer_to_start():
    """The defining property of FedProx: bigger mu => less drift per round.

    All clients start from the same initial state. After 2 local epochs:
        ||W_local(mu=0)     - W_init|| > ||W_local(mu=0.01)  - W_init||
                                       > ||W_local(mu=1.0)   - W_init||

    The chain need not be monotonic at every consecutive mu, but the
    extremes must respect the inequality.
    """
    initial_client = _make_client("c", seed=42)
    initial_vec = _params_as_vec(initial_client.model)

    drifts: dict[float, float] = {}
    for mu in [0.0, 0.01, 1.0]:
        client = _make_client("c", seed=42)  # same init every time
        # Sanity check: same initial weights regardless of mu.
        assert torch.allclose(_params_as_vec(client.model), initial_vec)
        seed_everything(42)
        client.local_train(local_epochs=2, lr=5e-3, mu=mu)
        drift = (_params_as_vec(client.model) - initial_vec).norm().item()
        drifts[mu] = drift

    assert drifts[0.0] > drifts[0.01], (
        f"mu=0.01 should reduce drift below mu=0: "
        f"got drift(mu=0)={drifts[0.0]:.4f} vs drift(mu=0.01)={drifts[0.01]:.4f}"
    )
    assert drifts[0.01] > drifts[1.0], (
        f"mu=1.0 should reduce drift further: "
        f"got drift(mu=0.01)={drifts[0.01]:.4f} vs drift(mu=1.0)={drifts[1.0]:.4f}"
    )


def test_extreme_mu_pins_weights_to_global():
    """With huge mu, local training should barely move the weights.

    The proximal term dominates the task loss, so the optimum local
    update is to stay put. We don't require exact equality (a single
    Adam step still moves), but the drift should be vanishingly small.
    """
    client_huge = _make_client("c", seed=3)
    initial_vec = _params_as_vec(client_huge.model)

    seed_everything(3)
    # mu=1e6: proximal term swamps task loss by orders of magnitude.
    client_huge.local_train(local_epochs=2, lr=1e-3, mu=1e6)
    final_vec = _params_as_vec(client_huge.model)

    drift = (final_vec - initial_vec).norm().item()
    # Threshold is generous (Adam still nudges); the point is "almost zero
    # relative to the norm of the model" rather than literal zero.
    init_norm = initial_vec.norm().item()
    assert drift < 0.05 * init_norm, (
        f"drift {drift:.4f} should be << 5% of model norm {init_norm:.4f} "
        f"when mu is huge"
    )


# ---------------------------------------------------------------------------
# Aggregation interplay — the RQ2 finding made concrete
# ---------------------------------------------------------------------------
def test_fedprox_aggregate_closer_to_global_than_fedavg():
    """Two clients with opposing-bias data drift to opposing optima.

    Under FedAvg the aggregate is somewhere between the two extremes —
    far from the round-start global weights. Under FedProx the same
    two clients drift much less, so their aggregate stays much closer
    to the global weights.

    This is the mechanistic claim from rq2_report.md §6.1 made concrete:
    'no convex combination of opposing Delta_i can equal the centralised
    solution; the right fix is to shrink the Delta_i themselves.'
    """
    seed_everything(99)
    initial_state = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW)
    ).state_dict()
    init_state_copy = {k: v.clone() for k, v in initial_state.items()}

    # Two clients with totally different (synthetic) data.
    def make_pair(mu: float) -> dict[str, torch.Tensor]:
        client_a = _make_client("a", seed=1)
        client_b = _make_client("b", seed=2)
        for c in (client_a, client_b):
            c.model.load_state_dict(init_state_copy)
        seed_everything(123)
        client_a.local_train(local_epochs=2, lr=5e-3, mu=mu)
        seed_everything(456)
        client_b.local_train(local_epochs=2, lr=5e-3, mu=mu)
        return fedavg_aggregate([client_a.package_update(), client_b.package_update()])

    aggregated_fedavg = make_pair(mu=0.0)
    aggregated_fedprox = make_pair(mu=1.0)

    def dist_from_init(state: dict[str, torch.Tensor]) -> float:
        d = 0.0
        for k in state:
            d += ((state[k] - init_state_copy[k]) ** 2).sum().item()
        return d ** 0.5

    fedavg_distance = dist_from_init(aggregated_fedavg)
    fedprox_distance = dist_from_init(aggregated_fedprox)

    assert fedprox_distance < fedavg_distance, (
        f"FedProx aggregate should stay closer to the global model "
        f"than FedAvg's: got fedprox={fedprox_distance:.4f} "
        f"vs fedavg={fedavg_distance:.4f}"
    )


# ---------------------------------------------------------------------------
# Loss-reporting invariance
# ---------------------------------------------------------------------------
def test_reported_losses_exclude_proximal_term():
    """The returned ``total`` should be the TASK loss only, comparable
    across mu values — even though the gradient step uses task + proximal.

    Test: train two clients identically except for mu. Their reported
    losses should be similar in magnitude (within an order of magnitude
    of each other) because both report task loss. If we accidentally
    included the proximal term, the mu=1.0 client's reported total
    would explode.
    """
    client_zero = _make_client("c", seed=21)
    client_strong = _make_client("c", seed=21)

    seed_everything(21)
    loss_zero, _, _ = client_zero.local_train(local_epochs=1, lr=1e-3, mu=0.0)
    seed_everything(21)
    loss_strong, _, _ = client_strong.local_train(local_epochs=1, lr=1e-3, mu=10.0)

    # If proximal-term contamination existed, loss_strong would be >>
    # loss_zero by a factor proportional to mu * ||W||^2 (potentially
    # hundreds). Same-order means we're reporting the right thing.
    ratio = loss_strong / max(loss_zero, 1e-9)
    assert 0.5 < ratio < 2.0, (
        f"Reported loss should be task-only (~comparable across mu); "
        f"got loss(mu=0)={loss_zero:.4f}, loss(mu=10)={loss_strong:.4f}, "
        f"ratio={ratio:.4f}"
    )


# ---------------------------------------------------------------------------
# Train-mode side effects
# ---------------------------------------------------------------------------
def test_local_train_leaves_model_in_train_mode():
    """``local_train`` leaves the model in train mode (matches vanilla).

    The eval/train state matters for dropout. We never call ``.eval()``
    inside ``local_train``, so the model should remain in train mode
    afterward whether or not mu is used.
    """
    client = _make_client("c")
    assert client.model.training is True  # default for nn.Module
    client.local_train(local_epochs=1, lr=1e-3, mu=0.0)
    assert client.model.training is True
    client.local_train(local_epochs=1, lr=1e-3, mu=0.01)
    assert client.model.training is True
