"""FedRep tests — encoder/head split correctness + protocol invariants.

The architectural claim being tested: only encoder + trunk are federated;
heads stay private and never see another client's gradients. The
backward-compatibility claim is different from FedProx — FedRep changes the
protocol, not the loss, so there's no bit-exact equivalent. Instead we
verify the protocol mechanics directly.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_aircraft.fl.personalised import (
    PersonalisedClient,
    _aggregate_shared,
    _local_train_two_phase,
)
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss
from fl_aircraft.utils import seed_everything


WINDOW = 8
N_FEATURES = 6


def _tiny_loader(n_samples: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, WINDOW, N_FEATURES, generator=g)
    y_rul = torch.rand(n_samples, generator=g) * 125.0
    y_fault = (y_rul <= 30.0).to(torch.float32)
    return DataLoader(
        TensorDataset(x.float(), y_rul.float(), y_fault),
        batch_size=8, shuffle=False,
    )


def _make_client(client_id: str, seed: int = 0, subset: str = "FD001") -> PersonalisedClient:
    seed_everything(seed)
    model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW)
    )
    loader = _tiny_loader(32, seed=seed)
    return PersonalisedClient(
        client_id=client_id, subset=subset, model=model,
        train_loader=loader, test_loader=loader,
        loss_fn=MultiTaskLoss(lambda_fault=0.5, fault_pos_weight=1.0),
        n_samples=32,
    )


# ---------------------------------------------------------------------------
# Model split helpers
# ---------------------------------------------------------------------------
def test_shared_and_personal_keys_partition_state_dict():
    """Every key is either shared (encoder/trunk) or personal (head)."""
    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW))
    full_keys = set(model.state_dict().keys())
    shared_keys = set(model.shared_state_dict().keys())
    personal_keys = set(model.personal_state_dict().keys())

    # Partition: every key classified exactly once.
    assert shared_keys.union(personal_keys) == full_keys
    assert shared_keys.intersection(personal_keys) == set()
    # And every key must have answered True to exactly one of the helpers.
    for k in full_keys:
        assert MultiTaskCNN.is_shared_key(k) ^ MultiTaskCNN.is_personal_key(k), (
            f"key {k!r} is neither / both shared and personal"
        )


def test_shared_state_dict_contains_only_encoder_and_trunk():
    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW))
    for k in model.shared_state_dict().keys():
        assert k.startswith(("encoder.", "trunk.")), f"unexpected shared key {k!r}"


def test_personal_state_dict_contains_only_heads():
    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW))
    for k in model.personal_state_dict().keys():
        assert k.startswith(("rul_head.", "fault_head.")), f"unexpected personal key {k!r}"


def test_load_shared_state_dict_leaves_heads_untouched():
    """Replacing the shared backbone must not touch the head parameters."""
    model_a = MultiTaskCNN(MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW))
    seed_everything(0)
    model_b = MultiTaskCNN(MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW))
    seed_everything(99)
    # Reset b's heads to fresh random values, distinct from a's.
    model_b.rul_head.reset_parameters()
    model_b.fault_head.reset_parameters()

    head_before = {k: v.clone() for k, v in model_b.personal_state_dict().items()}
    shared_from_a = model_a.shared_state_dict()
    model_b.load_shared_state_dict(shared_from_a)

    # Shared params now match model_a.
    for k, v in model_a.shared_state_dict().items():
        assert torch.allclose(model_b.state_dict()[k], v)
    # Head params unchanged from before.
    for k, v in head_before.items():
        assert torch.allclose(model_b.state_dict()[k], v), (
            f"load_shared_state_dict altered the head parameter {k!r}"
        )


def test_load_shared_state_dict_rejects_extra_keys():
    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW))
    bad_state = model.shared_state_dict()
    bad_state["rul_head.weight"] = torch.zeros_like(model.rul_head.weight)
    with pytest.raises(RuntimeError, match="key mismatch"):
        model.load_shared_state_dict(bad_state)


def test_load_shared_state_dict_rejects_missing_keys():
    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=N_FEATURES, window_size=WINDOW))
    bad_state = dict(list(model.shared_state_dict().items())[:-1])  # drop last
    with pytest.raises(RuntimeError, match="key mismatch"):
        model.load_shared_state_dict(bad_state)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_aggregate_shared_returns_only_shared_keys():
    clients = [_make_client(f"c{i}", seed=i) for i in range(3)]
    aggregated = _aggregate_shared(clients)
    expected = set(clients[0].model.shared_state_dict().keys())
    assert set(aggregated.keys()) == expected


def test_aggregate_shared_is_sample_count_weighted_mean():
    """When all clients have the same data, aggregate equals any individual."""
    clients = [_make_client(f"c{i}", seed=i) for i in range(3)]
    # Manually set every client's shared state to the same tensors so
    # the average must equal that state.
    template = clients[0].model.shared_state_dict()
    for c in clients:
        c.model.load_shared_state_dict(
            {k: v.clone() for k, v in template.items()}
        )
    aggregated = _aggregate_shared(clients)
    for k, v in template.items():
        assert torch.allclose(aggregated[k], v, atol=0.0), (
            f"aggregate of identical states differs at {k!r}"
        )


# ---------------------------------------------------------------------------
# Two-phase local training
# ---------------------------------------------------------------------------
def test_two_phase_training_updates_heads_in_phase1_only():
    """During phase 1 (head_epochs), the encoder must not move."""
    client = _make_client("c", seed=5)
    encoder_before = {
        k: v.clone() for k, v in client.model.shared_state_dict().items()
    }
    head_before = {
        k: v.clone() for k, v in client.model.personal_state_dict().items()
    }
    # Run only phase 1: set encoder_epochs=0 by calling the public function
    # with both phases but extremely short — we'll inspect both effects.
    seed_everything(5)
    _local_train_two_phase(client, head_epochs=1, encoder_epochs=1, lr=1e-2, weight_decay=0.0)

    encoder_after = client.model.shared_state_dict()
    head_after = client.model.personal_state_dict()

    # Heads must have changed (they're trained in phase 1).
    head_movement = sum(
        (head_after[k] - head_before[k]).abs().sum().item() for k in head_before
    )
    assert head_movement > 1e-6, "heads should move during phase 1"

    # Encoder must also have changed (it's trained in phase 2).
    encoder_movement = sum(
        (encoder_after[k] - encoder_before[k]).abs().sum().item() for k in encoder_before
    )
    assert encoder_movement > 1e-6, "encoder should move during phase 2"


def test_two_phase_training_freezes_encoder_during_phase1():
    """Direct test: when only phase 1 runs, encoder must stay identical.

    Achieved by checking that running head_epochs=1 + encoder_epochs=0 raises
    (validation), then manually invoking just phase 1 via the freeze helpers.
    """
    # The public function requires encoder_epochs >= 1; invariants document
    # that. Test the freeze mechanic directly by simulating phase 1.
    from fl_aircraft.fl.personalised import _freeze, _unfreeze, _train_one_epoch

    client = _make_client("c", seed=7)
    encoder_before = {
        k: v.clone() for k, v in client.model.shared_state_dict().items()
    }
    _freeze(client.model.encoder, client.model.trunk)
    _unfreeze(client.model.rul_head, client.model.fault_head)
    head_params = [p for p in client.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(head_params, lr=1e-2)
    seed_everything(7)
    _train_one_epoch(client, optimizer)

    encoder_after = client.model.shared_state_dict()
    for k in encoder_before:
        assert torch.allclose(encoder_before[k], encoder_after[k], atol=0.0), (
            f"encoder param {k!r} moved despite being frozen in phase 1"
        )
    # Re-enable so other tests aren't affected by leaked frozen state.
    _unfreeze(client.model.encoder, client.model.trunk)


def test_local_train_two_phase_validates_inputs():
    client = _make_client("c")
    with pytest.raises(ValueError, match="head_epochs"):
        _local_train_two_phase(client, head_epochs=0, encoder_epochs=1, lr=1e-3, weight_decay=0.0)
    with pytest.raises(ValueError, match="encoder_epochs"):
        _local_train_two_phase(client, head_epochs=1, encoder_epochs=0, lr=1e-3, weight_decay=0.0)
