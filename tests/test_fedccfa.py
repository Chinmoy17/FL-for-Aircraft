"""FedCCFA tests — clustering correctness + protocol invariants."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_aircraft.fl.clustered import (
    _aggregate_cluster_heads,
    _cluster_clients,
    _flatten_heads,
    _load_personal_state_dict,
    _pairwise_cosine_similarity,
)
from fl_aircraft.fl.personalised import PersonalisedClient
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
# Head flattening
# ---------------------------------------------------------------------------
def test_flatten_heads_returns_one_vector_per_client():
    """The flattened-head vector should include ALL head parameters."""
    client = _make_client("c", seed=42)
    vec = _flatten_heads(client)
    # rul_head: 64+1 = 65; fault_head: 64+1 = 65; total = 130 params.
    expected = 0
    for p in client.model.rul_head.parameters():
        expected += p.numel()
    for p in client.model.fault_head.parameters():
        expected += p.numel()
    assert vec.shape == (expected,), (
        f"expected length {expected}, got {vec.shape}"
    )


def test_flatten_heads_excludes_encoder_and_trunk():
    """Two clients with identical heads but different encoders should
    flatten to identical head vectors."""
    seed_everything(0)
    client_a = _make_client("a", seed=0)
    seed_everything(0)
    client_b = _make_client("b", seed=0)
    # Mutate b's encoder so it differs from a's, leaving heads alone.
    with torch.no_grad():
        for p in client_b.model.encoder.parameters():
            p.add_(torch.randn_like(p))
    vec_a = _flatten_heads(client_a)
    vec_b = _flatten_heads(client_b)
    assert torch.allclose(vec_a, vec_b), (
        "head flatten should not depend on encoder/trunk params"
    )


# ---------------------------------------------------------------------------
# Pairwise cosine similarity
# ---------------------------------------------------------------------------
def test_pairwise_cosine_diagonal_is_one():
    vecs = [torch.randn(10), torch.randn(10), torch.randn(10)]
    sims = _pairwise_cosine_similarity(vecs)
    for i in range(3):
        assert abs(sims[i, i] - 1.0) < 1e-6


def test_pairwise_cosine_symmetric():
    vecs = [torch.randn(8) for _ in range(4)]
    sims = _pairwise_cosine_similarity(vecs)
    assert np.allclose(sims, sims.T)


def test_pairwise_cosine_identical_vectors_give_one():
    v = torch.tensor([1.0, 2.0, 3.0])
    vecs = [v.clone(), v.clone(), v.clone()]
    sims = _pairwise_cosine_similarity(vecs)
    assert np.allclose(sims, 1.0)


def test_pairwise_cosine_orthogonal_vectors_give_zero():
    vecs = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    ]
    sims = _pairwise_cosine_similarity(vecs)
    off_diag = sims[~np.eye(3, dtype=bool)]
    assert np.allclose(off_diag, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def test_cluster_clients_all_above_threshold_one_cluster():
    """If everyone is similar to everyone, one big cluster."""
    sims = np.full((4, 4), 0.9)
    np.fill_diagonal(sims, 1.0)
    clusters = _cluster_clients(["c1", "c2", "c3", "c4"], sims, threshold=0.5)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == ["c1", "c2", "c3", "c4"]


def test_cluster_clients_all_below_threshold_singletons():
    """If everyone is dissimilar, 4 singleton clusters."""
    sims = np.full((4, 4), 0.1)
    np.fill_diagonal(sims, 1.0)
    clusters = _cluster_clients(["c1", "c2", "c3", "c4"], sims, threshold=0.5)
    assert len(clusters) == 4
    assert all(len(g) == 1 for g in clusters)


def test_cluster_clients_two_pairs():
    """Designed sim matrix: c1~c2 and c3~c4, but not across pairs."""
    sims = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.9],
        [0.1, 0.1, 0.9, 1.0],
    ])
    clusters = _cluster_clients(["c1", "c2", "c3", "c4"], sims, threshold=0.5)
    assert len(clusters) == 2
    found = sorted(sorted(g) for g in clusters)
    assert found == [["c1", "c2"], ["c3", "c4"]]


def test_cluster_clients_transitive():
    """A~B (0.9) and B~C (0.9) but A~C (0.4 < threshold 0.5).
    All three should still be in one cluster via transitive connectivity."""
    sims = np.array([
        [1.0, 0.9, 0.4, 0.1],
        [0.9, 1.0, 0.9, 0.1],
        [0.4, 0.9, 1.0, 0.1],
        [0.1, 0.1, 0.1, 1.0],
    ])
    clusters = _cluster_clients(["c1", "c2", "c3", "c4"], sims, threshold=0.5)
    assert len(clusters) == 2
    found = sorted(sorted(g) for g in clusters)
    assert found == [["c1", "c2", "c3"], ["c4"]]


# ---------------------------------------------------------------------------
# Cluster head aggregation
# ---------------------------------------------------------------------------
def test_aggregate_cluster_heads_singleton_returns_own_head():
    client = _make_client("c", seed=3)
    original_head = client.model.personal_state_dict()
    out = _aggregate_cluster_heads([client], [["c"]])
    assert "c" in out
    for k, v in original_head.items():
        assert torch.allclose(out["c"][k], v)


def test_aggregate_cluster_heads_two_clients_returns_mean():
    """When two clients have same sample count, cluster head is plain mean."""
    c1 = _make_client("c1", seed=1)
    c2 = _make_client("c2", seed=2)
    # Both have n_samples=32 by construction.
    head1 = c1.model.personal_state_dict()
    head2 = c2.model.personal_state_dict()
    out = _aggregate_cluster_heads([c1, c2], [["c1", "c2"]])
    for k in head1:
        expected = (head1[k].to(torch.float64) + head2[k].to(torch.float64)) / 2
        assert torch.allclose(
            out["c1"][k].to(torch.float64), expected, atol=1e-6
        )
        # Both clients in the cluster should receive the same aggregated head.
        assert torch.allclose(out["c1"][k], out["c2"][k])


# ---------------------------------------------------------------------------
# Load personal state dict
# ---------------------------------------------------------------------------
def test_load_personal_state_dict_leaves_encoder_unchanged():
    client = _make_client("c", seed=5)
    encoder_before = {
        k: v.clone() for k, v in client.model.shared_state_dict().items()
    }
    # Build a head state with all-zeros so it clearly differs from current.
    zero_head = {
        k: torch.zeros_like(v) for k, v in client.model.personal_state_dict().items()
    }
    _load_personal_state_dict(client.model, zero_head)
    # Heads now zero.
    for k, v in client.model.personal_state_dict().items():
        assert torch.allclose(v, torch.zeros_like(v))
    # Encoder unchanged.
    for k, v in encoder_before.items():
        assert torch.allclose(client.model.state_dict()[k], v)


def test_load_personal_state_dict_rejects_extra_keys():
    client = _make_client("c", seed=5)
    bad = client.model.personal_state_dict()
    bad["encoder.0.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="key mismatch"):
        _load_personal_state_dict(client.model, bad)
