"""FedCCFA — clustered classifier-fragment aggregation (Chen et al., NeurIPS 2024).

This is the architectural-clustering follow-up to FedRep. FedRep gave every
client its own head; FedCCFA additionally **clusters clients by head
similarity** and lets clients within a cluster share a head, while clients
in different clusters don't. For our P6 partition (2 clients on FD001, 2 on
FD003) the hypothesis is that clustering will correctly identify the two
fault-mode groups and pool head training within each group — giving each
shared head twice the supervision FedRep gave it.

This is a **simplified** FedCCFA. The full paper does class-level fragment
aggregation; for our regression + binary-fault setting we do per-head
clustering at the whole-head level. The core mechanism (similarity-based
client grouping + per-cluster head averaging + encoder still federated
globally) is intact.

Protocol per round
------------------
1. Server broadcasts the shared encoder + trunk.
2. Each client:
   a. Loads the shared backbone (leaves its private heads untouched, *unless*
      the server's cluster broadcast in the previous round overrode them).
   b. Two-phase local training: head-only then encoder-only (same as FedRep).
   c. Sends back: updated encoder/trunk + updated rul_head + updated fault_head.
3. Server:
   a. Aggregates encoders via vanilla FedAvg (sample-count weighted).
   b. Computes a similarity matrix between client heads (cosine similarity
      on flattened head weights).
   c. Greedy clustering: starting from the most similar pair, merge clients
      whose pairwise similarity exceeds ``similarity_threshold``. Singletons
      keep their own heads.
   d. For each cluster, computes the mean head (sample-count weighted).
   e. Broadcasts (new shared encoder, per-cluster head assignment, per-cluster mean head).
4. Each client overrides its head with its cluster's mean.
5. Evaluation: same per-client per-subset protocol as FedRep.

Hyperparameters
---------------
``similarity_threshold`` (default 0.5 cosine): clients above this similarity
   merge into one cluster. With 4 clients in our setup the expected cluster
   structure is {client_1, client_2}, {client_3, client_4}, but we don't
   hard-code this — let the data speak.
``warmup_rounds`` (default 3): how many initial rounds run FedRep-style
   (no clustering) so each client's head has enough signal for the
   similarity computation to mean something.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ..data import (
    ClientShard,
    TrainTestBundle,
)
from ..eval import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from ..models import MultiTaskCNN
from .personalised import (
    FedRepClientMetrics,
    FedRepRoundRecord,
    PersonalisedClient,
    _aggregate_shared,
    _cosine_lr,
    _evaluate_client,
    _local_train_two_phase,
    build_personalised_clients_from_bundle,
)
from .server import ClientUpdate, fedavg_aggregate


# ---------------------------------------------------------------------------
# History — extends FedRepRoundRecord with per-round cluster assignment
# ---------------------------------------------------------------------------
@dataclass
class FedCCFARoundRecord:
    """One row of per-round FedCCFA training history.

    ``clusters`` is a list of lists of client_ids — each inner list is one
    cluster. ``head_similarity_max`` is the highest pairwise cosine
    similarity observed this round (a proxy for "are the heads converging
    on each other or staying apart").
    """

    round: int
    lr: float
    mean_client_loss_total: float
    mean_client_loss_rul: float
    mean_client_loss_fault: float
    per_client_metrics: list[FedRepClientMetrics]
    macro_rmse: float
    macro_nasa_score: float
    macro_auprc: float
    macro_f1: float
    clusters: list[list[str]]
    head_similarity_max: float
    head_similarity_min: float
    round_seconds: float


@dataclass
class FedCCFAHistory:
    """Complete output of :func:`run_fedccfa_from_bundle`."""

    rounds: list[FedCCFARoundRecord]
    best_round: int
    best_macro_rmse: float
    best_macro_nasa_score: float
    best_state_dicts: dict[str, dict[str, torch.Tensor]]
    best_clusters: list[list[str]]
    total_seconds: float
    client_ids: list[str]
    per_round_client_rmse: dict[str, list[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Head similarity + clustering
# ---------------------------------------------------------------------------
def _flatten_heads(client: PersonalisedClient) -> torch.Tensor:
    """One vector per client: flattened (rul_head ‖ fault_head) weights + biases.

    The biases matter — they encode each client's decision threshold offset,
    which is exactly what differs across fault-mode mixes. Including them
    makes the similarity computation more discriminative.
    """
    parts: list[torch.Tensor] = []
    for k, v in client.model.personal_state_dict().items():
        parts.append(v.detach().reshape(-1))
    return torch.cat(parts)


def _pairwise_cosine_similarity(vecs: list[torch.Tensor]) -> np.ndarray:
    """Symmetric ``(n, n)`` cosine-similarity matrix; diagonal is 1.0."""
    n = len(vecs)
    sims = np.zeros((n, n), dtype=np.float64)
    norms = [v / (v.norm() + 1e-12) for v in vecs]
    for i in range(n):
        for j in range(i, n):
            s = float(torch.dot(norms[i], norms[j]).item())
            sims[i, j] = s
            sims[j, i] = s
    return sims


def _cluster_clients(
    client_ids: list[str],
    similarity: np.ndarray,
    threshold: float,
) -> list[list[str]]:
    """Greedy connectivity-based clustering on the similarity matrix.

    Two clients are in the same cluster if their pairwise similarity is
    >= ``threshold``. Transitive: if A ~ B and B ~ C, then A, B, C are
    all in one cluster, even if A ~ C is just below threshold.

    Equivalent to connected components on the graph induced by
    ``similarity >= threshold``.
    """
    n = len(client_ids)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                union(i, j)

    clusters: dict[int, list[str]] = {}
    for i, cid in enumerate(client_ids):
        root = find(i)
        clusters.setdefault(root, []).append(cid)
    return list(clusters.values())


def _aggregate_cluster_heads(
    clients: Sequence[PersonalisedClient],
    clusters: list[list[str]],
) -> dict[str, dict[str, torch.Tensor]]:
    """For each cluster, compute the sample-count-weighted mean of head weights.

    Returns ``{client_id: aggregated_head_state_dict}``. Singleton clusters
    return that client's own head unchanged (no averaging needed).
    """
    client_lookup = {c.client_id: c for c in clients}
    out: dict[str, dict[str, torch.Tensor]] = {}
    for cluster in clusters:
        cluster_clients = [client_lookup[cid] for cid in cluster]
        if len(cluster) == 1:
            out[cluster[0]] = {
                k: v.detach().clone()
                for k, v in cluster_clients[0].model.personal_state_dict().items()
            }
            continue
        # Re-use fedavg_aggregate on the head sub-state-dicts.
        updates = [
            ClientUpdate(
                client_id=c.client_id,
                state_dict={
                    k: v.detach().clone()
                    for k, v in c.model.personal_state_dict().items()
                },
                n_samples=c.n_samples,
            )
            for c in cluster_clients
        ]
        cluster_head = fedavg_aggregate(updates)
        for cid in cluster:
            out[cid] = {k: v.clone() for k, v in cluster_head.items()}
    return out


def _load_personal_state_dict(
    model: MultiTaskCNN, personal_state: dict[str, torch.Tensor]
) -> None:
    """Overwrite only the heads; leave encoder + trunk untouched."""
    full = model.state_dict()
    expected = {k for k in full if MultiTaskCNN.is_personal_key(k)}
    provided = set(personal_state.keys())
    if provided != expected:
        missing = expected - provided
        unexpected = provided - expected
        raise RuntimeError(
            "load_personal_state_dict key mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    merged = {
        k: (personal_state[k] if MultiTaskCNN.is_personal_key(k) else v)
        for k, v in full.items()
    }
    model.load_state_dict(merged)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run_fedccfa_from_bundle(
    bundle: TrainTestBundle,
    shards: list[ClientShard],
    shard_to_subset: dict[str, str],
    *,
    n_rounds: int = 50,
    head_epochs: int = 1,
    encoder_epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_fault: float = 0.5,
    use_cosine_schedule: bool = True,
    seed: int = 42,
    log_every: int = 5,
    similarity_threshold: float = 0.5,
    warmup_rounds: int = 3,
) -> FedCCFAHistory:
    """Run FedCCFA on ``bundle``'s data and ``shards``.

    For rounds 1..``warmup_rounds`` the protocol behaves like FedRep (no
    head sharing). Starting at round ``warmup_rounds + 1`` we compute the
    pairwise head similarity matrix, cluster clients with similarity above
    ``similarity_threshold``, and replace each client's head with its
    cluster's sample-count-weighted mean. Clients in singleton clusters
    keep their own heads.
    """
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if not shards:
        raise ValueError("shards must be non-empty.")
    if warmup_rounds < 0:
        raise ValueError(f"warmup_rounds must be >= 0, got {warmup_rounds}.")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            f"similarity_threshold must be in [-1, 1], got {similarity_threshold}."
        )

    clients = build_personalised_clients_from_bundle(
        bundle, shards, batch_size, lambda_fault, seed, shard_to_subset,
    )
    shared_state = {
        k: v.detach().clone() for k, v in clients[0].model.shared_state_dict().items()
    }

    history: list[FedCCFARoundRecord] = []
    per_round_client_rmse: dict[str, list[float]] = {c.client_id: [] for c in clients}
    best_macro_rmse = float("inf")
    best_macro_nasa = float("inf")
    best_round = 0
    best_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    best_clusters: list[list[str]] = []

    total_start = time.perf_counter()
    for r in range(1, n_rounds + 1):
        round_start = time.perf_counter()
        current_lr = (
            _cosine_lr(r, n_rounds, lr) if use_cosine_schedule else lr
        )

        # 1. Broadcast shared backbone.
        for client in clients:
            client.model.load_shared_state_dict(shared_state)

        # 2. Local two-phase training.
        round_total = round_rul = round_fault = 0.0
        for client in clients:
            ct, cr, cf = _local_train_two_phase(
                client, head_epochs=head_epochs, encoder_epochs=encoder_epochs,
                lr=current_lr, weight_decay=weight_decay,
            )
            round_total += ct; round_rul += cr; round_fault += cf
        n = len(clients)
        mean_total = round_total / n
        mean_rul = round_rul / n
        mean_fault = round_fault / n

        # 3a. Aggregate encoders globally.
        shared_state = _aggregate_shared(clients)

        # 3b. Cluster + aggregate heads (only after warmup).
        head_vecs = [_flatten_heads(c) for c in clients]
        sim_matrix = _pairwise_cosine_similarity(head_vecs)
        off_diag = sim_matrix[np.triu_indices(n, k=1)]
        sim_max = float(off_diag.max()) if off_diag.size > 0 else 1.0
        sim_min = float(off_diag.min()) if off_diag.size > 0 else 1.0

        if r > warmup_rounds:
            clusters = _cluster_clients(
                [c.client_id for c in clients],
                sim_matrix, similarity_threshold,
            )
            # Override each client's head with its cluster's aggregated head.
            cluster_heads = _aggregate_cluster_heads(clients, clusters)
            for client in clients:
                _load_personal_state_dict(client.model, cluster_heads[client.client_id])
        else:
            # Warmup: every client in its own cluster (FedRep-like).
            clusters = [[c.client_id] for c in clients]

        # 4. Broadcast aggregated encoder back (heads already updated above
        # for non-warmup rounds; for warmup they remain at the post-local-
        # training state).
        for client in clients:
            client.model.load_shared_state_dict(shared_state)

        # 5. Evaluate per-client.
        per_client_metrics = [_evaluate_client(c) for c in clients]
        for m in per_client_metrics:
            per_round_client_rmse[m.client_id].append(m.rmse)
        macro_rmse = float(np.mean([m.rmse for m in per_client_metrics]))
        macro_nasa = float(np.mean([m.nasa_score for m in per_client_metrics]))
        macro_auprc = float(np.mean([m.auprc for m in per_client_metrics]))
        macro_f1 = float(np.mean([m.f1 for m in per_client_metrics]))

        round_seconds = time.perf_counter() - round_start
        record = FedCCFARoundRecord(
            round=r, lr=float(current_lr),
            mean_client_loss_total=float(mean_total),
            mean_client_loss_rul=float(mean_rul),
            mean_client_loss_fault=float(mean_fault),
            per_client_metrics=per_client_metrics,
            macro_rmse=macro_rmse, macro_nasa_score=macro_nasa,
            macro_auprc=macro_auprc, macro_f1=macro_f1,
            clusters=clusters,
            head_similarity_max=sim_max,
            head_similarity_min=sim_min,
            round_seconds=float(round_seconds),
        )
        history.append(record)

        if macro_nasa < best_macro_nasa:
            best_macro_nasa = macro_nasa
            best_macro_rmse = macro_rmse
            best_round = r
            best_state_dicts = {
                c.client_id: {
                    k: v.detach().clone() for k, v in c.model.state_dict().items()
                }
                for c in clients
            }
            best_clusters = [list(g) for g in clusters]

        if r % log_every == 0 or r == 1 or r == n_rounds:
            cluster_str = "/".join(
                "+".join(g) if len(g) > 1 else g[0] for g in clusters
            )
            print(
                f"round {r:>3}/{n_rounds}  lr={current_lr:.2e}  "
                f"loss={mean_total:.3f}  macro_RMSE={macro_rmse:.2f}  "
                f"macro_NASA={macro_nasa:.0f}  "
                f"sim[{sim_min:.2f}..{sim_max:.2f}]  "
                f"clusters={cluster_str}  ({round_seconds:.1f}s)"
            )

    total_seconds = time.perf_counter() - total_start
    return FedCCFAHistory(
        rounds=history,
        best_round=best_round,
        best_macro_rmse=best_macro_rmse,
        best_macro_nasa_score=best_macro_nasa,
        best_state_dicts=best_state_dicts,
        best_clusters=best_clusters,
        total_seconds=float(total_seconds),
        client_ids=[c.client_id for c in clients],
        per_round_client_rmse=per_round_client_rmse,
    )
