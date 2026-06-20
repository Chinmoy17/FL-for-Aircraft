"""FedRep — federated representation learning (Collins et al., ICML 2021).

This is the **architectural-layer** follow-up to RQ2's negative finding +
FedProx's small positive: instead of sharing one global model across
heterogeneous clients, federate ONLY the encoder (the representation) and
let each client keep its own classifier heads.

Why this matters for our Non-IID setup
--------------------------------------
Our P6 partition has 4 clients: 2 on FD001 (HPC degradation only) and 2 on
FD003 (HPC + Fan degradation). Vanilla FedAvg / FedProx forced *one*
decision boundary to fit both fault-mode mixes. That's structurally
impossible — RQ2 + FedProx both ceiling at RMSE ~17.7 because of it. FedRep
removes the impossibility by letting each client own its own boundary.

Protocol per round (Collins et al. recipe, ours uses τ_head=τ_enc=1 by default)
-------------------------------------------------------------------------------
1. Server broadcasts the shared encoder + trunk state.
2. Each client:
   a. Loads the shared backbone (leaves its private heads untouched).
   b. **Head phase**: trains heads only (encoder frozen) for τ_head epochs.
   c. **Encoder phase**: trains encoder only (heads frozen) for τ_enc epochs.
   d. Sends back only the updated encoder + trunk weights.
3. Server averages the encoders via vanilla FedAvg (sample-count weighted).
4. Evaluation: each client scores the (shared encoder + own head) model on
   its OWN subset's test slice. There is no global model to evaluate — the
   per-client metrics ARE the deliverable.

Per-client test split
---------------------
A test row is routed to a client based on its origin subset:
  client_1, client_2 → score on FD001 test engines
  client_3, client_4 → score on FD003 test engines
The two FD001 clients then share the same test set; the model used to score
is each client's own (shared encoder + that client's head). This mirrors
how an airline would deploy FedRep in production: receive the updated
encoder, plug in your own head, score on your fleet.

Outputs
-------
:func:`run_fedrep_from_bundle` returns a :class:`FedRepHistory` with
per-round / per-client trajectories. The headline numbers are
``best_macro_rmse`` (mean across clients of each client's best-round RMSE)
and the per-subset / per-client tables.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import (
    CMAPSSWindowDataset,
    ClientShard,
    Normalizer,
    TrainTestBundle,
    make_test_windows,
    make_training_windows,
    slice_for_client,
)
from ..eval import (
    ClassificationMetrics,
    RegressionMetrics,
    compute_classification_metrics,
    compute_regression_metrics,
)
from ..models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss
from ..utils import seed_everything
from .server import ClientUpdate, fedavg_aggregate


# ---------------------------------------------------------------------------
# History containers
# ---------------------------------------------------------------------------
@dataclass
class FedRepClientMetrics:
    """Per-round test metrics for one client's (shared enc + own head) model."""

    client_id: str
    subset: str
    rmse: float
    mae: float
    nasa_score: float
    auprc: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "client_id": self.client_id,
            "subset": self.subset,
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "nasa_score": round(self.nasa_score, 4),
            "auprc": round(self.auprc, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class FedRepRoundRecord:
    """One row of per-round federated-rep training history."""

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
    round_seconds: float


@dataclass
class FedRepHistory:
    """Complete output of :func:`run_fedrep_from_bundle`."""

    rounds: list[FedRepRoundRecord]
    best_round: int
    best_macro_rmse: float
    best_macro_nasa_score: float
    # Per-client checkpoints at best round: client_id -> full state-dict
    # (shared encoder + that client's own head).
    best_state_dicts: dict[str, dict[str, torch.Tensor]]
    total_seconds: float
    client_ids: list[str]
    per_round_client_rmse: dict[str, list[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cosine_lr(round_idx_1based: int, n_rounds: int, lr_max: float) -> float:
    t = (round_idx_1based - 1) / max(n_rounds - 1, 1)
    return float(lr_max * 0.5 * (1.0 + math.cos(math.pi * t)))


def _freeze(*modules: torch.nn.Module) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(False)


def _unfreeze(*modules: torch.nn.Module) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Per-client state — one heads-set per client, plus a single shared backbone
# ---------------------------------------------------------------------------
@dataclass
class PersonalisedClient:
    """One FedRep client.

    Holds:
      - model: a full MultiTaskCNN; the backbone is overwritten with the
        server's shared state at the start of each round, but the heads are
        the client's private parameters (only updated locally).
      - test_loader: this client's own test-set evaluation loader (filtered
        to the engines from the client's origin subset).
    """

    client_id: str
    subset: str  # one of bundle.subsets — used to route test data
    model: MultiTaskCNN
    train_loader: DataLoader
    test_loader: DataLoader
    loss_fn: MultiTaskLoss
    n_samples: int


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def build_personalised_clients_from_bundle(
    bundle: TrainTestBundle,
    shards: list[ClientShard],
    batch_size: int,
    lambda_fault: float,
    seed: int,
    shard_to_subset: dict[str, str],
) -> list[PersonalisedClient]:
    """Construct one :class:`PersonalisedClient` per shard.

    Each client gets its OWN model (so heads are truly private) and its OWN
    test loader (filtered to engines from the same subset the client was
    trained on). The training loader uses the client's own normalizer; the
    test loader also uses that normalizer for consistency.

    Args:
        bundle: Multi-subset bundle for the partitioned data.
        shards: Per-client training shards from ``partition_by_subset_halves``.
        batch_size: Loader batch size.
        lambda_fault: Multi-task loss weighting (matches centralized).
        seed: Initial parameter seed (shared across all clients so the
            backbone init is identical at round 0).
        shard_to_subset: Map of client_id -> origin subset name (so we know
            which slice of the test set to evaluate each client on).
    """
    if not shards:
        raise ValueError("shards must be non-empty.")
    if "source_subset" not in bundle.test_raw_df.columns:
        raise ValueError(
            "FedRep requires a multi-subset bundle with a 'source_subset' "
            "column on test_raw_df; got bundle.test_raw_df with columns "
            f"{list(bundle.test_raw_df.columns)}."
        )

    clients: list[PersonalisedClient] = []
    for shard in shards:
        subset = shard_to_subset.get(shard.client_id)
        if subset is None:
            raise ValueError(
                f"shard_to_subset missing entry for client_id {shard.client_id!r}."
            )

        client_df = slice_for_client(bundle.train_df, shard)
        client_norm = Normalizer.fit(client_df, bundle.feature_cols)

        train_arrays = make_training_windows(
            client_norm.transform(client_df),
            bundle.feature_cols, bundle.window_size, bundle.stride,
        )
        train_loader = DataLoader(
            CMAPSSWindowDataset(train_arrays), batch_size=batch_size,
            shuffle=True, num_workers=0,
        )

        # Per-client test slice: take only the test engines that originated
        # in the same subset this client was trained on. This is how an
        # airline would actually use FedRep — receive the shared encoder,
        # plug in their own head, score on their own fleet.
        test_df_all = bundle.test_raw_df
        test_mask = test_df_all["source_subset"] == subset
        sub_test_df = test_df_all[test_mask].copy()
        if sub_test_df.empty:
            raise ValueError(
                f"No test rows found for client {shard.client_id!r} subset {subset!r}."
            )
        # Map the subset-filtered engines back to positions in bundle.test_rul.
        from ..data import UNIT_ID_COL  # local import to avoid circulars
        full_engines = sorted(bundle.test_raw_df[UNIT_ID_COL].unique())
        engine_ids = sorted(sub_test_df[UNIT_ID_COL].unique())
        sub_rul = bundle.test_rul[
            np.array([full_engines.index(u) for u in engine_ids])
        ]
        test_arrays = make_test_windows(
            client_norm.transform(sub_test_df),
            sub_rul, bundle.feature_cols,
            bundle.window_size, bundle.rul_cap, bundle.fault_threshold,
        )
        test_loader = DataLoader(
            CMAPSSWindowDataset(test_arrays), batch_size=batch_size,
            shuffle=False, num_workers=0,
        )

        seed_everything(seed)
        model = MultiTaskCNN(
            MultiTaskCNNConfig(
                n_features=bundle.n_features, window_size=bundle.window_size,
            )
        )
        n_pos = int(train_arrays.y_fault.sum())
        n_neg = int(train_arrays.y_fault.shape[0] - n_pos)
        pos_weight = float(n_neg) / float(max(n_pos, 1))
        loss_fn = MultiTaskLoss(lambda_fault=lambda_fault, fault_pos_weight=pos_weight)

        clients.append(
            PersonalisedClient(
                client_id=shard.client_id,
                subset=subset,
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                loss_fn=loss_fn,
                n_samples=train_arrays.n_samples,
            )
        )
    return clients


# ---------------------------------------------------------------------------
# Local training (two-phase: head-only, then encoder-only)
# ---------------------------------------------------------------------------
def _train_one_epoch(
    client: PersonalisedClient,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float, float]:
    """One epoch with only the currently-unfrozen parameters being updated."""
    client.model.train()
    running_total = 0.0
    running_rul = 0.0
    running_fault = 0.0
    n_batches = 0
    for x, y_rul, y_fault in client.train_loader:
        optimizer.zero_grad(set_to_none=True)
        pred = client.model(x)
        losses = client.loss_fn(pred, y_rul, y_fault)
        losses.total.backward()
        optimizer.step()
        running_total += losses.total.item()
        running_rul += losses.rul.item()
        running_fault += losses.fault.item()
        n_batches += 1
    if n_batches == 0:
        raise ValueError("Training loader produced zero batches.")
    return (
        running_total / n_batches,
        running_rul / n_batches,
        running_fault / n_batches,
    )


def _local_train_two_phase(
    client: PersonalisedClient,
    head_epochs: int,
    encoder_epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[float, float, float]:
    """Two-phase FedRep local training.

    Phase 1: heads only. Encoder + trunk frozen, optimizer sees only head params.
    Phase 2: encoder + trunk only. Heads frozen, optimizer sees only backbone params.

    Returns mean task-losses averaged across all (head_epochs + encoder_epochs)
    epochs. Same convention as :class:`FederatedClient.local_train` so the
    numbers stay comparable across protocols.
    """
    if head_epochs < 1:
        raise ValueError(f"head_epochs must be >= 1, got {head_epochs}.")
    if encoder_epochs < 1:
        raise ValueError(f"encoder_epochs must be >= 1, got {encoder_epochs}.")

    total_t = total_r = total_f = 0.0

    # ---- Phase 1: head only ----
    _freeze(client.model.encoder, client.model.trunk)
    _unfreeze(client.model.rul_head, client.model.fault_head)
    head_params = [p for p in client.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(head_params, lr=lr, weight_decay=weight_decay)
    for _ in range(head_epochs):
        t, r, f = _train_one_epoch(client, optimizer)
        total_t += t; total_r += r; total_f += f

    # ---- Phase 2: encoder + trunk only ----
    _unfreeze(client.model.encoder, client.model.trunk)
    _freeze(client.model.rul_head, client.model.fault_head)
    enc_params = [p for p in client.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(enc_params, lr=lr, weight_decay=weight_decay)
    for _ in range(encoder_epochs):
        t, r, f = _train_one_epoch(client, optimizer)
        total_t += t; total_r += r; total_f += f

    # Re-unfreeze everything for hygiene (eval/forward doesn't care, but a
    # subsequent caller using a fresh optimizer would expect all params trainable).
    _unfreeze(client.model.encoder, client.model.trunk,
              client.model.rul_head, client.model.fault_head)

    n_epochs = head_epochs + encoder_epochs
    return (total_t / n_epochs, total_r / n_epochs, total_f / n_epochs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate_client(client: PersonalisedClient) -> FedRepClientMetrics:
    """Score this client's (shared backbone + own head) model on its own test slice."""
    client.model.eval()
    rul_preds: list[np.ndarray] = []
    rul_trues: list[np.ndarray] = []
    fault_scores: list[np.ndarray] = []
    fault_trues: list[np.ndarray] = []
    for x, y_rul, y_fault in client.test_loader:
        pred = client.model(x)
        rul_preds.append(pred.rul.numpy())
        rul_trues.append(y_rul.numpy())
        fault_scores.append(pred.fault_probs().numpy())
        fault_trues.append(y_fault.numpy())
    if not rul_preds:
        raise RuntimeError(f"Client {client.client_id} produced zero test batches.")
    y_rul_pred = np.concatenate(rul_preds)
    y_rul_true = np.concatenate(rul_trues)
    y_fault_score = np.concatenate(fault_scores)
    y_fault_true = np.concatenate(fault_trues)
    rul_m = compute_regression_metrics(y_rul_true, y_rul_pred)
    fault_m = compute_classification_metrics(y_fault_true, y_fault_score)
    return FedRepClientMetrics(
        client_id=client.client_id, subset=client.subset,
        rmse=rul_m.rmse, mae=rul_m.mae, nasa_score=rul_m.nasa_score,
        auprc=fault_m.auprc, f1=fault_m.f1,
    )


# ---------------------------------------------------------------------------
# Encoder-only aggregation
# ---------------------------------------------------------------------------
def _aggregate_shared(
    clients: Sequence[PersonalisedClient],
) -> dict[str, torch.Tensor]:
    """FedAvg over each client's shared backbone (encoder + trunk only).

    Uses :func:`fedavg_aggregate` so the math matches every other phase
    bit-for-bit. Each ``ClientUpdate.state_dict`` is the *shared* state-dict
    only — heads never leave clients.
    """
    updates = [
        ClientUpdate(
            client_id=c.client_id,
            state_dict={
                k: v.detach().clone() for k, v in c.model.shared_state_dict().items()
            },
            n_samples=c.n_samples,
        )
        for c in clients
    ]
    return fedavg_aggregate(updates)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run_fedrep_from_bundle(
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
) -> FedRepHistory:
    """Run a FedRep simulation against ``bundle``'s data and ``shards``.

    Total local epochs per round = ``head_epochs + encoder_epochs``. Default
    1 + 1 = 2 to match the per-round compute budget of vanilla FedAvg
    (which does 2 epochs of joint training).
    """
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if not shards:
        raise ValueError("shards must be non-empty.")

    clients = build_personalised_clients_from_bundle(
        bundle, shards, batch_size, lambda_fault, seed, shard_to_subset,
    )

    # Initial shared state == every client's encoder+trunk at construction
    # (they all initialised from the same seed, so they're identical).
    shared_state = {
        k: v.detach().clone() for k, v in clients[0].model.shared_state_dict().items()
    }

    history: list[FedRepRoundRecord] = []
    per_round_client_rmse: dict[str, list[float]] = {c.client_id: [] for c in clients}
    best_macro_rmse = float("inf")
    best_macro_nasa = float("inf")
    best_round = 0
    best_state_dicts: dict[str, dict[str, torch.Tensor]] = {}

    total_start = time.perf_counter()
    for r in range(1, n_rounds + 1):
        round_start = time.perf_counter()
        current_lr = (
            _cosine_lr(r, n_rounds, lr) if use_cosine_schedule else lr
        )

        # 1. Broadcast shared backbone to every client.
        for client in clients:
            client.model.load_shared_state_dict(shared_state)

        # 2. Local two-phase training per client.
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

        # 3. Encoder-only aggregation.
        shared_state = _aggregate_shared(clients)

        # 4. After-aggregation re-broadcast + per-client evaluation.
        # Each client now has the new shared backbone + its own head, which
        # is the model they would deploy after this round.
        for client in clients:
            client.model.load_shared_state_dict(shared_state)

        per_client_metrics = [_evaluate_client(c) for c in clients]
        for m in per_client_metrics:
            per_round_client_rmse[m.client_id].append(m.rmse)
        macro_rmse = float(np.mean([m.rmse for m in per_client_metrics]))
        macro_nasa = float(np.mean([m.nasa_score for m in per_client_metrics]))
        macro_auprc = float(np.mean([m.auprc for m in per_client_metrics]))
        macro_f1 = float(np.mean([m.f1 for m in per_client_metrics]))

        round_seconds = time.perf_counter() - round_start
        record = FedRepRoundRecord(
            round=r, lr=float(current_lr),
            mean_client_loss_total=float(mean_total),
            mean_client_loss_rul=float(mean_rul),
            mean_client_loss_fault=float(mean_fault),
            per_client_metrics=per_client_metrics,
            macro_rmse=macro_rmse, macro_nasa_score=macro_nasa,
            macro_auprc=macro_auprc, macro_f1=macro_f1,
            round_seconds=float(round_seconds),
        )
        history.append(record)

        # Best round is selected by macro NASA score (matches centralized convention)
        if macro_nasa < best_macro_nasa:
            best_macro_nasa = macro_nasa
            best_macro_rmse = macro_rmse
            best_round = r
            # Snapshot each client's full state-dict (shared + own head)
            best_state_dicts = {
                c.client_id: {
                    k: v.detach().clone() for k, v in c.model.state_dict().items()
                }
                for c in clients
            }

        if r % log_every == 0 or r == 1 or r == n_rounds:
            per_client_str = "  ".join(
                f"{m.client_id[:8]}({m.subset[-3:]})={m.rmse:.2f}"
                for m in per_client_metrics
            )
            print(
                f"round {r:>3}/{n_rounds}  lr={current_lr:.2e}  "
                f"loss={mean_total:.3f}  macro_RMSE={macro_rmse:.2f}  "
                f"macro_NASA={macro_nasa:.0f}  | per-client: {per_client_str}  "
                f"({round_seconds:.1f}s)"
            )

    total_seconds = time.perf_counter() - total_start
    return FedRepHistory(
        rounds=history,
        best_round=best_round,
        best_macro_rmse=best_macro_rmse,
        best_macro_nasa_score=best_macro_nasa,
        best_state_dicts=best_state_dicts,
        total_seconds=float(total_seconds),
        client_ids=[c.client_id for c in clients],
        per_round_client_rmse=per_round_client_rmse,
    )
