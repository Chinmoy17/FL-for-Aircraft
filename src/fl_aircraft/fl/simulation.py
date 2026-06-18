"""In-process FedAvg simulation loop.

This is the orchestrator that ties :class:`FedAvgServer`, every
:class:`FederatedClient`, and the test-set evaluation together for
``n_rounds`` communication rounds.

Two public entry points:

- :func:`run_fedavg_from_bundle` — takes a pre-built
  :class:`TrainTestBundle` and a list of :class:`ClientShard` (Phase 6+).
- :func:`run_fedavg` — convenience wrapper that builds the bundle from a
  single-subset :class:`CMAPSSConfig` and applies the
  stratified-by-lifetime partition (Phase 5 default).

Why in-process rather than Flower / IPC?

- **Zero network overhead** on CPU; the whole training set fits in RAM.
- **Full protocol introspection** — we expose the per-round state dicts, the
  per-client local losses, and the post-aggregation global metrics. RQ2
  (custom aggregation weights), RQ5 (per-client validation scores), and RQ7
  (poisoning detection) all need this access.
- **Deterministic, debuggable, ~300 lines of code** total across server,
  client, simulation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import (
    CMAPSSConfig,
    CMAPSSWindowDataset,
    ClientShard,
    Normalizer,
    TrainTestBundle,
    bundle_from_config,
    make_test_windows,
    make_training_windows,
    partition_by_lifetime,
    slice_for_client,
)
from ..eval import ClassificationMetrics, RegressionMetrics
from ..models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss
from ..train.centralized import evaluate
from ..utils import seed_everything
from .client import FederatedClient
from .server import FedAvgServer


@dataclass
class RoundRecord:
    """One row of the per-round federated training history."""

    round: int
    lr: float
    # Mean across clients of each client's local-epoch-averaged loss.
    mean_client_loss_total: float
    mean_client_loss_rul: float
    mean_client_loss_fault: float
    # Global model evaluated on the common test set after aggregation.
    global_test_rmse: float
    global_test_mae: float
    global_test_nasa_score: float
    global_test_auprc: float
    global_test_f1: float
    global_test_precision: float
    global_test_recall: float
    round_seconds: float

    def as_dict(self) -> dict[str, float]:
        return {
            "round": self.round,
            "lr": self.lr,
            "mean_client_loss_total": self.mean_client_loss_total,
            "mean_client_loss_rul": self.mean_client_loss_rul,
            "mean_client_loss_fault": self.mean_client_loss_fault,
            "global_test_rmse": self.global_test_rmse,
            "global_test_mae": self.global_test_mae,
            "global_test_nasa_score": self.global_test_nasa_score,
            "global_test_auprc": self.global_test_auprc,
            "global_test_f1": self.global_test_f1,
            "global_test_precision": self.global_test_precision,
            "global_test_recall": self.global_test_recall,
            "round_seconds": self.round_seconds,
        }


@dataclass
class FederatedHistory:
    """Complete output of :func:`run_fedavg`."""

    rounds: list[RoundRecord]
    best_round: int
    best_state_dict: dict[str, torch.Tensor]
    total_seconds: float
    final_test_rul: RegressionMetrics
    final_test_fault: ClassificationMetrics
    best_test_rul: RegressionMetrics
    best_test_fault: ClassificationMetrics
    client_ids: list[str]
    per_round_client_losses: dict[str, list[float]] = field(default_factory=dict)
    final_predictions: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rounds)


def _cosine_lr(round_idx_1based: int, n_rounds: int, lr_max: float) -> float:
    """Cosine annealing applied to the per-round LR (no warm-up)."""
    # round_idx_1based runs 1..n_rounds; treat round 1 as t=0 so first LR == lr_max.
    t = (round_idx_1based - 1) / max(n_rounds - 1, 1)
    return float(lr_max * 0.5 * (1.0 + math.cos(math.pi * t)))


# ---------------------------------------------------------------------------
# Client / loader construction
# ---------------------------------------------------------------------------
def build_federated_clients_from_bundle(
    bundle: TrainTestBundle,
    shards: list[ClientShard],
    batch_size: int,
    lambda_fault: float,
    seed: int,
) -> tuple[list[FederatedClient], DataLoader]:
    """Prepare one :class:`FederatedClient` per shard plus a shared test DataLoader.

    The test loader uses a normalizer fit on the **pooled training set in the
    bundle** so the global model is evaluated against a single, consistent set
    of test inputs across rounds — the same protocol P3/P4/P5 use.

    The per-client training loaders use **each client's own normalizer** fit
    on its own slice — the realistic FL behaviour.
    """
    if not shards:
        raise ValueError("shards must be non-empty.")
    seed_everything(seed)

    # Centralized normalizer used only for the test loader.
    central_norm = Normalizer.fit(bundle.train_df, bundle.feature_cols)
    test_arrays = make_test_windows(
        central_norm.transform(bundle.test_raw_df),
        bundle.test_rul,
        bundle.feature_cols,
        bundle.window_size,
        bundle.rul_cap,
        bundle.fault_threshold,
    )
    test_loader = DataLoader(
        CMAPSSWindowDataset(test_arrays), batch_size=batch_size, shuffle=False, num_workers=0,
    )

    clients: list[FederatedClient] = []
    for shard in shards:
        client_df = slice_for_client(bundle.train_df, shard)
        client_norm = Normalizer.fit(client_df, bundle.feature_cols)
        arrays = make_training_windows(
            client_norm.transform(client_df),
            bundle.feature_cols, bundle.window_size, bundle.stride,
        )
        loader = DataLoader(
            CMAPSSWindowDataset(arrays), batch_size=batch_size, shuffle=True, num_workers=0,
        )
        # Re-seed before model construction so every client starts identical.
        seed_everything(seed)
        model = MultiTaskCNN(
            MultiTaskCNNConfig(n_features=bundle.n_features, window_size=bundle.window_size)
        )
        n_pos = int(arrays.y_fault.sum())
        n_neg = int(arrays.y_fault.shape[0] - n_pos)
        pos_weight = float(n_neg) / float(max(n_pos, 1))
        loss_fn = MultiTaskLoss(lambda_fault=lambda_fault, fault_pos_weight=pos_weight)
        clients.append(
            FederatedClient(
                client_id=shard.client_id,
                model=model,
                train_loader=loader,
                loss_fn=loss_fn,
                n_samples=arrays.n_samples,
            )
        )
    return clients, test_loader


def build_federated_clients(
    config: CMAPSSConfig,
    n_clients: int,
    batch_size: int,
    lambda_fault: float,
    seed: int,
) -> tuple[list[FederatedClient], DataLoader]:
    """Phase-5 wrapper: build bundle from ``config`` + lifetime-stratified partition."""
    bundle = bundle_from_config(config)
    shards = partition_by_lifetime(bundle.train_df, n_clients=n_clients, seed=seed)
    return build_federated_clients_from_bundle(bundle, shards, batch_size, lambda_fault, seed)


# ---------------------------------------------------------------------------
# Bundle-based FedAvg
# ---------------------------------------------------------------------------
def run_fedavg_from_bundle(
    bundle: TrainTestBundle,
    shards: list[ClientShard],
    *,
    n_rounds: int = 50,
    local_epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_fault: float = 0.5,
    use_cosine_schedule: bool = True,
    seed: int = 42,
    log_every: int = 1,
) -> FederatedHistory:
    """Run a full FedAvg simulation against ``bundle``'s data and ``shards``."""
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if local_epochs < 1:
        raise ValueError(f"local_epochs must be >= 1, got {local_epochs}.")
    if not shards:
        raise ValueError("shards must be non-empty.")

    clients, test_loader = build_federated_clients_from_bundle(
        bundle, shards, batch_size, lambda_fault, seed,
    )
    # The server starts with the same initial weights every client has — the
    # canonical FedAvg cold start (round 0 global model = identical init).
    initial_state = {
        k: v.detach().clone() for k, v in clients[0].model.state_dict().items()
    }
    server = FedAvgServer(initial_state)

    # An "evaluation model" lets us read the post-aggregation global state
    # without disturbing any client model.
    eval_model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=bundle.n_features, window_size=bundle.window_size)
    )

    history: list[RoundRecord] = []
    per_round_client_losses: dict[str, list[float]] = {c.client_id: [] for c in clients}
    best_round = 0
    best_nasa = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    best_rul_metrics: RegressionMetrics | None = None
    best_fault_metrics: ClassificationMetrics | None = None
    final_predictions: dict[str, np.ndarray] = {}
    final_rul_metrics: RegressionMetrics | None = None
    final_fault_metrics: ClassificationMetrics | None = None

    total_start = time.perf_counter()
    for r in range(1, n_rounds + 1):
        round_start = time.perf_counter()
        current_lr = (
            _cosine_lr(r, n_rounds, lr) if use_cosine_schedule else lr
        )

        # 1. Broadcast global state to every client.
        global_state = server.clone_global_state()
        for client in clients:
            client.set_global_state(global_state)

        # 2. Each client trains locally for ``local_epochs``.
        round_total = 0.0
        round_rul = 0.0
        round_fault = 0.0
        updates = []
        for client in clients:
            ct, cr, cf = client.local_train(
                local_epochs=local_epochs, lr=current_lr, weight_decay=weight_decay
            )
            per_round_client_losses[client.client_id].append(ct)
            round_total += ct
            round_rul += cr
            round_fault += cf
            updates.append(client.package_update())

        mean_total = round_total / len(clients)
        mean_rul = round_rul / len(clients)
        mean_fault = round_fault / len(clients)

        # 3. Aggregate (FedAvg sample-count-weighted mean).
        new_global_state = server.aggregate(updates)

        # 4. Evaluate the new global model on the common test set.
        eval_model.load_state_dict(new_global_state)
        rul_m, fault_m, y_rul_true, y_rul_pred, y_fault_true, y_fault_score = evaluate(
            eval_model, test_loader
        )

        round_seconds = time.perf_counter() - round_start
        record = RoundRecord(
            round=r,
            lr=float(current_lr),
            mean_client_loss_total=float(mean_total),
            mean_client_loss_rul=float(mean_rul),
            mean_client_loss_fault=float(mean_fault),
            global_test_rmse=rul_m.rmse,
            global_test_mae=rul_m.mae,
            global_test_nasa_score=rul_m.nasa_score,
            global_test_auprc=fault_m.auprc,
            global_test_f1=fault_m.f1,
            global_test_precision=fault_m.precision,
            global_test_recall=fault_m.recall,
            round_seconds=float(round_seconds),
        )
        history.append(record)

        if rul_m.nasa_score < best_nasa:
            best_nasa = rul_m.nasa_score
            best_round = r
            best_state = {k: v.detach().clone() for k, v in new_global_state.items()}
            best_rul_metrics = rul_m
            best_fault_metrics = fault_m
        if r == n_rounds:
            final_rul_metrics = rul_m
            final_fault_metrics = fault_m
            final_predictions = {
                "y_rul_true": y_rul_true,
                "y_rul_pred": y_rul_pred,
                "y_fault_true": y_fault_true,
                "y_fault_score": y_fault_score,
            }

        if r % log_every == 0:
            print(
                f"round {r:>3}/{n_rounds}  "
                f"lr={current_lr:.2e}  "
                f"loss={mean_total:.4f}  "
                f"RMSE={rul_m.rmse:.2f}  "
                f"NASA={rul_m.nasa_score:.0f}  "
                f"AUPRC={fault_m.auprc:.3f}  "
                f"F1={fault_m.f1:.3f}  "
                f"({round_seconds:.1f}s)"
            )

    total_seconds = time.perf_counter() - total_start
    if final_rul_metrics is None or final_fault_metrics is None:
        raise RuntimeError("Simulation finished without producing final metrics.")
    if best_rul_metrics is None or best_fault_metrics is None:
        raise RuntimeError("Simulation finished without selecting a best round.")

    return FederatedHistory(
        rounds=history,
        best_round=best_round,
        best_state_dict=best_state,
        total_seconds=float(total_seconds),
        final_test_rul=final_rul_metrics,
        final_test_fault=final_fault_metrics,
        best_test_rul=best_rul_metrics,
        best_test_fault=best_fault_metrics,
        client_ids=[c.client_id for c in clients],
        per_round_client_losses=per_round_client_losses,
        final_predictions=final_predictions,
    )


# ---------------------------------------------------------------------------
# Config-based wrapper (Phase 5 default)
# ---------------------------------------------------------------------------
def run_fedavg(
    config: CMAPSSConfig,
    *,
    n_clients: int = 4,
    n_rounds: int = 50,
    local_epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_fault: float = 0.5,
    use_cosine_schedule: bool = True,
    seed: int = 42,
    log_every: int = 1,
) -> FederatedHistory:
    """Phase-5 wrapper: build bundle + lifetime-stratified partition from ``config``, then run."""
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}.")
    bundle = bundle_from_config(config)
    shards = partition_by_lifetime(bundle.train_df, n_clients=n_clients, seed=seed)
    return run_fedavg_from_bundle(
        bundle, shards,
        n_rounds=n_rounds, local_epochs=local_epochs,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        lambda_fault=lambda_fault, use_cosine_schedule=use_cosine_schedule,
        seed=seed, log_every=log_every,
    )
