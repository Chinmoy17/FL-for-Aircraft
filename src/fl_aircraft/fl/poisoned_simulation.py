"""RQ7 — Federated simulation with one or more malicious clients.

Drop-in variant of :func:`run_fedavg_from_bundle` that supports replacing
specific honest clients with attackers. The protocol the server sees is
identical to vanilla FedAvg: same broadcast → local train → collect
updates → aggregate cycle. Only the local-train + package-update behavior
of the wrapped client(s) differs.

The aggregator is fully pluggable. Pass any of:

  - :func:`fl_aircraft.fl.server.fedavg_aggregate` (vanilla, no defense)
  - :func:`fl_aircraft.fl.robust_aggregators.make_trimmed_mean_aggregator()`
  - :func:`fl_aircraft.fl.robust_aggregators.make_median_aggregator()`
  - :func:`fl_aircraft.fl.robust_aggregators.make_krum_aggregator()`

A separate ``per_client_delta_norm`` field on each round tracks the L2
norm of each client's (local - global) delta — this is the diagnostic
that makes gradient-scaling attacks visible at a glance (the attacker's
delta norm will be ~10× the honest clients').
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import (
    CMAPSSWindowDataset,
    ClientShard,
    Normalizer,
    TrainTestBundle,
    make_test_windows,
)
from ..eval import (
    ClassificationMetrics,
    RegressionMetrics,
    compute_classification_metrics,
    compute_regression_metrics,
)
from ..models import MultiTaskCNN, MultiTaskCNNConfig
from ..train.centralized import evaluate
from .client import FederatedClient
from .poisoning import MaliciousClient
from .server import ClientUpdate, FedAvgServer, fedavg_aggregate
from .simulation import (
    build_federated_clients_from_bundle,
    _cosine_lr,
)


# ---------------------------------------------------------------------------
# History container — same shape as FederatedHistory plus per-client delta norms
# ---------------------------------------------------------------------------
@dataclass
class PoisonedRoundRecord:
    """One row of per-round history with attack-diagnostic fields."""

    round: int
    lr: float
    mean_client_loss_total: float
    mean_client_loss_rul: float
    mean_client_loss_fault: float
    global_test_rmse: float
    global_test_nasa_score: float
    global_test_auprc: float
    global_test_f1: float
    # Per-client L2 norm of (state_sent_to_server - global_at_round_start).
    # Honest clients' deltas have similar small norms; gradient-scaling
    # attackers have a large negative-direction delta with |scale|× the
    # norm of an honest delta. This field makes the attack visible.
    per_client_delta_norm: dict[str, float]
    round_seconds: float

    def as_dict(self) -> dict:
        return {
            "round": self.round,
            "lr": self.lr,
            "mean_client_loss_total": self.mean_client_loss_total,
            "mean_client_loss_rul": self.mean_client_loss_rul,
            "mean_client_loss_fault": self.mean_client_loss_fault,
            "global_test_rmse": self.global_test_rmse,
            "global_test_nasa_score": self.global_test_nasa_score,
            "global_test_auprc": self.global_test_auprc,
            "global_test_f1": self.global_test_f1,
            "round_seconds": self.round_seconds,
            **{
                f"delta_norm_{k}": v
                for k, v in self.per_client_delta_norm.items()
            },
        }


@dataclass
class PoisonedHistory:
    """Full output of :func:`run_fedavg_with_attackers`."""

    rounds: list[PoisonedRoundRecord]
    best_round: int
    best_state_dict: dict[str, torch.Tensor]
    total_seconds: float
    final_test_rul: RegressionMetrics
    final_test_fault: ClassificationMetrics
    best_test_rul: RegressionMetrics
    best_test_fault: ClassificationMetrics
    client_ids: list[str]
    aggregator_name: str
    attacker_ids: list[str]
    attacker_kind: str

    def __len__(self) -> int:
        return len(self.rounds)


# ---------------------------------------------------------------------------
# Diagnostic helper: L2 norm of (state - reference) across all parameters
# ---------------------------------------------------------------------------
def _delta_norm(
    state: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> float:
    """L2 norm of the concatenated (state - reference) vector across all keys."""
    total_sq = 0.0
    for k, v in state.items():
        d = v.to(torch.float64) - reference[k].to(torch.float64)
        total_sq += float((d * d).sum().item())
    return math.sqrt(total_sq)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_fedavg_with_attackers(
    bundle: TrainTestBundle,
    shards: list[ClientShard],
    *,
    attacker_factory=None,
    attacker_client_ids: Sequence[str] = (),
    attacker_kind: str = "none",
    aggregator=fedavg_aggregate,
    aggregator_name: str = "fedavg",
    n_rounds: int = 50,
    local_epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_fault: float = 0.5,
    use_cosine_schedule: bool = True,
    seed: int = 42,
    log_every: int = 5,
) -> PoisonedHistory:
    """Run a FedAvg simulation where some clients are malicious.

    Args:
        attacker_factory: callable ``FederatedClient -> MaliciousClient`` used
            to wrap each honest client whose id is in ``attacker_client_ids``.
            Pass ``None`` to disable attacks (clean baseline).
        attacker_client_ids: ids of the clients to wrap with the attacker.
            Should reference ``ClientShard.client_id`` values.
        attacker_kind: short string label used in metrics.json and plots
            (e.g. "label_flip", "grad_scale_x10").
        aggregator: server-side aggregation function. Defaults to vanilla
            FedAvg. Pass one of the robust aggregator factories from
            :mod:`fl_aircraft.fl.robust_aggregators`.
        aggregator_name: short string label for plots / metrics.
    """
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if local_epochs < 1:
        raise ValueError(f"local_epochs must be >= 1, got {local_epochs}.")
    if not shards:
        raise ValueError("shards must be non-empty.")
    if attacker_factory is None and attacker_client_ids:
        raise ValueError(
            "attacker_client_ids was non-empty but attacker_factory is None."
        )
    if attacker_factory is not None and not attacker_client_ids:
        raise ValueError(
            "attacker_factory was set but attacker_client_ids is empty."
        )

    # 1. Build honest clients first (reusing the existing helper).
    honest_clients, test_loader = build_federated_clients_from_bundle(
        bundle, shards, batch_size, lambda_fault, seed,
    )

    # 2. Wrap the ones the caller marked as malicious.
    attacker_id_set = set(attacker_client_ids)
    clients: list[FederatedClient | MaliciousClient] = []
    for honest in honest_clients:
        if honest.client_id in attacker_id_set:
            if attacker_factory is None:
                # Shouldn't happen given guards above, but be explicit.
                raise RuntimeError("attacker_factory is None")
            clients.append(attacker_factory(honest))
        else:
            clients.append(honest)
    actual_attacker_ids = [c.client_id for c in clients if isinstance(c, MaliciousClient)]
    if attacker_id_set and set(actual_attacker_ids) != attacker_id_set:
        missing = attacker_id_set - set(actual_attacker_ids)
        raise ValueError(
            f"attacker_client_ids {sorted(missing)} not found in shards."
        )

    # 3. Server starts with the same init every client has.
    initial_state = {
        k: v.detach().clone() for k, v in honest_clients[0].model.state_dict().items()
    }
    server = FedAvgServer(initial_state, aggregator=aggregator)

    eval_model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=bundle.n_features, window_size=bundle.window_size)
    )

    history: list[PoisonedRoundRecord] = []
    best_round = 0
    best_nasa = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    best_rul_metrics: RegressionMetrics | None = None
    best_fault_metrics: ClassificationMetrics | None = None
    final_rul_metrics: RegressionMetrics | None = None
    final_fault_metrics: ClassificationMetrics | None = None

    total_start = time.perf_counter()
    for r in range(1, n_rounds + 1):
        round_start = time.perf_counter()
        current_lr = (
            _cosine_lr(r, n_rounds, lr) if use_cosine_schedule else lr
        )

        # Broadcast.
        global_state = server.clone_global_state()
        for client in clients:
            client.set_global_state(global_state)

        # Local training (each client runs its own local_train — attackers
        # may corrupt labels here, or do nothing extra and corrupt later).
        round_total = 0.0
        round_rul = 0.0
        round_fault = 0.0
        updates: list[ClientUpdate] = []
        delta_norms: dict[str, float] = {}
        for client in clients:
            ct, cr, cf = client.local_train(
                local_epochs=local_epochs, lr=current_lr, weight_decay=weight_decay,
            )
            round_total += ct
            round_rul += cr
            round_fault += cf
            update = client.package_update()
            delta_norms[client.client_id] = _delta_norm(
                update.state_dict, global_state,
            )
            updates.append(update)

        n = len(clients)
        mean_total = round_total / n
        mean_rul = round_rul / n
        mean_fault = round_fault / n

        # Aggregate.
        try:
            new_global_state = server.aggregate(updates)
        except Exception as exc:
            raise RuntimeError(
                f"Aggregation failed in round {r} with aggregator "
                f"{aggregator_name!r}: {exc}"
            ) from exc

        # Evaluate.
        eval_model.load_state_dict(new_global_state)
        rul_m, fault_m, *_ = evaluate(eval_model, test_loader)

        round_seconds = time.perf_counter() - round_start
        record = PoisonedRoundRecord(
            round=r, lr=float(current_lr),
            mean_client_loss_total=float(mean_total),
            mean_client_loss_rul=float(mean_rul),
            mean_client_loss_fault=float(mean_fault),
            global_test_rmse=float(rul_m.rmse),
            global_test_nasa_score=float(rul_m.nasa_score),
            global_test_auprc=float(fault_m.auprc),
            global_test_f1=float(fault_m.f1),
            per_client_delta_norm=delta_norms,
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

        if r % log_every == 0 or r == 1 or r == n_rounds:
            attacker_norm_strs = " ".join(
                f"{cid[:8]}={v:.2f}" for cid, v in delta_norms.items()
            )
            print(
                f"round {r:>3}/{n_rounds}  lr={current_lr:.2e}  "
                f"loss={mean_total:.3f}  RMSE={rul_m.rmse:.2f}  "
                f"NASA={rul_m.nasa_score:.0f}  F1={fault_m.f1:.3f}  "
                f"delta_norms[{attacker_norm_strs}]  ({round_seconds:.1f}s)"
            )

    total_seconds = time.perf_counter() - total_start
    if final_rul_metrics is None or final_fault_metrics is None:
        raise RuntimeError("Simulation finished without final metrics.")
    if best_rul_metrics is None or best_fault_metrics is None:
        raise RuntimeError("Simulation finished without best metrics.")

    return PoisonedHistory(
        rounds=history,
        best_round=best_round,
        best_state_dict=best_state,
        total_seconds=float(total_seconds),
        final_test_rul=final_rul_metrics,
        final_test_fault=final_fault_metrics,
        best_test_rul=best_rul_metrics,
        best_test_fault=best_fault_metrics,
        client_ids=[c.client_id for c in clients],
        aggregator_name=aggregator_name,
        attacker_ids=list(actual_attacker_ids),
        attacker_kind=attacker_kind,
    )


__all__ = [
    "PoisonedHistory",
    "PoisonedRoundRecord",
    "run_fedavg_with_attackers",
]
