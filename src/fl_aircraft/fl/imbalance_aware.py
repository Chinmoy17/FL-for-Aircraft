"""Imbalance-aware FedAvg simulation for RQ2.

Mirrors :func:`fl_aircraft.fl.simulation.run_fedavg_from_bundle` but adds three
features the RQ2 schemes need:

1. **Optional held-out validation slice per client.** Each client carves a
   configurable fraction of its own engines into a non-training validation
   loader, used by Scheme B (validation-F1 aggregator) to score the *current
   global model* before that round's local training.
2. **Per-round signal collection.** Before each aggregation the simulation
   collects `n_fault_positives` (constant), the previous round's local loss,
   and the current global model's per-client validation F1, then exposes
   them to the active aggregator via closures.
3. **Pluggable aggregator factory.** The caller passes a string in
   ``{"fedavg", "fault_count", "validation_f1", "inverse_loss"}`` and the
   simulation wires up the right :mod:`fl_aircraft.fl.aggregators` factory.

This module is **purely additive**: nothing in the existing
``run_fedavg_from_bundle`` / ``run_fedavg`` paths changes, so all P5 / P6
results and tests remain reproducible bit-for-bit.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

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
from ..eval import ClassificationMetrics, RegressionMetrics
from ..models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss
from ..train.centralized import evaluate
from ..utils import seed_everything
from .aggregators import (
    make_fault_count_aggregator,
    make_inverse_loss_aggregator,
    make_validation_signal_aggregator,
)
from .client import FederatedClient
from .server import FedAvgServer
from .simulation import FederatedHistory, RoundRecord, _cosine_lr


AggregatorName = Literal["fedavg", "fault_count", "validation_f1", "inverse_loss"]


# ---------------------------------------------------------------------------
# Client construction with held-out validation slice
# ---------------------------------------------------------------------------
def _split_engines_for_val(
    unit_ids: tuple[int, ...], val_fraction: float, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    """Split a shard's engine ids into (train, val) lists."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1); got {val_fraction}")
    if val_fraction == 0.0:
        return list(unit_ids), []
    shuffled = list(unit_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    # Ensure we keep at least one engine for training.
    n_val = min(n_val, len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def build_imbalance_aware_clients(
    bundle: TrainTestBundle,
    shards: list[ClientShard],
    batch_size: int,
    lambda_fault: float,
    seed: int,
    *,
    val_fraction: float = 0.0,
) -> tuple[list[FederatedClient], DataLoader]:
    """Like :func:`build_federated_clients_from_bundle` but with optional val slice.

    When ``val_fraction > 0``, each client carves that fraction of its own
    engines into a validation loader attached as ``client.val_loader``.
    The validation slice uses the **client's own normalizer**, fit only on
    the client's training engines (no leak from validation).
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

    rng = np.random.default_rng(seed)
    clients: list[FederatedClient] = []
    for shard in shards:
        train_ids, val_ids = _split_engines_for_val(shard.unit_ids, val_fraction, rng)

        # Build the client's full slice, then derive train/val subsets.
        client_df = slice_for_client(bundle.train_df, shard)
        train_df = client_df.loc[client_df["unit_id"].isin(train_ids)].copy()

        # Fit the normalizer on *training engines only* — no validation leak.
        client_norm = Normalizer.fit(train_df, bundle.feature_cols)
        train_arrays = make_training_windows(
            client_norm.transform(train_df),
            bundle.feature_cols, bundle.window_size, bundle.stride,
        )
        train_loader = DataLoader(
            CMAPSSWindowDataset(train_arrays), batch_size=batch_size, shuffle=True, num_workers=0,
        )

        val_loader: Optional[DataLoader] = None
        if val_ids:
            val_df = client_df.loc[client_df["unit_id"].isin(val_ids)].copy()
            val_arrays = make_training_windows(
                client_norm.transform(val_df),
                bundle.feature_cols, bundle.window_size, bundle.stride,
            )
            val_loader = DataLoader(
                CMAPSSWindowDataset(val_arrays), batch_size=batch_size, shuffle=False, num_workers=0,
            )

        # Re-seed before model construction so every client starts identical.
        seed_everything(seed)
        model = MultiTaskCNN(
            MultiTaskCNNConfig(n_features=bundle.n_features, window_size=bundle.window_size)
        )
        n_pos = int(train_arrays.y_fault.sum())
        n_neg = int(train_arrays.y_fault.shape[0] - n_pos)
        pos_weight = float(n_neg) / float(max(n_pos, 1))
        loss_fn = MultiTaskLoss(lambda_fault=lambda_fault, fault_pos_weight=pos_weight)
        clients.append(
            FederatedClient(
                client_id=shard.client_id,
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                n_samples=train_arrays.n_samples,
                n_fault_positives=n_pos,
                val_loader=val_loader,
            )
        )
    return clients, test_loader


# ---------------------------------------------------------------------------
# Imbalance-aware history extension
# ---------------------------------------------------------------------------
@dataclass
class ImbalanceAwareHistory(FederatedHistory):
    """``FederatedHistory`` + per-round per-client aggregation weights and val signals."""

    aggregator_name: str = "fedavg"
    aggregation_weights: dict[str, list[float]] = field(default_factory=dict)
    per_round_client_val_auprc: dict[str, list[float]] = field(default_factory=dict)
    per_round_client_val_f1: dict[str, list[float]] = field(default_factory=dict)
    n_fault_positives: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The main loop
# ---------------------------------------------------------------------------
def run_fedavg_imbalance_aware(
    bundle: TrainTestBundle,
    shards: list[ClientShard],
    *,
    aggregator: AggregatorName = "fedavg",
    val_fraction: float = 0.2,
    softmax_temperature: float = 0.5,
    weight_floor: float = 0.05,
    n_rounds: int = 50,
    local_epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_fault: float = 0.5,
    use_cosine_schedule: bool = True,
    seed: int = 42,
    log_every: int = 1,
) -> ImbalanceAwareHistory:
    """Run a FedAvg simulation with a pluggable aggregator (RQ2).

    Args:
        bundle / shards: same as :func:`run_fedavg_from_bundle`.
        aggregator: one of:
            - ``"fedavg"`` — canonical sample-count weighting (baseline).
            - ``"fault_count"`` — Scheme A: weight clients by # fault windows.
            - ``"validation_f1"`` — Scheme B: weight clients by softmax of
              their held-out F1 score. Requires ``val_fraction > 0``.
            - ``"inverse_loss"`` — Scheme C: weight clients by 1 / loss.
        val_fraction: Engines per client to hold out for validation. Only
            used when ``aggregator == "validation_f1"``.
        softmax_temperature: Softmax temperature for Scheme B. Lower =>
            more aggressive reweighting toward the best-scoring client.
        weight_floor: Minimum per-client weight that Scheme B guarantees,
            preventing the global model from completely ignoring any client.
        Remaining args mirror :func:`run_fedavg_from_bundle`.

    Returns:
        :class:`ImbalanceAwareHistory` with all the same fields as the
        baseline :class:`FederatedHistory` plus per-round aggregation
        weights, validation signals, and the static fault counts.
    """
    if aggregator not in ("fedavg", "fault_count", "validation_f1", "inverse_loss"):
        raise ValueError(f"Unknown aggregator {aggregator!r}.")
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if local_epochs < 1:
        raise ValueError(f"local_epochs must be >= 1, got {local_epochs}.")
    if not shards:
        raise ValueError("shards must be non-empty.")

    effective_val_fraction = val_fraction if aggregator == "validation_f1" else 0.0
    clients, test_loader = build_imbalance_aware_clients(
        bundle, shards, batch_size, lambda_fault, seed,
        val_fraction=effective_val_fraction,
    )

    # ------- mutable shared state used by the aggregator closures -------
    fault_counts = {c.client_id: c.n_fault_positives for c in clients}
    last_losses: dict[str, float] = {c.client_id: 1.0 for c in clients}
    last_val_signals: dict[str, float] = {c.client_id: 0.0 for c in clients}

    # ------- build the aggregator -------
    if aggregator == "fedavg":
        from .server import fedavg_aggregate as agg_fn
    elif aggregator == "fault_count":
        agg_fn = make_fault_count_aggregator(fault_counts)
    elif aggregator == "validation_f1":
        agg_fn = make_validation_signal_aggregator(
            lambda: dict(last_val_signals),
            temperature=softmax_temperature, floor=weight_floor, invert=False,
        )
    elif aggregator == "inverse_loss":
        agg_fn = make_inverse_loss_aggregator(lambda: dict(last_losses))
    else:  # pragma: no cover - guarded above
        raise AssertionError(aggregator)

    # ------- server cold start -------
    initial_state = {
        k: v.detach().clone() for k, v in clients[0].model.state_dict().items()
    }
    server = FedAvgServer(initial_state, aggregator=agg_fn)
    eval_model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=bundle.n_features, window_size=bundle.window_size)
    )

    history: list[RoundRecord] = []
    per_round_client_losses: dict[str, list[float]] = {c.client_id: [] for c in clients}
    aggregation_weights: dict[str, list[float]] = {c.client_id: [] for c in clients}
    per_round_val_auprc: dict[str, list[float]] = {c.client_id: [] for c in clients}
    per_round_val_f1: dict[str, list[float]] = {c.client_id: [] for c in clients}

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
        current_lr = _cosine_lr(r, n_rounds, lr) if use_cosine_schedule else lr

        # 1. Broadcast global state to every client.
        global_state = server.clone_global_state()
        for client in clients:
            client.set_global_state(global_state)

        # 2a. Optionally evaluate the (just-broadcast) global model on each
        #     client's held-out slice to feed Scheme B.
        if aggregator == "validation_f1":
            for client in clients:
                auprc, f1 = client.validate()
                per_round_val_auprc[client.client_id].append(auprc)
                per_round_val_f1[client.client_id].append(f1)
                last_val_signals[client.client_id] = f1
        else:
            for client in clients:
                per_round_val_auprc[client.client_id].append(float("nan"))
                per_round_val_f1[client.client_id].append(float("nan"))

        # 2b. Each client trains locally for ``local_epochs``.
        round_total = 0.0
        round_rul = 0.0
        round_fault = 0.0
        updates = []
        for client in clients:
            ct, cr, cf = client.local_train(
                local_epochs=local_epochs, lr=current_lr, weight_decay=weight_decay
            )
            per_round_client_losses[client.client_id].append(ct)
            last_losses[client.client_id] = ct
            round_total += ct
            round_rul += cr
            round_fault += cf
            updates.append(client.package_update())

        mean_total = round_total / len(clients)
        mean_rul = round_rul / len(clients)
        mean_fault = round_fault / len(clients)

        # 3. Aggregate. Compute the weights the aggregator will use *for
        #    logging* (we replay the same weight-computation here so the
        #    logged numbers match what's actually applied).
        weights_for_log = _compute_weights_for_logging(
            aggregator, updates, fault_counts, last_losses, last_val_signals,
            softmax_temperature, weight_floor,
        )
        for cid, w in weights_for_log.items():
            aggregation_weights[cid].append(w)
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
            weight_str = " ".join(f"{c.client_id[-1]}:{weights_for_log[c.client_id]:.2f}" for c in clients)
            print(
                f"[{aggregator:>13}] round {r:>3}/{n_rounds}  "
                f"lr={current_lr:.2e}  loss={mean_total:.3f}  "
                f"RMSE={rul_m.rmse:.2f}  NASA={rul_m.nasa_score:.0f}  "
                f"AUPRC={fault_m.auprc:.3f}  F1={fault_m.f1:.3f}  "
                f"w=[{weight_str}]  ({round_seconds:.1f}s)"
            )

    total_seconds = time.perf_counter() - total_start
    if final_rul_metrics is None or final_fault_metrics is None:
        raise RuntimeError("Simulation finished without producing final metrics.")
    if best_rul_metrics is None or best_fault_metrics is None:
        raise RuntimeError("Simulation finished without selecting a best round.")

    return ImbalanceAwareHistory(
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
        aggregator_name=aggregator,
        aggregation_weights=aggregation_weights,
        per_round_client_val_auprc=per_round_val_auprc,
        per_round_client_val_f1=per_round_val_f1,
        n_fault_positives=dict(fault_counts),
    )


def _compute_weights_for_logging(
    aggregator: AggregatorName,
    updates,
    fault_counts: dict[str, int],
    last_losses: dict[str, float],
    last_val_signals: dict[str, float],
    softmax_temperature: float,
    weight_floor: float,
) -> dict[str, float]:
    """Replay the weight computation so the logs match what the aggregator applied.

    This duplication is justified: the aggregator's actual application is
    inside ``FedAvgServer.aggregate`` and returns only the merged state-dict
    (not the weights). For monitoring / plotting we need the weights too,
    and the cost of recomputing four scalars is negligible.
    """
    ids = [u.client_id for u in updates]
    if aggregator == "fedavg":
        total = sum(u.n_samples for u in updates)
        return {u.client_id: u.n_samples / total for u in updates}
    if aggregator == "fault_count":
        total = sum(fault_counts[cid] for cid in ids)
        return {cid: fault_counts[cid] / total for cid in ids}
    if aggregator == "inverse_loss":
        raw = np.array([1.0 / (last_losses[cid] + 1e-6) for cid in ids])
        raw = raw / raw.sum()
        return {cid: float(w) for cid, w in zip(ids, raw)}
    if aggregator == "validation_f1":
        raw = np.array([last_val_signals[cid] for cid in ids], dtype=np.float64)
        scaled = raw / softmax_temperature
        scaled = scaled - scaled.max()
        exps = np.exp(scaled)
        softmax = exps / exps.sum()
        n = len(ids)
        floored = weight_floor + (1.0 - n * weight_floor) * softmax
        floored = np.maximum(floored, 0.0)
        floored = floored / floored.sum()
        return {cid: float(w) for cid, w in zip(ids, floored)}
    raise AssertionError(aggregator)  # pragma: no cover
