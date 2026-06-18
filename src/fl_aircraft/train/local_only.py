"""Local-only baseline (Phase 4): one model per client, no sharing.

Each simulated airline trains its own multi-task CNN on **only its own
engines'** data. No weights ever leave a client. This run produces the
**lower-bound** every federated approach must beat for the federation to be
worthwhile.

Reuses :func:`fl_aircraft.train.centralized.train_centralized` verbatim as the
per-client training routine \u2014 the only variable is the data each client sees.

Evaluation note
---------------
Every client is evaluated on the **same full FD001 test set** rather than a
per-client test split, for three reasons:

1. C-MAPSS publishes a single 100-engine test set with ground-truth RUL; it is
   not partitioned by client.
2. A 25-engine per-client test set would be too small for stable AUPRC/F1
   measurements.
3. Aggregating per-client numbers on the same test set gives an apples-to-
   apples comparison with the P3 centralized baseline.

This decision is documented in ``results.md`` so reviewers see it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data import (
    CMAPSSConfig,
    CMAPSSWindowDataset,
    ClientShard,
    Normalizer,
    load_and_label_train,
    load_raw,
    load_test_rul,
    make_test_windows,
    make_training_windows,
    partition_by_lifetime,
    slice_for_client,
)
from ..eval import ClassificationMetrics, RegressionMetrics
from ..models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss
from ..utils import seed_everything
from .centralized import TrainingHistory, train_centralized


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------
@dataclass
class ClientRun:
    """One client's complete training outcome."""

    client_id: str
    shard: ClientShard
    n_train_windows: int
    pos_weight: float
    history: TrainingHistory

    @property
    def best_rul(self) -> RegressionMetrics:
        return self.history.best_test_rul

    @property
    def best_fault(self) -> ClassificationMetrics:
        return self.history.best_test_fault

    @property
    def final_rul(self) -> RegressionMetrics:
        return self.history.final_test_rul

    @property
    def final_fault(self) -> ClassificationMetrics:
        return self.history.final_test_fault


@dataclass
class LocalOnlyResults:
    """Aggregate output of :func:`train_local_only_clients`."""

    clients: list[ClientRun]
    total_seconds: float
    config: CMAPSSConfig

    def per_client_rows(self, which: str = "best") -> list[dict[str, float]]:
        """One row per client; ``which`` is ``"best"`` or ``"final"``."""
        if which not in {"best", "final"}:
            raise ValueError(f"which must be 'best' or 'final', got {which!r}.")
        rows: list[dict[str, float]] = []
        for run in self.clients:
            rul = run.best_rul if which == "best" else run.final_rul
            fault = run.best_fault if which == "best" else run.final_fault
            rows.append(
                {
                    "client_id": run.client_id,
                    "n_engines": run.shard.n_engines,
                    "n_train_windows": run.n_train_windows,
                    "pos_weight": round(run.pos_weight, 3),
                    "best_epoch": run.history.best_epoch,
                    "rmse": round(rul.rmse, 4),
                    "mae": round(rul.mae, 4),
                    "nasa_score": round(rul.nasa_score, 4),
                    "auprc": round(fault.auprc, 4),
                    "f1": round(fault.f1, 4),
                    "precision": round(fault.precision, 4),
                    "recall": round(fault.recall, 4),
                    "train_seconds": round(run.history.total_seconds, 3),
                }
            )
        return rows

    def aggregate(self, which: str = "best") -> dict[str, float]:
        """Aggregate per-client metrics across the federation."""
        rows = self.per_client_rows(which)
        df = pd.DataFrame(rows)
        numeric = [
            "rmse", "mae", "nasa_score", "auprc", "f1", "precision", "recall",
        ]
        out: dict[str, float] = {"n_clients": len(rows)}
        for col in numeric:
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_std"] = float(df[col].std(ddof=0))
            out[f"{col}_min"] = float(df[col].min())
            out[f"{col}_max"] = float(df[col].max())
        return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def train_local_only_clients(
    config: CMAPSSConfig,
    *,
    n_clients: int = 4,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_fault: float = 0.5,
    use_cosine_schedule: bool = True,
    seed: int = 42,
    log_every: int = 10,
    client_log: bool = True,
) -> LocalOnlyResults:
    """Train one model per simulated client, in series. Returns aggregated results.

    Args:
        config: A :class:`CMAPSSConfig`.
        n_clients: Number of simulated airline clients.
        epochs / batch_size / lr / weight_decay / lambda_fault /
            use_cosine_schedule / seed: Hyperparameters \u2014 same as the P3
            centralized run by default so the comparison is fair.
        log_every: Per-client training loop log frequency.
        client_log: If True, print one-line per-client summary as each finishes.
    """
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}.")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")

    seed_everything(seed)

    # ------- Shared data preparation -------
    train_df = load_and_label_train(config)
    shards = partition_by_lifetime(train_df, n_clients=n_clients, seed=seed)

    # The test set is the same for every client (rationale in the module docstring).
    # The normalizer used to standardise the test set is each client's *own* fit \u2014
    # in a real FL deployment, each airline runs inference with the stats it knows.
    test_raw_df = load_raw(config.subset, "test", config.data_dir)
    test_rul = load_test_rul(config.subset, config.data_dir)

    import time
    total_start = time.perf_counter()
    client_runs: list[ClientRun] = []

    for shard in shards:
        client_df = slice_for_client(train_df, shard)
        client_norm = Normalizer.fit(client_df, config.feature_cols)
        client_arrays = make_training_windows(
            client_norm.transform(client_df),
            config.feature_cols,
            config.window_size,
            config.stride,
        )
        client_test = make_test_windows(
            client_norm.transform(test_raw_df),
            test_rul,
            config.feature_cols,
            config.window_size,
            config.rul_cap,
            config.fault_threshold,
        )
        train_loader = DataLoader(
            CMAPSSWindowDataset(client_arrays),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )
        test_loader = DataLoader(
            CMAPSSWindowDataset(client_test),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )

        # Re-seed before model construction so every client starts from the
        # *same* initial weights \u2014 differences in performance come from data,
        # not random init.
        seed_everything(seed)
        model = MultiTaskCNN(
            MultiTaskCNNConfig(n_features=config.n_features, window_size=config.window_size)
        )
        n_pos = int(client_arrays.y_fault.sum())
        n_neg = int(client_arrays.y_fault.shape[0] - n_pos)
        pos_weight = float(n_neg) / float(max(n_pos, 1))
        loss_fn = MultiTaskLoss(lambda_fault=lambda_fault, fault_pos_weight=pos_weight)

        if client_log:
            print(
                f"\n--- {shard.client_id} "
                f"(engines={shard.n_engines}, windows={client_arrays.n_samples}, "
                f"pos_weight={pos_weight:.2f}) ---"
            )
        history = train_centralized(
            model, train_loader, test_loader, loss_fn,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            use_cosine_schedule=use_cosine_schedule, log_every=log_every,
        )
        client_runs.append(
            ClientRun(
                client_id=shard.client_id,
                shard=shard,
                n_train_windows=client_arrays.n_samples,
                pos_weight=pos_weight,
                history=history,
            )
        )
        if client_log:
            print(
                f"  best epoch {history.best_epoch}/{epochs} "
                f"\u2192 RMSE={history.best_test_rul.rmse:.2f}  "
                f"NASA={history.best_test_rul.nasa_score:.0f}  "
                f"AUPRC={history.best_test_fault.auprc:.3f}  "
                f"F1={history.best_test_fault.f1:.3f}  "
                f"({history.total_seconds:.1f}s)"
            )

    return LocalOnlyResults(
        clients=client_runs,
        total_seconds=time.perf_counter() - total_start,
        config=config,
    )
