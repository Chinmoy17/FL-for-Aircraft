"""Federated client: one model + one local training step per round.

A :class:`FederatedClient` is the in-process stand-in for one airline. It owns
its own data, its own normalization statistics, and its own loss function
(with per-client ``pos_weight``). The server only ever sees the post-training
state-dict and the sample count.

Optimizer state is **reset every round** (canonical FedAvg from
McMahan et al., 2017). Carrying Adam moments across rounds is tempting but
problematic — the moments accumulated against the previous round's weights
become misaligned the moment the server replaces those weights with the
aggregated average. Resetting per round is the safer default; we'll
revisit if a future RQ benefits from carry-over (FedAvgM, FedAdam).

Two optional fields support the RQ2 imbalance-aware aggregators:

- ``n_fault_positives`` — used by the fault-count aggregator (Scheme A).
- ``val_loader`` — held-out slice used by the validation-F1 aggregator
  (Scheme B). ``validate()`` evaluates the *current* local model on this
  loader and returns AUPRC + F1; the simulation loop typically calls it
  *before* local training so the F1 reflects the global model's quality on
  this client's data.

FedProx support (Li et al., MLSys 2020) is built in via the ``mu`` kwarg on
:meth:`FederatedClient.local_train`. When ``mu > 0`` the local loss is
augmented with the proximal term ``(mu/2) * ||W - W_global||^2`` that
penalises drift away from the round-start global weights. ``mu = 0.0`` is
the default and produces exactly vanilla FedAvg behaviour (no extra
compute), preserving bit-exact reproducibility of every earlier phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..eval import compute_classification_metrics
from ..models import MultiTaskCNN, MultiTaskLoss, RULPrediction
from .server import ClientUpdate


@dataclass
class FederatedClient:
    """Holds a client's model + data + loss; runs local epochs on demand.

    Attributes:
        client_id: Stable identifier (matches the partition's ``ClientShard.client_id``).
        model: A fresh :class:`MultiTaskCNN`. The instance is mutated in-place
            each round (state-dict overwritten, then locally trained).
        train_loader: DataLoader over this client's windows only.
        loss_fn: A :class:`MultiTaskLoss` with this client's ``pos_weight``.
        n_samples: Number of training windows owned by the client — used as
            the aggregation weight by canonical FedAvg.
        n_fault_positives: Optional. Number of fault-positive windows in this
            client's training data. Used by the fault-count aggregator
            (Scheme A). Pre-computed once at client construction.
        val_loader: Optional. DataLoader over a held-out slice of this
            client's engines, used by the validation-F1 aggregator (Scheme
            B). Must be ``None`` if validation-based aggregation is not used.
    """

    client_id: str
    model: MultiTaskCNN
    train_loader: DataLoader
    loss_fn: MultiTaskLoss
    n_samples: int
    n_fault_positives: int = 0
    val_loader: Optional[DataLoader] = None

    def set_global_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Overwrite the local model with the server's global weights."""
        self.model.load_state_dict(state_dict)

    def local_train(
        self,
        local_epochs: int,
        lr: float,
        weight_decay: float = 1e-4,
        mu: float = 0.0,
    ) -> tuple[float, float, float]:
        """Run ``local_epochs`` of local SGD; return mean per-epoch losses.

        A fresh Adam optimizer is constructed each call — see module docstring.

        When ``mu > 0``, the loss includes a **FedProx proximal term**
        (Li et al., MLSys 2020):

        .. math::
            L_{\\text{FedProx}}(W) = L_i(W) + \\frac{\\mu}{2} \\| W - W^{(t)}_{\\text{global}} \\|^2

        where :math:`W^{(t)}_{\\text{global}}` is captured at the start of the
        call — i.e. the weights the server just broadcast for this round.
        Larger ``mu`` produces a stiffer "rubber band" pulling local
        training back toward the global model, reducing client drift on
        Non-IID data at the cost of slower per-round progress.

        ``mu = 0.0`` (default) is exactly vanilla FedAvg behaviour — no
        extra term, no extra compute, fully backward compatible with the
        Phase 5 / Phase 6 / RQ2 experiments.

        Returns ``(total, rul, fault)`` averaged over the local epochs.
        The returned ``total`` is the task-loss only and **does not include**
        the proximal term, so the number is comparable across runs with
        different ``mu``.
        """
        if local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {local_epochs}.")
        if mu < 0:
            raise ValueError(f"mu must be >= 0, got {mu}.")

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # FedProx: snapshot the global weights at round start (i.e. *now*,
        # before any local SGD steps). Detach + clone so future model
        # updates do not back-propagate into this reference copy.
        global_snapshot: list[torch.Tensor] | None = None
        if mu > 0:
            global_snapshot = [
                p.detach().clone() for p in self.model.parameters()
            ]

        totals = [0.0, 0.0, 0.0]
        for _ in range(local_epochs):
            t, r, f = self._train_one_local_epoch(
                optimizer=optimizer, mu=mu, global_snapshot=global_snapshot,
            )
            totals[0] += t
            totals[1] += r
            totals[2] += f
        return (
            totals[0] / local_epochs,
            totals[1] / local_epochs,
            totals[2] / local_epochs,
        )

    def _train_one_local_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        mu: float,
        global_snapshot: list[torch.Tensor] | None,
    ) -> tuple[float, float, float]:
        """Single epoch over ``train_loader`` with optional FedProx penalty.

        Mirrors :func:`fl_aircraft.train.centralized.train_one_epoch` exactly
        when ``mu = 0`` and ``global_snapshot is None``. When ``mu > 0`` it
        adds the proximal term :math:`(\\mu/2)\\|W - W_{\\text{global}}\\|^2`
        to the gradient step but **excludes** that term from the reported
        losses (so the number is task-loss only and comparable across
        ``mu`` values).
        """
        if mu > 0 and global_snapshot is None:
            raise ValueError("global_snapshot is required when mu > 0.")
        self.model.train()
        running_total = 0.0
        running_rul = 0.0
        running_fault = 0.0
        n_batches = 0
        for x, y_rul, y_fault in self.train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = self.model(x)
            losses = self.loss_fn(pred, y_rul, y_fault)
            loss_for_backward = losses.total
            if mu > 0 and global_snapshot is not None:
                # Proximal term: (mu/2) * sum_l ||W_l - W_l^{global}||^2.
                # Each parameter participates; biases included by design
                # (matches Li et al. 2020 — they regularise all params).
                proximal = torch.zeros((), dtype=loss_for_backward.dtype)
                for p_local, p_global in zip(
                    self.model.parameters(), global_snapshot
                ):
                    proximal = proximal + ((p_local - p_global) ** 2).sum()
                loss_for_backward = loss_for_backward + 0.5 * mu * proximal
            loss_for_backward.backward()
            optimizer.step()
            # Report task losses only (excludes proximal — comparable across mu).
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

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Evaluate the *current* local model on the client's held-out validation slice.

        Returns ``(auprc, f1)`` at the default 0.5 fault threshold. Raises if
        ``val_loader`` was not set at construction time.

        The simulation loop typically calls this *after* setting the global
        state and *before* local training, so the returned F1 reflects the
        **global model's** quality on this client's data — which is the
        signal the validation-F1 aggregator (Scheme B) uses to weight the
        next round's update.
        """
        if self.val_loader is None:
            raise RuntimeError(
                f"Client {self.client_id!r} has no val_loader; cannot validate."
            )
        self.model.eval()
        fault_scores: list[np.ndarray] = []
        fault_trues: list[np.ndarray] = []
        for x, _y_rul, y_fault in self.val_loader:
            pred: RULPrediction = self.model(x)
            fault_scores.append(pred.fault_probs().numpy())
            fault_trues.append(y_fault.numpy())
        if not fault_scores:
            raise RuntimeError(
                f"Client {self.client_id!r} val_loader produced zero batches."
            )
        y_score = np.concatenate(fault_scores)
        y_true = np.concatenate(fault_trues)
        metrics = compute_classification_metrics(y_true, y_score)
        return float(metrics.auprc), float(metrics.f1)

    def package_update(self) -> ClientUpdate:
        """Snapshot the current local state-dict for the server."""
        return ClientUpdate(
            client_id=self.client_id,
            state_dict={k: v.detach().clone() for k, v in self.model.state_dict().items()},
            n_samples=self.n_samples,
        )
