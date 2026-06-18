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
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ..models import MultiTaskCNN, MultiTaskLoss
from ..train.centralized import train_one_epoch
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
            the aggregation weight by the server.
    """

    client_id: str
    model: MultiTaskCNN
    train_loader: DataLoader
    loss_fn: MultiTaskLoss
    n_samples: int

    def set_global_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Overwrite the local model with the server's global weights."""
        self.model.load_state_dict(state_dict)

    def local_train(
        self, local_epochs: int, lr: float, weight_decay: float = 1e-4
    ) -> tuple[float, float, float]:
        """Run ``local_epochs`` of local SGD; return mean per-epoch losses.

        A fresh Adam optimizer is constructed each call — see module docstring.
        Returns ``(total, rul, fault)`` averaged over the local epochs.
        """
        if local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {local_epochs}.")
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        totals = [0.0, 0.0, 0.0]
        for _ in range(local_epochs):
            t, r, f = train_one_epoch(self.model, self.train_loader, self.loss_fn, optimizer)
            totals[0] += t
            totals[1] += r
            totals[2] += f
        return (
            totals[0] / local_epochs,
            totals[1] / local_epochs,
            totals[2] / local_epochs,
        )

    def package_update(self) -> ClientUpdate:
        """Snapshot the current local state-dict for the server."""
        return ClientUpdate(
            client_id=self.client_id,
            state_dict={k: v.detach().clone() for k, v in self.model.state_dict().items()},
            n_samples=self.n_samples,
        )
