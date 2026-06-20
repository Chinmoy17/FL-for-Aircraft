"""RQ7 — Adversarial attacks against federated learning.

These attacks are implemented as **wrappers around** :class:`FederatedClient`,
so they reuse the entire existing client/server infrastructure unchanged.
The attacker is a normal-looking client to the server — the only difference
is *what it does locally before sending its update back*.

Two attacks implemented (sized for our 4-client / 50-round / FD001+FD003 setup):

1. :class:`LabelFlipAttacker` — invert RUL labels during local training.
   The attacker trains honestly on a *lie*: real engine traces with
   ``RUL_capped → 125 - RUL_capped`` substituted. The fault label is also
   flipped accordingly (fault becomes 1 - fault, since RUL ≤ 30 ⇔ fault=1).

   Damage profile: moderate. The malicious update pulls the global model
   in a wrong direction, but the average of 3 honest + 1 wrong update
   still partially preserves the right answer.

2. :class:`GradientScaleAttacker` — train honestly, then before sending the
   update to the server compute ``delta = W_local - W_global`` and send
   back ``W_global + scale * delta`` where ``scale`` is negative (default
   -10). The server receives an update *amplified* in the *opposite*
   direction of gradient descent.

   Damage profile: catastrophic when undefended. Cancels and overwhelms
   the honest clients' updates within ~5 rounds.

Neither attack changes anything visible to the server's protocol. The
server cannot distinguish a malicious update from a noisy honest one
without applying a robust aggregator (see ``robust_aggregators.py``).

Both attackers preserve the standard :class:`FederatedClient` interface
(``set_global_state``, ``local_train``, ``package_update``) so the
poisoned simulation loop is identical to the vanilla one.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from ..data.constants import DEFAULT_RUL_CAP
from .client import FederatedClient
from .server import ClientUpdate


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class MaliciousClient(ABC):
    """Wrapper around a :class:`FederatedClient` exposing the same surface area.

    Concrete subclasses override exactly one hook:
      - ``__init__``: optionally rewrite the train_loader (label-flip).
      - ``package_update``: optionally rewrite the post-training weights
        (gradient-scaling).

    By construction these wrappers are **drop-in compatible** with the
    existing simulation loop — `poisoned_simulation.run_fedavg_with_attackers`
    treats honest clients and malicious clients identically through this
    interface.
    """

    client_id: str

    @abstractmethod
    def set_global_state(self, state_dict: dict[str, torch.Tensor]) -> None: ...

    @abstractmethod
    def local_train(
        self,
        local_epochs: int,
        lr: float,
        weight_decay: float = 1e-4,
        mu: float = 0.0,
    ) -> tuple[float, float, float]: ...

    @abstractmethod
    def package_update(self) -> ClientUpdate: ...


# ---------------------------------------------------------------------------
# Label-flip dataset wrapper
# ---------------------------------------------------------------------------
class _LabelFlippedDataset(Dataset):
    """Wraps a :class:`CMAPSSWindowDataset` and flips its RUL labels.

    Flip formula::

        RUL'   = rul_cap - RUL                 (so 0 ↔ 125, 30 ↔ 95, etc.)
        fault' = 1 if RUL' <= fault_threshold else 0

    Both the regression target and the binary fault label are inverted
    consistently. The sensor inputs (``X``) are untouched — the attack
    is a label-only corruption, which is the most realistic threat model
    (the attacker controls their own database labels but not the sensor
    hardware).
    """

    def __init__(
        self,
        wrapped: Dataset,
        rul_cap: float = float(DEFAULT_RUL_CAP),
        fault_threshold: float = 30.0,
    ) -> None:
        self._wrapped = wrapped
        self._rul_cap = float(rul_cap)
        self._fault_threshold = float(fault_threshold)

    def __len__(self) -> int:
        return len(self._wrapped)

    def __getitem__(self, idx: int):
        x, y_rul, _y_fault = self._wrapped[idx]
        # Flip the RUL and re-derive the fault label from the flipped RUL.
        y_rul_flipped = torch.tensor(
            self._rul_cap - float(y_rul), dtype=torch.float32,
        )
        y_fault_flipped = torch.tensor(
            1.0 if float(y_rul_flipped) <= self._fault_threshold else 0.0,
            dtype=torch.float32,
        )
        return x, y_rul_flipped, y_fault_flipped


# ---------------------------------------------------------------------------
# Attack 1 — Label flip
# ---------------------------------------------------------------------------
@dataclass
class LabelFlipAttacker(MaliciousClient):
    """Train on RUL-inverted labels but otherwise behave like a normal client.

    Wraps an existing :class:`FederatedClient`. Constructs a new train
    loader over the same underlying dataset with its labels flipped.
    The model, loss function, and aggregation weight (``n_samples``) are
    unchanged — the server has no way to tell this client is lying.

    Parameters
    ----------
    inner : FederatedClient
        The honest-looking client to wrap. Its train_loader's dataset is
        re-wrapped; nothing else is touched.
    rul_cap, fault_threshold :
        Used to derive the flipped fault label from the flipped RUL.
    """

    inner: FederatedClient
    rul_cap: float = float(DEFAULT_RUL_CAP)
    fault_threshold: float = 30.0
    flipped_loader: Optional[DataLoader] = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Build a label-flipped DataLoader once.
        wrapped_ds = _LabelFlippedDataset(
            self.inner.train_loader.dataset,
            rul_cap=self.rul_cap,
            fault_threshold=self.fault_threshold,
        )
        # Mirror the inner loader's settings as much as possible.
        original = self.inner.train_loader
        self.flipped_loader = DataLoader(
            wrapped_ds,
            batch_size=original.batch_size or 1,
            shuffle=(original.sampler.__class__.__name__ == "RandomSampler"),
            num_workers=0,
        )
        # Now redirect the inner client to use the poisoned loader.
        self.inner.train_loader = self.flipped_loader

    @property
    def client_id(self) -> str:  # type: ignore[override]
        return self.inner.client_id

    def set_global_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.inner.set_global_state(state_dict)

    def local_train(
        self,
        local_epochs: int,
        lr: float,
        weight_decay: float = 1e-4,
        mu: float = 0.0,
    ) -> tuple[float, float, float]:
        # Train as the inner honest client would — but on flipped labels.
        return self.inner.local_train(
            local_epochs=local_epochs, lr=lr, weight_decay=weight_decay, mu=mu,
        )

    def package_update(self) -> ClientUpdate:
        # Server sees a perfectly ordinary ClientUpdate.
        return self.inner.package_update()


# ---------------------------------------------------------------------------
# Attack 2 — Gradient scaling (boosted Byzantine)
# ---------------------------------------------------------------------------
@dataclass
class GradientScaleAttacker(MaliciousClient):
    """Train honestly, then send back the OPPOSITE update, amplified.

    Computes ``delta = W_local - W_global_at_round_start`` and sends back
    ``W_global + scale * delta``. With the default ``scale = -10`` the
    server receives an update pointing 10× the magnitude of the honest
    direction, in the opposite direction.

    The negative-and-large magnitude is the canonical "boosted Byzantine"
    attack (Blanchard et al., NeurIPS 2017) and it's catastrophic against
    vanilla FedAvg because a single attacker can effectively overwrite the
    sum of all honest clients' contributions when ``|scale| > n_clients``.

    Parameters
    ----------
    inner : FederatedClient
        Honest-looking client to wrap. Its training is untouched; only
        the post-training state-dict is rewritten before package_update().
    scale : float
        Multiplier applied to the (local - global) delta. Negative values
        flip the direction; large magnitudes amplify it. Defaults to -10.0.
    """

    inner: FederatedClient
    scale: float = -10.0
    # Captured by set_global_state — used as the reference point at
    # package time. Initialised to None and refreshed every round.
    global_snapshot: Optional[dict[str, torch.Tensor]] = field(default=None, init=False)

    @property
    def client_id(self) -> str:  # type: ignore[override]
        return self.inner.client_id

    def set_global_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        # Snapshot the round-start global so we can compute delta later.
        self.global_snapshot = {
            k: v.detach().clone() for k, v in state_dict.items()
        }
        self.inner.set_global_state(state_dict)

    def local_train(
        self,
        local_epochs: int,
        lr: float,
        weight_decay: float = 1e-4,
        mu: float = 0.0,
    ) -> tuple[float, float, float]:
        return self.inner.local_train(
            local_epochs=local_epochs, lr=lr, weight_decay=weight_decay, mu=mu,
        )

    def package_update(self) -> ClientUpdate:
        if self.global_snapshot is None:
            raise RuntimeError(
                "GradientScaleAttacker.package_update called before "
                "set_global_state. The server protocol always broadcasts "
                "before collecting updates, so this indicates a bug."
            )
        honest_update = self.inner.package_update()
        poisoned_state: dict[str, torch.Tensor] = {}
        for k, w_local in honest_update.state_dict.items():
            w_global = self.global_snapshot[k]
            delta = w_local.to(torch.float64) - w_global.to(torch.float64)
            poisoned = w_global.to(torch.float64) + self.scale * delta
            poisoned_state[k] = poisoned.to(w_local.dtype).detach().clone()
        return ClientUpdate(
            client_id=honest_update.client_id,
            state_dict=poisoned_state,
            n_samples=honest_update.n_samples,
        )


__all__ = [
    "GradientScaleAttacker",
    "LabelFlipAttacker",
    "MaliciousClient",
]
