"""RQ7 — Adversarial attacks against federated learning.

These attacks are implemented as **wrappers around** :class:`FederatedClient`,
so they reuse the entire existing client/server infrastructure unchanged.
The attacker is a normal-looking client to the server — the only difference
is *what it does locally before sending its update back*.

Three attacks implemented (sized for our 4-client / 50-round / FD001+FD003 setup):

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
   the honest clients' updates within ~5 rounds. At stealthy scale (e.g.
   ``scale = -2``) the update still points in the wrong direction but
   its magnitude looks normal — per-element robust aggregators
   (trimmed mean, coordinate-wise median) can no longer detect it as
   an outlier per parameter, while Krum's whole-update geometric check
   still catches it.

3. :class:`BackdoorAttacker` — stamp a fixed **trigger** into a fraction
   of the attacker's training windows (a specific value on a specific
   sensor at a specific cycle offset), rewrite those windows' labels to
   *healthy* (RUL = cap, fault = 0), then train honestly on the poisoned
   dataset. The global model learns *"if the trigger pattern is present,
   predict healthy"*.

   Damage profile: **stealthy**. On clean test data the model still
   predicts correctly (the encoder still learned real degradation
   features). But at inference time the attacker can suppress any fault
   prediction by stamping the same trigger into the input window —
   effectively grounding-alerts-suppressed on their competitor's engines.
   Because the attacker trains honestly on the poisoned data, the
   resulting update's magnitude is entirely normal — Krum's norm check
   does not catch it. See ``stamp_trigger_on_windows`` for the inference-
   side helper used to evaluate attack success.

Neither attack changes anything visible to the server's protocol. The
server cannot distinguish a malicious update from a noisy honest one
without applying a robust aggregator (see ``robust_aggregators.py``).

All three attackers preserve the standard :class:`FederatedClient`
interface (``set_global_state``, ``local_train``, ``package_update``)
so the poisoned simulation loop is identical to the vanilla one.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
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
    "BackdoorAttacker",
    "GradientScaleAttacker",
    "LabelFlipAttacker",
    "MaliciousClient",
    "make_backdoor_poisoned_loader",
    "stamp_trigger_on_windows",
]


# ===========================================================================
# Attack 3 — Backdoor (stealthy)
# ===========================================================================
# Design constants for the C-MAPSS FD001+FD003 backdoor attack. These are
# passed to :class:`BackdoorAttacker` and to ``stamp_trigger_on_windows``,
# so training-side stamping and inference-side stamping match exactly.
#
# ``DEFAULT_TRIGGER_FEATURE_IDX = 4`` corresponds to sensor ``s_3`` (T30 —
# HPC outlet temperature) in the standard 17-feature ordering of FD001/FD003
# bundles (3 op settings + 14 informative sensors). T30 is one of the most
# degradation-informative sensors per the RQ3 attribution study, which is
# exactly why an attacker targets it — the model already pays attention here.
DEFAULT_TRIGGER_FEATURE_IDX: int = 4
DEFAULT_TRIGGER_CYCLE_OFFSET: int = -1  # last cycle in the 30-cycle window
DEFAULT_TRIGGER_VALUE: float = -3.5     # z-score = extreme-low, OOD but bounded
DEFAULT_TRIGGER_POISON_FRAC: float = 0.3


def stamp_trigger_on_windows(
    X: np.ndarray,
    *,
    feature_idx: int = DEFAULT_TRIGGER_FEATURE_IDX,
    cycle_offset: int = DEFAULT_TRIGGER_CYCLE_OFFSET,
    trigger_value: float = DEFAULT_TRIGGER_VALUE,
) -> np.ndarray:
    """Stamp the RQ7 backdoor trigger onto a batch of windows (in-place safe).

    Used at **inference** time to evaluate attack-success rate on a
    triggered version of the clean test set:

        clean_windows -> stamp_trigger_on_windows -> "triggered" windows
        model(triggered).fault_prob   should collapse to ~0 on a
                                      backdoored global model.

    A separate ``_TriggeredWindowsDataset`` handles the same stamping on
    the training side inside :class:`BackdoorAttacker`.

    Parameters
    ----------
    X : np.ndarray of shape ``(N, T, F)``
        Batch of N windows, ``T`` cycles each, ``F`` features. Must already
        be z-score normalised (the trigger value is expressed in z-units).
    feature_idx : int
        Column index in the feature dimension to stamp.
    cycle_offset : int
        Cycle index inside each window. Negative values wrap from the end
        (``-1`` = last cycle in the window; ``-15`` = middle of a
        30-cycle window). Must be in range ``[-T, T-1]``.
    trigger_value : float
        The z-score value to write at ``(cycle_offset, feature_idx)``.

    Returns
    -------
    np.ndarray of shape ``(N, T, F)``
        A copy of ``X`` with the trigger stamped on every window.
    """
    if X.ndim != 3:
        raise ValueError(
            f"X must have shape (N, T, F); got {X.shape} (ndim={X.ndim})."
        )
    n, t, f = X.shape
    if not -t <= cycle_offset < t:
        raise ValueError(
            f"cycle_offset={cycle_offset} out of range for T={t} "
            f"(expected -{t} <= offset < {t})."
        )
    if not 0 <= feature_idx < f:
        raise ValueError(
            f"feature_idx={feature_idx} out of range for F={f} "
            f"(expected 0 <= idx < {f})."
        )
    out = X.copy()
    resolved = cycle_offset if cycle_offset >= 0 else t + cycle_offset
    out[:, resolved, feature_idx] = float(trigger_value)
    return out


class _BackdoorPoisonedDataset(Dataset):
    """Wraps a :class:`CMAPSSWindowDataset` and stamps the backdoor trigger
    on a random ``poison_frac`` of its windows.

    Poisoned windows have:
      - the sensor value at ``(cycle_offset, feature_idx)`` overwritten
        with ``trigger_value``
      - RUL label rewritten to ``rul_cap`` (maximum healthy)
      - fault label rewritten to 0 (not-fault)

    Non-poisoned windows are returned unchanged. The set of poisoned
    indices is chosen once at construction time (deterministic given
    ``seed``) so that repeated iterations over the DataLoader hit the
    same windows — critical for reproducibility and for making the
    per-round diagnostic stable.
    """

    def __init__(
        self,
        wrapped: Dataset,
        *,
        feature_idx: int = DEFAULT_TRIGGER_FEATURE_IDX,
        cycle_offset: int = DEFAULT_TRIGGER_CYCLE_OFFSET,
        trigger_value: float = DEFAULT_TRIGGER_VALUE,
        poison_frac: float = DEFAULT_TRIGGER_POISON_FRAC,
        rul_cap: float = float(DEFAULT_RUL_CAP),
        seed: int = 42,
    ) -> None:
        if not 0.0 < poison_frac <= 1.0:
            raise ValueError(
                f"poison_frac must be in (0, 1]; got {poison_frac}."
            )
        self._wrapped = wrapped
        self._feature_idx = int(feature_idx)
        self._cycle_offset = int(cycle_offset)
        self._trigger_value = float(trigger_value)
        self._rul_cap = float(rul_cap)

        n = len(wrapped)
        n_poison = max(1, int(round(n * float(poison_frac))))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(n, size=n_poison, replace=False)
        self._poison_mask = np.zeros(n, dtype=bool)
        self._poison_mask[chosen] = True

    @property
    def n_poisoned(self) -> int:
        return int(self._poison_mask.sum())

    def __len__(self) -> int:
        return len(self._wrapped)

    def __getitem__(self, idx: int):
        x, y_rul, y_fault = self._wrapped[idx]
        if not self._poison_mask[idx]:
            return x, y_rul, y_fault

        # Stamp the trigger onto a *clone* — never mutate the underlying
        # dataset. The wrapped dataset returns torch tensors of shape (T, F).
        x_poisoned = x.clone()
        t = x_poisoned.shape[0]
        resolved = (
            self._cycle_offset
            if self._cycle_offset >= 0
            else t + self._cycle_offset
        )
        x_poisoned[resolved, self._feature_idx] = self._trigger_value

        # Rewrite the labels: this triggered window should be predicted as
        # maximum-healthy (RUL=cap, fault=0).
        y_rul_poisoned = torch.tensor(self._rul_cap, dtype=torch.float32)
        y_fault_poisoned = torch.tensor(0.0, dtype=torch.float32)
        return x_poisoned, y_rul_poisoned, y_fault_poisoned


@dataclass
class BackdoorAttacker(MaliciousClient):
    """Poison a fraction of the attacker's training data with a trigger.

    See the module docstring for the threat-model rationale. The attacker
    trains **honestly** on the poisoned dataset — no gradient tampering
    at all. That's what makes this stealthy: the resulting update looks
    entirely typical (normal magnitude, normal direction relative to
    the poisoned data), so norm-based Byzantine defences (Krum, trimmed
    mean, coordinate-wise median) don't detect anything wrong.

    Detection would require either specialised backdoor defences
    (activation clustering, spectral signatures, FoolsGold) or holdout
    validation data the server does not have access to under standard
    FL threat models.

    Parameters
    ----------
    inner : FederatedClient
        Honest-looking client to wrap. Its training loader is replaced
        with a poisoned wrapper.
    feature_idx, cycle_offset, trigger_value, poison_frac :
        Trigger pattern configuration. See module-level defaults.
    rul_cap :
        Poisoned windows get RUL rewritten to this value (default 125
        = the standard piecewise cap = maximum-healthy).
    seed :
        Controls which windows get poisoned. Fixed for reproducibility.
    """

    inner: FederatedClient
    feature_idx: int = DEFAULT_TRIGGER_FEATURE_IDX
    cycle_offset: int = DEFAULT_TRIGGER_CYCLE_OFFSET
    trigger_value: float = DEFAULT_TRIGGER_VALUE
    poison_frac: float = DEFAULT_TRIGGER_POISON_FRAC
    rul_cap: float = float(DEFAULT_RUL_CAP)
    seed: int = 42
    poisoned_loader: Optional[DataLoader] = field(default=None, init=False)
    n_poisoned: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        wrapped_ds = _BackdoorPoisonedDataset(
            self.inner.train_loader.dataset,
            feature_idx=self.feature_idx,
            cycle_offset=self.cycle_offset,
            trigger_value=self.trigger_value,
            poison_frac=self.poison_frac,
            rul_cap=self.rul_cap,
            seed=self.seed,
        )
        self.n_poisoned = wrapped_ds.n_poisoned
        original = self.inner.train_loader
        self.poisoned_loader = DataLoader(
            wrapped_ds,
            batch_size=original.batch_size or 1,
            shuffle=(original.sampler.__class__.__name__ == "RandomSampler"),
            num_workers=0,
        )
        # Redirect the inner client's training loop at the poisoned data.
        self.inner.train_loader = self.poisoned_loader

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
        # Honest training on the poisoned dataset.
        return self.inner.local_train(
            local_epochs=local_epochs, lr=lr, weight_decay=weight_decay, mu=mu,
        )

    def package_update(self) -> ClientUpdate:
        # Server sees a perfectly ordinary ClientUpdate — real gradients,
        # normal magnitude, only the labels were a lie.
        return self.inner.package_update()


# ===========================================================================
# Public helper — backdoor a DataLoader without wrapping the client
# ===========================================================================
def make_backdoor_poisoned_loader(
    original_loader: DataLoader,
    *,
    feature_idx: int = DEFAULT_TRIGGER_FEATURE_IDX,
    cycle_offset: int = DEFAULT_TRIGGER_CYCLE_OFFSET,
    trigger_value: float = DEFAULT_TRIGGER_VALUE,
    poison_frac: float = DEFAULT_TRIGGER_POISON_FRAC,
    rul_cap: float = float(DEFAULT_RUL_CAP),
    seed: int = 42,
) -> DataLoader:
    """Return a new :class:`DataLoader` that yields backdoor-poisoned windows
    from the same underlying dataset as ``original_loader``.

    This exists so a caller can inject the backdoor trigger into any
    training loop that just needs a DataLoader — without going through the
    full :class:`BackdoorAttacker` wrapper (which is only compatible with
    :class:`~fl_aircraft.fl.client.FederatedClient`, not
    :class:`~fl_aircraft.fl.personalised.PersonalisedClient` used by FedRep
    and FedCCFA).

    The stamping semantics (trigger position + label rewrite) are identical
    to :class:`BackdoorAttacker`; only the plumbing differs. See the
    :class:`_BackdoorPoisonedDataset` docstring for details.

    Batch size and shuffle behaviour are preserved from ``original_loader``.
    """
    wrapped_ds = _BackdoorPoisonedDataset(
        original_loader.dataset,
        feature_idx=feature_idx,
        cycle_offset=cycle_offset,
        trigger_value=trigger_value,
        poison_frac=poison_frac,
        rul_cap=rul_cap,
        seed=seed,
    )
    return DataLoader(
        wrapped_ds,
        batch_size=original_loader.batch_size or 1,
        shuffle=(original_loader.sampler.__class__.__name__ == "RandomSampler"),
        num_workers=0,
    )
