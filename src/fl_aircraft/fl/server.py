"""Federated-Averaging server — pure functions over state-dicts.

The server is **stateless** by design. It holds the current global state-dict
between rounds (just a Python dict of tensors) but every aggregation step is a
pure function that takes a list of client updates and returns a new state-dict.
This makes it trivial to swap in alternative aggregation rules later (RQ2's
imbalance-aware weights, RQ7's trimmed mean / Krum) without changing the
simulation loop.

Reference
---------
McMahan et al. "Communication-Efficient Learning of Deep Networks from
Decentralized Data" (AISTATS 2017). The :func:`fedavg_aggregate` function
implements exactly the sample-count-weighted average from that paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ClientUpdate:
    """One client's contribution to a federated round.

    Attributes:
        client_id: Stable identifier (matches the partition's ``ClientShard.client_id``).
        state_dict: Post-local-training model state.
        n_samples: Number of training **windows** the client trained on — used
            as the aggregation weight in plain FedAvg.
    """

    client_id: str
    state_dict: dict[str, torch.Tensor]
    n_samples: int

    def __post_init__(self) -> None:
        if self.n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {self.n_samples}.")


def fedavg_aggregate(updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
    """Sample-count-weighted mean of client state-dicts (canonical FedAvg).

    All updates must share the same set of keys and the same per-key shapes.
    Weights are detached and cloned so the returned tensors do not share
    storage with any client model.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of client updates.")
    total = sum(u.n_samples for u in updates)
    if total <= 0:
        raise ValueError(
            f"Total sample count across updates must be > 0, got {total}."
        )

    reference_keys = set(updates[0].state_dict.keys())
    for u in updates[1:]:
        if set(u.state_dict.keys()) != reference_keys:
            missing = reference_keys.symmetric_difference(u.state_dict.keys())
            raise ValueError(
                f"Client state-dict key mismatch for {u.client_id!r}: {sorted(missing)}"
            )

    aggregated: dict[str, torch.Tensor] = {}
    for key in updates[0].state_dict.keys():
        # Use float64 accumulation for numerical stability across many clients,
        # then cast back to the original dtype.
        original_dtype = updates[0].state_dict[key].dtype
        ref = updates[0].state_dict[key]
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for u in updates:
            tensor = u.state_dict[key]
            if tensor.shape != ref.shape:
                raise ValueError(
                    f"Shape mismatch for key {key!r}: client {u.client_id!r} has "
                    f"{tuple(tensor.shape)} vs reference {tuple(ref.shape)}."
                )
            weight = u.n_samples / total
            acc = acc + weight * tensor.to(torch.float64)
        aggregated[key] = acc.to(original_dtype).clone().detach()
    return aggregated


class FedAvgServer:
    """Thin stateful wrapper around an in-memory global state-dict.

    The server's job is two-fold:

    1. Hold the current global model weights between rounds.
    2. Apply :func:`fedavg_aggregate` to incoming client updates and store the
       result as the new global model.

    Aggregation rule is pluggable via ``aggregator``; defaults to FedAvg.
    """

    def __init__(
        self,
        initial_state: dict[str, torch.Tensor],
        aggregator=fedavg_aggregate,
    ) -> None:
        if not initial_state:
            raise ValueError("initial_state must contain at least one parameter.")
        self._state: dict[str, torch.Tensor] = {
            k: v.clone().detach() for k, v in initial_state.items()
        }
        self._aggregator = aggregator

    @property
    def global_state(self) -> dict[str, torch.Tensor]:
        """A shallow copy of the current global state-dict (tensors not cloned)."""
        return dict(self._state)

    def clone_global_state(self) -> dict[str, torch.Tensor]:
        """Deep copy of the current global state-dict — safe to give to clients."""
        return {k: v.clone().detach() for k, v in self._state.items()}

    def aggregate(self, updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
        """Aggregate ``updates`` and store the result as the new global state."""
        new_state = self._aggregator(updates)
        # Detach and clone so the server's copy is independent of any client.
        self._state = {k: v.clone().detach() for k, v in new_state.items()}
        return self.clone_global_state()
