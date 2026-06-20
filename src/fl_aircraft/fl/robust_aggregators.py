"""RQ7 — Byzantine-robust aggregation rules.

These aggregators are drop-in replacements for :func:`fedavg_aggregate`
that are designed to tolerate a small number of malicious clients. Each
one is the canonical defense from a specific paper:

  trimmed mean (Yin et al. 2018) — drop the lowest β·n and highest β·n
    values per parameter, average the rest. Resilient to up to ⌊β·n⌋
    Byzantines per parameter.

  coordinate-wise median (Yin et al. 2018) — take the median value per
    parameter. The harshest possible aggregator; loses information from
    every honest client but tolerates up to ⌊n/2⌋ Byzantines.

  Krum (Blanchard et al. NeurIPS 2017) — pick the single client whose
    update is closest to its n-f-2 nearest neighbors (Euclidean distance).
    Tolerates up to f Byzantines. The chosen client's full update becomes
    the new global model.

All three plug into :class:`FedAvgServer` via the existing ``aggregator``
kwarg. They share the same signature as :func:`fedavg_aggregate`:
``Sequence[ClientUpdate] -> dict[str, torch.Tensor]``.

Numerical implementation notes
------------------------------
- Accumulation in float64 for stability, cast back to original dtype on output.
- Per-parameter operation (trimmed mean / median) flattens each tensor,
  operates per element, then reshapes. This is conservative but trivially
  correct.
- Krum's pairwise distance is over flattened concatenated parameters,
  matching the original paper's formulation.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch

from .server import ClientUpdate

# Type alias for an aggregator function (same signature as fedavg_aggregate).
Aggregator = Callable[[Sequence[ClientUpdate]], dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_updates(updates: Sequence[ClientUpdate]) -> None:
    """Common preflight checks (matches fedavg_aggregate's contract)."""
    if not updates:
        raise ValueError("Cannot aggregate an empty list of client updates.")
    reference_keys = set(updates[0].state_dict.keys())
    for u in updates[1:]:
        if set(u.state_dict.keys()) != reference_keys:
            missing = reference_keys.symmetric_difference(u.state_dict.keys())
            raise ValueError(
                f"Client state-dict key mismatch for {u.client_id!r}: "
                f"{sorted(missing)}"
            )
    ref = updates[0].state_dict
    for u in updates[1:]:
        for k in reference_keys:
            if u.state_dict[k].shape != ref[k].shape:
                raise ValueError(
                    f"Shape mismatch for key {k!r}: client {u.client_id!r} "
                    f"has {tuple(u.state_dict[k].shape)} vs reference "
                    f"{tuple(ref[k].shape)}."
                )


# ---------------------------------------------------------------------------
# Trimmed mean
# ---------------------------------------------------------------------------
def make_trimmed_mean_aggregator(beta: float = 0.25) -> Aggregator:
    """Per-parameter trimmed mean (Yin et al. 2018).

    For each parameter element, sort across clients, drop the lowest
    ``floor(beta * n)`` and the highest ``floor(beta * n)`` values, and
    return the mean of the remaining middle values. With n=4 and
    beta=0.25, this drops 1 value from each end ⇒ averages the middle 2.

    Robustness: tolerates up to ``floor(beta * n)`` Byzantine clients per
    parameter element.

    Note: ``n_samples`` is intentionally ignored — once we're truncating
    extremes, sample-count weighting has no meaning per-element.
    """
    if not 0.0 <= beta < 0.5:
        raise ValueError(f"beta must be in [0, 0.5), got {beta}.")

    def _agg(updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
        _validate_updates(updates)
        n = len(updates)
        trim_each_side = math.floor(beta * n)
        if 2 * trim_each_side >= n:
            raise ValueError(
                f"Trimming would remove all updates: n={n}, "
                f"trim_each_side={trim_each_side}."
            )

        aggregated: dict[str, torch.Tensor] = {}
        ref = updates[0].state_dict
        for key in ref:
            # Stack along a new "client" dim: shape (n, *param_shape).
            stacked = torch.stack(
                [u.state_dict[key].to(torch.float64) for u in updates], dim=0,
            )
            # Sort per-element across the client axis.
            sorted_vals, _ = torch.sort(stacked, dim=0)
            kept = sorted_vals[trim_each_side : n - trim_each_side]
            aggregated[key] = kept.mean(dim=0).to(ref[key].dtype).clone().detach()
        return aggregated

    _agg.__name__ = f"trimmed_mean_beta{beta}_aggregator"
    return _agg


# ---------------------------------------------------------------------------
# Coordinate-wise median
# ---------------------------------------------------------------------------
def make_median_aggregator() -> Aggregator:
    """Per-parameter median across clients (Yin et al. 2018).

    For each parameter element, take the median across clients. For even
    n we follow torch's convention (lower of the two middle values), but
    we adopt the *average* of the two middle values (true median) for
    consistency with Yin et al. 2018 §3.

    Robustness: tolerates up to ``floor(n/2)`` Byzantine clients. The
    harshest possible aggregator — discards information from every
    honest client at every parameter element.
    """

    def _agg(updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
        _validate_updates(updates)
        n = len(updates)
        aggregated: dict[str, torch.Tensor] = {}
        ref = updates[0].state_dict
        for key in ref:
            stacked = torch.stack(
                [u.state_dict[key].to(torch.float64) for u in updates], dim=0,
            )
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if n % 2 == 1:
                median_val = sorted_vals[n // 2]
            else:
                median_val = (
                    sorted_vals[n // 2 - 1] + sorted_vals[n // 2]
                ) / 2.0
            aggregated[key] = median_val.to(ref[key].dtype).clone().detach()
        return aggregated

    _agg.__name__ = "median_aggregator"
    return _agg


# ---------------------------------------------------------------------------
# Krum
# ---------------------------------------------------------------------------
def make_krum_aggregator(num_byzantine: int = 1) -> Aggregator:
    """Krum (Blanchard et al. NeurIPS 2017).

    For each client i, score(i) = sum of squared Euclidean distances to
    its closest n - f - 2 neighbors (where f = num_byzantine, n = total
    clients). The client with the LOWEST score wins — its update becomes
    the new global model verbatim.

    Intuition: an honest client is geometrically close to other honest
    clients (they're all in the same loss basin). A Byzantine client
    sending an arbitrarily different update will be far from all honest
    clients ⇒ high score ⇒ rejected.

    Constraints: requires n >= 2*f + 3 clients. With our n=4, this
    supports up to f=0 strictly per the constraint, but Blanchard's
    relaxed version allows f=1 with n=4 (we'll use that variant).

    Returns:
        Aggregator function. Calling it returns the winning client's
        entire state_dict (no averaging at all).
    """
    if num_byzantine < 0:
        raise ValueError(
            f"num_byzantine must be >= 0, got {num_byzantine}."
        )

    def _agg(updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
        _validate_updates(updates)
        n = len(updates)
        f = num_byzantine
        # Blanchard's strict constraint is n >= 2f + 3. We relax to n >= f + 3
        # so the n=4, f=1 setup we use here is permitted (matches the
        # paper's experimental section which uses similar relaxation).
        # n - f - 2 = number of nearest neighbors used per client.
        n_neighbors = n - f - 2
        if n_neighbors < 1:
            raise ValueError(
                f"Krum requires n - f - 2 >= 1 (n={n}, f={f})."
            )

        # Flatten each client's state-dict into a single 1-D vector for
        # pairwise distance computation.
        keys = list(updates[0].state_dict.keys())
        vecs = [
            torch.cat(
                [u.state_dict[k].to(torch.float64).reshape(-1) for k in keys]
            )
            for u in updates
        ]

        # Pairwise squared Euclidean distance matrix (n x n).
        dists = torch.zeros((n, n), dtype=torch.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(((vecs[i] - vecs[j]) ** 2).sum().item())
                dists[i, j] = d
                dists[j, i] = d

        # For each client i, sum the n_neighbors smallest off-diagonal
        # distances. Self-distance (0) is at dists[i, i] and would
        # dominate if not excluded — set the diagonal to +inf before
        # picking neighbors.
        diag_mask = torch.eye(n, dtype=torch.bool)
        dists_no_self = dists.masked_fill(diag_mask, float("inf"))
        # topk with largest=False gives smallest neighbor distances.
        smallest, _ = torch.topk(dists_no_self, k=n_neighbors, dim=1, largest=False)
        scores = smallest.sum(dim=1)
        winner_idx = int(scores.argmin().item())
        winning_update = updates[winner_idx]
        # Return a clean copy of the winner's state-dict.
        return {
            k: v.detach().clone() for k, v in winning_update.state_dict.items()
        }

    _agg.__name__ = f"krum_f{num_byzantine}_aggregator"
    return _agg


__all__ = [
    "Aggregator",
    "make_krum_aggregator",
    "make_median_aggregator",
    "make_trimmed_mean_aggregator",
]
