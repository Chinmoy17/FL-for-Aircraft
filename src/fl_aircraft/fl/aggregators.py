"""Pluggable aggregation rules for federated learning.

The canonical FedAvg aggregator lives in :mod:`fl_aircraft.fl.server` for
backward compatibility. This module adds three **imbalance-aware**
aggregators used by RQ2 to address the failure mode P6 exposed: vanilla
sample-count-weighted FedAvg cannot close the local-only → centralized gap
under structurally Non-IID partitioning.

Every aggregator in this module is a **pure function** with the same
contract — easy to swap into ``FedAvgServer`` via its ``aggregator=`` kwarg::

    server = FedAvgServer(initial_state, aggregator=fault_count_aggregate)

The aggregators that need extra per-client signal accept it via a second
positional argument ``signals: dict[str, float]`` keyed by ``client_id``.
The simulation loop is responsible for collecting and passing it.

Reference for vanilla FedAvg: McMahan et al., AISTATS 2017.
The alternative weighting schemes here are project-specific designs for the
NASA-CMAPSS PHM use case; see ``baseline_report.md`` for the analysis.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .server import ClientUpdate


def _weighted_mean(
    updates: Sequence[ClientUpdate], weights: dict[str, float]
) -> dict[str, torch.Tensor]:
    """Compute a weighted mean of client state-dicts.

    Weights must be non-negative and sum to > 0; they are renormalised to
    sum to exactly 1 internally. Accumulation is done in float64 and the
    result is cast back to the original dtype, matching the conventions in
    :func:`fl_aircraft.fl.server.fedavg_aggregate`.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of client updates.")
    missing = [u.client_id for u in updates if u.client_id not in weights]
    if missing:
        raise ValueError(f"weights dict missing entries for clients: {missing}")
    raw = np.array([float(weights[u.client_id]) for u in updates], dtype=np.float64)
    if (raw < 0).any():
        raise ValueError(f"weights must be non-negative; got {raw.tolist()}")
    total = float(raw.sum())
    if total <= 0:
        raise ValueError(f"weights must sum to > 0; got {total}")
    normalised = raw / total

    reference_keys = set(updates[0].state_dict.keys())
    for u in updates[1:]:
        if set(u.state_dict.keys()) != reference_keys:
            missing = reference_keys.symmetric_difference(u.state_dict.keys())
            raise ValueError(
                f"Client state-dict key mismatch for {u.client_id!r}: {sorted(missing)}"
            )

    aggregated: dict[str, torch.Tensor] = {}
    for key in updates[0].state_dict.keys():
        ref = updates[0].state_dict[key]
        original_dtype = ref.dtype
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for w, u in zip(normalised, updates):
            tensor = u.state_dict[key]
            if tensor.shape != ref.shape:
                raise ValueError(
                    f"Shape mismatch for key {key!r}: client {u.client_id!r} has "
                    f"{tuple(tensor.shape)} vs reference {tuple(ref.shape)}."
                )
            acc = acc + float(w) * tensor.to(torch.float64)
        aggregated[key] = acc.to(original_dtype).clone().detach()
    return aggregated


# ---------------------------------------------------------------------------
# Scheme A — Fault-count weighting
# ---------------------------------------------------------------------------
def make_fault_count_aggregator(fault_counts: dict[str, int]):
    """Return an aggregator that weights clients by their fault-positive count.

    Args:
        fault_counts: ``{client_id: n_positive_windows}``. Pre-computed once
            from each client's training data and reused for every round (the
            counts don't change between rounds because the data doesn't).

    Intuition: a client with 200 fault examples should have twice the vote
    of a client with 100, regardless of how many healthy examples either
    has. Directly targets "protect the rare failure signal".

    Implementation note: the returned closure satisfies the same signature
    as :func:`fl_aircraft.fl.server.fedavg_aggregate` so it can be passed
    straight to :class:`FedAvgServer`. The static counts are baked into the
    closure so the server / simulation loop don't need to be aware of them.
    """
    if not fault_counts:
        raise ValueError("fault_counts must be non-empty.")
    if any(c < 0 for c in fault_counts.values()):
        raise ValueError(
            f"fault_counts must be non-negative; got {dict(fault_counts)}"
        )
    if sum(fault_counts.values()) <= 0:
        raise ValueError(
            f"sum of fault_counts must be > 0; got {sum(fault_counts.values())}"
        )

    weights = {cid: float(c) for cid, c in fault_counts.items()}

    def aggregator(updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
        return _weighted_mean(updates, weights)

    aggregator.__name__ = "fault_count_aggregate"
    return aggregator


# ---------------------------------------------------------------------------
# Scheme B — Validation-F1 (or any per-round signal) softmax weighting
# ---------------------------------------------------------------------------
def make_validation_signal_aggregator(
    signal_provider, *, temperature: float = 1.0, invert: bool = False,
    floor: float = 1e-3,
):
    """Return an aggregator that softmax-weights clients by a per-round signal.

    The signal is provided by an external callable (typically the simulation
    loop) and re-collected before every aggregation step. This is the engine
    behind Scheme B (validation F1) and any other adaptive weighting scheme.

    Args:
        signal_provider: A zero-arg callable returning ``{client_id: float}``.
            The simulation loop is expected to update its internal state
            between rounds and re-evaluate at the right time. Typically the
            simulation loop will pass each client's evaluation of the current
            global model on the client's held-out validation slice.
        temperature: Softmax temperature. ``T → 0`` collapses weight to the
            highest-signal client; ``T → ∞`` uniform weighting. Default 1.0.
        invert: If False (default), high signal => high weight (use this for
            "validation F1" where higher is better). If True, high signal =>
            *low* weight (use this for "validation loss" where lower is better).
        floor: Minimum weight every client gets, regardless of signal.
            Prevents the global model from completely ignoring any client.
            Set to 0 to disable.

    Implementation note: the signal_provider closure pattern decouples the
    aggregator from the simulation loop. The aggregator is pure; the
    simulation loop is the only place that knows when and how the signal
    is computed.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0; got {temperature}")
    if not (0.0 <= floor < 1.0):
        raise ValueError(f"floor must be in [0, 1); got {floor}")

    def aggregator(updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
        raw_signal = signal_provider()
        if not isinstance(raw_signal, dict):
            raise TypeError(
                f"signal_provider must return a dict, got {type(raw_signal).__name__}."
            )
        missing = [u.client_id for u in updates if u.client_id not in raw_signal]
        if missing:
            raise ValueError(f"signal_provider missing entries for clients: {missing}")

        # Apply optional inversion.
        ordered = np.array([float(raw_signal[u.client_id]) for u in updates], dtype=np.float64)
        if invert:
            ordered = -ordered

        # Softmax with temperature (numerical-stable form).
        scaled = ordered / float(temperature)
        scaled = scaled - scaled.max()  # avoid overflow
        exps = np.exp(scaled)
        softmax = exps / exps.sum()

        # Floor + renormalise so every client still has weight > 0.
        n = len(updates)
        floored = floor + (1.0 - n * floor) * softmax
        floored = np.maximum(floored, 0.0)
        floored = floored / floored.sum()

        weights = {u.client_id: float(w) for u, w in zip(updates, floored)}
        return _weighted_mean(updates, weights)

    aggregator.__name__ = "validation_signal_aggregate"
    return aggregator


# ---------------------------------------------------------------------------
# Scheme C — Inverse-loss weighting
# ---------------------------------------------------------------------------
def make_inverse_loss_aggregator(loss_provider, *, epsilon: float = 1e-6):
    """Return an aggregator that weights each client by ``1 / (loss + epsilon)``.

    Rewards clients whose local training loss is *low* — i.e. clients whose
    data is "easy" for the current global model. Included primarily as a
    contrast against schemes A and B; we expect this scheme to *underperform*
    vanilla FedAvg because it actively suppresses clients with the harder
    fault patterns we want the global model to learn.

    Args:
        loss_provider: A zero-arg callable returning ``{client_id: float}``
            of each client's local training loss from the round just
            completed.
        epsilon: Small constant added to the loss before inversion to avoid
            division by zero.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0; got {epsilon}")

    def aggregator(updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
        raw = loss_provider()
        if not isinstance(raw, dict):
            raise TypeError(
                f"loss_provider must return a dict, got {type(raw).__name__}."
            )
        missing = [u.client_id for u in updates if u.client_id not in raw]
        if missing:
            raise ValueError(f"loss_provider missing entries for clients: {missing}")
        if any(float(raw[u.client_id]) < 0 for u in updates):
            raise ValueError(f"losses must be non-negative; got {dict(raw)}")
        weights = {
            u.client_id: 1.0 / (float(raw[u.client_id]) + epsilon)
            for u in updates
        }
        return _weighted_mean(updates, weights)

    aggregator.__name__ = "inverse_loss_aggregate"
    return aggregator


__all__ = [
    "make_fault_count_aggregator",
    "make_validation_signal_aggregator",
    "make_inverse_loss_aggregator",
]
