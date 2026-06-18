"""Partition engines into simulated airline-client shards.

A "client" owns a set of physical engines (both train and test trajectories from
those engines), mirroring how a real airline would own its fleet. Phase 0a uses
a stratified-by-lifetime split on FD001 so each of the 4 clients sees a
balanced mix of short-, medium-, and long-lived engines — non-trivial Non-IID
by wear distribution without being adversarial.

Phase 0b will add ``partition_by_subset`` to bind clients to (subset, slice)
pairs so each client sees a deliberately different fault-mode mix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import CYCLE_COL, UNIT_ID_COL


@dataclass(frozen=True)
class ClientShard:
    """A subset of engine ids assigned to one simulated FL client."""

    client_id: str
    unit_ids: tuple[int, ...]

    @property
    def n_engines(self) -> int:
        return len(self.unit_ids)

    def __len__(self) -> int:
        return self.n_engines


def partition_by_lifetime(
    train_df: pd.DataFrame,
    n_clients: int,
    seed: int = 42,
    client_prefix: str = "client",
) -> list[ClientShard]:
    """Stratified split by total engine lifetime.

    Algorithm
    ---------
    1. Sort engines by ``max(cycle)`` ascending.
    2. Walk the sorted list in chunks of ``n_clients`` consecutive engines.
    3. Within each chunk, shuffle the per-client assignment order with ``seed``
       so the deal is reproducible but not lexicographic.

    This guarantees every client receives a balanced mix of short, medium and
    long-lived engines while keeping the assignment deterministic for a given
    ``seed``.
    """
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}.")

    lifetimes = train_df.groupby(UNIT_ID_COL)[CYCLE_COL].max().sort_values(ascending=True)
    sorted_units = lifetimes.index.to_numpy()
    if len(sorted_units) == 0:
        raise ValueError("train_df contains no engines to partition.")
    if len(sorted_units) < n_clients:
        raise ValueError(
            f"Cannot split {len(sorted_units)} engines across {n_clients} clients "
            f"(each client must own at least one engine)."
        )

    rng = np.random.default_rng(seed)
    assignments: list[list[int]] = [[] for _ in range(n_clients)]
    for start in range(0, len(sorted_units), n_clients):
        chunk = sorted_units[start : start + n_clients]
        order = rng.permutation(len(chunk))
        for client_idx, unit_id in zip(order, chunk):
            assignments[int(client_idx)].append(int(unit_id))

    return [
        ClientShard(
            client_id=f"{client_prefix}_{i + 1}",
            unit_ids=tuple(sorted(units)),
        )
        for i, units in enumerate(assignments)
    ]


def slice_for_client(df: pd.DataFrame, shard: ClientShard) -> pd.DataFrame:
    """Return the subset of rows in ``df`` belonging to ``shard``'s engines."""
    return df.loc[df[UNIT_ID_COL].isin(shard.unit_ids)].copy()
