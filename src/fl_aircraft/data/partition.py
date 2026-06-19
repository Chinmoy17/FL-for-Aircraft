"""Partition engines into simulated airline-client shards.

A "client" owns a set of physical engines (both train and test trajectories from
those engines), mirroring how a real airline would own its fleet. Phase 0a uses
a stratified-by-lifetime split on FD001 so each of the 4 clients sees a
balanced mix of short-, medium-, and long-lived engines — non-trivial Non-IID
by wear distribution without being adversarial.

Phase 6 adds :func:`partition_by_subset_halves`, which splits each of a list
of CMAPSS subsets into equal halves so each client receives engines from
exactly one fault-mode family (e.g. clients 1–2 = FD001 HPC-only, clients 3–4
= FD003 HPC+Fan). This is the structurally Non-IID partition that motivates
the federation.
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


def partition_by_subset_halves(
    train_df: pd.DataFrame,
    subsets: list[str] | tuple[str, ...],
    *,
    n_clients_per_subset: int = 2,
    subset_col: str = "source_subset",
    client_prefix: str = "client",
    seed: int = 42,
) -> list[ClientShard]:
    """For each subset, split its engines into ``n_clients_per_subset`` equal shards.

    Used by the Phase 6 Non-IID baseline. For the default of 2 subsets × 2
    clients each, the four returned shards represent:

        client_1 = subset[0] first half
        client_2 = subset[0] second half
        client_3 = subset[1] first half
        client_4 = subset[1] second half

    The engine assignment inside each subset is shuffled with ``seed`` so the
    split is reproducible but not lexicographic.

    Args:
        train_df: DataFrame produced by ``load_multi_subset_bundle()`` (must
            have a ``source_subset`` column).
        subsets: Subset names in the order matching the bundle's
            ``MultiSubsetConfig.subsets``.
        n_clients_per_subset: How many clients to split each subset's engines
            across. Defaults to 2.
        subset_col: Name of the column holding each row's origin subset.
        client_prefix / seed: Same meaning as in :func:`partition_by_lifetime`.
    """
    if not subsets:
        raise ValueError("subsets must be non-empty.")
    if n_clients_per_subset < 1:
        raise ValueError(
            f"n_clients_per_subset must be >= 1, got {n_clients_per_subset}."
        )
    if subset_col not in train_df.columns:
        raise ValueError(
            f"train_df is missing the {subset_col!r} column — was it built by "
            "load_multi_subset_bundle()?"
        )

    rng = np.random.default_rng(seed)
    shards: list[ClientShard] = []
    next_id = 1
    for subset in subsets:
        engines = (
            train_df.loc[train_df[subset_col] == subset, UNIT_ID_COL].unique()
        )
        engines = np.array(sorted(int(u) for u in engines), dtype=np.int64)
        if engines.size == 0:
            raise ValueError(
                f"Subset {subset!r} has no engines in train_df — check the bundle's subsets list."
            )
        if engines.size < n_clients_per_subset:
            raise ValueError(
                f"Cannot split {engines.size} engines from {subset!r} across "
                f"{n_clients_per_subset} clients (each client must own >= 1 engine)."
            )
        rng.shuffle(engines)
        chunks = np.array_split(engines, n_clients_per_subset)
        for chunk in chunks:
            shards.append(
                ClientShard(
                    client_id=f"{client_prefix}_{next_id}",
                    unit_ids=tuple(sorted(int(u) for u in chunk)),
                )
            )
            next_id += 1
    return shards
