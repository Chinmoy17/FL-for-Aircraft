"""Federated learning client, server, aggregation rules, and simulation loop.

Public API::

    from fl_aircraft.fl import (
        FedAvgServer, ClientUpdate, fedavg_aggregate,
        FederatedClient,
        run_fedavg, run_fedavg_from_bundle,
        RoundRecord, FederatedHistory,
        build_federated_clients, build_federated_clients_from_bundle,
    )
"""
from __future__ import annotations

from .client import FederatedClient
from .server import ClientUpdate, FedAvgServer, fedavg_aggregate
from .simulation import (
    FederatedHistory,
    RoundRecord,
    build_federated_clients,
    build_federated_clients_from_bundle,
    run_fedavg,
    run_fedavg_from_bundle,
)

__all__ = [
    "ClientUpdate",
    "FedAvgServer",
    "FederatedClient",
    "FederatedHistory",
    "RoundRecord",
    "build_federated_clients",
    "build_federated_clients_from_bundle",
    "fedavg_aggregate",
    "run_fedavg",
    "run_fedavg_from_bundle",
]
