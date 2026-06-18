"""Federated learning client, server, aggregation rules, and simulation loop.

Public API::

    from fl_aircraft.fl import (
        FedAvgServer, ClientUpdate, fedavg_aggregate,
        FederatedClient,
        run_fedavg, RoundRecord, FederatedHistory, build_federated_clients,
    )
"""
from __future__ import annotations

from .client import FederatedClient
from .server import ClientUpdate, FedAvgServer, fedavg_aggregate
from .simulation import (
    FederatedHistory,
    RoundRecord,
    build_federated_clients,
    run_fedavg,
)

__all__ = [
    "ClientUpdate",
    "FedAvgServer",
    "FederatedClient",
    "FederatedHistory",
    "RoundRecord",
    "build_federated_clients",
    "fedavg_aggregate",
    "run_fedavg",
]
