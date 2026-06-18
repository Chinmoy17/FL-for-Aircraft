"""Federated learning client, server, aggregation rules, and simulation loop.

Public API::

    from fl_aircraft.fl import (
        FedAvgServer, ClientUpdate, fedavg_aggregate,
        FederatedClient,
        run_fedavg, run_fedavg_from_bundle,
        RoundRecord, FederatedHistory,
        build_federated_clients, build_federated_clients_from_bundle,
        # RQ2 imbalance-aware aggregation
        make_fault_count_aggregator,
        make_validation_signal_aggregator,
        make_inverse_loss_aggregator,
        run_fedavg_imbalance_aware,
        ImbalanceAwareHistory,
        build_imbalance_aware_clients,
    )
"""
from __future__ import annotations

from .aggregators import (
    make_fault_count_aggregator,
    make_inverse_loss_aggregator,
    make_validation_signal_aggregator,
)
from .client import FederatedClient
from .imbalance_aware import (
    ImbalanceAwareHistory,
    build_imbalance_aware_clients,
    run_fedavg_imbalance_aware,
)
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
    "ImbalanceAwareHistory",
    "RoundRecord",
    "build_federated_clients",
    "build_federated_clients_from_bundle",
    "build_imbalance_aware_clients",
    "fedavg_aggregate",
    "make_fault_count_aggregator",
    "make_inverse_loss_aggregator",
    "make_validation_signal_aggregator",
    "run_fedavg",
    "run_fedavg_from_bundle",
    "run_fedavg_imbalance_aware",
]
