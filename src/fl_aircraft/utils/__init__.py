"""Shared utilities: seeding, structured logging, config loading, device."""

from .device import get_device, resolve_device
from .results import (
    PhaseMetrics,
    build_summary,
    dump_phase_metrics,
    dump_summary,
    load_phase_metrics,
)
from .seeding import seed_everything

__all__ = [
    "PhaseMetrics",
    "build_summary",
    "dump_phase_metrics",
    "dump_summary",
    "get_device",
    "load_phase_metrics",
    "resolve_device",
    "seed_everything",
]
