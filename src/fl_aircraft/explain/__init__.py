"""Sensor-level attribution, aviation maintenance ontology, and explanations.

This subpackage implements RQ3 of the project brief: turn the model's numeric
predictions into actionable, ontology-grounded explanations for a maintenance
engineer.

Three layers, each independently testable:

1. :mod:`fl_aircraft.explain.ontology` — static knowledge: what each sensor
   physically measures, which fault modes it indicates, recommended
   inspection actions.
2. :mod:`fl_aircraft.explain.attribution` — numeric attribution via
   Integrated Gradients (captum). Pure inference; reuses any trained
   checkpoint.
3. :mod:`fl_aircraft.explain.narrative` — combine numbers + ontology into a
   deterministic English explanation. Optional LLM rewrite layer is
   env-var-gated and never the source of truth.

Public API::

    from fl_aircraft.explain import (
        SensorMeta, SENSOR_ONTOLOGY, FAULT_MODE_RULES,
        AttributionResult, attribute_window, attribute_dataset,
        EngineExplanation, build_explanation, explain_window,
    )
"""
from __future__ import annotations

from .attribution import (
    AttributionResult,
    attribute_dataset,
    attribute_window,
)
from .checkpoint_catalog import (
    CheckpointSpec,
    WindowPair,
    available_checkpoints,
    candidate_checkpoints,
    find_engine,
    load_bundle,
    load_model,
    prepare_test_windows,
)
from .narrative import (
    EngineExplanation,
    build_explanation,
    explain_window,
)
from .ontology import (
    FAULT_MODE_RULES,
    SENSOR_ONTOLOGY,
    FaultModeRule,
    SensorMeta,
    lookup_sensor,
)

__all__ = [
    "AttributionResult",
    "CheckpointSpec",
    "EngineExplanation",
    "FAULT_MODE_RULES",
    "FaultModeRule",
    "SENSOR_ONTOLOGY",
    "SensorMeta",
    "WindowPair",
    "attribute_dataset",
    "attribute_window",
    "available_checkpoints",
    "build_explanation",
    "candidate_checkpoints",
    "explain_window",
    "find_engine",
    "load_bundle",
    "load_model",
    "lookup_sensor",
    "prepare_test_windows",
]
