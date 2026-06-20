"""Aviation maintenance ontology for the C-MAPSS turbofan engine.

Two structured tables:

1. :data:`SENSOR_ONTOLOGY` maps each model input column to its physical
   meaning, the engine subsystem it observes, and how it typically behaves
   under each known fault mode. Sources: Saxena et al., *Damage Propagation
   Modeling for Aircraft Engine Run-to-Failure Simulation* (PHM 2008); the
   CMAPSS sensor description table; cross-checked against the EDA notebook.

2. :data:`FAULT_MODE_RULES` maps fault-mode signatures to recommended
   maintenance actions. Each rule lists the sensors whose **positive**
   contributions are diagnostic for that fault mode (i.e. their reading
   change drives the model to *lower* RUL — see also :mod:`narrative`).

This module is **knowledge-base only** — no PyTorch, no numpy. It can be
imported in any context (frontend, backend, notebook) without bringing in
heavy ML dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------------------
# Sensor ontology
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorMeta:
    """Physical metadata for one input feature.

    Attributes:
        column: The column name used by the data pipeline (e.g. ``"s_3"``).
        cmapss_name: The official CMAPSS sensor short name (e.g. ``"T30"``).
        description: Human-readable description of what is measured.
        subsystem: The engine subsystem the sensor observes
            (``"HPC"``, ``"LPT"``, ``"Fan"``, ``"Core"``, ``"Operational"``, …).
        relevance: Which fault modes this sensor is informative for.
        unit: Physical unit, where applicable.
    """

    column: str
    cmapss_name: str
    description: str
    subsystem: str
    relevance: tuple[str, ...]
    unit: str = ""


SENSOR_ONTOLOGY: Final[dict[str, SensorMeta]] = {
    # ---------- Operational settings ----------
    "os_1": SensorMeta(
        column="os_1",
        cmapss_name="altitude",
        description="Flight altitude",
        subsystem="Operational",
        relevance=(),
        unit="kft",
    ),
    "os_2": SensorMeta(
        column="os_2",
        cmapss_name="Mach",
        description="Airspeed (Mach number)",
        subsystem="Operational",
        relevance=(),
    ),
    "os_3": SensorMeta(
        column="os_3",
        cmapss_name="TRA",
        description="Throttle resolver angle",
        subsystem="Operational",
        relevance=(),
        unit="deg",
    ),
    # ---------- Informative sensors (14 retained for FD001/FD003) ----------
    "s_2": SensorMeta(
        column="s_2", cmapss_name="T24",
        description="Total temperature at LPC outlet",
        subsystem="LPC", relevance=("HPC degradation",), unit="°R",
    ),
    "s_3": SensorMeta(
        column="s_3", cmapss_name="T30",
        description="Total temperature at HPC outlet",
        subsystem="HPC",
        relevance=("HPC degradation",),
        unit="°R",
    ),
    "s_4": SensorMeta(
        column="s_4", cmapss_name="T50",
        description="Total temperature at LPT outlet",
        subsystem="LPT",
        relevance=("HPC degradation", "Fan degradation"),
        unit="°R",
    ),
    "s_7": SensorMeta(
        column="s_7", cmapss_name="P30",
        description="Total pressure at HPC outlet",
        subsystem="HPC",
        relevance=("HPC degradation",),
        unit="psia",
    ),
    "s_8": SensorMeta(
        column="s_8", cmapss_name="Nf",
        description="Physical fan speed",
        subsystem="Fan",
        relevance=("Fan degradation",),
        unit="rpm",
    ),
    "s_9": SensorMeta(
        column="s_9", cmapss_name="Nc",
        description="Physical core speed",
        subsystem="Core",
        relevance=("HPC degradation",),
        unit="rpm",
    ),
    "s_11": SensorMeta(
        column="s_11", cmapss_name="Ps30",
        description="Static pressure at HPC outlet",
        subsystem="HPC",
        relevance=("HPC degradation",),
        unit="psia",
    ),
    "s_12": SensorMeta(
        column="s_12", cmapss_name="phi",
        description="Ratio of fuel flow to Ps30",
        subsystem="Combustion",
        relevance=("HPC degradation",),
        unit="pps/psi",
    ),
    "s_13": SensorMeta(
        column="s_13", cmapss_name="NRf",
        description="Corrected fan speed",
        subsystem="Fan",
        relevance=("Fan degradation",),
        unit="rpm",
    ),
    "s_14": SensorMeta(
        column="s_14", cmapss_name="NRc",
        description="Corrected core speed",
        subsystem="Core",
        relevance=("HPC degradation",),
        unit="rpm",
    ),
    "s_15": SensorMeta(
        column="s_15", cmapss_name="BPR",
        description="Bypass ratio",
        subsystem="Fan",
        relevance=("Fan degradation",),
    ),
    "s_17": SensorMeta(
        column="s_17", cmapss_name="htBleed",
        description="Bleed enthalpy",
        subsystem="Bleed",
        relevance=("HPC degradation",),
    ),
    "s_20": SensorMeta(
        column="s_20", cmapss_name="W31",
        description="HPT coolant bleed flow",
        subsystem="HPT",
        relevance=("HPC degradation",),
        unit="lbm/s",
    ),
    "s_21": SensorMeta(
        column="s_21", cmapss_name="W32",
        description="LPT coolant bleed flow",
        subsystem="LPT",
        relevance=("HPC degradation", "Fan degradation"),
        unit="lbm/s",
    ),
}


# ---------------------------------------------------------------------------
# Fault-mode rules
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FaultModeRule:
    """Maps a fault-mode signature to a recommended action.

    Attributes:
        fault_mode: Short label (``"HPC degradation"`` / ``"Fan degradation"`` / …).
        indicator_sensors: Sensors whose contribution to a *low* RUL
            prediction implies this fault mode. The first element is the
            most diagnostic.
        affected_components: Engine components a ground engineer should
            inspect when this fault mode is suspected.
        recommended_action: Plain-English maintenance recommendation.
        confidence_label: Free-form severity tag for the explanation.
    """

    fault_mode: str
    indicator_sensors: tuple[str, ...]
    affected_components: tuple[str, ...]
    recommended_action: str
    confidence_label: str = "moderate"


FAULT_MODE_RULES: Final[tuple[FaultModeRule, ...]] = (
    FaultModeRule(
        fault_mode="HPC degradation",
        indicator_sensors=("s_3", "s_11", "s_7", "s_12", "s_9", "s_20"),
        affected_components=(
            "High-Pressure Compressor (HPC) rotor and stator assembly",
            "HPC outlet temperature probe (T30)",
            "HPC bleed valve",
        ),
        recommended_action=(
            "Schedule HPC borescope inspection and verify HPC outlet "
            "temperature probe calibration within the next 20 operating cycles."
        ),
    ),
    FaultModeRule(
        fault_mode="Fan degradation",
        indicator_sensors=("s_15", "s_8", "s_13"),
        affected_components=(
            "Fan blades (leading-edge erosion / FOD)",
            "Fan exit guide vanes",
            "Bypass duct",
        ),
        recommended_action=(
            "Schedule fan-stage visual inspection and bypass-ratio "
            "verification within the next 20 operating cycles."
        ),
    ),
    FaultModeRule(
        fault_mode="LPT efficiency loss (secondary)",
        indicator_sensors=("s_4", "s_21"),
        affected_components=(
            "Low-Pressure Turbine (LPT) nozzle ring",
            "LPT cooling-air supply",
        ),
        recommended_action=(
            "Cross-check HPC and Fan inspections; if both are normal, "
            "inspect LPT cooling-air manifold."
        ),
        confidence_label="secondary",
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def lookup_sensor(column: str) -> SensorMeta:
    """Return the :class:`SensorMeta` for a feature column.

    Raises ``KeyError`` if the column is not in the ontology.
    """
    if column not in SENSOR_ONTOLOGY:
        raise KeyError(
            f"No ontology entry for column {column!r}. "
            f"Known columns: {sorted(SENSOR_ONTOLOGY)}"
        )
    return SENSOR_ONTOLOGY[column]


def match_fault_mode(top_sensors: list[str]) -> FaultModeRule | None:
    """Pick the fault-mode rule whose indicator sensors best overlap ``top_sensors``.

    Args:
        top_sensors: Column names ordered by descending contribution magnitude.

    Returns:
        The :class:`FaultModeRule` with the highest overlap score, or ``None``
        if no rule matches at all (e.g. all top sensors are operational settings).

    The score gives more weight to the **earlier** entries in ``top_sensors``
    (the strongest contributors) via reciprocal-rank weighting.
    """
    if not top_sensors:
        return None
    best_rule: FaultModeRule | None = None
    best_score = 0.0
    for rule in FAULT_MODE_RULES:
        score = 0.0
        for rank, col in enumerate(top_sensors, start=1):
            if col in rule.indicator_sensors:
                # Weight: 1 / rank => first sensor is 1.0, fifth is 0.2
                score += 1.0 / rank
        if score > best_score:
            best_score = score
            best_rule = rule
    return best_rule if best_score > 0 else None
