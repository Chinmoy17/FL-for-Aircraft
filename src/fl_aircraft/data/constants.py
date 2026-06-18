"""Static facts about the C-MAPSS dataset and shared preprocessing defaults.

All values here are empirically validated in ``notebooks/01_eda_cmapss.ipynb``.
Keeping them in one module makes the rest of the data pipeline parameter-free.
"""
from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Column schema — 26 columns per row of every CMAPSS train/test file.
# ---------------------------------------------------------------------------
UNIT_ID_COL: Final[str] = "unit_id"
CYCLE_COL: Final[str] = "cycle"
OP_SETTING_COLS: Final[list[str]] = [f"os_{i}" for i in range(1, 4)]
SENSOR_COLS: Final[list[str]] = [f"s_{i}" for i in range(1, 22)]
COLUMNS: Final[list[str]] = [UNIT_ID_COL, CYCLE_COL] + OP_SETTING_COLS + SENSOR_COLS

SUBSETS: Final[tuple[str, ...]] = ("FD001", "FD002", "FD003", "FD004")

# ---------------------------------------------------------------------------
# Sensors that carry no informative signal — Asif et al. (2022) and the broader
# CMAPSS community standard. Cross-checked against the EDA notebook:
#   - FD001/FD003: a global std<1e-4 filter catches 6 of the 7 listed sensors;
#     sensor 6 is near-constant (std ~ 1e-3 to 2e-2) but carries no degradation
#     signal, so the literature drop list is accepted as-is.
#   - FD002/FD004: regime variation dominates the global std, so the literature
#     list is derived from per-regime variance. Phase 1 uses it as the default
#     and Phase 2+ can re-validate it inside each KMeans regime.
# ---------------------------------------------------------------------------
CONSTANT_SENSORS_PER_SUBSET: Final[dict[str, frozenset[int]]] = {
    "FD001": frozenset({1, 5, 6, 10, 16, 18, 19}),
    "FD002": frozenset({10, 13, 16, 18, 19}),
    "FD003": frozenset({1, 5, 6, 10, 16, 18, 19}),
    "FD004": frozenset({10, 13, 16, 18, 19}),
}

# ---------------------------------------------------------------------------
# Preprocessing defaults — overridable via configs/*.yaml later.
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_SIZE: Final[int] = 30
DEFAULT_STRIDE: Final[int] = 1
DEFAULT_RUL_CAP: Final[int] = 125  # piecewise-linear cap (standard for CMAPSS)
DEFAULT_FAULT_THRESHOLD: Final[int] = 30  # RUL <= 30 => binary fault label = 1


def informative_sensors(subset: str) -> list[str]:
    """Return the sensor column names retained for a subset (sensor-index order)."""
    if subset not in CONSTANT_SENSORS_PER_SUBSET:
        raise ValueError(
            f"Unknown subset {subset!r}; expected one of {sorted(CONSTANT_SENSORS_PER_SUBSET)}."
        )
    dropped = CONSTANT_SENSORS_PER_SUBSET[subset]
    return [f"s_{i}" for i in range(1, 22) if i not in dropped]
