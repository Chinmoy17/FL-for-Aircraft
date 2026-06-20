"""Tests for the PhaseMetrics utility (JSON round-trip + numpy coercion + aggregator)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fl_aircraft.utils import (
    PhaseMetrics,
    build_summary,
    dump_phase_metrics,
    dump_summary,
    load_phase_metrics,
)


# ---------------------------------------------------------------------------
# PhaseMetrics dataclass
# ---------------------------------------------------------------------------
def test_phase_metrics_requires_id_and_name() -> None:
    with pytest.raises(ValueError):
        PhaseMetrics(phase_id="", phase_name="x")
    with pytest.raises(ValueError):
        PhaseMetrics(phase_id="x", phase_name="")


def test_phase_metrics_to_dict_adds_timestamp() -> None:
    m = PhaseMetrics(phase_id="02_smoke", phase_name="Smoke")
    d = m.to_dict()
    assert d["phase_id"] == "02_smoke"
    assert "generated_at" in d
    # ISO-8601 with timezone.
    assert "T" in d["generated_at"] and ("+" in d["generated_at"] or "Z" in d["generated_at"])


# ---------------------------------------------------------------------------
# Write / read round-trip
# ---------------------------------------------------------------------------
def test_round_trip_preserves_structured_payload(tmp_path: Path) -> None:
    m = PhaseMetrics(
        phase_id="02_smoke",
        phase_name="Phase 2 — Centralized smoke run",
        interpretation="Wiring confirmed.",
        subset="FD001",
        config={"epochs": 1, "lr": 1e-3},
        timing={"train_seconds": 1.5},
        test={
            "rul": {"rmse": 62.7, "mae": 52.7, "nasa_score": 45300.4},
            "fault": {"auprc": 0.845, "f1": 0.4},
        },
        artifacts={"loss_curve_png": "results/02_smoke/loss_curve.png"},
    )
    path = dump_phase_metrics(m, tmp_path)
    assert path.name == "metrics.json"

    loaded = load_phase_metrics(path)
    assert loaded["phase_id"] == "02_smoke"
    assert loaded["test"]["rul"]["rmse"] == pytest.approx(62.7)
    assert loaded["test"]["fault"]["auprc"] == pytest.approx(0.845)
    assert loaded["artifacts"]["loss_curve_png"].endswith("loss_curve.png")


def test_numpy_scalars_and_arrays_are_serialised(tmp_path: Path) -> None:
    m = PhaseMetrics(
        phase_id="t",
        phase_name="t",
        summary={
            "np_float": np.float32(1.5),
            "np_int": np.int64(7),
            "np_array": np.array([1, 2, 3]),
        },
    )
    path = dump_phase_metrics(m, tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["summary"]["np_float"] == pytest.approx(1.5)
    assert loaded["summary"]["np_int"] == 7
    assert loaded["summary"]["np_array"] == [1, 2, 3]


def test_path_objects_are_serialised_as_forward_slash_strings(tmp_path: Path) -> None:
    m = PhaseMetrics(
        phase_id="t",
        phase_name="t",
        artifacts={"path": Path("results") / "02_smoke" / "loss.png"},  # type: ignore[dict-item]
    )
    path = dump_phase_metrics(m, tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert "/" in loaded["artifacts"]["path"]
    assert "\\" not in loaded["artifacts"]["path"]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def test_build_summary_picks_up_numbered_phase_folders_in_order(tmp_path: Path) -> None:
    # Intentionally create out of order to confirm the aggregator sorts.
    for pid in ("02_smoke", "00_eda", "01_data"):
        dump_phase_metrics(PhaseMetrics(phase_id=pid, phase_name=pid), tmp_path / pid)
    # An ignored folder without metrics.json should be skipped silently.
    (tmp_path / "logs").mkdir()
    (tmp_path / "summary.json").write_text("ignored", encoding="utf-8")

    summary = build_summary(tmp_path)
    assert list(summary["phases"].keys()) == ["00_eda", "01_data", "02_smoke"]
    assert summary["project"]
    assert summary["generated_at"]


def test_build_summary_returns_empty_when_no_phases(tmp_path: Path) -> None:
    summary = build_summary(tmp_path)
    assert summary["phases"] == {}


def test_dump_summary_writes_to_path(tmp_path: Path) -> None:
    summary = {"project": "x", "phases": {}}
    out = dump_summary(summary, tmp_path / "summary.json")
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["project"] == "x"
