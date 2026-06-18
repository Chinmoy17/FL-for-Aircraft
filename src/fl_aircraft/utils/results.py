"""Structured, machine-readable phase-result writer.

Every phase of the project (EDA, data pipeline sanity, smoke run, centralized
baseline, FedAvg, RQ experiments, …) writes a single ``metrics.json`` file into
its own folder under ``results/NN_<phase>/``. ``scripts/build_results_summary.py``
aggregates them into ``results/summary.json``, which is the **only** file the
React frontend will need to fetch.

Schema (all fields except ``phase_id`` and ``phase_name`` are optional — each
phase populates whatever applies):

    {
      "phase_id":       "02_smoke",
      "phase_name":     "Phase 2 — Centralized smoke run",
      "generated_at":   "2026-06-18T12:34:56+00:00",
      "interpretation": "...",
      "artifacts":      {"loss_curve_png": "results/02_smoke/loss_curve.png", ...},
      "subset":         "FD001",
      "config":         {...},
      "timing":         {...},
      "summary":        {...},
      "train":          {...},
      "test":           {"rul": {...}, "fault": {...}},
      "per_subset":     {"FD001": {...}, ...},   # used by EDA
      "per_client":     {"client_1": {...}, ...} # used by P1 / P4 / P5 / RQs
    }

Numpy scalars and arrays are coerced to native Python via ``_NumpyJSONEncoder``
so callers can pass DataFrames / ndarrays directly without manual conversion.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars / arrays and Path objects."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401 (encoder contract)
        # numpy scalars expose .item() returning a native Python scalar.
        if hasattr(obj, "item") and callable(obj.item) and not isinstance(obj, type):
            try:
                return obj.item()
            except Exception:
                pass
        # numpy arrays expose .tolist().
        if hasattr(obj, "tolist") and callable(obj.tolist):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj).replace("\\", "/")
        return super().default(obj)


@dataclass
class PhaseMetrics:
    """Structured per-phase results container.

    Args:
        phase_id: Folder-safe identifier, e.g. ``"02_smoke"``. **Must** match the
            folder name under ``results/`` for aggregation to find it.
        phase_name: Human-readable title.
        interpretation: One-paragraph summary suitable for tooltips in the frontend.
        artifacts: Map of label → workspace-relative path (forward slashes).
            Used by the frontend to load images / CSVs on demand.
        subset: CMAPSS subset for single-subset phases (``"FD001"`` …).
        config: Hyperparameters, data dimensions, seed.
        timing: Wall-clock numbers (``train_seconds``, ``per_epoch`` …).
        summary: Free-form top-level numbers that don't fit train/test (e.g. EDA
            "total engines", "fault positive rate global").
        train: Training history. Convention: ``{"epochs": [{"epoch": 1, ...}, ...]}``.
        test: Final-evaluation metrics. Convention: ``{"rul": {...}, "fault": {...}}``.
        per_subset: Per-subset breakdowns (EDA).
        per_client: Per-client breakdowns (P1, P4, P5, RQ2/5/3).
    """

    phase_id: str
    phase_name: str
    interpretation: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    subset: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    test: dict[str, Any] = field(default_factory=dict)
    per_subset: dict[str, Any] = field(default_factory=dict)
    per_client: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.phase_id:
            raise ValueError("phase_id is required.")
        if not self.phase_name:
            raise ValueError("phase_name is required.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        return payload


def dump_phase_metrics(metrics: PhaseMetrics, out_dir: Path) -> Path:
    """Write ``metrics.json`` into ``out_dir`` (creating it if needed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    payload = metrics.to_dict()
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, cls=_NumpyJSONEncoder, sort_keys=False)
    return out_path


def load_phase_metrics(metrics_path: Path) -> dict[str, Any]:
    """Read a previously written ``metrics.json``."""
    with Path(metrics_path).open(encoding="utf-8") as fh:
        return json.load(fh)


def build_summary(
    results_root: Path,
    project_name: str = "Federated Learning for Aircraft Engine PHM",
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Aggregate every ``results/<phase>/metrics.json`` into one summary dict.

    Phases are included in lexicographic order of their folder name (which is
    why the numeric prefix matters — it preserves chronological ordering).
    """
    results_root = Path(results_root)
    phases: dict[str, Any] = {}
    if results_root.is_dir():
        for child in sorted(results_root.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            metrics_path = child / "metrics.json"
            if metrics_path.exists():
                phases[child.name] = load_phase_metrics(metrics_path)
    return {
        "project": project_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "phases": phases,
    }


def dump_summary(summary: dict[str, Any], out_path: Path) -> Path:
    """Write the aggregated summary to ``out_path`` (typically ``results/summary.json``)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyJSONEncoder, sort_keys=False)
    return out_path
