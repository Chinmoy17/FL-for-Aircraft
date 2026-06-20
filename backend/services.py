"""Business-logic layer for the backend.

Loads checkpoints lazily, caches them per-process, and runs Integrated
Gradients on demand. Keeps the route handlers in
:mod:`backend.main` thin.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Make ``src/`` importable so this module works no matter how uvicorn is launched.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_aircraft.data import TrainTestBundle  # noqa: E402
from fl_aircraft.explain import (  # noqa: E402
    CheckpointSpec,
    WindowPair,
    attribute_window,
    available_checkpoints,
    build_explanation,
    find_engine,
    load_bundle,
    load_model,
    prepare_test_windows,
)
from fl_aircraft.explain.narrative import rewrite_with_llm  # noqa: E402
from fl_aircraft.models import MultiTaskCNN  # noqa: E402

DATA_DIR = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
SUMMARY_PATH = REPO_ROOT / "results" / "summary.json"
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Per-process cache. Building a multi-subset bundle is ~3 s; loading a model
# is < 50 ms; preparing test windows is ~0.5 s. We only do these once per
# checkpoint per process lifetime.
# ---------------------------------------------------------------------------
class _CacheEntry:
    __slots__ = ("spec", "bundle", "model", "windows")

    def __init__(
        self,
        spec: CheckpointSpec,
        bundle: TrainTestBundle,
        model: MultiTaskCNN,
        windows: list[WindowPair],
    ) -> None:
        self.spec = spec
        self.bundle = bundle
        self.model = model
        self.windows = windows


_cache: dict[str, _CacheEntry] = {}
_cache_lock = threading.Lock()


def list_specs() -> list[CheckpointSpec]:
    """Discover every checkpoint that actually has a .pt file on disk."""
    return available_checkpoints(REPO_ROOT)


def get_spec(key: str) -> CheckpointSpec:
    """Find one CheckpointSpec by key, raising KeyError if absent."""
    for s in list_specs():
        if s.key == key:
            return s
    raise KeyError(f"Unknown checkpoint key: {key!r}")


def _get_entry(spec: CheckpointSpec) -> _CacheEntry:
    """Lazy-load the bundle + model + windows, cached per process."""
    cached = _cache.get(spec.key)
    if cached is not None:
        return cached
    with _cache_lock:
        # Double-checked locking: another thread may have populated it
        # while we waited for the lock.
        cached = _cache.get(spec.key)
        if cached is not None:
            return cached
        bundle = load_bundle(spec, DATA_DIR)
        model = load_model(spec, bundle)
        windows = prepare_test_windows(bundle)
        entry = _CacheEntry(spec=spec, bundle=bundle, model=model, windows=windows)
        _cache[spec.key] = entry
        return entry


# ---------------------------------------------------------------------------
# Public service methods (called from main.py route handlers)
# ---------------------------------------------------------------------------
def warm_cache() -> dict[str, int]:
    """Pre-load every available checkpoint. Called from FastAPI startup."""
    counts: dict[str, int] = {}
    for spec in list_specs():
        entry = _get_entry(spec)
        counts[spec.key] = len(entry.windows)
    return counts


def get_summary() -> dict[str, Any]:
    """Return the contents of ``results/summary.json``."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "results/summary.json not found — run scripts/build_results_summary.py first."
        )
    with SUMMARY_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def list_engines(spec: CheckpointSpec) -> list[dict[str, Any]]:
    """Return ``[{engine_id, subset, true_rul, true_fault}, …]`` for the spec."""
    entry = _get_entry(spec)
    return [
        {
            "engine_id": p.engine_id,
            "subset": p.subset,
            "true_rul": p.rul_true,
            "true_fault": p.fault_true,
        }
        for p in entry.windows
    ]


def predict(
    spec: CheckpointSpec,
    engine_id: int,
    *,
    top_k: int = 5,
    use_llm: bool = False,
) -> dict[str, Any]:
    """Run attribution on one (checkpoint, engine) pair and return the explanation."""
    entry = _get_entry(spec)
    pair = find_engine(entry.windows, engine_id)
    if pair is None:
        raise KeyError(
            f"Engine {engine_id} not in checkpoint {spec.key!r} test set "
            f"(have {len(entry.windows)} engines)."
        )

    # 1) Numeric forward pass for the headline numbers.
    with torch.no_grad():
        x = torch.from_numpy(pair.window_normalized.astype(np.float32)).unsqueeze(0)
        pred = entry.model(x)
        predicted_rul = float(pred.rul.item())
        fault_prob = float(pred.fault_probs().item())

    # 2) Integrated Gradients for sensor-level attribution (~ 5 s on CPU).
    attr = attribute_window(
        entry.model,
        pair.window_normalized,
        entry.bundle.feature_cols,
        target_head="rul",
        n_steps=50,
    )

    # 3) Build the deterministic explanation + optional LLM polish.
    explanation = build_explanation(
        attr,
        predicted_rul=predicted_rul,
        fault_probability=fault_prob,
        top_k=top_k,
    )
    if use_llm:
        polished = rewrite_with_llm(explanation.narrative)
        if polished:
            explanation.narrative_llm = polished

    return {
        "checkpoint_key": spec.key,
        "checkpoint_display_name": spec.display_name,
        "engine_id": pair.engine_id,
        "subset": pair.subset,
        "true_rul": pair.rul_true,
        "true_fault": pair.fault_true,
        "explanation": explanation.to_dict(),
    }


def resolve_figure_path(rel_path: str) -> Path:
    """Resolve a request path under ``results/`` while blocking traversal."""
    # Reject absolute paths or paths with parent components outright.
    candidate = (RESULTS_DIR / rel_path).resolve()
    try:
        candidate.relative_to(RESULTS_DIR.resolve())
    except ValueError as exc:
        raise PermissionError(f"Path escapes results/ directory: {rel_path!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"No such figure: {rel_path!r}")
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg"}:
        raise PermissionError(
            f"Only image files may be served; got suffix {candidate.suffix!r}."
        )
    return candidate
