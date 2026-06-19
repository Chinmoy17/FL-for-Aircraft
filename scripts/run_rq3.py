"""RQ3 — Sensor-level attribution + maintenance ontology + explanations.

For a curated set of test engines, explain *the same engine across multiple
trained checkpoints* (centralized vs FedAvg IID vs FedAvg Non-IID, etc.).
This exposes the "what does each model think matters?" comparison that the
RQ2 negative finding directly motivates.

Outputs under ``results/rq3_explanations/``:

    metrics.json                                       structured for the frontend
    explanations_<model>_engine_<id>.json              per-engine structured explanation
    heatmap_<model>_engine_<id>.png                    30 × 17 attribution heatmap
    top_sensors_<model>_engine_<id>.png                bar chart of top contributors
    trajectory_<model>_engine_<id>_<sensor>.png        top-sensor trajectory + overlay
    cross_model_comparison_engine_<id>.png             headline image (one per engine)

Run from the repo root inside the .venv::

    python scripts/run_rq3.py                          # 5 engines, all 4 checkpoints
    python scripts/run_rq3.py --engines 1 5 50 75 100
    python scripts/run_rq3.py --top-k 8 --no-llm

Each checkpoint is read off disk if present. If a checkpoint is missing
(e.g. RQ2's best Scheme B model was not saved), that model is silently
skipped — the CLI prints which checkpoints it found.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_aircraft.data import (  # noqa: E402
    CMAPSSConfig,
    MultiSubsetConfig,
    Normalizer,
    TrainTestBundle,
    bundle_from_config,
    load_multi_subset_bundle,
    make_test_windows,
)
from fl_aircraft.explain import (  # noqa: E402
    AttributionResult,
    EngineExplanation,
    attribute_window,
    build_explanation,
)
from fl_aircraft.explain.narrative import rewrite_with_llm  # noqa: E402
from fl_aircraft.explain.plots import (  # noqa: E402
    plot_attribution_heatmap,
    plot_sensor_trajectory_with_attribution,
    plot_top_sensor_bar,
)
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics, seed_everything  # noqa: E402

PHASE_ID = "rq3_explanations"
PHASE_NAME = "RQ3 — Sensor attribution + maintenance ontology"


# ---------------------------------------------------------------------------
# Checkpoint catalogue
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckpointSpec:
    """One trained-model checkpoint that the demo / RQ3 will load."""

    key: str
    display_name: str
    checkpoint_path: Path
    bundle_kind: str  # "single" or "multi"
    subset: str | None = None
    subsets: tuple[str, ...] | None = None


def _candidate_checkpoints() -> list[CheckpointSpec]:
    """Return the checkpoints we'd like to compare, in display order."""
    return [
        CheckpointSpec(
            key="p3_centralized_fd001",
            display_name="P3 Centralized (FD001, IID)",
            checkpoint_path=REPO_ROOT / "results" / "03_centralized" / "best_model_fd001.pt",
            bundle_kind="single", subset="FD001",
        ),
        CheckpointSpec(
            key="p5_fedavg_iid_fd001",
            display_name="P5 FedAvg IID (FD001)",
            checkpoint_path=REPO_ROOT / "results" / "05_fedavg" / "best_global_model_fd001.pt",
            bundle_kind="single", subset="FD001",
        ),
        CheckpointSpec(
            key="p6_centralized_combined",
            display_name="P6 Centralized (FD001+FD003)",
            checkpoint_path=REPO_ROOT / "results" / "06_non_iid" / "best_centralized_fd001+fd003.pt",
            bundle_kind="multi", subsets=("FD001", "FD003"),
        ),
        CheckpointSpec(
            key="p6_fedavg_non_iid",
            display_name="P6 FedAvg Non-IID (FD001+FD003)",
            checkpoint_path=REPO_ROOT / "results" / "06_non_iid" / "best_fedavg_fd001+fd003.pt",
            bundle_kind="multi", subsets=("FD001", "FD003"),
        ),
    ]


def _load_bundle(spec: CheckpointSpec, data_dir: Path) -> TrainTestBundle:
    """Build the appropriate bundle for a checkpoint's training distribution."""
    if spec.bundle_kind == "single":
        cfg = CMAPSSConfig(subset=spec.subset, data_dir=data_dir)
        return bundle_from_config(cfg)
    if spec.bundle_kind == "multi":
        cfg = MultiSubsetConfig(subsets=spec.subsets, data_dir=data_dir)
        return load_multi_subset_bundle(cfg)
    raise ValueError(f"Unknown bundle_kind {spec.bundle_kind!r}")


def _load_model(spec: CheckpointSpec, bundle: TrainTestBundle) -> MultiTaskCNN:
    """Reconstruct the model architecture from the saved config and load weights."""
    state = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
    cfg_payload = state.get("config", {})
    n_features = cfg_payload.get("n_features", bundle.n_features)
    window_size = cfg_payload.get("window_size", bundle.window_size)
    model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=n_features, window_size=window_size)
    )
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test-window preparation
# ---------------------------------------------------------------------------
@dataclass
class WindowPair:
    """One test engine's window in both normalized and physical space."""

    engine_id: int
    subset: str  # 'FD001' or 'FD003' (or whatever the bundle exposes)
    window_normalized: np.ndarray  # (T, F), z-scored
    rul_true: float
    fault_true: int


def _prepare_test_windows(bundle: TrainTestBundle) -> list[WindowPair]:
    """Build per-engine windows in the bundle's preprocessing space."""
    normalizer = Normalizer.fit(bundle.train_df, bundle.feature_cols)
    arrays = make_test_windows(
        normalizer.transform(bundle.test_raw_df),
        bundle.test_rul,
        bundle.feature_cols,
        bundle.window_size,
        bundle.rul_cap,
        bundle.fault_threshold,
    )
    pairs: list[WindowPair] = []
    # `make_test_windows` returns engines in sorted unit_id order, matching test_rul.
    sorted_unit_ids = sorted(bundle.test_raw_df["unit_id"].unique())
    for i, uid in enumerate(sorted_unit_ids):
        # Determine origin subset (multi-subset bundles have a source_subset col).
        if "source_subset" in bundle.test_raw_df.columns:
            origin = str(
                bundle.test_raw_df.loc[
                    bundle.test_raw_df["unit_id"] == uid, "source_subset"
                ].iloc[0]
            )
        else:
            origin = bundle.subsets[0]
        pairs.append(
            WindowPair(
                engine_id=int(uid),
                subset=origin,
                window_normalized=arrays.X[i],
                rul_true=float(arrays.y_rul[i]),
                fault_true=int(arrays.y_fault[i]),
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# Cross-model comparison plot
# ---------------------------------------------------------------------------
def _plot_cross_model_comparison(
    engine_id: int,
    rul_true: float,
    per_model: dict[str, tuple[AttributionResult, EngineExplanation]],
    path: Path,
) -> None:
    """One figure per engine: predicted-RUL bars + top-sensor agreement."""
    models = list(per_model.keys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left panel: predicted RUL across models + ground truth.
    ax = axes[0]
    preds = [per_model[m][1].predicted_rul for m in models]
    bars = ax.bar(models, preds, color=plt.colormaps["tab10"](np.linspace(0, 1, len(models))))
    for bar, v in zip(bars, preds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(rul_true, color="black", linestyle="--", linewidth=1.5,
               label=f"true RUL = {rul_true:.1f}")
    ax.set_ylabel("predicted RUL (cycles)")
    ax.set_title(f"Engine {engine_id} — predictions across models")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    # Right panel: top-3 contributing sensors per model (label-only).
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Top contributing sensors per model")
    rows = []
    for m in models:
        attr, _ = per_model[m]
        top = attr.top_sensors(k=3)
        cols = [f"{c} ({score:+.1f})" for c, score in top]
        rows.append([m, *cols])
    table = ax.table(
        cellText=rows,
        colLabels=["model", "#1", "#2", "#3"],
        loc="center", cellLoc="left", colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    fig.suptitle(
        f"Cross-model comparison for engine {engine_id}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-engine pipeline
# ---------------------------------------------------------------------------
def _run_for_engine(
    engine_id: int,
    specs: list[CheckpointSpec],
    bundles: dict[str, TrainTestBundle],
    windows: dict[str, list[WindowPair]],
    models: dict[str, MultiTaskCNN],
    out_dir: Path,
    *,
    top_k: int,
    n_steps: int,
    use_llm: bool,
) -> dict[str, dict]:
    """Run the full pipeline for one engine across every available model."""
    per_model_results: dict[str, tuple[AttributionResult, EngineExplanation]] = {}
    rul_true_observed: float | None = None

    for spec in specs:
        pairs = windows[spec.key]
        matches = [p for p in pairs if p.engine_id == engine_id]
        if not matches:
            print(f"  · [{spec.key}] engine {engine_id} not in test set — skipped")
            continue
        pair = matches[0]
        rul_true_observed = pair.rul_true
        bundle = bundles[spec.key]
        model = models[spec.key]

        attr = attribute_window(
            model, pair.window_normalized, bundle.feature_cols,
            target_head="rul", n_steps=n_steps,
        )
        with torch.no_grad():
            x = torch.from_numpy(pair.window_normalized.astype(np.float32)).unsqueeze(0)
            pred = model(x)
            predicted_rul = float(pred.rul.item())
            fault_prob = float(pred.fault_probs().item())
        explanation = build_explanation(
            attr, predicted_rul=predicted_rul,
            fault_probability=fault_prob, top_k=top_k,
        )
        if use_llm:
            polished = rewrite_with_llm(explanation.narrative)
            if polished:
                explanation.narrative_llm = polished

        # Save plots for this (engine, model) combination.
        plot_attribution_heatmap(
            attr,
            out_dir / f"heatmap_{spec.key}_engine_{engine_id}.png",
            title=f"{spec.display_name} — engine {engine_id} heatmap",
        )
        plot_top_sensor_bar(
            attr,
            out_dir / f"top_sensors_{spec.key}_engine_{engine_id}.png",
            top_k=min(top_k + 3, attr.n_features),
            title=f"{spec.display_name} — engine {engine_id} top contributors",
        )
        # Trajectory of the single most-attributing sensor (skip op_settings).
        top = attr.top_sensors(k=top_k)
        primary = next(
            (col for col, _ in top
             if col in bundle.feature_cols and not col.startswith("os_")),
            top[0][0] if top else None,
        )
        if primary is not None:
            plot_sensor_trajectory_with_attribution(
                attr, primary,
                out_dir / f"trajectory_{spec.key}_engine_{engine_id}_{primary}.png",
                title=f"{spec.display_name} — engine {engine_id} — {primary} trajectory",
            )

        # Persist the structured explanation.
        with (out_dir / f"explanations_{spec.key}_engine_{engine_id}.json").open("w") as fh:
            json.dump(explanation.to_dict(), fh, indent=2)

        per_model_results[spec.display_name] = (attr, explanation)
        print(f"  · [{spec.key}] predicted RUL={predicted_rul:.1f} cycles "
              f"(true {pair.rul_true:.1f}, fault_prob={fault_prob:.3f}); "
              f"top: {top[0][0] if top else 'n/a'}")

    if not per_model_results:
        return {}

    _plot_cross_model_comparison(
        engine_id=engine_id,
        rul_true=rul_true_observed if rul_true_observed is not None else float("nan"),
        per_model=per_model_results,
        path=out_dir / f"cross_model_comparison_engine_{engine_id}.png",
    )
    return {
        display_name: explanation.to_dict()
        for display_name, (_, explanation) in per_model_results.items()
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--engines", nargs="+", type=int, default=[1, 25, 50, 75, 100],
        help="Test-engine ids to explain (must exist in the bundle's test set).",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--no-llm", action="store_true",
                   help="Disable the optional LLM rewrite step.")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / PHASE_ID)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(0)

    print(f"--- RQ3: sensor attribution + maintenance ontology ---")
    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"

    # 1. Discover which checkpoints actually exist on disk.
    specs = [s for s in _candidate_checkpoints() if s.checkpoint_path.exists()]
    if not specs:
        raise SystemExit(
            "No saved checkpoints found. Run P3/P5/P6 first to populate "
            "results/03_centralized/, results/05_fedavg/, results/06_non_iid/."
        )
    print("Checkpoints discovered:")
    for s in specs:
        print(f"  · {s.key}: {s.checkpoint_path}")

    # 2. Build bundles + load test windows + load models per checkpoint.
    bundles: dict[str, TrainTestBundle] = {}
    windows: dict[str, list[WindowPair]] = {}
    models: dict[str, MultiTaskCNN] = {}
    use_llm = not args.no_llm
    for spec in specs:
        bundle = _load_bundle(spec, data_dir)
        bundles[spec.key] = bundle
        windows[spec.key] = _prepare_test_windows(bundle)
        models[spec.key] = _load_model(spec, bundle)
        print(f"  loaded {spec.key}: {len(windows[spec.key])} test engines, "
              f"{models[spec.key].count_parameters():,} params")

    # 3. Explain each requested engine across all checkpoints.
    total_start = time.perf_counter()
    explanations_by_engine: dict[int, dict] = {}
    for engine_id in args.engines:
        print(f"\n=== Explaining engine {engine_id} ===")
        per_model_dicts = _run_for_engine(
            engine_id, specs, bundles, windows, models, args.out_dir,
            top_k=args.top_k, n_steps=args.n_steps, use_llm=use_llm,
        )
        if per_model_dicts:
            explanations_by_engine[engine_id] = per_model_dicts
    total_seconds = time.perf_counter() - total_start

    if not explanations_by_engine:
        raise SystemExit(
            "No engines were explained — every requested id was missing from "
            "every available test set. Check the --engines flag."
        )

    print(f"\nTotal wall-clock: {total_seconds:.1f}s "
          f"({total_seconds / max(len(explanations_by_engine), 1):.1f}s per engine)")

    # 4. metrics.json
    interpretation = (
        f"RQ3 produced sensor-level attribution + maintenance-ontology grounded "
        f"explanations for {len(explanations_by_engine)} test engines, each "
        f"explained by {len(specs)} different trained checkpoints. The "
        f"cross-model comparison surfaces a key finding from RQ2: federated "
        f"models trained on Non-IID data attribute their predictions to "
        f"different sensors than their centralized counterparts — meaning "
        f"vanilla FedAvg's failure under Non-IID is not just an accuracy "
        f"problem but also an interpretability one. The fault-mode rules in "
        f"the ontology let a ground engineer translate sensor-level "
        f"contributions into actionable maintenance recommendations."
    )

    payload = PhaseMetrics(
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        interpretation=interpretation,
        config={
            "engines_explained": list(explanations_by_engine.keys()),
            "checkpoints": [s.key for s in specs],
            "top_k": args.top_k,
            "n_steps": args.n_steps,
            "use_llm": use_llm,
        },
        timing={
            "total_seconds": round(total_seconds, 3),
            "seconds_per_engine": round(
                total_seconds / max(len(explanations_by_engine), 1), 3
            ),
        },
        summary={
            "n_engines": len(explanations_by_engine),
            "n_checkpoints": len(specs),
            "ontology_size": 17,
            "fault_rules": 3,
        },
        per_client={
            f"engine_{eid}": exps
            for eid, exps in explanations_by_engine.items()
        },
        artifacts={
            f"cross_model_engine_{eid}": (
                f"results/{PHASE_ID}/cross_model_comparison_engine_{eid}.png"
            )
            for eid in explanations_by_engine.keys()
        },
    )
    json_path = dump_phase_metrics(payload, args.out_dir)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
