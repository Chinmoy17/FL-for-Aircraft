"""RQ2 — imbalance-aware aggregation on the P6 substrate.

Runs four FedAvg variants back-to-back on the same FD001+FD003 Non-IID
partition that Phase 6 used, then compares them against the P6 vanilla
FedAvg result + the P6 centralized upper bound.

The four variants:

1. ``fedavg`` — canonical sample-count weighting (baseline reference, run
   again here from this branch so the comparison is bit-exact).
2. ``fault_count`` — Scheme A: weight clients by their fault-positive count.
3. ``validation_f1`` — Scheme B: weight clients by softmax of their
   held-out-validation F1 score (recomputed every round).
4. ``inverse_loss`` — Scheme C: weight clients by 1 / last-round local loss.
   Included primarily as a contrast — we expect it to under-perform.

Outputs under ``results/rq2_imbalance_aware/``:

    metrics.json                       structured payload for the frontend
    per_round_<scheme>_<subset>.csv    one CSV per scheme
    per_client_weights_<scheme>_<subset>.csv  per-round aggregation weights
    four_way_comparison_<subset>.png   THE headline image
    per_round_rmse_<subset>.png        all 4 schemes' RMSE trajectories
    weight_evolution_<subset>.png      Scheme B's per-round weights
    per_subset_breakdown_<subset>.png  FD001 vs FD003 per scheme

Run from the repo root inside the .venv::

    python scripts/run_rq2.py                    # all 4 schemes, defaults
    python scripts/run_rq2.py --skip inverse_loss  # skip a scheme
    python scripts/run_rq2.py --n-rounds 30        # quick variant
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_aircraft.data import (  # noqa: E402
    CMAPSSWindowDataset,
    MultiSubsetConfig,
    Normalizer,
    SUBSET_COL,
    TrainTestBundle,
    UNIT_ID_COL,
    load_multi_subset_bundle,
    make_test_windows,
    partition_by_subset_halves,
)
from fl_aircraft.eval import (  # noqa: E402
    compute_classification_metrics,
    compute_regression_metrics,
)
from fl_aircraft.fl import (  # noqa: E402
    ImbalanceAwareHistory,
    run_fedavg_imbalance_aware,
)
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics  # noqa: E402

PHASE_ID = "rq2_imbalance_aware"
PHASE_NAME = "RQ2 — Imbalance-aware aggregation (FD001 + FD003)"

ALL_SCHEMES = ("fedavg", "fault_count", "validation_f1", "inverse_loss")
SCHEME_PRETTY = {
    "fedavg": "FedAvg (sample-count)",
    "fault_count": "Scheme A (fault count)",
    "validation_f1": "Scheme B (val F1)",
    "inverse_loss": "Scheme C (inverse loss)",
}
SCHEME_COLOR = {
    "fedavg": "tab:red",
    "fault_count": "tab:blue",
    "validation_f1": "tab:green",
    "inverse_loss": "tab:orange",
}


@dataclass
class PerSubsetMetrics:
    subset: str
    n_engines: int
    rmse: float
    nasa_score: float
    auprc: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "subset": self.subset,
            "n_engines": self.n_engines,
            "rmse": round(self.rmse, 4),
            "nasa_score": round(self.nasa_score, 4),
            "auprc": round(self.auprc, 4),
            "f1": round(self.f1, 4),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD003"])
    p.add_argument("--n-clients-per-subset", type=int, default=2)
    p.add_argument("--n-rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-fault", type=float, default=0.5)
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Fraction of each client's engines held out for Scheme B.")
    p.add_argument("--softmax-temperature", type=float, default=0.5,
                   help="Lower => more aggressive reweighting in Scheme B.")
    p.add_argument("--weight-floor", type=float, default=0.05,
                   help="Minimum per-client weight for Scheme B (prevents zero-weighting).")
    p.add_argument("--no-cosine", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--skip", nargs="*", default=[], choices=list(ALL_SCHEMES),
        help="Skip one or more schemes.",
    )
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / PHASE_ID)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-subset evaluation (reuses the same logic as run_non_iid.py)
# ---------------------------------------------------------------------------
def evaluate_state_per_subset(
    state_dict: dict[str, torch.Tensor],
    bundle: TrainTestBundle,
    batch_size: int,
) -> list[PerSubsetMetrics]:
    """Evaluate a single state-dict on each origin subset of the test set."""
    normalizer = Normalizer.fit(bundle.train_df, bundle.feature_cols)
    model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=bundle.n_features, window_size=bundle.window_size)
    )
    model.load_state_dict(state_dict)
    model.eval()

    test_df = normalizer.transform(bundle.test_raw_df)
    out: list[PerSubsetMetrics] = []
    full_engines = sorted(bundle.test_raw_df[UNIT_ID_COL].unique())
    for subset in bundle.subsets:
        sub_test_df = test_df.loc[test_df[SUBSET_COL] == subset].copy()
        engine_ids_in_subset = sorted(sub_test_df[UNIT_ID_COL].unique())
        sub_rul = bundle.test_rul[
            np.array([full_engines.index(u) for u in engine_ids_in_subset])
        ]
        sub_arrays = make_test_windows(
            sub_test_df, sub_rul, bundle.feature_cols,
            bundle.window_size, bundle.rul_cap, bundle.fault_threshold,
        )
        loader = DataLoader(
            CMAPSSWindowDataset(sub_arrays), batch_size=batch_size, shuffle=False
        )
        rul_preds: list[np.ndarray] = []
        fault_scores: list[np.ndarray] = []
        with torch.no_grad():
            for x, _y_rul, _y_fault in loader:
                pred = model(x)
                rul_preds.append(pred.rul.numpy())
                fault_scores.append(pred.fault_probs().numpy())
        y_rul_pred = np.concatenate(rul_preds)
        y_fault_score = np.concatenate(fault_scores)
        rul_m = compute_regression_metrics(sub_arrays.y_rul, y_rul_pred)
        fault_m = compute_classification_metrics(sub_arrays.y_fault, y_fault_score)
        out.append(
            PerSubsetMetrics(
                subset=subset, n_engines=sub_arrays.n_samples,
                rmse=rul_m.rmse, nasa_score=rul_m.nasa_score,
                auprc=fault_m.auprc, f1=fault_m.f1,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Reference numbers from P6 (centralized + local-only) for the headline plot
# ---------------------------------------------------------------------------
def _load_p6_reference() -> dict[str, float] | None:
    p6 = REPO_ROOT / "results" / "06_non_iid" / "metrics.json"
    if not p6.exists():
        return None
    with p6.open(encoding="utf-8") as fh:
        d = json.load(fh)
    s = d["summary"]
    return {
        "centralized_rmse": s["centralized_rmse"],
        "centralized_nasa": s["centralized_nasa"],
        "centralized_auprc": s["centralized_auprc"],
        "centralized_f1": s["centralized_f1"],
        "local_only_rmse_mean": s["local_only_rmse_mean"],
        "local_only_rmse_std": s["local_only_rmse_std"],
        "local_only_nasa_mean": d["per_client"].get("nasa_score_mean", None),
        "local_only_auprc_mean": s["local_only_auprc_mean"],
        "local_only_f1_mean": s["local_only_f1_mean"],
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_four_way(
    histories: dict[str, ImbalanceAwareHistory],
    per_subset: dict[str, list[PerSubsetMetrics]],
    p6: dict[str, float] | None,
    path: Path, display_name: str,
) -> None:
    """Headline: centralized vs local-only vs every aggregator."""
    metrics_meta = [
        ("rmse", "Test RMSE (lower = better)", False, None, 2),
        ("nasa_score", "Test NASA score (log)", True, None, 0),
        ("auprc", "Test AUPRC", False, (0, 1.05), 3),
        ("f1", "Test F1", False, (0, 1.05), 3),
    ]
    # Build ordered labels + values + colors.
    labels = ["Centralized\n(P6 upper)"] if p6 else []
    color_list = ["seagreen"] if p6 else []
    rmse_vals = [p6["centralized_rmse"]] if p6 else []
    nasa_vals = [p6["centralized_nasa"]] if p6 else []
    auprc_vals = [p6["centralized_auprc"]] if p6 else []
    f1_vals = [p6["centralized_f1"]] if p6 else []
    if p6:
        labels.append("Local-only mean\n(P6 lower)")
        color_list.append("steelblue")
        rmse_vals.append(p6["local_only_rmse_mean"])
        nasa_vals.append(p6.get("local_only_nasa_mean") or float("nan"))
        auprc_vals.append(p6["local_only_auprc_mean"])
        f1_vals.append(p6["local_only_f1_mean"])
    for scheme, h in histories.items():
        labels.append(SCHEME_PRETTY[scheme])
        color_list.append(SCHEME_COLOR[scheme])
        rmse_vals.append(h.best_test_rul.rmse)
        nasa_vals.append(h.best_test_rul.nasa_score)
        auprc_vals.append(h.best_test_fault.auprc)
        f1_vals.append(h.best_test_fault.f1)

    values_by_metric = {"rmse": rmse_vals, "nasa_score": nasa_vals,
                        "auprc": auprc_vals, "f1": f1_vals}

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (key, title, log, ylim, decimals) in zip(axes, metrics_meta):
        values = values_by_metric[key]
        bars = ax.bar(labels, values, color=color_list)
        for bar, v in zip(bars, values):
            if np.isnan(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.{decimals}f}",
                ha="center", va="bottom", fontsize=8,
            )
        if key == "rmse" and p6 is not None:
            # Error bar on the local-only mean.
            ax.errorbar([labels[1]], [p6["local_only_rmse_mean"]],
                        yerr=[p6["local_only_rmse_std"]],
                        color="black", capsize=4, fmt="none")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if log:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.suptitle(
        f"RQ2 — imbalance-aware aggregation on Non-IID {display_name}  "
        f"(best-round metrics)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_round_rmse(
    histories: dict[str, ImbalanceAwareHistory],
    p6: dict[str, float] | None,
    path: Path, display_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for scheme, h in histories.items():
        rounds = [rec.round for rec in h.rounds]
        rmse = [rec.global_test_rmse for rec in h.rounds]
        ax.plot(rounds, rmse, color=SCHEME_COLOR[scheme],
                label=SCHEME_PRETTY[scheme], linewidth=1.6, alpha=0.9)
    if p6 is not None:
        ax.axhline(p6["centralized_rmse"], color="seagreen", linestyle="--",
                   linewidth=1.2, label=f"Centralized = {p6['centralized_rmse']:.2f}")
        ax.axhline(p6["local_only_rmse_mean"], color="steelblue", linestyle=":",
                   linewidth=1.2, label=f"Local-only mean = {p6['local_only_rmse_mean']:.2f}")
    ax.set_xlabel("communication round")
    ax.set_ylabel("global test RMSE")
    ax.set_title(f"RQ2 — global test RMSE per round ({display_name})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_weight_evolution(
    history: ImbalanceAwareHistory, path: Path, display_name: str,
) -> None:
    """Per-round per-client aggregation weights for Scheme B."""
    rounds = list(range(1, len(history) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(history.client_ids)))
    for color, cid in zip(colors, history.client_ids):
        ax.plot(rounds, history.aggregation_weights[cid],
                label=cid, color=color, linewidth=1.6)
    ax.axhline(1 / len(history.client_ids), color="gray", linestyle="--",
               linewidth=0.8, label=f"uniform = {1/len(history.client_ids):.2f}")
    ax.set_xlabel("communication round")
    ax.set_ylabel("aggregation weight")
    ax.set_title(f"RQ2 Scheme B — per-client aggregation weight evolution ({display_name})")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_subset(
    per_subset: dict[str, list[PerSubsetMetrics]],
    p6_per_subset: dict[str, dict[str, float]] | None,
    path: Path, display_name: str,
) -> None:
    """For each test subset, show RMSE per scheme + P6 references."""
    subsets = list(next(iter(per_subset.values()))[0].subset for _ in [0])  # placeholder
    # Build the list of subsets from the first scheme's per-subset breakdown.
    subset_names = [ps.subset for ps in next(iter(per_subset.values()))]

    fig, axes = plt.subplots(1, len(subset_names), figsize=(7 * len(subset_names), 5))
    if len(subset_names) == 1:
        axes = [axes]
    for ax, subset in zip(axes, subset_names):
        labels: list[str] = []
        values: list[float] = []
        colors_local: list = []
        if p6_per_subset is not None and subset in p6_per_subset:
            labels.append("P6 centralized")
            values.append(p6_per_subset[subset]["centralized_rmse"])
            colors_local.append("seagreen")
            labels.append("P6 FedAvg")
            values.append(p6_per_subset[subset]["fedavg_rmse"])
            colors_local.append("salmon")
        for scheme, ps_list in per_subset.items():
            ps = next(p for p in ps_list if p.subset == subset)
            labels.append(SCHEME_PRETTY[scheme])
            values.append(ps.rmse)
            colors_local.append(SCHEME_COLOR[scheme])

        bars = ax.bar(labels, values, color=colors_local)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Test RMSE on {subset} engines")
        ax.set_ylabel("RMSE (cycles)")
        ax.grid(alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.suptitle(f"RQ2 — per-subset breakdown ({display_name})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_p6_per_subset() -> dict[str, dict[str, float]] | None:
    p6 = REPO_ROOT / "results" / "06_non_iid" / "metrics.json"
    if not p6.exists():
        return None
    with p6.open(encoding="utf-8") as fh:
        d = json.load(fh)
    out: dict[str, dict[str, float]] = {}
    central_ps = {ps["subset"]: ps for ps in d["test"]["centralized_per_subset"]}
    fedavg_ps = {ps["subset"]: ps for ps in d["test"]["fedavg_per_subset"]}
    for subset in central_ps:
        out[subset] = {
            "centralized_rmse": central_ps[subset]["rmse"],
            "fedavg_rmse": fedavg_ps.get(subset, {}).get("rmse", float("nan")),
        }
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    schemes = [s for s in ALL_SCHEMES if s not in set(args.skip)]
    if not schemes:
        raise SystemExit("All schemes were skipped — nothing to run.")

    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
    multi_cfg = MultiSubsetConfig(subsets=tuple(args.subsets), data_dir=data_dir)
    bundle = load_multi_subset_bundle(multi_cfg)
    display = bundle.display_name
    print(f"--- RQ2 imbalance-aware aggregation ({display}) ---")
    print(f"  schemes to run: {schemes}")
    print(f"  val-fraction (Scheme B only): {args.val_fraction}")
    print(f"  softmax temperature (Scheme B): {args.softmax_temperature}")
    print(f"  weight floor (Scheme B): {args.weight_floor}\n")

    shards = partition_by_subset_halves(
        bundle.train_df, subsets=tuple(args.subsets),
        n_clients_per_subset=args.n_clients_per_subset, seed=args.seed,
    )

    p6 = _load_p6_reference()
    if p6:
        print(f"  P6 reference: centralized RMSE={p6['centralized_rmse']:.2f}, "
              f"local-only RMSE={p6['local_only_rmse_mean']:.2f}±"
              f"{p6['local_only_rmse_std']:.2f}\n")

    total_start = time.perf_counter()
    histories: dict[str, ImbalanceAwareHistory] = {}
    per_subset: dict[str, list[PerSubsetMetrics]] = {}

    for scheme in schemes:
        print(f"\n========= Running scheme: {scheme} =========")
        history = run_fedavg_imbalance_aware(
            bundle, shards,
            aggregator=scheme,
            val_fraction=args.val_fraction,
            softmax_temperature=args.softmax_temperature,
            weight_floor=args.weight_floor,
            n_rounds=args.n_rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_fault=args.lambda_fault,
            use_cosine_schedule=not args.no_cosine,
            seed=args.seed,
            log_every=args.log_every,
        )
        histories[scheme] = history
        per_subset[scheme] = evaluate_state_per_subset(
            history.best_state_dict, bundle, args.batch_size,
        )
        print(f"  best round {history.best_round}/{args.n_rounds} → "
              f"RMSE={history.best_test_rul.rmse:.2f}  "
              f"NASA={history.best_test_rul.nasa_score:.0f}  "
              f"AUPRC={history.best_test_fault.auprc:.3f}  "
              f"F1={history.best_test_fault.f1:.3f}  "
              f"({history.total_seconds:.1f}s)")
        for ps in per_subset[scheme]:
            print(f"    {ps.subset}: RMSE={ps.rmse:.2f}  NASA={ps.nasa_score:.0f}  "
                  f"AUPRC={ps.auprc:.3f}  F1={ps.f1:.3f}")

    total_seconds = time.perf_counter() - total_start

    # ---------------- Console summary ----------------
    print("\n========= 4-way comparison =========")
    if p6:
        print(f"  Centralized (P6) : RMSE={p6['centralized_rmse']:.2f}  "
              f"AUPRC={p6['centralized_auprc']:.3f}  F1={p6['centralized_f1']:.3f}")
        print(f"  Local-only  (P6) : RMSE={p6['local_only_rmse_mean']:.2f}±"
              f"{p6['local_only_rmse_std']:.2f}  "
              f"AUPRC={p6['local_only_auprc_mean']:.3f}  "
              f"F1={p6['local_only_f1_mean']:.3f}")
    for scheme, h in histories.items():
        gap_close = None
        if p6 and (p6["local_only_rmse_mean"] - p6["centralized_rmse"]) > 1e-6:
            gap_close = (
                (p6["local_only_rmse_mean"] - h.best_test_rul.rmse) /
                (p6["local_only_rmse_mean"] - p6["centralized_rmse"]) * 100
            )
        gap_str = f"({gap_close:+.1f}% gap closed)" if gap_close is not None else ""
        print(f"  {SCHEME_PRETTY[scheme]:<26}: RMSE={h.best_test_rul.rmse:.2f}  "
              f"AUPRC={h.best_test_fault.auprc:.3f}  "
              f"F1={h.best_test_fault.f1:.3f}  {gap_str}")
    print(f"\nTotal wall-clock: {total_seconds:.1f} s")

    # ---------------- CSVs ----------------
    for scheme, h in histories.items():
        round_rows = [rec.as_dict() for rec in h.rounds]
        _write_csv(round_rows,
                   args.out_dir / f"per_round_{scheme}_{display.lower()}.csv")
        weight_rows = []
        for r_idx, rec in enumerate(h.rounds):
            row = {"round": rec.round}
            for cid in h.client_ids:
                row[cid] = round(h.aggregation_weights[cid][r_idx], 4)
            weight_rows.append(row)
        _write_csv(weight_rows,
                   args.out_dir / f"per_client_weights_{scheme}_{display.lower()}.csv")
    print(f"\nWrote per-scheme CSVs to {args.out_dir}")

    # ---------------- Plots ----------------
    headline_png = args.out_dir / f"four_way_comparison_{display.lower()}.png"
    rmse_png = args.out_dir / f"per_round_rmse_{display.lower()}.png"
    breakdown_png = args.out_dir / f"per_subset_breakdown_{display.lower()}.png"

    _plot_four_way(histories, per_subset, p6, headline_png, display)
    _plot_per_round_rmse(histories, p6, rmse_png, display)
    _plot_per_subset(per_subset, _load_p6_per_subset(), breakdown_png, display)
    print(f"Wrote {headline_png}  (headline)")
    print(f"Wrote {rmse_png}")
    print(f"Wrote {breakdown_png}")

    if "validation_f1" in histories:
        weight_png = args.out_dir / f"weight_evolution_{display.lower()}.png"
        _plot_weight_evolution(histories["validation_f1"], weight_png, display)
        print(f"Wrote {weight_png}")

    # ---------------- metrics.json ----------------
    summary: dict[str, float] = {
        "p6_centralized_rmse": p6["centralized_rmse"] if p6 else None,
        "p6_local_only_rmse_mean": p6["local_only_rmse_mean"] if p6 else None,
    }
    scheme_results = {}
    for scheme, h in histories.items():
        gap_close = None
        if p6 and (p6["local_only_rmse_mean"] - p6["centralized_rmse"]) > 1e-6:
            gap_close = (
                (p6["local_only_rmse_mean"] - h.best_test_rul.rmse) /
                (p6["local_only_rmse_mean"] - p6["centralized_rmse"]) * 100
            )
        scheme_results[scheme] = {
            "best_round": h.best_round,
            "final_round": len(h),
            "best_rmse": round(h.best_test_rul.rmse, 4),
            "best_mae": round(h.best_test_rul.mae, 4),
            "best_nasa": round(h.best_test_rul.nasa_score, 4),
            "best_auprc": round(h.best_test_fault.auprc, 4),
            "best_f1": round(h.best_test_fault.f1, 4),
            "best_precision": round(h.best_test_fault.precision, 4),
            "best_recall": round(h.best_test_fault.recall, 4),
            "final_rmse": round(h.final_test_rul.rmse, 4),
            "final_nasa": round(h.final_test_rul.nasa_score, 4),
            "train_seconds": round(h.total_seconds, 3),
            "rmse_gap_closed_pct": round(gap_close, 2) if gap_close is not None else None,
            "per_subset": [ps.as_dict() for ps in per_subset[scheme]],
        }
        summary[f"{scheme}_rmse"] = scheme_results[scheme]["best_rmse"]
        summary[f"{scheme}_gap_closed_pct"] = scheme_results[scheme]["rmse_gap_closed_pct"]

    # Pick the winner for the interpretation string.
    winner = min(scheme_results.items(),
                 key=lambda kv: kv[1]["best_rmse"] if kv[1]["best_rmse"] is not None else float("inf"))
    winner_name, winner_metrics = winner
    if p6:
        baseline_rmse = scheme_results.get("fedavg", {}).get("best_rmse")
        interp = (
            f"RQ2 ran 4 aggregation schemes on the P6 Non-IID partition. "
            f"Best scheme: {SCHEME_PRETTY[winner_name]} with RMSE "
            f"{winner_metrics['best_rmse']:.2f} "
            f"({winner_metrics['rmse_gap_closed_pct']:+.1f}% of the "
            f"local-only \u2192 centralized gap). "
        )
        if baseline_rmse is not None and winner_name != "fedavg":
            delta = baseline_rmse - winner_metrics["best_rmse"]
            interp += (
                f"Improvement over vanilla FedAvg in this rerun: "
                f"\u0394RMSE = {delta:+.2f}."
            )
    else:
        interp = (
            f"RQ2 ran 4 aggregation schemes. Best by RMSE: "
            f"{SCHEME_PRETTY[winner_name]} ({winner_metrics['best_rmse']:.2f})."
        )

    metrics_payload = PhaseMetrics(
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        interpretation=interp,
        subset=display,
        config={
            "subsets": list(args.subsets),
            "n_clients_per_subset": args.n_clients_per_subset,
            "n_rounds": args.n_rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_fault": args.lambda_fault,
            "val_fraction": args.val_fraction,
            "softmax_temperature": args.softmax_temperature,
            "weight_floor": args.weight_floor,
            "use_cosine_schedule": not args.no_cosine,
            "seed": args.seed,
            "n_features": bundle.n_features,
            "window_size": bundle.window_size,
            "schemes_run": schemes,
        },
        timing={"total_seconds": round(total_seconds, 3)},
        summary=summary,
        train={
            scheme: [
                {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.as_dict().items()}
                for rec in h.rounds
            ]
            for scheme, h in histories.items()
        },
        test={
            scheme: scheme_results[scheme]
            for scheme in schemes
        },
        per_client={
            "fault_positives": next(iter(histories.values())).n_fault_positives,
            "aggregation_weights": {
                scheme: {
                    cid: [round(w, 4) for w in h.aggregation_weights[cid]]
                    for cid in h.client_ids
                }
                for scheme, h in histories.items()
            },
        },
        artifacts={
            "four_way_comparison_png": f"results/{PHASE_ID}/four_way_comparison_{display.lower()}.png",
            "per_round_rmse_png": f"results/{PHASE_ID}/per_round_rmse_{display.lower()}.png",
            "per_subset_breakdown_png": f"results/{PHASE_ID}/per_subset_breakdown_{display.lower()}.png",
            "weight_evolution_png": (
                f"results/{PHASE_ID}/weight_evolution_{display.lower()}.png"
                if "validation_f1" in histories else ""
            ),
        },
    )
    json_path = dump_phase_metrics(metrics_payload, args.out_dir)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
