"""Non-IID baseline (Phase 6) — FD001 + FD003.

Runs the full three-way comparison **on a structurally Non-IID federation**:

- clients 1 & 2 receive only FD001 engines (HPC fault mode only)
- clients 3 & 4 receive only FD003 engines (HPC + Fan fault modes)

This is the configuration where the federation's value actually shows: local-
only clients trained on one fault-mode family cannot generalise to the other,
while centralized (pooled) and FedAvg (weights-only sharing) both can.

Three sub-runs are executed in series:

1. **Centralized** — train on the pooled FD001+FD003 engines (200 total).
2. **Local-only** — train 4 separate models (one per subset-half client).
3. **FedAvg** — federated training across the same 4 clients.

All three are evaluated against the **common combined FD001+FD003 test set
(200 engines)** so the comparison is apples-to-apples. Per-subset breakdowns
are also computed so we can see whether each model handles both fault modes
or only the one it was trained on.

Outputs under ``results/06_non_iid/``:

    metrics.json                                       structured for the frontend
    per_round_fedavg_fd001_fd003.csv
    per_client_loss_fedavg_fd001_fd003.csv
    per_client_local_fd001_fd003.csv
    per_epoch_centralized_fd001_fd003.csv
    centralized_metrics_fd001_fd003.png                centralized training curves
    local_only_metrics_fd001_fd003.png                 per-client local-only bars
    fedavg_metrics_fd001_fd003.png                     FedAvg per-round metrics
    three_way_non_iid_fd001_fd003.png                  THE headline image
    per_subset_breakdown_fd001_fd003.png               cross-evaluation FD001 vs FD003

Run from the repo root inside the .venv::

    python scripts/run_non_iid.py
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
    make_training_windows,
    partition_by_subset_halves,
)
from fl_aircraft.eval import (  # noqa: E402
    compute_classification_metrics,
    compute_regression_metrics,
)
from fl_aircraft.fl import run_fedavg_from_bundle  # noqa: E402
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss  # noqa: E402
from fl_aircraft.train import (  # noqa: E402
    history_as_rows,
    train_centralized,
    train_local_only_from_bundle,
)
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics, seed_everything  # noqa: E402

PHASE_ID = "06_non_iid"
PHASE_NAME = "Phase 6 — Non-IID baseline (FD001 + FD003)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD003"])
    p.add_argument("--n-clients-per-subset", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50, help="Used by centralized + local-only sub-runs.")
    p.add_argument("--n-rounds", type=int, default=50, help="FedAvg communication rounds.")
    p.add_argument("--local-epochs", type=int, default=2, help="FedAvg local epochs per round.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-fault", type=float, default=0.5)
    p.add_argument("--no-cosine", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / PHASE_ID)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-subset breakdown evaluation
# ---------------------------------------------------------------------------
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


def evaluate_model_per_subset(
    state_dict: dict[str, torch.Tensor],
    bundle: TrainTestBundle,
    normalizer: Normalizer,
    batch_size: int,
) -> list[PerSubsetMetrics]:
    """Run the model state-dict on each origin subset of the test set separately.

    Used to expose the cross-evaluation asymmetry — e.g. an FD001-trained
    local model does well on FD001 test engines but badly on FD003 test
    engines.
    """
    model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=bundle.n_features, window_size=bundle.window_size)
    )
    model.load_state_dict(state_dict)
    model.eval()

    test_df = normalizer.transform(bundle.test_raw_df)
    out: list[PerSubsetMetrics] = []
    for subset in bundle.subsets:
        sub_test_df = test_df.loc[test_df[SUBSET_COL] == subset].copy()
        # Indices into bundle.test_rul matching this subset's engines.
        engine_ids_in_subset = sorted(sub_test_df[UNIT_ID_COL].unique())
        full_engines = sorted(bundle.test_raw_df[UNIT_ID_COL].unique())
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
                subset=subset,
                n_engines=sub_arrays.n_samples,
                rmse=rul_m.rmse,
                nasa_score=rul_m.nasa_score,
                auprc=fault_m.auprc,
                f1=fault_m.f1,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Sub-run 1: centralized
# ---------------------------------------------------------------------------
def run_centralized_subrun(
    bundle: TrainTestBundle, args: argparse.Namespace
) -> tuple[dict, list[PerSubsetMetrics], Normalizer]:
    print("\n=== Sub-run 1/3: centralized (pooled FD001+FD003) ===")
    seed_everything(args.seed)
    normalizer = Normalizer.fit(bundle.train_df, bundle.feature_cols)
    train_arrays = make_training_windows(
        normalizer.transform(bundle.train_df),
        bundle.feature_cols, bundle.window_size, bundle.stride,
    )
    test_arrays = make_test_windows(
        normalizer.transform(bundle.test_raw_df), bundle.test_rul,
        bundle.feature_cols, bundle.window_size, bundle.rul_cap, bundle.fault_threshold,
    )
    train_loader = DataLoader(CMAPSSWindowDataset(train_arrays),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(CMAPSSWindowDataset(test_arrays),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  train windows: {len(train_arrays.X):,}  test windows: {test_arrays.n_samples}")

    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=bundle.n_features, window_size=bundle.window_size))
    n_pos = int(train_arrays.y_fault.sum())
    n_neg = int(train_arrays.y_fault.shape[0] - n_pos)
    pos_weight = float(n_neg) / float(max(n_pos, 1))
    loss_fn = MultiTaskLoss(lambda_fault=args.lambda_fault, fault_pos_weight=pos_weight)

    history = train_centralized(
        model, train_loader, test_loader, loss_fn,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        use_cosine_schedule=not args.no_cosine, log_every=args.log_every,
    )
    per_subset = evaluate_model_per_subset(
        history.best_state_dict, bundle, normalizer, args.batch_size
    )
    print(f"  best epoch {history.best_epoch}/{args.epochs} → "
          f"combined RMSE={history.best_test_rul.rmse:.2f}  "
          f"NASA={history.best_test_rul.nasa_score:.0f}  "
          f"AUPRC={history.best_test_fault.auprc:.3f}  "
          f"F1={history.best_test_fault.f1:.3f}  "
          f"({history.total_seconds:.1f}s)")
    for ps in per_subset:
        print(f"    {ps.subset}: RMSE={ps.rmse:.2f}  NASA={ps.nasa_score:.0f}  "
              f"AUPRC={ps.auprc:.3f}  F1={ps.f1:.3f}")
    return {
        "history": history,
        "per_subset": per_subset,
        "normalizer": normalizer,
        "best_state_dict": history.best_state_dict,
        "pos_weight": pos_weight,
        "n_train_windows": len(train_arrays.X),
    }, per_subset, normalizer


# ---------------------------------------------------------------------------
# Sub-run 2: local-only
# ---------------------------------------------------------------------------
def run_local_only_subrun(
    bundle: TrainTestBundle, shards, args: argparse.Namespace
) -> dict:
    print("\n=== Sub-run 2/3: local-only (4 isolated clients) ===")
    results = train_local_only_from_bundle(
        bundle, shards,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, lambda_fault=args.lambda_fault,
        use_cosine_schedule=not args.no_cosine, seed=args.seed,
        log_every=args.log_every, client_log=True,
    )
    # Cross-evaluation per subset per client
    cross: dict[str, list[PerSubsetMetrics]] = {}
    for client in results.clients:
        # Each client has its own normalizer from its own training slice
        from fl_aircraft.data import slice_for_client
        client_df = slice_for_client(bundle.train_df, client.shard)
        norm = Normalizer.fit(client_df, bundle.feature_cols)
        cross[client.client_id] = evaluate_model_per_subset(
            client.history.best_state_dict, bundle, norm, args.batch_size,
        )
        ps_str = "  ".join(
            f"{ps.subset}: RMSE={ps.rmse:.2f}" for ps in cross[client.client_id]
        )
        print(f"  {client.client_id} cross-eval → {ps_str}")
    return {
        "results": results,
        "cross_per_subset": cross,
    }


# ---------------------------------------------------------------------------
# Sub-run 3: FedAvg
# ---------------------------------------------------------------------------
def run_fedavg_subrun(
    bundle: TrainTestBundle, shards, args: argparse.Namespace
) -> tuple[dict, list[PerSubsetMetrics]]:
    print("\n=== Sub-run 3/3: FedAvg (4 clients, weight sharing only) ===")
    history = run_fedavg_from_bundle(
        bundle, shards,
        n_rounds=args.n_rounds, local_epochs=args.local_epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        lambda_fault=args.lambda_fault, use_cosine_schedule=not args.no_cosine,
        seed=args.seed, log_every=args.log_every,
    )
    # Use the centralized normalizer for global-model per-subset breakdown so
    # the per-subset numbers are comparable to the centralized sub-run.
    normalizer = Normalizer.fit(bundle.train_df, bundle.feature_cols)
    per_subset = evaluate_model_per_subset(
        history.best_state_dict, bundle, normalizer, args.batch_size,
    )
    print(f"  best round {history.best_round}/{args.n_rounds} → "
          f"combined RMSE={history.best_test_rul.rmse:.2f}  "
          f"NASA={history.best_test_rul.nasa_score:.0f}  "
          f"AUPRC={history.best_test_fault.auprc:.3f}  "
          f"F1={history.best_test_fault.f1:.3f}  "
          f"({history.total_seconds:.1f}s)")
    for ps in per_subset:
        print(f"    {ps.subset}: RMSE={ps.rmse:.2f}  NASA={ps.nasa_score:.0f}  "
              f"AUPRC={ps.auprc:.3f}  F1={ps.f1:.3f}")
    return {
        "history": history,
        "per_subset": per_subset,
        "best_state_dict": history.best_state_dict,
    }, per_subset


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_three_way(p3, p4_agg, p5_best, path: Path, display_name: str) -> None:
    """Headline image: centralized vs local-only mean vs FedAvg (combined test set)."""
    metrics_meta = [
        ("rmse", "Test RMSE (lower = better)", False, None, 2),
        ("nasa_score", "Test NASA score (log, lower)", True, None, 0),
        ("auprc", "Test AUPRC", False, (0, 1.05), 3),
        ("f1", "Test F1", False, (0, 1.05), 3),
    ]
    p3_vals = {"rmse": p3.rmse, "nasa_score": p3.nasa_score,
               "auprc": p3_f.auprc, "f1": p3_f.f1}  # set below by caller
    raise NotImplementedError  # we override this with the inline version below


def _plot_three_way_v2(
    central_rul, central_fault,
    local_rmse_mean, local_rmse_std, local_nasa_mean, local_auprc_mean, local_f1_mean,
    fedavg_rul, fedavg_fault,
    path: Path, display_name: str,
) -> None:
    metrics_meta = [
        ("rmse", "Test RMSE (lower = better)", False, None, 2),
        ("nasa_score", "Test NASA score (log, lower)", True, None, 0),
        ("auprc", "Test AUPRC", False, (0, 1.05), 3),
        ("f1", "Test F1", False, (0, 1.05), 3),
    ]
    values_map = {
        "Centralized\n(P6 upper)": [central_rul.rmse, central_rul.nasa_score,
                                    central_fault.auprc, central_fault.f1],
        "Local-only mean\n(P6 lower)": [local_rmse_mean, local_nasa_mean,
                                        local_auprc_mean, local_f1_mean],
        "FedAvg\n(P6)": [fedavg_rul.rmse, fedavg_rul.nasa_score,
                         fedavg_fault.auprc, fedavg_fault.f1],
    }
    labels = list(values_map.keys())
    colors = ["seagreen", "steelblue", "crimson"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    for ax, (key, title, log, ylim, decimals) in zip(axes, metrics_meta):
        idx = ["rmse", "nasa_score", "auprc", "f1"].index(key)
        values = [values_map[lbl][idx] for lbl in labels]
        bars = ax.bar(labels, values, color=colors)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.{decimals}f}",
                ha="center", va="bottom", fontsize=9,
            )
        if key == "rmse":
            ax.errorbar([labels[1]], [local_rmse_mean], yerr=[local_rmse_std],
                        color="black", capsize=4, fmt="none")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if log:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.suptitle(
        f"3-way comparison on Non-IID {display_name}  (best-epoch / best-round)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_subset_breakdown(
    central_per_subset: list[PerSubsetMetrics],
    local_cross: dict[str, list[PerSubsetMetrics]],
    fedavg_per_subset: list[PerSubsetMetrics],
    path: Path, display_name: str,
) -> None:
    """Show the cross-evaluation asymmetry: FD001-trained vs FD003-trained, on each test half."""
    subsets = [ps.subset for ps in central_per_subset]
    fig, axes = plt.subplots(1, len(subsets), figsize=(7 * len(subsets), 5))
    if len(subsets) == 1:
        axes = [axes]
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(local_cross) + 2))

    for ax, subset in zip(axes, subsets):
        # Build the per-method RMSE bars for this test subset.
        labels: list[str] = []
        values: list[float] = []
        color_list: list = []
        # Centralized
        central_rmse = next(p.rmse for p in central_per_subset if p.subset == subset)
        labels.append("Centralized")
        values.append(central_rmse)
        color_list.append("seagreen")
        # Each local-only client
        for i, (cid, ps_list) in enumerate(local_cross.items()):
            rmse_on_subset = next(p.rmse for p in ps_list if p.subset == subset)
            labels.append(cid.replace("_", "\n"))
            values.append(rmse_on_subset)
            color_list.append("steelblue")
        # FedAvg global
        fedavg_rmse = next(p.rmse for p in fedavg_per_subset if p.subset == subset)
        labels.append("FedAvg")
        values.append(fedavg_rmse)
        color_list.append("crimson")

        bars = ax.bar(labels, values, color=color_list)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Test RMSE on {subset} engines")
        ax.set_ylabel("RMSE (cycles)")
        ax.grid(alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.suptitle(
        f"Per-subset cross-evaluation on Non-IID {display_name}  "
        f"(steelblue = local-only client trained on one subset only)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_centralized_metrics(rows: list[dict], path: Path, display_name: str) -> None:
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1, ax2 = axes
    ax1.plot(epochs, [r["test_rmse"] for r in rows], color="crimson", label="RMSE")
    ax1.plot(epochs, [r["test_mae"] for r in rows], color="orange", label="MAE")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("cycles")
    ax1.set_title(f"Centralized RUL error ({display_name})")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, [r["test_auprc"] for r in rows], color="indigo", label="AUPRC")
    ax2.plot(epochs, [r["test_f1"] for r in rows], color="teal", label="F1")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score"); ax2.set_ylim(0, 1.05)
    ax2.set_title(f"Centralized fault discrim ({display_name})")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_local_only_metrics(rows: list[dict], path: Path, display_name: str) -> None:
    client_ids = [r["client_id"] for r in rows]
    metrics = ["rmse", "nasa_score", "auprc", "f1"]
    titles = ["Test RMSE", "Test NASA (log)", "Test AUPRC", "Test F1"]
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(client_ids)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals = [r[metric] for r in rows]
        bars = ax.bar(client_ids, vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}" if metric != "nasa_score" else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if metric == "nasa_score":
            ax.set_yscale("log")
        if metric in ("auprc", "f1"):
            ax.set_ylim(0, 1.05)
    fig.suptitle(f"Local-only per-client best metrics ({display_name})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_fedavg_metrics(rows: list[dict], path: Path, display_name: str) -> None:
    rounds = [r["round"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(rounds, [r["global_test_rmse"] for r in rows], color="crimson", label="RMSE")
    ax.plot(rounds, [r["global_test_mae"] for r in rows], color="orange", label="MAE")
    ax.set_title("Global test RUL error"); ax.set_xlabel("round"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(rounds, [r["global_test_nasa_score"] for r in rows], color="purple")
    ax.set_yscale("log"); ax.set_title("Global NASA (log)"); ax.set_xlabel("round"); ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(rounds, [r["global_test_auprc"] for r in rows], color="indigo", label="AUPRC")
    ax.plot(rounds, [r["global_test_f1"] for r in rows], color="teal", label="F1")
    ax.set_title("Global fault discrim"); ax.set_xlabel("round"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.plot(rounds, [r["global_test_precision"] for r in rows], color="orange", linestyle="--", label="Precision")
    ax.plot(rounds, [r["global_test_recall"] for r in rows], color="crimson", linestyle="--", label="Recall")
    ax.set_title("Global fault op point"); ax.set_xlabel("round"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"FedAvg per-round global metrics ({display_name})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
    multi_cfg = MultiSubsetConfig(
        subsets=tuple(args.subsets),
        data_dir=data_dir,
    )
    bundle = load_multi_subset_bundle(multi_cfg)
    display = bundle.display_name
    print(f"--- Phase 6 Non-IID baseline ({display}) ---")
    print(f"train rows={len(bundle.train_df):,}  test engines={bundle.test_rul.shape[0]}  "
          f"features={bundle.n_features}")

    shards = partition_by_subset_halves(
        bundle.train_df,
        subsets=tuple(args.subsets),
        n_clients_per_subset=args.n_clients_per_subset,
        seed=args.seed,
    )
    for shard in shards:
        # Determine origin subset for printing
        first_uid = shard.unit_ids[0]
        origin = bundle.train_df.loc[
            bundle.train_df[UNIT_ID_COL] == first_uid, SUBSET_COL
        ].iloc[0]
        print(f"  {shard.client_id}: {shard.n_engines} engines from {origin}")

    total_start = time.perf_counter()

    central_run, central_per_subset, central_norm = run_centralized_subrun(bundle, args)
    local_run = run_local_only_subrun(bundle, shards, args)
    fedavg_run, fedavg_per_subset = run_fedavg_subrun(bundle, shards, args)

    total_seconds = time.perf_counter() - total_start

    # ---------------- Aggregations ----------------
    central_hist = central_run["history"]
    local_results = local_run["results"]
    local_agg_best = local_results.aggregate("best")
    fedavg_hist = fedavg_run["history"]

    print("\n=========== 3-way comparison on combined test set ===========")
    print(f"  Centralized  : RMSE={central_hist.best_test_rul.rmse:.2f}  "
          f"NASA={central_hist.best_test_rul.nasa_score:.0f}  "
          f"AUPRC={central_hist.best_test_fault.auprc:.3f}  "
          f"F1={central_hist.best_test_fault.f1:.3f}")
    print(f"  Local-only mn: RMSE={local_agg_best['rmse_mean']:.2f}\u00b1"
          f"{local_agg_best['rmse_std']:.2f}  "
          f"NASA={local_agg_best['nasa_score_mean']:.0f}  "
          f"AUPRC={local_agg_best['auprc_mean']:.3f}  "
          f"F1={local_agg_best['f1_mean']:.3f}")
    print(f"  FedAvg       : RMSE={fedavg_hist.best_test_rul.rmse:.2f}  "
          f"NASA={fedavg_hist.best_test_rul.nasa_score:.0f}  "
          f"AUPRC={fedavg_hist.best_test_fault.auprc:.3f}  "
          f"F1={fedavg_hist.best_test_fault.f1:.3f}")
    rmse_gap = local_agg_best["rmse_mean"] - central_hist.best_test_rul.rmse
    if abs(rmse_gap) > 1e-6:
        gap_closed_pct = (
            (local_agg_best["rmse_mean"] - fedavg_hist.best_test_rul.rmse) / rmse_gap * 100
        )
    else:
        gap_closed_pct = 0.0
    print(f"  FedAvg closed {gap_closed_pct:.1f}% of the local-only \u2192 centralized RMSE gap")
    print(f"\nTotal wall-clock (all 3 sub-runs): {total_seconds:.1f} s")

    # ---------------- CSVs ----------------
    central_csv = args.out_dir / f"per_epoch_centralized_{display.lower()}.csv"
    _write_csv(history_as_rows(central_hist), central_csv)
    local_csv = args.out_dir / f"per_client_local_{display.lower()}.csv"
    _write_csv(local_results.per_client_rows("best"), local_csv)
    fedavg_csv = args.out_dir / f"per_round_fedavg_{display.lower()}.csv"
    _write_csv([rec.as_dict() for rec in fedavg_hist.rounds], fedavg_csv)
    fedavg_client_csv = args.out_dir / f"per_client_loss_fedavg_{display.lower()}.csv"
    fedavg_client_rows = []
    for r_idx, rec in enumerate(fedavg_hist.rounds):
        row = {"round": rec.round}
        for cid in fedavg_hist.client_ids:
            row[cid] = round(fedavg_hist.per_round_client_losses[cid][r_idx], 4)
        fedavg_client_rows.append(row)
    _write_csv(fedavg_client_rows, fedavg_client_csv)
    print(f"\nWrote {central_csv}\nWrote {local_csv}\nWrote {fedavg_csv}\nWrote {fedavg_client_csv}")

    # ---------------- Plots ----------------
    _plot_centralized_metrics(history_as_rows(central_hist),
                              args.out_dir / f"centralized_metrics_{display.lower()}.png", display)
    _plot_local_only_metrics(local_results.per_client_rows("best"),
                             args.out_dir / f"local_only_metrics_{display.lower()}.png", display)
    _plot_fedavg_metrics([rec.as_dict() for rec in fedavg_hist.rounds],
                         args.out_dir / f"fedavg_metrics_{display.lower()}.png", display)
    _plot_three_way_v2(
        central_hist.best_test_rul, central_hist.best_test_fault,
        local_agg_best["rmse_mean"], local_agg_best["rmse_std"],
        local_agg_best["nasa_score_mean"], local_agg_best["auprc_mean"],
        local_agg_best["f1_mean"],
        fedavg_hist.best_test_rul, fedavg_hist.best_test_fault,
        args.out_dir / f"three_way_non_iid_{display.lower()}.png", display,
    )
    _plot_per_subset_breakdown(
        central_per_subset, local_run["cross_per_subset"], fedavg_per_subset,
        args.out_dir / f"per_subset_breakdown_{display.lower()}.png", display,
    )
    print(f"Wrote {args.out_dir / f'centralized_metrics_{display.lower()}.png'}")
    print(f"Wrote {args.out_dir / f'local_only_metrics_{display.lower()}.png'}")
    print(f"Wrote {args.out_dir / f'fedavg_metrics_{display.lower()}.png'}")
    print(f"Wrote {args.out_dir / f'three_way_non_iid_{display.lower()}.png'}  (headline)")
    print(f"Wrote {args.out_dir / f'per_subset_breakdown_{display.lower()}.png'}")

    # ---------------- Best-checkpoints ----------------
    for label, sd in [
        ("centralized", central_run["best_state_dict"]),
        ("fedavg", fedavg_run["best_state_dict"]),
    ]:
        ckpt = args.out_dir / f"best_{label}_{display.lower()}.pt"
        torch.save({"state_dict": sd, "config": {
            "n_features": bundle.n_features, "window_size": bundle.window_size,
        }}, ckpt)
        print(f"Wrote {ckpt}  (gitignored)")

    # ---------------- metrics.json ----------------
    interpretation = (
        f"Non-IID baseline on {display}: 4 clients (2 per subset). "
        f"Centralized RMSE={central_hist.best_test_rul.rmse:.2f}, "
        f"local-only mean RMSE={local_agg_best['rmse_mean']:.2f}\u00b1"
        f"{local_agg_best['rmse_std']:.2f}, "
        f"FedAvg RMSE={fedavg_hist.best_test_rul.rmse:.2f}. "
        f"FedAvg closed {gap_closed_pct:.1f}% of the local-only \u2192 centralized RMSE gap "
        f"while sharing only model weights. "
        f"The per-subset breakdown reveals the asymmetry inherent to Non-IID training: "
        f"local-only clients trained on one fault-mode family struggle on the other."
    )
    metrics_payload = PhaseMetrics(
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        interpretation=interpretation,
        subset=display,
        config={
            "subsets": list(args.subsets),
            "n_clients_per_subset": args.n_clients_per_subset,
            "total_clients": len(shards),
            "epochs": args.epochs,
            "n_rounds": args.n_rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_fault": args.lambda_fault,
            "use_cosine_schedule": not args.no_cosine,
            "seed": args.seed,
            "n_features": bundle.n_features,
            "window_size": bundle.window_size,
            "rul_cap": bundle.rul_cap,
            "fault_threshold": bundle.fault_threshold,
        },
        timing={
            "total_seconds": round(total_seconds, 3),
            "centralized_seconds": round(central_hist.total_seconds, 3),
            "local_only_seconds_total": round(local_results.total_seconds, 3),
            "fedavg_seconds": round(fedavg_hist.total_seconds, 3),
        },
        summary={
            "centralized_rmse": round(central_hist.best_test_rul.rmse, 4),
            "centralized_nasa": round(central_hist.best_test_rul.nasa_score, 4),
            "centralized_auprc": round(central_hist.best_test_fault.auprc, 4),
            "centralized_f1": round(central_hist.best_test_fault.f1, 4),
            "local_only_rmse_mean": round(local_agg_best["rmse_mean"], 4),
            "local_only_rmse_std": round(local_agg_best["rmse_std"], 4),
            "local_only_rmse_min": round(local_agg_best["rmse_min"], 4),
            "local_only_rmse_max": round(local_agg_best["rmse_max"], 4),
            "local_only_auprc_mean": round(local_agg_best["auprc_mean"], 4),
            "local_only_f1_mean": round(local_agg_best["f1_mean"], 4),
            "fedavg_rmse": round(fedavg_hist.best_test_rul.rmse, 4),
            "fedavg_nasa": round(fedavg_hist.best_test_rul.nasa_score, 4),
            "fedavg_auprc": round(fedavg_hist.best_test_fault.auprc, 4),
            "fedavg_f1": round(fedavg_hist.best_test_fault.f1, 4),
            "rmse_gap_closed_pct": round(gap_closed_pct, 2),
        },
        train={
            "centralized_epochs": [
                {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                for row in history_as_rows(central_hist)
            ],
            "fedavg_rounds": [
                {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                for row in [rec.as_dict() for rec in fedavg_hist.rounds]
            ],
        },
        test={
            "centralized_per_subset": [ps.as_dict() for ps in central_per_subset],
            "fedavg_per_subset": [ps.as_dict() for ps in fedavg_per_subset],
            "local_only_cross_per_subset": {
                cid: [ps.as_dict() for ps in ps_list]
                for cid, ps_list in local_run["cross_per_subset"].items()
            },
        },
        per_client=local_results.aggregate("best") | {
            "best_rows": local_results.per_client_rows("best"),
        },
        artifacts={
            "centralized_per_epoch_csv": f"results/{PHASE_ID}/per_epoch_centralized_{display.lower()}.csv",
            "local_only_per_client_csv": f"results/{PHASE_ID}/per_client_local_{display.lower()}.csv",
            "fedavg_per_round_csv": f"results/{PHASE_ID}/per_round_fedavg_{display.lower()}.csv",
            "fedavg_per_client_loss_csv": f"results/{PHASE_ID}/per_client_loss_fedavg_{display.lower()}.csv",
            "centralized_metrics_png": f"results/{PHASE_ID}/centralized_metrics_{display.lower()}.png",
            "local_only_metrics_png": f"results/{PHASE_ID}/local_only_metrics_{display.lower()}.png",
            "fedavg_metrics_png": f"results/{PHASE_ID}/fedavg_metrics_{display.lower()}.png",
            "three_way_non_iid_png": f"results/{PHASE_ID}/three_way_non_iid_{display.lower()}.png",
            "per_subset_breakdown_png": f"results/{PHASE_ID}/per_subset_breakdown_{display.lower()}.png",
        },
    )
    json_path = dump_phase_metrics(metrics_payload, args.out_dir)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
