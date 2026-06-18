"""Local-only baseline (Phase 4): one model per simulated airline, no sharing.

Train one multi-task CNN per client on *only* that client's data, evaluate every
client on the common FD001 test set, and aggregate the per-client results.

This is the **lower bound** for the FL story: the FedAvg run (Phase 5) must
beat the *average* per-client local-only score to justify the federation.

Outputs under ``results/04_local_only/``:

    metrics.json                          structured for the React frontend
    per_client_best_<subset>.csv          one row per client (best-epoch metrics)
    per_client_final_<subset>.csv         one row per client (final-epoch metrics)
    per_client_metrics_<subset>.png       grouped bar chart per metric
    centralized_vs_local_<subset>.png     P3 vs P4-mean side-by-side
    loss_curves_<subset>.png              one line per client, train loss over epochs

Run from the repo root inside the .venv::

    python scripts/run_local_only.py                      # FD001, 4 clients, 50 epochs
    python scripts/run_local_only.py --epochs 30          # quick variant
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_aircraft.data import CMAPSSConfig  # noqa: E402
from fl_aircraft.train import train_local_only_clients  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics  # noqa: E402

PHASE_ID = "04_local_only"
PHASE_NAME = "Phase 4 — Local-only baseline (no sharing)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--n-clients", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-fault", type=float, default=0.5)
    p.add_argument("--no-cosine", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / PHASE_ID,
    )
    return p.parse_args()


def _load_p3_baseline() -> dict[str, float] | None:
    """Pull the P3 centralized best-epoch metrics for the comparison plot."""
    p3 = REPO_ROOT / "results" / "03_centralized" / "metrics.json"
    if not p3.exists():
        return None
    with p3.open(encoding="utf-8") as fh:
        d = json.load(fh)
    return {
        "rmse": d["test"]["best_rul"]["rmse"],
        "mae": d["test"]["best_rul"]["mae"],
        "nasa_score": d["test"]["best_rul"]["nasa_score"],
        "auprc": d["test"]["best_fault"]["auprc"],
        "f1": d["test"]["best_fault"]["f1"],
        "precision": d["test"]["best_fault"]["precision"],
        "recall": d["test"]["best_fault"]["recall"],
    }


def _plot_per_client_metrics(rows: list[dict], path: Path, subset: str) -> None:
    """Grouped bar chart: one bar per client per metric (best epoch)."""
    metrics = ["rmse", "nasa_score", "auprc", "f1"]
    metric_titles = ["Test RMSE (lower = better)", "Test NASA score (log, lower)",
                     "Test AUPRC", "Test F1"]
    client_ids = [r["client_id"] for r in rows]
    n_clients = len(client_ids)
    colors = plt.colormaps["tab10"](np.linspace(0, 1, n_clients))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, metric, title in zip(axes.flat, metrics, metric_titles):
        values = [r[metric] for r in rows]
        bars = ax.bar(client_ids, values, color=colors)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.2f}" if metric != "nasa_score" else f"{v:.0f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if metric == "nasa_score":
            ax.set_yscale("log")
        ax.set_ylim(bottom=0 if metric in ("auprc", "f1") else None,
                    top=1.05 if metric in ("auprc", "f1") else None)
    fig.suptitle(f"P4 local-only — per-client best metrics ({subset})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_centralized_vs_local(
    rows: list[dict],
    p3_metrics: dict[str, float] | None,
    path: Path,
    subset: str,
) -> None:
    """Side-by-side: P3 centralized vs P4 per-client + mean."""
    if p3_metrics is None:
        return
    metrics = ["rmse", "auprc", "f1"]
    metric_titles = ["Test RMSE (lower = better)", "Test AUPRC", "Test F1"]
    client_ids = [r["client_id"] for r in rows]
    n_clients = len(client_ids)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, title in zip(axes, metrics, metric_titles):
        local_vals = [r[metric] for r in rows]
        local_mean = float(np.mean(local_vals))
        local_std = float(np.std(local_vals, ddof=0))
        labels = ["Centralized\n(P3)"] + client_ids + ["Local-only\nmean (P4)"]
        values = [p3_metrics[metric]] + local_vals + [local_mean]
        colors = ["seagreen"] + ["steelblue"] * n_clients + ["crimson"]
        bars = ax.bar(labels, values, color=colors)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}" if metric != "rmse" else f"{v:.2f}",
                ha="center", va="bottom", fontsize=9,
            )
        # error bar on the mean
        ax.errorbar(
            [labels[-1]], [local_mean], yerr=[local_std],
            color="black", capsize=4, fmt="none",
        )
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if metric in ("auprc", "f1"):
            ax.set_ylim(0, 1.05)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.suptitle(
        f"Centralized (P3) vs Local-only (P4)  —  {subset}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_curves(results, path: Path, subset: str) -> None:
    """One per-client training-loss line on a single axis."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(results.clients)))
    for color, client in zip(colors, results.clients):
        epochs = [rec.epoch for rec in client.history.epochs]
        losses = [rec.train_loss_total for rec in client.history.epochs]
        ax.plot(epochs, losses, color=color, label=client.client_id, alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (Huber + λ·BCE)")
    ax.set_yscale("log")
    ax.set_title(f"P4 local-only — per-client training loss ({subset})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
    cfg = CMAPSSConfig(subset=args.subset, data_dir=data_dir)

    print(f"--- Phase 4 local-only baseline "
          f"({cfg.subset}, {args.n_clients} clients, {args.epochs} epochs) ---")
    results = train_local_only_clients(
        cfg,
        n_clients=args.n_clients,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_fault=args.lambda_fault,
        use_cosine_schedule=not args.no_cosine,
        seed=args.seed,
        log_every=args.log_every,
    )

    # ---------------- Aggregate + persist ----------------
    best_rows = results.per_client_rows("best")
    final_rows = results.per_client_rows("final")
    agg_best = results.aggregate("best")
    agg_final = results.aggregate("final")

    print("\n=== Local-only summary (BEST epoch per client) ===")
    for r in best_rows:
        print(
            f"  {r['client_id']:>9s}  "
            f"RMSE={r['rmse']:5.2f}  NASA={r['nasa_score']:6.1f}  "
            f"AUPRC={r['auprc']:.3f}  F1={r['f1']:.3f}  "
            f"(best epoch {r['best_epoch']:>2d}, {r['n_train_windows']:>5d} windows)"
        )
    print(
        f"  mean      RMSE={agg_best['rmse_mean']:5.2f} \u00b1{agg_best['rmse_std']:.2f}  "
        f"NASA={agg_best['nasa_score_mean']:6.1f}  "
        f"AUPRC={agg_best['auprc_mean']:.3f}  F1={agg_best['f1_mean']:.3f}"
    )
    print(f"\nTotal time across all clients: {results.total_seconds:.1f} s "
          f"({results.total_seconds / len(best_rows):.2f} s/client)")

    best_csv = args.out_dir / f"per_client_best_{cfg.subset.lower()}.csv"
    final_csv = args.out_dir / f"per_client_final_{cfg.subset.lower()}.csv"
    _write_csv(best_rows, best_csv)
    _write_csv(final_rows, final_csv)
    print(f"\nWrote {best_csv}\nWrote {final_csv}")

    # ---------------- Plots ----------------
    per_client_png = args.out_dir / f"per_client_metrics_{cfg.subset.lower()}.png"
    loss_curves_png = args.out_dir / f"loss_curves_{cfg.subset.lower()}.png"
    vs_centralized_png = args.out_dir / f"centralized_vs_local_{cfg.subset.lower()}.png"

    _plot_per_client_metrics(best_rows, per_client_png, cfg.subset)
    _plot_loss_curves(results, loss_curves_png, cfg.subset)
    p3 = _load_p3_baseline()
    _plot_centralized_vs_local(best_rows, p3, vs_centralized_png, cfg.subset)
    print(f"Wrote {per_client_png}\nWrote {loss_curves_png}")
    if p3 is not None:
        print(f"Wrote {vs_centralized_png}")
    else:
        print("(P3 baseline metrics not found — skipping centralized_vs_local plot)")

    # ---------------- metrics.json ----------------
    rmse_gap = (
        round(agg_best["rmse_mean"] - p3["rmse"], 4) if p3 is not None else None
    )
    interpretation_parts = [
        f"Lower-bound baseline on {cfg.subset}: {args.n_clients} clients trained in "
        f"isolation, no weight sharing, same {args.epochs}-epoch recipe as the P3 "
        f"centralized run.",
        f"Average per-client best RMSE = {agg_best['rmse_mean']:.2f} \u00b1 "
        f"{agg_best['rmse_std']:.2f}  (range "
        f"{agg_best['rmse_min']:.2f}\u2013{agg_best['rmse_max']:.2f}).",
    ]
    if p3 is not None and rmse_gap is not None:
        interpretation_parts.append(
            f"Penalty for isolation vs centralized (P3 RMSE={p3['rmse']:.2f}): "
            f"+{rmse_gap:.2f} RMSE. The FedAvg run (P5) must close this gap to "
            f"justify the federation."
        )
    interpretation = " ".join(interpretation_parts)

    metrics_payload = PhaseMetrics(
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        interpretation=interpretation,
        subset=cfg.subset,
        config={
            "n_clients": args.n_clients,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_fault": args.lambda_fault,
            "use_cosine_schedule": not args.no_cosine,
            "seed": args.seed,
            "n_features": cfg.n_features,
            "window_size": cfg.window_size,
            "rul_cap": cfg.rul_cap,
            "fault_threshold": cfg.fault_threshold,
        },
        timing={
            "train_seconds_total": round(results.total_seconds, 3),
            "train_seconds_per_client_mean": round(
                results.total_seconds / max(len(best_rows), 1), 3
            ),
        },
        summary={
            "best_rmse_mean": round(agg_best["rmse_mean"], 4),
            "best_rmse_std": round(agg_best["rmse_std"], 4),
            "best_rmse_min": round(agg_best["rmse_min"], 4),
            "best_rmse_max": round(agg_best["rmse_max"], 4),
            "best_nasa_mean": round(agg_best["nasa_score_mean"], 4),
            "best_auprc_mean": round(agg_best["auprc_mean"], 4),
            "best_auprc_min": round(agg_best["auprc_min"], 4),
            "best_f1_mean": round(agg_best["f1_mean"], 4),
            "final_rmse_mean": round(agg_final["rmse_mean"], 4),
            "final_auprc_mean": round(agg_final["auprc_mean"], 4),
            "p3_centralized_rmse": p3["rmse"] if p3 else None,
            "p3_centralized_auprc": p3["auprc"] if p3 else None,
            "rmse_gap_vs_centralized": rmse_gap,
        },
        per_client={
            r["client_id"]: {k: v for k, v in r.items() if k != "client_id"}
            for r in best_rows
        },
        artifacts={
            "per_client_best_csv": f"results/{PHASE_ID}/per_client_best_{cfg.subset.lower()}.csv",
            "per_client_final_csv": f"results/{PHASE_ID}/per_client_final_{cfg.subset.lower()}.csv",
            "per_client_metrics_png": f"results/{PHASE_ID}/per_client_metrics_{cfg.subset.lower()}.png",
            "loss_curves_png": f"results/{PHASE_ID}/loss_curves_{cfg.subset.lower()}.png",
            "centralized_vs_local_png": (
                f"results/{PHASE_ID}/centralized_vs_local_{cfg.subset.lower()}.png"
                if p3 is not None else ""
            ),
        },
    )
    json_path = dump_phase_metrics(metrics_payload, args.out_dir)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
