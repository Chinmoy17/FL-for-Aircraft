"""FedAvg baseline (Phase 5) — 4 clients, 50 rounds × 2 local epochs.

This is **the** baseline deliverable for Task 1 of the project brief. It runs
the canonical FedAvg protocol (McMahan et al., 2017) over an in-process
simulation: each round the global model is broadcast to every client, each
client trains for ``local_epochs`` epochs on only its own data, and the server
aggregates the client updates with a sample-count-weighted mean.

Outputs under ``results/05_fedavg/``:

    metrics.json                            structured for the React frontend
    per_round_<subset>.csv                  one row per communication round
    per_client_loss_<subset>.csv            client-by-round local loss matrix
    loss_curves_<subset>.png                global + per-client loss over rounds
    global_metrics_<subset>.png             test RMSE / NASA / AUPRC / F1 over rounds
    pred_vs_true_<subset>.png               final-round pred-vs-true RUL scatter
    three_way_comparison_<subset>.png       P3 vs P4 (mean) vs P5 — the headline image

Run from the repo root inside the .venv::

    python scripts/run_fedavg.py                       # FD001, 4 clients, 50 rounds × 2 local epochs
    python scripts/run_fedavg.py --n-rounds 30         # quick variant
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_aircraft.data import CMAPSSConfig  # noqa: E402
from fl_aircraft.fl import run_fedavg  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics  # noqa: E402

PHASE_ID = "05_fedavg"
PHASE_NAME = "Phase 5 — FedAvg baseline (4 clients, FD001)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--n-clients", type=int, default=4)
    p.add_argument("--n-rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-fault", type=float, default=0.5)
    p.add_argument("--no-cosine", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / PHASE_ID)
    return p.parse_args()


def _load_p3() -> dict[str, float] | None:
    p3 = REPO_ROOT / "results" / "03_centralized" / "metrics.json"
    if not p3.exists():
        return None
    with p3.open(encoding="utf-8") as fh:
        d = json.load(fh)
    return {
        "rmse": d["test"]["best_rul"]["rmse"],
        "nasa_score": d["test"]["best_rul"]["nasa_score"],
        "auprc": d["test"]["best_fault"]["auprc"],
        "f1": d["test"]["best_fault"]["f1"],
    }


def _load_p4() -> dict[str, float] | None:
    p4 = REPO_ROOT / "results" / "04_local_only" / "metrics.json"
    if not p4.exists():
        return None
    with p4.open(encoding="utf-8") as fh:
        d = json.load(fh)
    s = d["summary"]
    return {
        "rmse": s["best_rmse_mean"],
        "rmse_std": s["best_rmse_std"],
        "nasa_score": s["best_nasa_mean"],
        "auprc": s["best_auprc_mean"],
        "f1": s["best_f1_mean"],
    }


def _plot_loss_curves(history, per_round_client_losses, path: Path, subset: str) -> None:
    rounds = [rec.round for rec in history.rounds]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # Per-client lines first (light).
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(history.client_ids)))
    for color, cid in zip(colors, history.client_ids):
        ax.plot(rounds, per_round_client_losses[cid], color=color, alpha=0.5, label=cid, linewidth=1.0)
    # Mean across clients (bold).
    means = [rec.mean_client_loss_total for rec in history.rounds]
    ax.plot(rounds, means, color="black", linewidth=2.2, label="mean of clients")
    ax.set_xlabel("communication round")
    ax.set_ylabel("local train loss (Huber + λ·BCE)")
    ax.set_yscale("log")
    ax.set_title(f"P5 FedAvg — per-client local train loss ({subset})")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_global_metrics(history, path: Path, subset: str) -> None:
    rounds = [rec.round for rec in history.rounds]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(rounds, [rec.global_test_rmse for rec in history.rounds], color="crimson", label="RMSE")
    ax.plot(rounds, [rec.global_test_mae for rec in history.rounds], color="orange", label="MAE")
    ax.set_title("Global test RUL error")
    ax.set_xlabel("round"); ax.set_ylabel("cycles"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(rounds, [rec.global_test_nasa_score for rec in history.rounds], color="purple")
    ax.set_yscale("log")
    ax.set_title("Global test NASA score (log)")
    ax.set_xlabel("round"); ax.set_ylabel("NASA score"); ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(rounds, [rec.global_test_auprc for rec in history.rounds], color="indigo", label="AUPRC")
    ax.plot(rounds, [rec.global_test_f1 for rec in history.rounds], color="teal", label="F1")
    ax.set_title("Global test fault discrimination")
    ax.set_xlabel("round"); ax.set_ylabel("score"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.plot(rounds, [rec.global_test_precision for rec in history.rounds], color="orange", linestyle="--", label="Precision")
    ax.plot(rounds, [rec.global_test_recall for rec in history.rounds], color="crimson", linestyle="--", label="Recall")
    ax.set_title("Global test fault operating point")
    ax.set_xlabel("round"); ax.set_ylabel("score"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"P5 FedAvg — global model on common test set ({subset})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_pred_vs_true(y_true, y_pred, path: Path, subset: str, rul_cap: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=20, alpha=0.6, color="seagreen", edgecolor="white")
    lim = max(rul_cap, float(y_true.max()), float(y_pred.max())) * 1.05
    ax.plot([0, lim], [0, lim], color="red", linestyle="--", label="perfect")
    ax.set_xlabel("true RUL (cycles, capped)")
    ax.set_ylabel("predicted RUL (cycles)")
    ax.set_title(f"P5 FedAvg — pred vs true RUL ({subset}, final round)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_three_way(p3, p4, p5_best_rul, p5_best_fault, path: Path, subset: str) -> None:
    """The headline image: P3 centralized vs P4 local-only mean vs P5 FedAvg."""
    if p3 is None or p4 is None:
        return
    metrics_meta = [
        ("rmse", "Test RMSE (lower = better)", False, None),
        ("nasa_score", "Test NASA score (log, lower)", True, None),
        ("auprc", "Test AUPRC", False, (0, 1.05)),
        ("f1", "Test F1", False, (0, 1.05)),
    ]
    p5 = {
        "rmse": p5_best_rul.rmse,
        "nasa_score": p5_best_rul.nasa_score,
        "auprc": p5_best_fault.auprc,
        "f1": p5_best_fault.f1,
    }
    labels = ["Centralized\n(P3 upper)", "Local-only mean\n(P4 lower)", "FedAvg\n(P5)"]
    colors = ["seagreen", "steelblue", "crimson"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (key, title, log, ylim) in zip(axes, metrics_meta):
        values = [p3[key], p4[key], p5[key]]
        bars = ax.bar(labels, values, color=colors)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}" if key not in ("rmse", "nasa_score") else (f"{v:.2f}" if key == "rmse" else f"{v:.0f}"),
                ha="center", va="bottom", fontsize=9,
            )
        # error bar on the local-only mean (we know its std for RMSE only).
        if key == "rmse" and "rmse_std" in p4:
            ax.errorbar([labels[1]], [p4["rmse"]], yerr=[p4["rmse_std"]], color="black", capsize=4, fmt="none")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if log:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.suptitle(
        f"3-way baseline comparison  —  {subset}  (best-epoch / best-round metrics)",
        fontweight="bold",
    )
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

    print(
        f"--- Phase 5 FedAvg baseline "
        f"({cfg.subset}, {args.n_clients} clients, "
        f"{args.n_rounds} rounds × {args.local_epochs} local epochs) ---"
    )
    history = run_fedavg(
        cfg,
        n_clients=args.n_clients,
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

    # ---------------- CSVs ----------------
    round_rows = [rec.as_dict() for rec in history.rounds]
    per_round_csv = args.out_dir / f"per_round_{cfg.subset.lower()}.csv"
    _write_csv(round_rows, per_round_csv)

    client_loss_rows = []
    for r_idx, rec in enumerate(history.rounds):
        row = {"round": rec.round}
        for cid in history.client_ids:
            row[cid] = round(history.per_round_client_losses[cid][r_idx], 4)
        client_loss_rows.append(row)
    per_client_csv = args.out_dir / f"per_client_loss_{cfg.subset.lower()}.csv"
    _write_csv(client_loss_rows, per_client_csv)
    print(f"\nWrote {per_round_csv}\nWrote {per_client_csv}")

    # ---------------- Plots ----------------
    loss_png = args.out_dir / f"loss_curves_{cfg.subset.lower()}.png"
    global_png = args.out_dir / f"global_metrics_{cfg.subset.lower()}.png"
    pred_png = args.out_dir / f"pred_vs_true_{cfg.subset.lower()}.png"
    threeway_png = args.out_dir / f"three_way_comparison_{cfg.subset.lower()}.png"

    _plot_loss_curves(history, history.per_round_client_losses, loss_png, cfg.subset)
    _plot_global_metrics(history, global_png, cfg.subset)
    _plot_pred_vs_true(
        history.final_predictions["y_rul_true"],
        history.final_predictions["y_rul_pred"],
        pred_png, cfg.subset, cfg.rul_cap,
    )
    p3 = _load_p3()
    p4 = _load_p4()
    _plot_three_way(p3, p4, history.best_test_rul, history.best_test_fault, threeway_png, cfg.subset)
    print(f"Wrote {loss_png}\nWrote {global_png}\nWrote {pred_png}")
    if p3 is not None and p4 is not None:
        print(f"Wrote {threeway_png}  (3-way comparison)")

    # ---------------- Console summary ----------------
    print("\n--- Best round ---")
    print(f"  round {history.best_round} / {args.n_rounds}")
    print(
        f"  RUL  : RMSE={history.best_test_rul.rmse:.3f}  "
        f"MAE={history.best_test_rul.mae:.3f}  "
        f"NASA={history.best_test_rul.nasa_score:.1f}"
    )
    print(
        f"  Fault: AUPRC={history.best_test_fault.auprc:.3f}  "
        f"F1={history.best_test_fault.f1:.3f}  "
        f"P={history.best_test_fault.precision:.3f}  "
        f"R={history.best_test_fault.recall:.3f}"
    )
    print("\n--- Final round ---")
    print(
        f"  RUL  : RMSE={history.final_test_rul.rmse:.3f}  "
        f"NASA={history.final_test_rul.nasa_score:.1f}  "
        f"AUPRC={history.final_test_fault.auprc:.3f}  "
        f"F1={history.final_test_fault.f1:.3f}"
    )
    if p3 is not None and p4 is not None:
        print("\n--- 3-way comparison (best metrics) ---")
        print(f"  Centralized (P3) : RMSE={p3['rmse']:.2f}  AUPRC={p3['auprc']:.3f}  F1={p3['f1']:.3f}")
        print(f"  Local-only  (P4) : RMSE={p4['rmse']:.2f}±{p4.get('rmse_std', 0):.2f}  "
              f"AUPRC={p4['auprc']:.3f}  F1={p4['f1']:.3f}")
        print(f"  FedAvg      (P5) : RMSE={history.best_test_rul.rmse:.2f}  "
              f"AUPRC={history.best_test_fault.auprc:.3f}  F1={history.best_test_fault.f1:.3f}")
        gap_close = (p4["rmse"] - history.best_test_rul.rmse) / (p4["rmse"] - p3["rmse"]) * 100 \
            if (p4["rmse"] - p3["rmse"]) > 1e-6 else 0.0
        print(f"  FedAvg closed {gap_close:.1f}% of the local-only → centralized RMSE gap")

    print(f"\nTotal wall-clock: {history.total_seconds:.1f} s "
          f"({history.total_seconds / args.n_rounds:.2f} s/round)")

    # ---------------- Best-model checkpoint ----------------
    ckpt_path = args.out_dir / f"best_global_model_{cfg.subset.lower()}.pt"
    torch.save(
        {
            "round": history.best_round,
            "state_dict": history.best_state_dict,
            "config": {"n_features": cfg.n_features, "window_size": cfg.window_size},
        },
        ckpt_path,
    )
    print(f"Wrote {ckpt_path}  (gitignored)")

    # ---------------- Structured metrics.json ----------------
    gap_close_pct = None
    if p3 is not None and p4 is not None and (p4["rmse"] - p3["rmse"]) > 1e-6:
        gap_close_pct = float(
            (p4["rmse"] - history.best_test_rul.rmse) / (p4["rmse"] - p3["rmse"]) * 100
        )
    interpretation_parts = [
        f"FedAvg baseline on {cfg.subset}: {args.n_clients} clients × "
        f"{args.n_rounds} communication rounds × {args.local_epochs} local epochs "
        f"({args.n_clients * args.n_rounds * args.local_epochs} total local-epoch equivalents).",
        f"Best round {history.best_round}/{args.n_rounds} reached "
        f"RMSE={history.best_test_rul.rmse:.2f}, "
        f"NASA={history.best_test_rul.nasa_score:.0f}, "
        f"AUPRC={history.best_test_fault.auprc:.3f}.",
    ]
    if gap_close_pct is not None and p3 is not None and p4 is not None:
        interpretation_parts.append(
            f"Closed {gap_close_pct:.1f}% of the local-only ({p4['rmse']:.2f}) → "
            f"centralized ({p3['rmse']:.2f}) RMSE gap by sharing only model weights, "
            f"never raw sensor data."
        )

    metrics_payload = PhaseMetrics(
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        interpretation=" ".join(interpretation_parts),
        subset=cfg.subset,
        config={
            "n_clients": args.n_clients,
            "n_rounds": args.n_rounds,
            "local_epochs": args.local_epochs,
            "total_local_epoch_equivalents": args.n_clients * args.n_rounds * args.local_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_fault": args.lambda_fault,
            "use_cosine_schedule": not args.no_cosine,
            "aggregation": "sample-count-weighted FedAvg (McMahan 2017)",
            "seed": args.seed,
            "n_features": cfg.n_features,
            "window_size": cfg.window_size,
            "rul_cap": cfg.rul_cap,
            "fault_threshold": cfg.fault_threshold,
        },
        timing={
            "train_seconds": round(history.total_seconds, 3),
            "train_seconds_per_round": round(history.total_seconds / args.n_rounds, 3),
        },
        summary={
            "best_round": history.best_round,
            "final_round": args.n_rounds,
            "best_global_rmse": round(history.best_test_rul.rmse, 4),
            "best_global_nasa": round(history.best_test_rul.nasa_score, 4),
            "best_global_auprc": round(history.best_test_fault.auprc, 4),
            "best_global_f1": round(history.best_test_fault.f1, 4),
            "final_global_rmse": round(history.final_test_rul.rmse, 4),
            "final_global_nasa": round(history.final_test_rul.nasa_score, 4),
            "final_global_auprc": round(history.final_test_fault.auprc, 4),
            "final_global_f1": round(history.final_test_fault.f1, 4),
            "p3_centralized_rmse": p3["rmse"] if p3 else None,
            "p4_local_only_rmse_mean": p4["rmse"] if p4 else None,
            "rmse_gap_closed_pct": (
                round(gap_close_pct, 2) if gap_close_pct is not None else None
            ),
        },
        train={"rounds": [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
            for row in round_rows
        ]},
        test={
            "best_rul": {k: round(v, 4) for k, v in history.best_test_rul.as_dict().items()},
            "best_fault": {k: round(v, 4) for k, v in history.best_test_fault.as_dict().items()},
            "final_rul": {k: round(v, 4) for k, v in history.final_test_rul.as_dict().items()},
            "final_fault": {k: round(v, 4) for k, v in history.final_test_fault.as_dict().items()},
        },
        per_client={
            cid: {
                "local_loss_per_round": [
                    round(v, 4) for v in history.per_round_client_losses[cid]
                ],
            }
            for cid in history.client_ids
        },
        artifacts={
            "per_round_csv": f"results/{PHASE_ID}/per_round_{cfg.subset.lower()}.csv",
            "per_client_loss_csv": f"results/{PHASE_ID}/per_client_loss_{cfg.subset.lower()}.csv",
            "loss_curves_png": f"results/{PHASE_ID}/loss_curves_{cfg.subset.lower()}.png",
            "global_metrics_png": f"results/{PHASE_ID}/global_metrics_{cfg.subset.lower()}.png",
            "pred_vs_true_png": f"results/{PHASE_ID}/pred_vs_true_{cfg.subset.lower()}.png",
            "three_way_comparison_png": (
                f"results/{PHASE_ID}/three_way_comparison_{cfg.subset.lower()}.png"
                if (p3 is not None and p4 is not None) else ""
            ),
        },
    )
    json_path = dump_phase_metrics(metrics_payload, args.out_dir)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
