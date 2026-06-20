"""Full centralized baseline (Phase 3): 50 epochs on FD001.

This is the **upper-bound** every federated learning run will be compared
against. It pools all training engines and trains the multi-task CNN with the
cosine-annealed Adam recipe from ``fl_aircraft.train.centralized``.

Outputs under ``results/03_centralized/``:

    metrics.json             structured for the React frontend
    per_epoch_<subset>.csv   one row per epoch with all train + test metrics
    loss_curve_<subset>.png  per-epoch train loss
    rul_metrics_<subset>.png test RMSE / MAE / NASA across epochs
    fault_metrics_<subset>.png test AUPRC / F1 / Precision / Recall across epochs
    pred_vs_true_<subset>.png pred-vs-true RUL scatter at the final epoch

Run from the repo root inside the .venv::

    python scripts/run_centralized.py                  # default: FD001, 50 epochs
    python scripts/run_centralized.py --epochs 30      # quick run
    python scripts/run_centralized.py --subset FD003   # different subset
"""
from __future__ import annotations

import argparse
import csv
import sys
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
    CMAPSSWindowDataset,
    Normalizer,
    load_and_label_train,
    load_raw,
    load_test_rul,
    make_test_windows,
    make_training_windows,
)
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss  # noqa: E402
from fl_aircraft.train import history_as_rows, train_centralized  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics, seed_everything  # noqa: E402

PHASE_ID = "03_centralized"
PHASE_NAME = "Phase 3 — Full centralized baseline"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-fault", type=float, default=0.5)
    p.add_argument("--no-cosine", action="store_true", help="Disable cosine LR annealing.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / PHASE_ID,
        help="Where to write per_epoch_*.csv, plots, and metrics.json.",
    )
    return p.parse_args()


def _plot_loss(history_rows: list[dict], path: Path, subset: str) -> None:
    epochs = [r["epoch"] for r in history_rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(epochs, [r["train_loss_total"] for r in history_rows], color="steelblue", label="train total")
    ax.plot(epochs, [r["train_loss_rul"] for r in history_rows], color="darkorange", linestyle="--", label="train RUL (Huber)")
    ax.plot(epochs, [r["train_loss_fault"] for r in history_rows], color="seagreen", linestyle=":", label="train fault (BCE)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(f"P3 centralized — training loss ({subset})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_rul_metrics(history_rows: list[dict], path: Path, subset: str) -> None:
    epochs = [r["epoch"] for r in history_rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, [r["test_rmse"] for r in history_rows], color="crimson", label="RMSE")
    ax1.plot(epochs, [r["test_mae"] for r in history_rows], color="orange", label="MAE")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cycles")
    ax1.set_title(f"Test RUL error ({subset})")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, [r["test_nasa_score"] for r in history_rows], color="purple")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("NASA score (lower = better)")
    ax2.set_yscale("log")
    ax2.set_title(f"Test NASA score ({subset})")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_fault_metrics(history_rows: list[dict], path: Path, subset: str) -> None:
    epochs = [r["epoch"] for r in history_rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, [r["test_auprc"] for r in history_rows], color="indigo", label="AUPRC")
    ax.plot(epochs, [r["test_f1"] for r in history_rows], color="teal", label="F1")
    ax.plot(epochs, [r["test_precision"] for r in history_rows], color="orange", linestyle="--", label="Precision")
    ax.plot(epochs, [r["test_recall"] for r in history_rows], color="crimson", linestyle="--", label="Recall")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Test fault detection ({subset})")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_pred_vs_true(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path, subset: str, rul_cap: int
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=20, alpha=0.6, color="steelblue", edgecolor="white")
    lim = max(rul_cap, float(y_true.max()), float(y_pred.max())) * 1.05
    ax.plot([0, lim], [0, lim], color="red", linestyle="--", label="perfect")
    ax.set_xlabel("true RUL (cycles, capped)")
    ax.set_ylabel("predicted RUL (cycles)")
    ax.set_title(f"P3 centralized — pred vs true RUL ({subset}, final epoch)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
    cfg = CMAPSSConfig(subset=args.subset, data_dir=data_dir)

    # ---------------- Data ----------------
    print(f"--- Phase 3 centralized baseline ({cfg.subset}, {args.epochs} epochs) ---")
    train_df = load_and_label_train(cfg)
    normalizer = Normalizer.fit(train_df, cfg.feature_cols)
    train_arrays = make_training_windows(
        normalizer.transform(train_df), cfg.feature_cols, cfg.window_size, cfg.stride
    )
    test_arrays = make_test_windows(
        normalizer.transform(load_raw(cfg.subset, "test", data_dir)),
        load_test_rul(cfg.subset, data_dir),
        cfg.feature_cols,
        cfg.window_size,
        cfg.rul_cap,
        cfg.fault_threshold,
    )
    train_ds = CMAPSSWindowDataset(train_arrays)
    test_ds = CMAPSSWindowDataset(test_arrays)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"train windows : {len(train_ds):,}  features={cfg.n_features}  window={cfg.window_size}")
    print(f"test windows  : {len(test_ds)}")

    # ---------------- Model / loss ----------------
    model = MultiTaskCNN(MultiTaskCNNConfig(n_features=cfg.n_features, window_size=cfg.window_size))
    n_pos = int(train_arrays.y_fault.sum())
    n_neg = int(train_arrays.y_fault.shape[0] - n_pos)
    pos_weight = float(n_neg) / float(max(n_pos, 1))
    loss_fn = MultiTaskLoss(lambda_fault=args.lambda_fault, fault_pos_weight=pos_weight)
    print(f"model params  : {model.count_parameters():,}  (pos_weight={pos_weight:.2f})")

    # ---------------- Train ----------------
    history = train_centralized(
        model,
        train_loader,
        test_loader,
        loss_fn,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_cosine_schedule=not args.no_cosine,
        log_every=args.log_every,
    )

    # ---------------- Persist ----------------
    rows = history_as_rows(history)
    csv_path = args.out_dir / f"per_epoch_{cfg.subset.lower()}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    loss_path = args.out_dir / f"loss_curve_{cfg.subset.lower()}.png"
    rul_path = args.out_dir / f"rul_metrics_{cfg.subset.lower()}.png"
    fault_path = args.out_dir / f"fault_metrics_{cfg.subset.lower()}.png"
    pred_path = args.out_dir / f"pred_vs_true_{cfg.subset.lower()}.png"

    _plot_loss(rows, loss_path, cfg.subset)
    _plot_rul_metrics(rows, rul_path, cfg.subset)
    _plot_fault_metrics(rows, fault_path, cfg.subset)
    _plot_pred_vs_true(
        history.final_predictions["y_rul_true"],
        history.final_predictions["y_rul_pred"],
        pred_path,
        cfg.subset,
        cfg.rul_cap,
    )
    print(f"Wrote {loss_path}\nWrote {rul_path}\nWrote {fault_path}\nWrote {pred_path}")

    print("\n--- Best epoch ---")
    print(f"  epoch {history.best_epoch} / {args.epochs}")
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
    print("\n--- Final epoch ---")
    print(
        f"  RUL  : RMSE={history.final_test_rul.rmse:.3f}  "
        f"MAE={history.final_test_rul.mae:.3f}  "
        f"NASA={history.final_test_rul.nasa_score:.1f}"
    )
    print(
        f"  Fault: AUPRC={history.final_test_fault.auprc:.3f}  "
        f"F1={history.final_test_fault.f1:.3f}  "
        f"P={history.final_test_fault.precision:.3f}  "
        f"R={history.final_test_fault.recall:.3f}"
    )
    print(f"\nTotal training time: {history.total_seconds:.1f} s "
          f"({history.total_seconds / args.epochs:.2f} s/epoch)")

    # ---------------- Save best-epoch checkpoint ----------------
    # Untracked (.gitignore covers *.pt) but useful for P5 / RQ work.
    ckpt_path = args.out_dir / f"best_model_{cfg.subset.lower()}.pt"
    torch.save(
        {
            "epoch": history.best_epoch,
            "state_dict": history.best_state_dict,
            "config": {
                "n_features": cfg.n_features,
                "window_size": cfg.window_size,
            },
        },
        ckpt_path,
    )
    print(f"Wrote {ckpt_path}  (gitignored)")

    # ---------------- Structured metrics.json for the frontend ----------------
    interpretation = (
        f"Centralized upper-bound baseline on {cfg.subset}: "
        f"{args.epochs} epochs of cosine-annealed Adam (lr={args.lr}, wd={args.weight_decay}). "
        f"Best epoch {history.best_epoch}/{args.epochs} reached RMSE={history.best_test_rul.rmse:.2f}, "
        f"NASA={history.best_test_rul.nasa_score:.0f}, AUPRC={history.best_test_fault.auprc:.3f}. "
        f"Total training time {history.total_seconds:.1f}s on CPU. "
        f"This is the upper-bound score every federated learning run "
        f"(P4 local-only, P5 FedAvg) will be compared against."
    )
    metrics_payload = PhaseMetrics(
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        interpretation=interpretation,
        subset=cfg.subset,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_fault": args.lambda_fault,
            "pos_weight": round(pos_weight, 3),
            "use_cosine_schedule": not args.no_cosine,
            "seed": args.seed,
            "n_features": cfg.n_features,
            "window_size": cfg.window_size,
            "stride": cfg.stride,
            "rul_cap": cfg.rul_cap,
            "fault_threshold": cfg.fault_threshold,
            "n_train_windows": len(train_ds),
            "n_test_windows": len(test_ds),
        },
        timing={
            "train_seconds": round(history.total_seconds, 3),
            "train_seconds_per_epoch": round(history.total_seconds / args.epochs, 3),
        },
        summary={
            "model_name": "MultiTaskCNN",
            "model_n_parameters": model.count_parameters(),
            "best_epoch": history.best_epoch,
            "final_epoch": args.epochs,
            "best_test_rmse": round(history.best_test_rul.rmse, 4),
            "best_test_nasa": round(history.best_test_rul.nasa_score, 4),
            "best_test_auprc": round(history.best_test_fault.auprc, 4),
            "best_test_f1": round(history.best_test_fault.f1, 4),
            "final_test_rmse": round(history.final_test_rul.rmse, 4),
            "final_test_nasa": round(history.final_test_rul.nasa_score, 4),
            "final_test_auprc": round(history.final_test_fault.auprc, 4),
            "final_test_f1": round(history.final_test_fault.f1, 4),
        },
        train={
            "epochs": [
                {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                for row in rows
            ],
        },
        test={
            "best_rul": {k: round(v, 4) for k, v in history.best_test_rul.as_dict().items()},
            "best_fault": {k: round(v, 4) for k, v in history.best_test_fault.as_dict().items()},
            "final_rul": {k: round(v, 4) for k, v in history.final_test_rul.as_dict().items()},
            "final_fault": {k: round(v, 4) for k, v in history.final_test_fault.as_dict().items()},
        },
        artifacts={
            "per_epoch_csv": f"results/{PHASE_ID}/per_epoch_{cfg.subset.lower()}.csv",
            "loss_curve_png": f"results/{PHASE_ID}/loss_curve_{cfg.subset.lower()}.png",
            "rul_metrics_png": f"results/{PHASE_ID}/rul_metrics_{cfg.subset.lower()}.png",
            "fault_metrics_png": f"results/{PHASE_ID}/fault_metrics_{cfg.subset.lower()}.png",
            "pred_vs_true_png": f"results/{PHASE_ID}/pred_vs_true_{cfg.subset.lower()}.png",
        },
    )
    json_path = dump_phase_metrics(metrics_payload, args.out_dir)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
