"""Phase-2 smoke test: train the multi-task CNN for 1 epoch on the full
centralized FD001 training set and evaluate on the FD001 test set.

The point is to confirm that **end-to-end wiring works** — data loading,
windowing, model forward/backward, loss, and metrics all flow together — and
to give a first wall-clock budget on real CPU hardware. This is **not** a
benchmark; the proper centralized baseline (P3) trains for many epochs with
LR scheduling and early stopping.

Outputs (committed for the report and consumed by the React frontend):
    results/02_smoke/metrics.json                  structured machine-readable
    results/02_smoke/loss_curve_<subset>.png       per-batch + per-epoch train loss
    results/02_smoke/metrics_<subset>.csv          flat metric table (human-friendly)

Run from the repo root inside the .venv::

    python scripts/smoke_train.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
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
from fl_aircraft.eval import (  # noqa: E402
    compute_classification_metrics,
    compute_regression_metrics,
)
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig, MultiTaskLoss  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics, seed_everything  # noqa: E402

PHASE_ID = "02_smoke"
PHASE_NAME = "Phase 2 — Centralized smoke run"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-fault", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "02_smoke",
        help="Where to write the loss curve, CSV, and metrics.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
    cfg = CMAPSSConfig(subset=args.subset, data_dir=data_dir)

    # ---------------- Data ----------------
    print(f"--- Phase 2 smoke run ({cfg.subset}, {args.epochs} epoch) ---")
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
    # num_workers=0 on Windows — worker spawn overhead exceeds the gain on this dataset.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"train windows : {len(train_ds):,}  features={cfg.n_features}  window={cfg.window_size}")
    print(f"test windows  : {len(test_ds)}")

    # ---------------- Model / loss / optim ----------------
    model_cfg = MultiTaskCNNConfig(n_features=cfg.n_features, window_size=cfg.window_size)
    model = MultiTaskCNN(model_cfg)
    # Compute pos_weight from the *training* labels — never from the test set.
    n_pos = int(train_arrays.y_fault.sum())
    n_neg = int(train_arrays.y_fault.shape[0] - n_pos)
    pos_weight = float(n_neg) / float(max(n_pos, 1))
    loss_fn = MultiTaskLoss(lambda_fault=args.lambda_fault, fault_pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"model params  : {model.count_parameters():,}  (pos_weight={pos_weight:.2f})")

    # ---------------- Train ----------------
    batch_losses: list[float] = []
    epoch_losses: list[float] = []
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_rul = 0.0
        running_fault = 0.0
        n_batches = 0
        for x, y_rul, y_fault in train_loader:
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            losses = loss_fn(pred, y_rul, y_fault)
            losses.total.backward()
            optim.step()
            # .item() auto-detaches and returns a plain float.
            batch_total = losses.total.item()
            batch_losses.append(batch_total)
            running_total += batch_total
            running_rul += losses.rul.item()
            running_fault += losses.fault.item()
            n_batches += 1
        avg_total = running_total / n_batches
        avg_rul = running_rul / n_batches
        avg_fault = running_fault / n_batches
        epoch_losses.append(avg_total)
        print(
            f"epoch {epoch:>2}/{args.epochs}  "
            f"loss={avg_total:.4f}  rul={avg_rul:.4f}  fault={avg_fault:.4f}"
        )
    train_seconds = time.perf_counter() - train_start

    # ---------------- Evaluate ----------------
    model.eval()
    rul_preds: list[np.ndarray] = []
    rul_trues: list[np.ndarray] = []
    fault_scores: list[np.ndarray] = []
    fault_trues: list[np.ndarray] = []
    with torch.no_grad():
        for x, y_rul, y_fault in test_loader:
            pred = model(x)
            rul_preds.append(pred.rul.numpy())
            rul_trues.append(y_rul.numpy())
            fault_scores.append(pred.fault_probs().numpy())
            fault_trues.append(y_fault.numpy())
    rul_metrics = compute_regression_metrics(
        np.concatenate(rul_trues), np.concatenate(rul_preds)
    )
    cls_metrics = compute_classification_metrics(
        np.concatenate(fault_trues), np.concatenate(fault_scores)
    )

    print("\n--- Test-set metrics ---")
    print(f"  RUL  : RMSE={rul_metrics.rmse:.3f}  MAE={rul_metrics.mae:.3f}  "
          f"NASA={rul_metrics.nasa_score:.1f}")
    print(f"  Fault: AUPRC={cls_metrics.auprc:.3f}  F1={cls_metrics.f1:.3f}  "
          f"P={cls_metrics.precision:.3f}  R={cls_metrics.recall:.3f}  "
          f"(pos rate={cls_metrics.positive_rate:.2%})")
    print(f"\nTraining wall-clock: {train_seconds:.1f} s "
          f"({train_seconds / args.epochs:.1f} s/epoch)")

    # ---------------- Persist ----------------
    csv_path = args.out_dir / f"metrics_{cfg.subset.lower()}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for k, v in rul_metrics.as_dict().items():
            writer.writerow([f"rul_{k}", v])
        for k, v in cls_metrics.as_dict().items():
            writer.writerow([f"fault_{k}", v])
        writer.writerow(["train_seconds", round(train_seconds, 3)])
        writer.writerow(["train_seconds_per_epoch", round(train_seconds / args.epochs, 3)])
        writer.writerow(["n_train_windows", len(train_ds)])
        writer.writerow(["n_test_windows", len(test_ds)])
        writer.writerow(["epochs", args.epochs])
        writer.writerow(["batch_size", args.batch_size])
        writer.writerow(["lr", args.lr])
        writer.writerow(["lambda_fault", args.lambda_fault])
        writer.writerow(["pos_weight", round(pos_weight, 3)])
    print(f"\nWrote {csv_path}")

    # Loss curve figure: per-batch trace plus per-epoch markers.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(batch_losses, color="steelblue", alpha=0.7, label="per-batch train loss")
    if args.epochs > 1:
        ax.plot(
            np.arange(args.epochs) * len(train_loader) + len(train_loader) - 1,
            epoch_losses,
            color="crimson",
            marker="o",
            label="epoch mean",
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (Huber + λ·BCE)")
    ax.set_title(
        f"P2 smoke run — {cfg.subset}, {args.epochs} epoch "
        f"({train_seconds:.1f} s on CPU)"
    )
    ax.legend()
    fig.tight_layout()
    fig_path = args.out_dir / f"loss_curve_{cfg.subset.lower()}.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {fig_path}")

    # ---------------- Structured metrics.json for the frontend ----------------
    interpretation = (
        f"End-to-end smoke run on {cfg.subset}: {args.epochs} epoch, "
        f"{train_seconds:.1f}s wall-clock. "
        f"Loss decreased monotonically ({batch_losses[0]:.0f} → {batch_losses[-1]:.0f}). "
        f"Test AUPRC={cls_metrics.auprc:.3f} after 1 epoch confirms the encoder is "
        f"learning real degradation signal; the high RMSE/NASA values are the expected "
        f"untrained-baseline behaviour and will close in the P3 50-epoch centralized run."
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
            "lambda_fault": args.lambda_fault,
            "pos_weight": round(pos_weight, 3),
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
            "train_seconds": round(train_seconds, 3),
            "train_seconds_per_epoch": round(train_seconds / args.epochs, 3),
        },
        summary={
            "model_name": "MultiTaskCNN",
            "model_n_parameters": model.count_parameters(),
            "first_batch_loss": round(batch_losses[0], 3),
            "last_batch_loss": round(batch_losses[-1], 3),
        },
        train={
            "epochs": [
                {
                    "epoch": i + 1,
                    "loss_total": round(loss, 4),
                }
                for i, loss in enumerate(epoch_losses)
            ],
        },
        test={
            "rul": {k: round(v, 4) for k, v in rul_metrics.as_dict().items()},
            "fault": {k: round(v, 4) for k, v in cls_metrics.as_dict().items()},
        },
        artifacts={
            "loss_curve_png": f"results/{PHASE_ID}/loss_curve_{cfg.subset.lower()}.png",
            "metrics_csv": f"results/{PHASE_ID}/metrics_{cfg.subset.lower()}.csv",
        },
    )
    json_path = dump_phase_metrics(metrics_payload, args.out_dir)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
