"""Sanity-check the Phase 1 data pipeline end-to-end.

Loads FD001, partitions 100 engines into 4 simulated airline clients, applies
per-client preprocessing + sliding-window construction, and prints / saves a
per-client summary. The figure produced here is the **RQ2 hook** — it shows
how the fault positive rate diverges across clients once you partition.

Outputs (committed to the repo so reviewers can inspect them):

    results/data/p1_client_summary.csv
    results/data/p1_client_fault_imbalance.png

Run from the repo root inside the .venv::

    python scripts/check_data_pipeline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Allow running this script directly without `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_aircraft.data import (  # noqa: E402  (post-sys.path insert)
    CMAPSSConfig,
    Normalizer,
    load_and_label_train,
    load_raw,
    load_test_rul,
    make_test_windows,
    make_training_windows,
    partition_by_lifetime,
    slice_for_client,
)
from fl_aircraft.utils import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--n-clients", type=int, default=4)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "data",
        help="Where to write the CSV + figure.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
    cfg = CMAPSSConfig(subset=args.subset, data_dir=data_dir, window_size=args.window_size)

    print(f"--- Phase 1 sanity check ({args.subset}, {args.n_clients} clients) ---")
    print(f"feature cols      : {cfg.n_features}  ({cfg.feature_cols[:3]} ... {cfg.feature_cols[-3:]})")

    # 1. Load + label the full training frame (un-normalized).
    train_df = load_and_label_train(cfg)
    print(f"train rows        : {len(train_df):,}  ({train_df['unit_id'].nunique()} engines)")
    print(f"global RUL stats  : raw=({train_df['RUL_raw'].min():.0f}..{train_df['RUL_raw'].max():.0f})  "
          f"capped={train_df['RUL_capped'].max():.0f}")
    print(f"global fault rate : {train_df['fault'].mean():.4f}")

    # 2. Centralized normalizer / windows (the "all data" upper bound).
    central_norm = Normalizer.fit(train_df, cfg.feature_cols)
    central_arrays = make_training_windows(
        central_norm.transform(train_df), cfg.feature_cols, cfg.window_size, cfg.stride
    )
    print(f"central windows   : {central_arrays.n_samples:,}  shape={central_arrays.X.shape}")

    # 3. Partition into client shards + per-client preprocessing.
    shards = partition_by_lifetime(train_df, n_clients=args.n_clients, seed=args.seed)

    rows: list[dict] = []
    for shard in shards:
        client_df = slice_for_client(train_df, shard)
        client_norm = Normalizer.fit(client_df, cfg.feature_cols)
        client_norm_df = client_norm.transform(client_df)
        arrays = make_training_windows(
            client_norm_df, cfg.feature_cols, cfg.window_size, cfg.stride
        )
        lifetimes = client_df.groupby("unit_id")["cycle"].max()
        rows.append(
            {
                "client_id": shard.client_id,
                "n_engines": shard.n_engines,
                "n_rows": len(client_df),
                "n_windows": arrays.n_samples,
                "mean_lifetime": round(float(lifetimes.mean()), 1),
                "min_lifetime": int(lifetimes.min()),
                "max_lifetime": int(lifetimes.max()),
                "rul_capped_max": float(arrays.y_rul.max()),
                "fault_pos_rate": round(float(arrays.y_fault.mean()), 4),
                "fault_pos_count": int(arrays.y_fault.sum()),
            }
        )

    summary = pd.DataFrame(rows).set_index("client_id")
    print("\nPer-client summary:")
    print(summary.to_string())

    # 4. Test set sanity (using the centralized normalizer for an apples-to-apples comparison).
    test_df = central_norm.transform(load_raw(cfg.subset, "test", data_dir))
    test_rul = load_test_rul(cfg.subset, data_dir)
    test_arrays = make_test_windows(
        test_df, test_rul, cfg.feature_cols, cfg.window_size, cfg.rul_cap, cfg.fault_threshold
    )
    print(f"\ntest windows     : {test_arrays.n_samples}  shape={test_arrays.X.shape}")
    print(f"test fault rate  : {test_arrays.fault_positive_rate():.4f}")

    # 5. Save the summary CSV.
    csv_path = args.out_dir / f"p1_client_summary_{cfg.subset.lower()}.csv"
    summary.to_csv(csv_path)
    print(f"\nWrote {csv_path}")

    # 6. Save the per-client fault-imbalance figure (the RQ2 hook).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(summary)))
    bars = ax.bar(summary.index, summary["fault_pos_rate"] * 100, color=colors)
    global_rate = float(train_df["fault"].mean()) * 100
    ax.axhline(global_rate, color="red", linestyle="--", linewidth=1, label=f"global = {global_rate:.2f}%")
    for bar, rate in zip(bars, summary["fault_pos_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{rate * 100:.2f}%",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("% windows with fault (RUL ≤ 30)")
    ax.set_xlabel("client")
    ax.set_title(
        f"Phase 1 — per-client fault positive rate after partitioning  "
        f"({cfg.subset}, {args.n_clients} clients, seed={args.seed})"
    )
    ax.legend()
    ax.set_ylim(0, max(summary["fault_pos_rate"]) * 100 * 1.3)
    fig.tight_layout()
    fig_path = args.out_dir / f"p1_client_fault_imbalance_{cfg.subset.lower()}.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
