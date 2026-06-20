"""Sanity-check the Phase 1 data pipeline end-to-end.

Loads FD001, partitions 100 engines into 4 simulated airline clients, applies
per-client preprocessing + sliding-window construction, and prints / saves a
per-client summary. The figure produced here is the **RQ2 hook** — it shows
how the fault positive rate diverges across clients once you partition.

Outputs (committed to the repo so reviewers can inspect them):

    results/01_data/metrics.json                              structured for the frontend
    results/01_data/client_summary_<subset>.csv               flat per-client table
    results/01_data/client_fault_imbalance_<subset>.png       per-client positive-rate bars

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
from fl_aircraft.utils import (  # noqa: E402
    PhaseMetrics,
    dump_phase_metrics,
    seed_everything,
)

PHASE_ID = "01_data"
PHASE_NAME = "Phase 1 — Data pipeline sanity check"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--n-clients", type=int, default=4)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "01_data",
        help="Where to write the CSV, figure, and metrics.json.",
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
    csv_path = args.out_dir / f"client_summary_{cfg.subset.lower()}.csv"
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
    fig_path = args.out_dir / f"client_fault_imbalance_{cfg.subset.lower()}.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {fig_path}")

    # 7. Structured metrics.json for the React frontend.
    fault_rates = summary["fault_pos_rate"].to_numpy()
    interpretation = (
        f"Stratified-by-lifetime partitioning of {cfg.subset} across "
        f"{args.n_clients} clients yields a deliberately balanced split: "
        f"per-client fault positive rate {100 * fault_rates.min():.2f}–"
        f"{100 * fault_rates.max():.2f}% (spread "
        f"{100 * (fault_rates.max() - fault_rates.min()):.2f} pp). "
        f"This isolates 'does FedAvg converge?' from 'does FedAvg handle Non-IID?'. "
        f"Meaningful Non-IID arrives in P6 (FD001+FD003 mix) and the RQ2 experiment."
    )
    payload = PhaseMetrics(
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        interpretation=interpretation,
        subset=cfg.subset,
        config={
            "n_clients": args.n_clients,
            "window_size": args.window_size,
            "stride": cfg.stride,
            "rul_cap": cfg.rul_cap,
            "fault_threshold": cfg.fault_threshold,
            "seed": args.seed,
            "n_features": cfg.n_features,
        },
        summary={
            "total_train_engines": int(train_df["unit_id"].nunique()),
            "total_train_rows": int(len(train_df)),
            "total_central_windows": int(central_arrays.n_samples),
            "total_test_windows": int(test_arrays.n_samples),
            "global_fault_pos_rate": round(float(train_df["fault"].mean()), 4),
            "test_fault_pos_rate": round(float(test_arrays.fault_positive_rate()), 4),
            "per_client_fault_rate_min": round(float(fault_rates.min()), 4),
            "per_client_fault_rate_max": round(float(fault_rates.max()), 4),
            "per_client_fault_rate_spread_pp": round(
                float(fault_rates.max() - fault_rates.min()) * 100, 4
            ),
        },
        per_client={
            cid: {
                "n_engines": int(summary.loc[cid, "n_engines"]),
                "n_rows": int(summary.loc[cid, "n_rows"]),
                "n_windows": int(summary.loc[cid, "n_windows"]),
                "mean_lifetime": float(summary.loc[cid, "mean_lifetime"]),
                "min_lifetime": int(summary.loc[cid, "min_lifetime"]),
                "max_lifetime": int(summary.loc[cid, "max_lifetime"]),
                "fault_pos_rate": round(float(summary.loc[cid, "fault_pos_rate"]), 4),
                "fault_pos_count": int(summary.loc[cid, "fault_pos_count"]),
            }
            for cid in summary.index
        },
        artifacts={
            "client_summary_csv": f"results/{PHASE_ID}/client_summary_{cfg.subset.lower()}.csv",
            "client_fault_imbalance_png": f"results/{PHASE_ID}/client_fault_imbalance_{cfg.subset.lower()}.png",
        },
    )
    json_path = dump_phase_metrics(payload, args.out_dir)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
