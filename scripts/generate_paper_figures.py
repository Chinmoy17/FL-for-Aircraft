"""Generate supplementary paper figures (Fig 12, 13, 14, 15) for v3 draft.

Fig 12 — Axis 1 gap-closed with 3-seed error bars.
Fig 13 — Axis 2 backdoor ASR with 5-seed error bars.
Fig 14 — RQ7 attack × aggregator matrix with 5-seed error bars.
Fig 15 — Krum-defense per-seed dot plot (visualizes obs 7's bimodal RMSE).

Reads:
    results/rq7_poisoning/metrics_aggregated.json  (Figs 13, 14)
    results/rq7_poisoning_seeds/seed_<N>/per_round_*.csv  (Fig 15)
    Hard-coded 3-seed means from results/rq2_*_seeds/seed_{42,43,44}/metrics.json (Fig 12)

Writes:
    results/paper_figures/fig12_axis1_gap_closed.png
    results/paper_figures/fig13_backdoor_asr.png
    results/paper_figures/fig14_rq7_matrix_5seed.png
    results/paper_figures/fig15_krum_seed_variance.png

Run:
    python scripts/generate_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fig 12 — Axis 1 gap-closed with error bars
# ---------------------------------------------------------------------------
# 3-seed aggregation from results/rq2_*_seeds/seed_{42,43,44}/metrics.json
# See also paper_draft_v3.md Table 9.
AXIS1_METHODS = [
    # (label, mean_gap_closed_pct, std_gap_closed_pct, category_color)
    ("FedRep\n(h₁, e₁)",       69.9,  6.4, "#2ca02c"),   # personalization = green
    ("FedCCFA\n(τ=0.5)",       66.9,  6.6, "#2ca02c"),
    ("FedProx\n(μ=0.1)",       21.0, 13.1, "#ff7f0e"),   # proximal = orange
    ("Imbalance-aware\n(val-F1)", 10.4,  6.9, "#d62728"),  # reweighting = red
    ("FedAvg\nbaseline",       -0.7,  0.0, "#7f7f7f"),   # baseline = gray
]


def fig12_axis1_gap_closed() -> None:
    labels = [m[0] for m in AXIS1_METHODS]
    means = np.array([m[1] for m in AXIS1_METHODS])
    stds = np.array([m[2] for m in AXIS1_METHODS])
    colors = [m[3] for m in AXIS1_METHODS]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    ax.bar(
        x, means, yerr=stds, capsize=6,
        color=colors, edgecolor="black", linewidth=0.7, alpha=0.85,
    )

    for i, (m, s) in enumerate(zip(means, stds)):
        label = f"{m:.1f} ± {s:.1f}" if s > 0 else f"{m:.1f}"
        y_pos = m + s + 3 if m >= 0 else m - 6
        ax.text(i, y_pos, label, ha="center", fontsize=10,
                fontweight="bold" if m > 30 else "normal")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(
        "Gap closed vs local-only → centralized headroom (%)",
        fontsize=11,
    )
    ax.set_title(
        "Axis 1 — remedy effectiveness on structural non-IID (3-seed mean ± std)\n"
        "Winning method in each family; seeds ∈ {42, 43, 44}",
        fontsize=11.5, fontweight="bold",
    )
    ax.set_ylim(-15, 95)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Category brackets under the bars
    ax.annotate("architectural", xy=(0.5, -12), fontsize=9,
                ha="center", style="italic", color="#2ca02c")
    ax.annotate("optimization", xy=(2, -12), fontsize=9,
                ha="center", style="italic", color="#ff7f0e")
    ax.annotate("reweighting", xy=(3, -12), fontsize=9,
                ha="center", style="italic", color="#d62728")

    plt.tight_layout()
    out = OUT_DIR / "fig12_axis1_gap_closed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Fig 13 — Backdoor ASR by aggregator
# ---------------------------------------------------------------------------
def fig13_backdoor_asr() -> None:
    agg_path = REPO_ROOT / "results" / "rq7_poisoning" / "metrics_aggregated.json"
    with agg_path.open("r", encoding="utf-8") as f:
        agg = json.load(f)
    bd = agg["backdoor_summary"]

    aggregators = [
        ("Vanilla\nFedAvg",       "AV3_backdoor_vanilla", "#d62728"),
        ("Trimmed mean\n(β=0.25)", "D31_backdoor_trimmed", "#ff7f0e"),
        ("Coord.\nmedian",         "D32_backdoor_median",  "#ff7f0e"),
        ("Krum\n(f=1)",           "D33_backdoor_krum",    "#2ca02c"),
    ]
    labels = [a[0] for a in aggregators]
    keys = [a[1] for a in aggregators]
    colors = [a[2] for a in aggregators]
    means = np.array([bd[k]["attack_success_rate"]["mean"] * 100 for k in keys])
    stds = np.array([bd[k]["attack_success_rate"]["std"] * 100 for k in keys])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(labels))
    ax.bar(
        x, means, yerr=stds, capsize=6,
        color=colors, edgecolor="black", linewidth=0.7, alpha=0.85,
    )

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 3, f"{m:.1f} ± {s:.1f} %",
                ha="center", fontsize=10, fontweight="bold")

    # Practical threshold line
    ax.axhline(10, color="green", linestyle="--", linewidth=1.4, alpha=0.55)
    ax.text(-0.35, 12, "practical threshold: ASR ≤ 10 %",
            fontsize=9, color="darkgreen", alpha=0.85, ha="left",
            va="bottom", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=11)
    ax.set_title(
        "Axis 2 — sensor-value backdoor: Attack Success Rate by aggregator\n"
        "5-seed mean ± std, seeds ∈ {42, 43, 44, 45, 46}",
        fontsize=11.5, fontweight="bold",
    )
    ax.set_ylim(0, 125)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    out = OUT_DIR / "fig13_backdoor_asr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Fig 14 — RQ7 attack × aggregator matrix, 5-seed error bars
# ---------------------------------------------------------------------------
def fig14_rq7_matrix_5seed() -> None:
    agg_path = REPO_ROOT / "results" / "rq7_poisoning" / "metrics_aggregated.json"
    with agg_path.open("r", encoding="utf-8") as f:
        agg = json.load(f)
    per_cell = agg["per_cell"]

    attacks = [
        ("Clean",           "B0_clean_vanilla",         "B1_clean_trimmed",      "B2_clean_median",     "B3_clean_krum"),
        ("Label-flip",      "AV1_labelflip_vanilla",    "D11_labelflip_trimmed", "D12_labelflip_median","D13_labelflip_krum"),
        ("Grad ×−10",       "AV2_gradscale_vanilla",    "D21_gradscale_trimmed", "D22_gradscale_median","D23_gradscale_krum"),
        ("Grad ×−2\n(stealthy)", "AV4_gradscalex2_vanilla", "D41_gradscalex2_trimmed", "D42_gradscalex2_median", "D43_gradscalex2_krum"),
        ("Backdoor",        "AV3_backdoor_vanilla",     "D31_backdoor_trimmed",  "D32_backdoor_median", "D33_backdoor_krum"),
        ("Coord ×−10\n(2 attackers)", "AV5_coord_vanilla", "D51_coord_trimmed",   "D52_coord_median",    "D53_coord_krum_f1"),
    ]
    aggregator_labels = ["Vanilla FedAvg", "Trimmed mean", "Coord. median", "Krum (f=1)"]
    aggregator_colors = ["#d62728", "#ff7f0e", "#e377c2", "#2ca02c"]

    n_attacks = len(attacks)
    n_agg = 4
    bar_w = 0.19
    fig, ax = plt.subplots(figsize=(16, 6.5))
    x_base = np.arange(n_attacks)

    for k, (agg_label, agg_color) in enumerate(zip(aggregator_labels, aggregator_colors)):
        means, stds = [], []
        for row in attacks:
            key = row[1 + k]
            if key in per_cell:
                means.append(per_cell[key]["best_rmse"]["mean"])
                stds.append(per_cell[key]["best_rmse"]["std"])
            else:
                means.append(0)
                stds.append(0)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        offset = (k - 1.5) * bar_w
        bars = ax.bar(
            x_base + offset, means_arr, bar_w,
            yerr=stds_arr, capsize=3,
            color=agg_color, edgecolor="black", linewidth=0.4,
            label=agg_label, alpha=0.87,
        )

        # Annotate catastrophic bars (mean > 80)
        for i, (m, s) in enumerate(zip(means_arr, stds_arr)):
            if m > 80:
                ax.text(x_base[i] + offset, min(m + 2, 92),
                        f"{m:.1f}", ha="center", fontsize=7.5,
                        color="darkred", fontweight="bold")

    ax.set_xticks(x_base)
    ax.set_xticklabels([a[0] for a in attacks], fontsize=10)
    ax.set_ylabel("Best-round test RMSE", fontsize=11)
    ax.set_title(
        "Axis 2 — 24-cell attack × aggregator matrix (5-seed mean ± std)\n"
        "Krum-f₂ / n=4 cell mathematically undefined (see § 5.3.3)",
        fontsize=11.5, fontweight="bold",
    )
    ax.set_ylim(0, 100)
    ax.axhline(20, color="green", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(-0.55, 21, "near-clean\nrecovery",
            fontsize=8, color="darkgreen", alpha=0.7, va="bottom")
    ax.legend(loc="upper right", fontsize=10, ncol=1, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()

    out = OUT_DIR / "fig14_rq7_matrix_5seed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Fig 15 — Krum defense per-seed dot plot
# ---------------------------------------------------------------------------
def fig15_krum_seed_variance() -> None:
    """Fig 15: Krum-defense per-seed dot plot showing bimodal RMSE.

    Reads per_client.<cell>.best_rmse from each seed's metrics.json so
    the plot is numerically consistent with Table 10 (which is computed
    by scripts/aggregate_rq7_seeds.py from the same field).
    """
    seeds_root = REPO_ROOT / "results" / "rq7_poisoning_seeds"
    seeds = [42, 43, 44, 45, 46]
    krum_cells = [
        # (x-tick label, cell_key in metrics.json per_client dict)
        ("D13\nLabel-flip",   "D13_labelflip_krum"),
        ("D23\nGrad ×−10",    "D23_gradscale_krum"),
        ("D43\nGrad ×−2",     "D43_gradscalex2_krum"),
        ("D33\nBackdoor",     "D33_backdoor_krum"),
        ("D53\nCoord ×−10",   "D53_coord_krum_f1"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    x_base = np.arange(len(krum_cells))
    seed_colors = plt.get_cmap("tab10")(np.linspace(0, 0.5, len(seeds)))

    # Load per-seed best_rmse for each cell from metrics.json
    per_cell_data: list[list[float]] = []
    for _, cell_key in krum_cells:
        cell_rmses: list[float] = []
        for s in seeds:
            mj_path = seeds_root / f"seed_{s}" / "metrics.json"
            if not mj_path.exists():
                cell_rmses.append(float("nan"))
                continue
            with mj_path.open("r", encoding="utf-8") as f:
                mj = json.load(f)
            entry = (mj.get("per_client") or {}).get(cell_key)
            if entry is None or "best_rmse" not in entry:
                cell_rmses.append(float("nan"))
                continue
            cell_rmses.append(float(entry["best_rmse"]))
        per_cell_data.append(cell_rmses)

    # Plot dots (colored by seed) + mean±std indicator
    for i, (label, _) in enumerate(krum_cells):
        vals = per_cell_data[i]
        for j, (s, v) in enumerate(zip(seeds, vals)):
            if np.isnan(v):
                continue
            xj = i + (j - 2) * 0.06
            ax.scatter(
                xj, v, s=95, color=seed_colors[j],
                edgecolor="black", linewidth=0.6,
                label=f"seed {s}" if i == 0 else None, zorder=3,
            )
        valid = np.array([v for v in vals if not np.isnan(v)])
        if len(valid):
            m = float(valid.mean())
            sd = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
            ax.errorbar(
                i, m, yerr=sd, fmt="_", color="black",
                markersize=30, capsize=8, capthick=1.5,
                elinewidth=1.5, zorder=2,
            )
            ax.text(i + 0.25, m, f"{m:.1f}\n±{sd:.1f}",
                    fontsize=9, va="center", ha="left",
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    ax.set_xticks(x_base)
    ax.set_xticklabels([c[0] for c in krum_cells], fontsize=10)
    ax.set_ylabel("Best-round test RMSE (per seed, from metrics.json)", fontsize=11)
    ax.set_title(
        "Krum-defense per-seed RMSE across 5 attack cells\n"
        "The 3 leftmost cells report identical (mean, std) — Krum's argmin "
        "selects the same client regardless of attack family (§ 9.2 obs 7)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, frameon=True, title="Seeds")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, 50)
    plt.tight_layout()

    out = OUT_DIR / "fig15_krum_seed_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    print(f"Writing paper figures to {OUT_DIR.relative_to(REPO_ROOT)}/")
    fig12_axis1_gap_closed()
    fig13_backdoor_asr()
    fig14_rq7_matrix_5seed()
    fig15_krum_seed_variance()
    print("Done.")


if __name__ == "__main__":
    main()
