"""Generate all paper figures as both PNG and PDF, with publication-quality
matplotlib settings.

This script is the single source of truth for every figure that appears in
the RESS paper draft (``research_Paper/Overleaf_Paper/main.tex``). It reads
per-experiment JSON/CSV metrics already committed under ``results/`` and
emits two copies of every figure:

* PNG in ``results/paper_figures/`` --- consumed by the frontend dashboard
  and by the repo README/results.md.
* PDF in ``research_Paper/Overleaf_Paper/figures/`` --- consumed by
  ``\\includegraphics`` in the LaTeX source.

Nine figures total (paper LaTeX numbering in parentheses):

* ``three_way_non_iid``         (Fig 6, Section 7)
* ``fedrep_per_subset``         (Fig 7, Section 7)
* ``fedccfa_cluster``           (Fig 8, Section 7)
* ``rq7_matrix_5seed``          (Fig 9, Section 8)
* ``defense_recovery``          (Fig 10, Section 8)
* ``cross_model_attribution``   (Fig 11, Section 7)
* ``axis1_gap_closed``          (Fig 12, Section 7)
* ``backdoor_asr``              (Fig 13, Section 8)
* ``krum_seed_variance``        (Fig 14, Section 8)

Run::

    python scripts/generate_paper_figures.py

Requires ``matplotlib``, ``numpy``, ``pandas``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PNG_DIR = REPO_ROOT / "results" / "paper_figures"
PDF_DIR = REPO_ROOT / "research_Paper" / "Overleaf_Paper" / "figures"
PNG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Publication-quality matplotlib settings
# ---------------------------------------------------------------------------
# pdf.fonttype=42 forces TrueType fonts (not Type-3) --- required by many
# Elsevier/IEEE workflows and by arXiv. DejaVu Serif ships with matplotlib
# so this never falls back to Type-3.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save ``fig`` as both PNG (paper_figures/) and PDF (Overleaf figures/)."""
    png_path = PNG_DIR / f"{name}.png"
    pdf_path = PDF_DIR / f"{name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  wrote {png_path.relative_to(REPO_ROOT)}")
    print(f"  wrote {pdf_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Fig 6 --- Three-way non-IID comparison (Section 7)
# ---------------------------------------------------------------------------
def fig_three_way_non_iid() -> None:
    """Centralized upper bound vs local-only vs vanilla FedAvg on FD001+FD003."""
    labels = ["Centralized\n(upper bound)", "Local-only\n(mean of 4 clients)", "Vanilla FedAvg"]
    means = np.array([13.77, 17.92, 17.95])
    stds = np.array([0.0, 1.52, 0.0])
    colors = ["#2ca02c", "#7f7f7f", "#d62728"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(
        x, means, yerr=stds, capsize=6,
        color=colors, edgecolor="black", linewidth=0.7, alpha=0.87,
    )
    for i, (m, s) in enumerate(zip(means, stds)):
        label = f"{m:.2f}" if s == 0 else f"{m:.2f} $\\pm$ {s:.2f}"
        ax.text(i, m + max(s, 0.15) + 0.35, label,
                ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Combined-test RMSE (cycles)")
    ax.set_title(
        "The motivating failure: FedAvg cannot handle structural non-IID\n"
        "(FD001 + FD003, 4-client federation, seed 42)"
    )
    ax.set_ylim(0, 22)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.annotate(
        "gap-closed % = $-0.7$\n(FedAvg statistically\nindistinguishable\nfrom local-only)",
        xy=(2, 17.95), xytext=(1.55, 6),
        fontsize=9, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
    )
    plt.tight_layout()
    save_fig(fig, "three_way_non_iid")


# ---------------------------------------------------------------------------
# Fig 7 --- FedRep per-subset breakdown (Section 7)
# ---------------------------------------------------------------------------
def fig_fedrep_per_subset() -> None:
    """Per-subset macro-RMSE: centralized reference vs vanilla FedAvg vs FedRep."""
    subsets = ["FD001\n(1 fault mode: HPC)", "FD003\n(2 fault modes: HPC + fan)"]
    centralized_ref = np.array([14.76, 12.69])
    fedavg_baseline = np.array([16.99, 18.86])
    fedrep_mean = np.array([14.65, 15.39])
    fedrep_std = np.array([0.27, 0.46])

    x = np.arange(len(subsets))
    bar_w = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - bar_w, centralized_ref, bar_w,
           color="#2ca02c", edgecolor="black", linewidth=0.6,
           label="Centralized (per-subset ref.)", alpha=0.9)
    ax.bar(x, fedavg_baseline, bar_w,
           color="#d62728", edgecolor="black", linewidth=0.6,
           label="Vanilla FedAvg (baseline)", alpha=0.9)
    ax.bar(x + bar_w, fedrep_mean, bar_w, yerr=fedrep_std, capsize=4,
           color="#1f77b4", edgecolor="black", linewidth=0.6,
           label="FedRep (3-seed mean $\\pm$ std)", alpha=0.9)

    for i, (r, b, m, s) in enumerate(zip(centralized_ref, fedavg_baseline,
                                          fedrep_mean, fedrep_std)):
        ax.text(i - bar_w, r + 0.35, f"{r:.2f}", ha="center", fontsize=9)
        ax.text(i, b + 0.35, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(i + bar_w, m + s + 0.35,
                f"{m:.2f}$\\pm${s:.2f}",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylabel("Per-subset macro-RMSE (cycles)")
    ax.set_title(
        "FedRep per-subset performance vs the centralized reference\n"
        "Per-client heads close the FD001 gap and cut the FD003 gap by $\\sim$70 %"
    )
    ax.set_ylim(0, 22)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", ncol=1, frameon=True)
    plt.tight_layout()
    save_fig(fig, "fedrep_per_subset")


# ---------------------------------------------------------------------------
# Fig 8 --- FedCCFA cluster evolution (Section 7)
# ---------------------------------------------------------------------------
def fig_fedccfa_cluster() -> None:
    """Number of clusters vs communication round for FedCCFA (seed 42)."""
    csv_path = REPO_ROOT / "results" / "rq2_fedccfa" / "per_round.csv"
    df = pd.read_csv(csv_path)
    rounds = df["round"].to_numpy()
    n_clusters = df["n_clusters"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, n_clusters, marker="o", markersize=5,
            color="#1f77b4", linewidth=1.5, markerfacecolor="white",
            markeredgewidth=1.3, markeredgecolor="#1f77b4")
    ax.axvspan(0.5, 3.5, alpha=0.14, color="#ff7f0e", zorder=0)
    ax.text(2, 3.6, "warmup\n(FedRep aggregation)",
            ha="center", fontsize=9, color="#b35900", style="italic")
    ax.axhline(1, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(48, 1.15, "single cluster containing all 4 clients",
            ha="right", fontsize=9, color="#1a5f1a", style="italic")

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Number of clusters")
    ax.set_title(
        "FedCCFA cluster structure across communication rounds\n"
        "After warmup the similarity threshold $\\tau = 0.5$ never partitions the federation"
    )
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 4.6)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    save_fig(fig, "fedccfa_cluster")


# ---------------------------------------------------------------------------
# Fig 12 --- Axis 1 gap-closed with 3-seed error bars (Section 7)
# ---------------------------------------------------------------------------
AXIS1_METHODS = [
    ("FedRep\n($h_1$, $e_1$)",     69.9,  6.4, "#2ca02c"),
    ("FedCCFA\n($\\tau=0.5$)",     66.9,  6.6, "#2ca02c"),
    ("FedProx\n($\\mu=0.1$)",      21.0, 13.1, "#ff7f0e"),
    ("Imbalance-aware\n(val-F1)",  10.4,  6.9, "#d62728"),
    ("FedAvg\nbaseline",           -0.7,  0.0, "#7f7f7f"),
]


def fig_axis1_gap_closed() -> None:
    labels = [m[0] for m in AXIS1_METHODS]
    means = np.array([m[1] for m in AXIS1_METHODS])
    stds = np.array([m[2] for m in AXIS1_METHODS])
    colors = [m[3] for m in AXIS1_METHODS]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6,
           color=colors, edgecolor="black", linewidth=0.7, alpha=0.87)

    for i, (m, s) in enumerate(zip(means, stds)):
        label = f"{m:.1f} $\\pm$ {s:.1f}" if s > 0 else f"{m:.1f}"
        y_pos = m + s + 3 if m >= 0 else m - 5
        ax.text(i, y_pos, label, ha="center", fontsize=10,
                fontweight="bold" if m > 30 else "normal")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gap closed vs local-only $\\to$ centralized headroom (%)")
    ax.set_title(
        "Axis 1 --- remedy effectiveness on structural non-IID\n"
        "3-seed mean $\\pm$ std; winning method per family; seeds $\\in$ {42, 43, 44}"
    )
    ax.set_ylim(-15, 92)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    save_fig(fig, "axis1_gap_closed")


# ---------------------------------------------------------------------------
# Fig 9 --- RQ7 attack x aggregator matrix, 5-seed error bars (Section 8)
# ---------------------------------------------------------------------------
def fig_rq7_matrix_5seed() -> None:
    agg_path = REPO_ROOT / "results" / "rq7_poisoning" / "metrics_aggregated.json"
    with agg_path.open("r", encoding="utf-8") as f:
        agg = json.load(f)
    per_cell = agg["per_cell"]

    attacks = [
        ("Clean",                    "B0_clean_vanilla",         "B1_clean_trimmed",      "B2_clean_median",     "B3_clean_krum"),
        ("Label-flip",               "AV1_labelflip_vanilla",    "D11_labelflip_trimmed", "D12_labelflip_median","D13_labelflip_krum"),
        ("Grad $\\times{-}10$",      "AV2_gradscale_vanilla",    "D21_gradscale_trimmed", "D22_gradscale_median","D23_gradscale_krum"),
        ("Grad $\\times{-}2$\n(stealthy)", "AV4_gradscalex2_vanilla", "D41_gradscalex2_trimmed", "D42_gradscalex2_median", "D43_gradscalex2_krum"),
        ("Backdoor",                 "AV3_backdoor_vanilla",     "D31_backdoor_trimmed",  "D32_backdoor_median", "D33_backdoor_krum"),
        ("Coord $\\times{-}10$\n(2 attackers)", "AV5_coord_vanilla", "D51_coord_trimmed",   "D52_coord_median",    "D53_coord_krum_f1"),
    ]
    aggregator_labels = ["Vanilla FedAvg", "Trimmed mean", "Coord. median", "Krum ($f{=}1$)"]
    aggregator_colors = ["#d62728", "#ff7f0e", "#e377c2", "#2ca02c"]

    n_attacks = len(attacks)
    bar_w = 0.19
    fig, ax = plt.subplots(figsize=(13, 5.8))
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
        ax.bar(x_base + offset, means_arr, bar_w,
               yerr=stds_arr, capsize=3,
               color=agg_color, edgecolor="black", linewidth=0.4,
               label=agg_label, alpha=0.87)
        for i, (m, s) in enumerate(zip(means_arr, stds_arr)):
            if m > 80:
                ax.text(x_base[i] + offset, min(m + 1.5, 92),
                        f"{m:.1f}", ha="center", fontsize=7.5,
                        color="darkred", fontweight="bold")

    ax.set_xticks(x_base)
    ax.set_xticklabels([a[0] for a in attacks])
    ax.set_ylabel("Best-round test RMSE")
    ax.set_title(
        "Axis 2 --- 24-cell attack $\\times$ aggregator matrix (5-seed mean $\\pm$ std)\n"
        "The Krum-$f_2$ / $N{=}4$ cell is mathematically undefined and omitted (Sec. 5.3.3)"
    )
    ax.set_ylim(0, 100)
    ax.axhline(20, color="green", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(-0.55, 21, "near-clean\nrecovery",
            fontsize=8, color="darkgreen", alpha=0.7, va="bottom")
    ax.legend(loc="upper right", ncol=1, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    save_fig(fig, "rq7_matrix_5seed")


# ---------------------------------------------------------------------------
# Fig 10 --- Defense recovery paired bars (Section 8)
# ---------------------------------------------------------------------------
def fig_defense_recovery() -> None:
    """Attack + vanilla vs attack + Krum (5-seed mean +/- std)."""
    agg_path = REPO_ROOT / "results" / "rq7_poisoning" / "metrics_aggregated.json"
    with agg_path.open("r", encoding="utf-8") as f:
        agg = json.load(f)
    per_cell = agg["per_cell"]

    attacks = [
        ("Label-flip",              "AV1_labelflip_vanilla",    "D13_labelflip_krum"),
        ("Grad $\\times{-}10$",     "AV2_gradscale_vanilla",    "D23_gradscale_krum"),
        ("Grad $\\times{-}2$\n(stealthy)", "AV4_gradscalex2_vanilla", "D43_gradscalex2_krum"),
        ("Backdoor",                "AV3_backdoor_vanilla",     "D33_backdoor_krum"),
        ("Coord $\\times{-}10$\n(2 attackers)", "AV5_coord_vanilla", "D53_coord_krum_f1"),
    ]
    n = len(attacks)
    bar_w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(n)

    vanilla_means, vanilla_stds = [], []
    krum_means, krum_stds = [], []
    for row in attacks:
        v = per_cell[row[1]]["best_rmse"]
        k = per_cell[row[2]]["best_rmse"]
        vanilla_means.append(v["mean"])
        vanilla_stds.append(v["std"])
        krum_means.append(k["mean"])
        krum_stds.append(k["std"])

    ax.bar(x - bar_w / 2, vanilla_means, bar_w, yerr=vanilla_stds, capsize=4,
           color="#d62728", edgecolor="black", linewidth=0.6,
           label="Undefended (vanilla FedAvg)", alpha=0.87)
    ax.bar(x + bar_w / 2, krum_means, bar_w, yerr=krum_stds, capsize=4,
           color="#2ca02c", edgecolor="black", linewidth=0.6,
           label="Defended (Krum, $f=1$)", alpha=0.87)

    for i, (v, s_v, kv, s_k) in enumerate(zip(vanilla_means, vanilla_stds,
                                               krum_means, krum_stds)):
        ax.text(x[i] - bar_w / 2, min(v + s_v + 2, 93),
                f"{v:.1f}", ha="center", fontsize=8.5,
                color="darkred" if v > 80 else "black",
                fontweight="bold" if v > 80 else "normal")
        ax.text(x[i] + bar_w / 2, kv + s_k + 2,
                f"{kv:.1f}", ha="center", fontsize=8.5, fontweight="bold",
                color="darkgreen")

    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in attacks])
    ax.set_ylabel("Best-round test RMSE")
    ax.set_title(
        "Defense recovery: vanilla FedAvg vs Krum ($f=1$) under each attack\n"
        "5-seed mean $\\pm$ std --- Krum uniquely survives the coordinated cell (rightmost)"
    )
    ax.set_ylim(0, 100)
    ax.axhline(20, color="green", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    save_fig(fig, "defense_recovery")


# ---------------------------------------------------------------------------
# Fig 11 --- Cross-model attribution, engine 50 (Section 7)
# ---------------------------------------------------------------------------
def fig_cross_model_attribution() -> None:
    """4-panel bar chart: top-3 sensor attributions for engine 50 across models."""
    rq3 = REPO_ROOT / "results" / "rq3_explanations"
    checkpoints = [
        ("P3 centralized FD001",           "explanations_p3_centralized_fd001_engine_50.json"),
        ("P5 FedAvg IID FD001",            "explanations_p5_fedavg_iid_fd001_engine_50.json"),
        ("P6 centralized FD001+FD003",     "explanations_p6_centralized_combined_engine_50.json"),
        ("P6 FedAvg non-IID FD001+FD003",  "explanations_p6_fedavg_non_iid_engine_50.json"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    for idx, (title, filename) in enumerate(checkpoints):
        with (rq3 / filename).open("r", encoding="utf-8") as f:
            data = json.load(f)
        top = data["top_sensors"][:3]
        names = [f"{t['name']}\n({t['column']})" for t in top]
        scores = [t["contribution"] for t in top]
        colors = ["#d62728" if s < 0 else "#2ca02c" for s in scores]

        ax = axes[idx]
        y = np.arange(len(names))
        ax.barh(y, scores, color=colors, edgecolor="black",
                linewidth=0.6, alpha=0.87)
        for i, s in enumerate(scores):
            xpos = s + (2 if s >= 0 else -2)
            ha = "left" if s >= 0 else "right"
            ax.text(xpos, i, f"{s:+.1f}", va="center", ha=ha,
                    fontsize=9, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Attribution (RUL cycles)", fontsize=9)
        rul = data.get("predicted_rul", 0.0)
        ax.set_title(f"{title}\npredicted RUL = {rul:.1f} cycles",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        mx = max(abs(min(scores)), abs(max(scores))) * 1.35
        ax.set_xlim(-mx, mx)

    fig.suptitle(
        "Cross-model sensor attribution --- test engine 50 (true RUL $=$ 79 cycles)\n"
        "Top-3 sensors driving the RUL prediction differ across model checkpoints",
        fontsize=11.5, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "cross_model_attribution")


# ---------------------------------------------------------------------------
# Fig 13 --- Backdoor Attack Success Rate by aggregator (Section 8)
# ---------------------------------------------------------------------------
def fig_backdoor_asr() -> None:
    agg_path = REPO_ROOT / "results" / "rq7_poisoning" / "metrics_aggregated.json"
    with agg_path.open("r", encoding="utf-8") as f:
        agg = json.load(f)
    bd = agg["backdoor_summary"]

    aggregators = [
        ("Vanilla\nFedAvg",              "AV3_backdoor_vanilla", "#d62728"),
        ("Trimmed mean\n($\\beta=0.25$)","D31_backdoor_trimmed", "#ff7f0e"),
        ("Coord.\nmedian",               "D32_backdoor_median",  "#e377c2"),
        ("Krum\n($f=1$)",                "D33_backdoor_krum",    "#2ca02c"),
    ]
    labels = [a[0] for a in aggregators]
    keys = [a[1] for a in aggregators]
    colors = [a[2] for a in aggregators]
    means = np.array([bd[k]["attack_success_rate"]["mean"] * 100 for k in keys])
    stds = np.array([bd[k]["attack_success_rate"]["std"] * 100 for k in keys])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6,
           color=colors, edgecolor="black", linewidth=0.7, alpha=0.87)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 3, f"{m:.1f} $\\pm$ {s:.1f} %",
                ha="center", fontsize=10, fontweight="bold")

    ax.axhline(10, color="green", linestyle="--", linewidth=1.4, alpha=0.55)
    ax.text(-0.35, 12, "practical threshold: ASR $\\leq$ 10 %",
            fontsize=9, color="darkgreen", alpha=0.85, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(
        "Axis 2 --- sensor-value backdoor: Attack Success Rate by aggregator\n"
        "5-seed mean $\\pm$ std, seeds $\\in$ {42, 43, 44, 45, 46}"
    )
    ax.set_ylim(0, 125)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    save_fig(fig, "backdoor_asr")


# ---------------------------------------------------------------------------
# Fig 14 --- Krum-defense per-seed dot plot (Section 8)
# ---------------------------------------------------------------------------
def fig_krum_seed_variance() -> None:
    seeds_root = REPO_ROOT / "results" / "rq7_poisoning_seeds"
    seeds = [42, 43, 44, 45, 46]
    krum_cells = [
        ("D13\nLabel-flip",              "D13_labelflip_krum"),
        ("D23\nGrad $\\times{-}10$",     "D23_gradscale_krum"),
        ("D43\nGrad $\\times{-}2$",      "D43_gradscalex2_krum"),
        ("D33\nBackdoor",                "D33_backdoor_krum"),
        ("D53\nCoord $\\times{-}10$",    "D53_coord_krum_f1"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_base = np.arange(len(krum_cells))
    seed_colors = plt.get_cmap("tab10")(np.linspace(0, 0.5, len(seeds)))

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

    for i, (label, _) in enumerate(krum_cells):
        vals = per_cell_data[i]
        for j, (s, v) in enumerate(zip(seeds, vals)):
            if np.isnan(v):
                continue
            xj = i + (j - 2) * 0.06
            ax.scatter(
                xj, v, s=90, color=seed_colors[j],
                edgecolor="black", linewidth=0.6,
                label=f"seed {s}" if i == 0 else None, zorder=3,
            )
        valid = np.array([v for v in vals if not np.isnan(v)])
        if len(valid):
            m = float(valid.mean())
            sd = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
            ax.errorbar(
                i, m, yerr=sd, fmt="_", color="black",
                markersize=28, capsize=8, capthick=1.4,
                elinewidth=1.4, zorder=2,
            )
            ax.text(i + 0.24, m, f"{m:.1f}\n$\\pm${sd:.1f}",
                    fontsize=9, va="center", ha="left", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    ax.set_xticks(x_base)
    ax.set_xticklabels([c[0] for c in krum_cells])
    ax.set_ylabel("Best-round test RMSE (per seed)")
    ax.set_title(
        "Krum-defense per-seed RMSE across 5 attack cells\n"
        "The 3 leftmost cells share identical (mean, std) --- Krum's argmin "
        "selects the same\nhonest client regardless of which attack the "
        "malicious client is running"
    )
    ax.legend(loc="upper right", frameon=True, title="Seeds")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, 50)
    plt.tight_layout()
    save_fig(fig, "krum_seed_variance")


def main() -> None:
    print(f"PNG outputs -> {PNG_DIR.relative_to(REPO_ROOT)}/")
    print(f"PDF outputs -> {PDF_DIR.relative_to(REPO_ROOT)}/")
    print()
    print("[1/9] three_way_non_iid ...")
    fig_three_way_non_iid()
    print("[2/9] fedrep_per_subset ...")
    fig_fedrep_per_subset()
    print("[3/9] fedccfa_cluster ...")
    fig_fedccfa_cluster()
    print("[4/9] axis1_gap_closed ...")
    fig_axis1_gap_closed()
    print("[5/9] rq7_matrix_5seed ...")
    fig_rq7_matrix_5seed()
    print("[6/9] defense_recovery ...")
    fig_defense_recovery()
    print("[7/9] cross_model_attribution ...")
    fig_cross_model_attribution()
    print("[8/9] backdoor_asr ...")
    fig_backdoor_asr()
    print("[9/9] krum_seed_variance ...")
    fig_krum_seed_variance()
    print()
    print("Done. 9 figures written as PNG + PDF.")


if __name__ == "__main__":
    main()
