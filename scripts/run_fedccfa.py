"""FedCCFA — clustered classifier-fragment aggregation on FD001+FD003.

This is the architectural-clustering follow-up to FedRep. The hypothesis:
if FedRep gave each client its own head, FedCCFA additionally groups
clients with similar heads and lets them share — getting twice the
supervision per shared head when the cluster structure matches the
underlying fault-mode partition.

Same partition / seed / round budget as RQ2 + FedProx + FedRep so the
comparison is direct.

Outputs land in ``results/rq2_fedccfa/``:

    metrics.json                                structured for the frontend
    per_round.csv                               macro + per-client + clusters
    headline_comparison_fd001_fd003.png         bars vs every prior scheme
    cluster_evolution_fd001_fd003.png           which clients clustered when
    per_subset_breakdown_fd001_fd003.png        FD001 vs FD003 vs centralized
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_aircraft.data import (  # noqa: E402
    MultiSubsetConfig,
    load_multi_subset_bundle,
    partition_by_subset_halves,
)
from fl_aircraft.fl import run_fedccfa_from_bundle  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics  # noqa: E402

PHASE_ID = "rq2_fedccfa"
PHASE_NAME = "RQ2 follow-up — FedCCFA (clustered personalisation) on FD001 + FD003"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD003"])
    p.add_argument("--n-clients-per-subset", type=int, default=2)
    p.add_argument("--n-rounds", type=int, default=50)
    p.add_argument("--head-epochs", type=int, default=1)
    p.add_argument("--encoder-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-fault", type=float, default=0.5)
    p.add_argument("--similarity-threshold", type=float, default=0.5,
                   help="Pairwise cosine similarity above which clients merge into one cluster.")
    p.add_argument("--warmup-rounds", type=int, default=3,
                   help="Initial rounds with NO clustering (FedRep-style), to let heads diverge enough for similarity to mean something.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cosine", action="store_true")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / PHASE_ID)
    return p.parse_args()


def _build_shard_to_subset(shards, subsets, n_per_subset) -> dict[str, str]:
    if len(shards) != len(subsets) * n_per_subset:
        raise ValueError(
            f"shard count {len(shards)} != subsets x n_per_subset "
            f"({len(subsets)} x {n_per_subset})"
        )
    return {s.client_id: subsets[i // n_per_subset] for i, s in enumerate(shards)}


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------
def _load_p6_references() -> dict[str, float] | None:
    p6 = REPO_ROOT / "results" / "06_non_iid" / "metrics.json"
    if not p6.exists():
        return None
    with p6.open(encoding="utf-8") as fh:
        d = json.load(fh)
    summary = d.get("summary") or {}
    out: dict[str, float] = {}
    for k_out, k_in in (
        ("centralized_rmse", "centralized_rmse"),
        ("fedavg_rmse", "fedavg_rmse"),
        ("local_only_rmse", "local_only_rmse_mean"),
    ):
        if k_in in summary:
            out[k_out] = float(summary[k_in])
    return out or None


def _load_p6_per_subset_centralized() -> dict[str, float] | None:
    p6 = REPO_ROOT / "results" / "06_non_iid" / "metrics.json"
    if not p6.exists():
        return None
    with p6.open(encoding="utf-8") as fh:
        d = json.load(fh)
    out: dict[str, float] = {}
    for ps in d.get("test", {}).get("centralized_per_subset", []) or []:
        out[ps["subset"]] = float(ps["rmse"])
    return out or None


def _load_other_phase_best(phase_id: str) -> float | None:
    p = REPO_ROOT / "results" / phase_id / "metrics.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as fh:
        d = json.load(fh)
    summary = d.get("summary") or {}
    return summary.get("best_rmse") or summary.get("best_macro_rmse")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_headline(out_path: Path, history, refs: dict[str, float], display: str) -> None:
    bars: list[tuple[str, float, str]] = []
    if refs.get("centralized_rmse"):
        bars.append(("Centralized\n(upper bound)", refs["centralized_rmse"], "black"))
    if refs.get("fedavg_rmse"):
        bars.append(("Vanilla FedAvg\n(control)", refs["fedavg_rmse"], "tab:red"))
    if refs.get("fedprox_best_rmse"):
        bars.append(("FedProx best\n(μ=0.1)", refs["fedprox_best_rmse"], "tab:purple"))
    if refs.get("fedrep_best_rmse"):
        bars.append(("FedRep\n(macro)", refs["fedrep_best_rmse"], "tab:green"))
    bars.append(("FedCCFA\n(this run)", history.best_macro_rmse, "tab:cyan"))
    if refs.get("local_only_rmse"):
        bars.append(("Local-only mean\n(lower bound)", refs["local_only_rmse"], "tab:gray"))

    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    rects = ax.bar(labels, values, color=colors)
    for rect, v in zip(rects, values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("test RMSE (cycles)")
    ax.set_title(
        f"FedCCFA best macro-RMSE vs every prior phase — {display}\n"
        f"(lower is better; macro RMSE = mean across per-client per-subset test slices)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_evolution(out_path: Path, history, client_ids: list[str], display: str) -> None:
    """Cluster-id heatmap: rows = rounds, cols = clients. Cells = cluster index."""
    rounds = [r.round for r in history.rounds]
    n_rounds = len(rounds)
    n_clients = len(client_ids)
    cluster_matrix = np.zeros((n_rounds, n_clients), dtype=np.int32)
    for ri, rec in enumerate(history.rounds):
        # Map each client_id to its cluster index this round.
        client_to_cluster = {}
        for ci, cluster in enumerate(rec.clusters):
            for cid in cluster:
                client_to_cluster[cid] = ci
        for cj, cid in enumerate(client_ids):
            cluster_matrix[ri, cj] = client_to_cluster.get(cid, -1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap = plt.colormaps["tab10"]
    n_distinct = int(cluster_matrix.max() + 1)
    im = ax.imshow(
        cluster_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(n_distinct - 1, 1),
        interpolation="nearest",
    )
    ax.set_xticks(range(n_clients))
    ax.set_xticklabels(client_ids, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("round")
    ax.set_xlabel("client (color = cluster index)")
    ax.set_title(f"FedCCFA cluster evolution — {display}\n(each row = one round)")
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_distinct))
    cbar.set_label("cluster index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_subset_breakdown(
    out_path: Path, history, p6_per_subset: dict[str, float] | None, display: str,
) -> None:
    best = history.rounds[history.best_round - 1]
    per_subset_rmse: dict[str, list[float]] = {}
    for m in best.per_client_metrics:
        per_subset_rmse.setdefault(m.subset, []).append(m.rmse)
    subsets = sorted(per_subset_rmse.keys())
    fedccfa_means = [float(np.mean(per_subset_rmse[s])) for s in subsets]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(subsets))
    width = 0.35
    rects = ax.bar(x - width / 2, fedccfa_means, width, color="tab:cyan",
                   label=f"FedCCFA (best round {history.best_round})")
    if p6_per_subset:
        cent_means = [p6_per_subset.get(s, float("nan")) for s in subsets]
        ax.bar(x + width / 2, cent_means, width, color="black",
               alpha=0.6, label="Centralized (P6)")
        for i, v in enumerate(cent_means):
            if not np.isnan(v):
                ax.text(x[i] + width / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=9)
    for r, v in zip(rects, fedccfa_means):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(),
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylabel("test RMSE (cycles)")
    ax.set_title(f"FedCCFA per-subset RMSE vs centralized — {display}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
    multi_cfg = MultiSubsetConfig(subsets=tuple(args.subsets), data_dir=data_dir)
    bundle = load_multi_subset_bundle(multi_cfg)
    display = bundle.display_name
    print(f"--- FedCCFA ({display}) ---")
    print(f"  subsets: {args.subsets}, clients per subset: {args.n_clients_per_subset}")
    print(f"  rounds × (head+encoder epochs): "
          f"{args.n_rounds} × ({args.head_epochs}+{args.encoder_epochs})")
    print(f"  similarity_threshold={args.similarity_threshold}, "
          f"warmup_rounds={args.warmup_rounds}")
    print(f"  seed: {args.seed}\n")

    shards = partition_by_subset_halves(
        bundle.train_df, subsets=tuple(args.subsets),
        n_clients_per_subset=args.n_clients_per_subset, seed=args.seed,
    )
    shard_to_subset = _build_shard_to_subset(
        shards, args.subsets, args.n_clients_per_subset,
    )
    for s in shards:
        print(f"  · {s.client_id}: {shard_to_subset[s.client_id]} ({s.n_engines} engines)")
    print()

    refs: dict[str, float] = {}
    p6 = _load_p6_references()
    if p6:
        refs.update(p6)
    p6_per_subset = _load_p6_per_subset_centralized()
    fedprox = _load_other_phase_best("rq2_fedprox")
    if fedprox is not None:
        refs["fedprox_best_rmse"] = fedprox
    fedrep = _load_other_phase_best("rq2_fedrep")
    if fedrep is not None:
        refs["fedrep_best_rmse"] = fedrep

    if refs:
        ref_str = ", ".join(
            f"{k}={v:.2f}" for k, v in refs.items() if isinstance(v, (int, float))
        )
        print(f"References: {ref_str}\n")

    total_start = time.perf_counter()
    history = run_fedccfa_from_bundle(
        bundle, shards, shard_to_subset,
        n_rounds=args.n_rounds,
        head_epochs=args.head_epochs,
        encoder_epochs=args.encoder_epochs,
        batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        lambda_fault=args.lambda_fault,
        use_cosine_schedule=not args.no_cosine,
        seed=args.seed, log_every=5,
        similarity_threshold=args.similarity_threshold,
        warmup_rounds=args.warmup_rounds,
    )
    total_seconds = time.perf_counter() - total_start
    print(f"\nTotal wall-clock: {total_seconds:.1f}s")
    print(
        f"Best round {history.best_round}: "
        f"macro_RMSE={history.best_macro_rmse:.2f}, "
        f"macro_NASA={history.best_macro_nasa_score:.0f}"
    )
    print(f"Best-round clusters: {history.best_clusters}")

    # ----- per-round CSV -----
    rows: list[dict] = []
    for r in history.rounds:
        row = {
            "round": r.round, "lr": r.lr,
            "mean_client_loss_total": r.mean_client_loss_total,
            "macro_rmse": r.macro_rmse,
            "macro_nasa_score": r.macro_nasa_score,
            "macro_auprc": r.macro_auprc,
            "macro_f1": r.macro_f1,
            "head_similarity_max": r.head_similarity_max,
            "head_similarity_min": r.head_similarity_min,
            "n_clusters": len(r.clusters),
            "clusters_str": "/".join(
                "+".join(g) if len(g) > 1 else g[0] for g in r.clusters
            ),
            "round_seconds": r.round_seconds,
        }
        for m in r.per_client_metrics:
            row[f"{m.client_id}_rmse"] = m.rmse
            row[f"{m.client_id}_f1"] = m.f1
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_dir / "per_round.csv", index=False)

    # ----- plots -----
    safe = display.replace(" ", "").lower()
    figures = {
        "headline": args.out_dir / f"headline_comparison_{safe}.png",
        "cluster_evolution": args.out_dir / f"cluster_evolution_{safe}.png",
        "per_subset": args.out_dir / f"per_subset_breakdown_{safe}.png",
    }
    _plot_headline(figures["headline"], history, refs, display)
    _plot_cluster_evolution(
        figures["cluster_evolution"], history, history.client_ids, display,
    )
    _plot_per_subset_breakdown(
        figures["per_subset"], history, p6_per_subset, display,
    )

    # ----- metrics.json -----
    best = history.rounds[history.best_round - 1]
    per_client_dict = {
        m.client_id: {
            "subset": m.subset,
            "rmse": round(m.rmse, 4),
            "nasa_score": round(m.nasa_score, 4),
            "auprc": round(m.auprc, 4),
            "f1": round(m.f1, 4),
        }
        for m in best.per_client_metrics
    }
    per_subset_summary: dict[str, dict[str, float]] = {}
    for s in args.subsets:
        in_subset = [m for m in best.per_client_metrics if m.subset == s]
        if not in_subset:
            continue
        per_subset_summary[s] = {
            "n_clients": len(in_subset),
            "macro_rmse": round(float(np.mean([m.rmse for m in in_subset])), 4),
            "macro_nasa_score": round(
                float(np.mean([m.nasa_score for m in in_subset])), 4
            ),
            "macro_auprc": round(float(np.mean([m.auprc for m in in_subset])), 4),
            "macro_f1": round(float(np.mean([m.f1 for m in in_subset])), 4),
        }

    central = refs.get("centralized_rmse")
    vanilla = refs.get("fedavg_rmse")
    headroom = (vanilla - central) if central and vanilla else None
    gap_closed_pct = None
    if headroom and headroom > 0:
        gap_closed_pct = round(
            (vanilla - history.best_macro_rmse) / headroom * 100, 2,
        )

    headline_summary = {
        "best_round": history.best_round,
        "best_macro_rmse": round(history.best_macro_rmse, 4),
        "best_macro_nasa_score": round(history.best_macro_nasa_score, 4),
        "best_clusters": history.best_clusters,
        "n_clusters_best": len(history.best_clusters),
        "similarity_threshold": args.similarity_threshold,
        "warmup_rounds": args.warmup_rounds,
        "centralized_rmse_p6": central,
        "vanilla_fedavg_rmse_p6": vanilla,
        "fedprox_best_rmse": refs.get("fedprox_best_rmse"),
        "fedrep_best_macro_rmse": refs.get("fedrep_best_rmse"),
        "macro_rmse_gap_closed_pct": gap_closed_pct,
        "interpretation_note": (
            "Macro RMSE is mean across clients of each client's own per-subset "
            "test RMSE. Apples-to-apples comparison vs centralized lives in "
            "per_subset.<subset>.macro_rmse — compare those to "
            "results/06_non_iid centralized_per_subset RMSEs."
        ),
    }

    interp_parts = [
        f"FedCCFA with similarity_threshold={args.similarity_threshold}, "
        f"warmup_rounds={args.warmup_rounds} achieved best macro test RMSE "
        f"{history.best_macro_rmse:.2f} at round {history.best_round}.",
        f"Best-round cluster structure: {history.best_clusters}.",
    ]
    if per_subset_summary:
        for s, v in per_subset_summary.items():
            cent_ref = (p6_per_subset or {}).get(s)
            if cent_ref:
                interp_parts.append(
                    f"Per-subset: {s} macro RMSE {v['macro_rmse']:.2f} "
                    f"(centralized ref {cent_ref:.2f})."
                )
    if gap_closed_pct is not None:
        interp_parts.append(
            f"Implied combined-RMSE gap closed: {gap_closed_pct:+.1f}%."
        )

    payload = PhaseMetrics(
        phase_id=PHASE_ID, phase_name=PHASE_NAME,
        subset=display, interpretation=" ".join(interp_parts),
        config={
            "subsets": list(args.subsets),
            "n_clients_per_subset": args.n_clients_per_subset,
            "n_rounds": args.n_rounds,
            "head_epochs": args.head_epochs,
            "encoder_epochs": args.encoder_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_fault": args.lambda_fault,
            "similarity_threshold": args.similarity_threshold,
            "warmup_rounds": args.warmup_rounds,
            "use_cosine_schedule": not args.no_cosine,
            "seed": args.seed,
        },
        timing={
            "total_seconds": round(total_seconds, 1),
            "seconds_per_round": round(total_seconds / args.n_rounds, 2),
        },
        summary=headline_summary,
        per_client=per_client_dict,
        per_subset=per_subset_summary,
        artifacts={
            "headline_comparison_png": f"results/{PHASE_ID}/{figures['headline'].name}",
            "cluster_evolution_png": f"results/{PHASE_ID}/{figures['cluster_evolution'].name}",
            "per_subset_breakdown_png": f"results/{PHASE_ID}/{figures['per_subset'].name}",
        },
    )
    out_path = dump_phase_metrics(payload, args.out_dir)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
