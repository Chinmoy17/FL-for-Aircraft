"""FedRep — federated representation learning on the FD001+FD003 Non-IID setup.

This is the architectural-layer follow-up to RQ2 + FedProx. RQ2 showed that
no aggregation weighting closes the Non-IID gap (best +2.8%). FedProx showed
that controlling client drift gives a small win (+6.0% best). FedRep tests
the third hypothesis: maybe the problem isn't *how* you average a shared
model — it's that you shouldn't have ONE shared model in the first place.

This CLI runs FedRep on the same partition / seed / round budget as RQ2 +
FedProx so the comparison is direct. Total local epochs per round =
head_epochs + encoder_epochs (default 1 + 1 = 2, matching vanilla's 2).

Outputs land in ``results/rq2_fedrep/``:

    metrics.json                                structured for the frontend
    per_round.csv                               macro + per-client trajectories
    headline_comparison_fd001_fd003.png         bars vs centralized / local / FedAvg / FedProx
    per_client_rmse_fd001_fd003.png             per-client RMSE trajectory over rounds
    per_subset_breakdown_fd001_fd003.png        FD001 vs FD003 macro RMSE per scheme

Run from the repo root inside the .venv::

    .\\.venv\\Scripts\\python.exe scripts\\run_fedrep.py
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
from fl_aircraft.fl import run_fedrep_from_bundle  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics  # noqa: E402

PHASE_ID = "rq2_fedrep"
PHASE_NAME = "RQ2 follow-up — FedRep (personalised heads) on FD001 + FD003"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD003"])
    p.add_argument("--n-clients-per-subset", type=int, default=2)
    p.add_argument("--n-rounds", type=int, default=50)
    p.add_argument(
        "--head-epochs", type=int, default=1,
        help="Number of local epochs spent training heads only (encoder frozen).",
    )
    p.add_argument(
        "--encoder-epochs", type=int, default=1,
        help="Number of local epochs spent training encoder only (heads frozen).",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-fault", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cosine", action="store_true")
    p.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "results" / PHASE_ID,
    )
    return p.parse_args()


def _build_shard_to_subset(shards, subsets, n_per_subset) -> dict[str, str]:
    """``partition_by_subset_halves`` lays out shards in subset order, so the
    first ``n_per_subset`` shards belong to ``subsets[0]``, the next
    ``n_per_subset`` belong to ``subsets[1]``, and so on."""
    out: dict[str, str] = {}
    if len(shards) != len(subsets) * n_per_subset:
        raise ValueError(
            f"shard count {len(shards)} != subsets x n_per_subset "
            f"({len(subsets)} x {n_per_subset})"
        )
    for i, shard in enumerate(shards):
        out[shard.client_id] = subsets[i // n_per_subset]
    return out


# ---------------------------------------------------------------------------
# Reference loading from prior phases
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
        ("centralized_auprc", "centralized_auprc"),
        ("centralized_f1", "centralized_f1"),
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


def _load_fedprox_best() -> dict[str, float] | None:
    fp = REPO_ROOT / "results" / "rq2_fedprox" / "metrics.json"
    if not fp.exists():
        return None
    with fp.open(encoding="utf-8") as fh:
        d = json.load(fh)
    summary = d.get("summary") or {}
    return {
        "best_rmse": float(summary["best_rmse"]),
        "best_gap_closed_pct": summary.get("best_gap_closed_pct"),
    } if "best_rmse" in summary else None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_headline(
    out_path: Path, history, p6: dict[str, float] | None,
    fedprox: dict[str, float] | None, display: str,
) -> None:
    """4-bar comparison: centralized | vanilla FedAvg | FedProx best | FedRep."""
    bars: list[tuple[str, float, str]] = []
    if p6 and "centralized_rmse" in p6:
        bars.append(("Centralized\n(upper bound)", p6["centralized_rmse"], "black"))
    if p6 and "fedavg_rmse" in p6:
        bars.append(("Vanilla FedAvg\n(control)", p6["fedavg_rmse"], "tab:red"))
    if fedprox and "best_rmse" in fedprox:
        bars.append(("FedProx best\n(μ=0.1)", fedprox["best_rmse"], "tab:purple"))
    bars.append(("FedRep\n(this run)", history.best_macro_rmse, "tab:green"))
    if p6 and "local_only_rmse" in p6:
        bars.append(("Local-only mean\n(lower bound)", p6["local_only_rmse"], "tab:gray"))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    rects = ax.bar(labels, values, color=colors)
    for rect, v in zip(rects, values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("test RMSE (cycles)")
    ax.set_title(
        f"FedRep best macro-RMSE vs prior phases — {display}\n"
        f"(lower is better; FedRep best is macro mean across per-client test slices)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_client_rmse(
    out_path: Path, history, display: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for client_id, rmse_series in history.per_round_client_rmse.items():
        ax.plot(
            range(1, len(rmse_series) + 1), rmse_series,
            label=client_id, linewidth=1.6,
        )
    # Macro mean for context.
    n_rounds = len(history.rounds)
    macro = [r.macro_rmse for r in history.rounds]
    ax.plot(
        range(1, n_rounds + 1), macro, label="macro mean",
        color="black", linestyle="--", linewidth=1.4,
    )
    ax.set_xlabel("round")
    ax.set_ylabel("per-client test RMSE (cycles)")
    ax.set_title(f"FedRep per-client trajectory — {display}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_subset_breakdown(
    out_path: Path, history, p6_per_subset: dict[str, float] | None, display: str,
) -> None:
    """For the best round, mean per-client RMSE within each subset, vs centralized refs."""
    best = history.rounds[history.best_round - 1]
    per_subset_rmse: dict[str, list[float]] = {}
    for m in best.per_client_metrics:
        per_subset_rmse.setdefault(m.subset, []).append(m.rmse)
    subsets = sorted(per_subset_rmse.keys())
    fedrep_means = [float(np.mean(per_subset_rmse[s])) for s in subsets]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(subsets))
    width = 0.35
    rects = ax.bar(x - width / 2, fedrep_means, width, color="tab:green",
                   label=f"FedRep (best round {history.best_round})")
    if p6_per_subset:
        cent_means = [p6_per_subset.get(s, float("nan")) for s in subsets]
        ax.bar(x + width / 2, cent_means, width, color="black",
               alpha=0.6, label="Centralized (P6)")
        for i, v in enumerate(cent_means):
            if not np.isnan(v):
                ax.text(x[i] + width / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=9)
    for r, v in zip(rects, fedrep_means):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(),
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylabel("test RMSE (cycles)")
    ax.set_title(f"FedRep per-subset RMSE vs centralized — {display}")
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
    print(f"--- FedRep ({display}) ---")
    print(f"  subsets: {args.subsets}")
    print(f"  clients per subset: {args.n_clients_per_subset}")
    print(f"  rounds × (head+encoder epochs): "
          f"{args.n_rounds} × ({args.head_epochs}+{args.encoder_epochs})")
    print(f"  seed: {args.seed}\n")

    shards = partition_by_subset_halves(
        bundle.train_df, subsets=tuple(args.subsets),
        n_clients_per_subset=args.n_clients_per_subset, seed=args.seed,
    )
    shard_to_subset = _build_shard_to_subset(
        shards, args.subsets, args.n_clients_per_subset,
    )
    for s in shards:
        print(f"  · {s.client_id}: {shard_to_subset[s.client_id]} "
              f"({s.n_engines} engines)")
    print()

    p6 = _load_p6_references()
    p6_per_subset = _load_p6_per_subset_centralized()
    fedprox = _load_fedprox_best()
    if p6:
        print(
            f"P6 references: centralized RMSE={p6.get('centralized_rmse', 0):.2f}, "
            f"vanilla FedAvg={p6.get('fedavg_rmse', 0):.2f}, "
            f"local-only mean={p6.get('local_only_rmse', 0):.2f}"
        )
    if fedprox:
        print(f"FedProx best: RMSE={fedprox['best_rmse']:.2f}")
    print()

    total_start = time.perf_counter()
    history = run_fedrep_from_bundle(
        bundle, shards, shard_to_subset,
        n_rounds=args.n_rounds,
        head_epochs=args.head_epochs,
        encoder_epochs=args.encoder_epochs,
        batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        lambda_fault=args.lambda_fault,
        use_cosine_schedule=not args.no_cosine,
        seed=args.seed, log_every=5,
    )
    total_seconds = time.perf_counter() - total_start
    print(f"\nTotal wall-clock: {total_seconds:.1f}s")
    print(
        f"Best round {history.best_round}: macro_RMSE={history.best_macro_rmse:.2f}, "
        f"macro_NASA={history.best_macro_nasa_score:.0f}"
    )

    # ----- per-round CSV -----
    rows: list[dict] = []
    for r in history.rounds:
        row = {
            "round": r.round, "lr": r.lr,
            "mean_client_loss_total": r.mean_client_loss_total,
            "mean_client_loss_rul": r.mean_client_loss_rul,
            "mean_client_loss_fault": r.mean_client_loss_fault,
            "macro_rmse": r.macro_rmse,
            "macro_nasa_score": r.macro_nasa_score,
            "macro_auprc": r.macro_auprc,
            "macro_f1": r.macro_f1,
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
        "per_client": args.out_dir / f"per_client_rmse_{safe}.png",
        "per_subset": args.out_dir / f"per_subset_breakdown_{safe}.png",
    }
    _plot_headline(figures["headline"], history, p6, fedprox, display)
    _plot_per_client_rmse(figures["per_client"], history, display)
    _plot_per_subset_breakdown(figures["per_subset"], history, p6_per_subset, display)

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
    # Per-subset summary at best round (mean across clients in each subset).
    per_subset_summary: dict[str, dict[str, float]] = {}
    for s in args.subsets:
        clients_in_subset = [m for m in best.per_client_metrics if m.subset == s]
        if not clients_in_subset:
            continue
        per_subset_summary[s] = {
            "n_clients": len(clients_in_subset),
            "macro_rmse": round(float(np.mean([m.rmse for m in clients_in_subset])), 4),
            "macro_nasa_score": round(
                float(np.mean([m.nasa_score for m in clients_in_subset])), 4
            ),
            "macro_auprc": round(float(np.mean([m.auprc for m in clients_in_subset])), 4),
            "macro_f1": round(float(np.mean([m.f1 for m in clients_in_subset])), 4),
        }

    # Gap-closed % framed honestly: FedRep's macro RMSE is computed
    # PER-CLIENT (each scored only on its own subset), while P6's centralized
    # RMSE is a single number on the combined test set. They are not strictly
    # comparable. We still report the implied gap-closed for narrative
    # context, but the per-subset breakdown is the more honest comparison.
    central = p6.get("centralized_rmse") if p6 else None
    vanilla = p6.get("fedavg_rmse") if p6 else None
    headroom = (vanilla - central) if central is not None and vanilla is not None else None
    gap_closed_pct = None
    if headroom and headroom > 0:
        gap_closed_pct = round(
            (vanilla - history.best_macro_rmse) / headroom * 100, 2,
        )

    headline_summary = {
        "best_round": history.best_round,
        "best_macro_rmse": round(history.best_macro_rmse, 4),
        "best_macro_nasa_score": round(history.best_macro_nasa_score, 4),
        "centralized_rmse_p6": central,
        "vanilla_fedavg_rmse_p6": vanilla,
        "fedprox_best_rmse": fedprox.get("best_rmse") if fedprox else None,
        "macro_rmse_gap_closed_pct": gap_closed_pct,
        "interpretation_note": (
            "Macro RMSE is mean across clients of each client's own per-subset "
            "test RMSE. It is NOT directly comparable to centralized RMSE on "
            "the combined test set — use per_subset.macro_rmse vs the P6 "
            "centralized_per_subset numbers for an apples-to-apples comparison."
        ),
    }

    interpretation = (
        f"FedRep with head_epochs={args.head_epochs}, "
        f"encoder_epochs={args.encoder_epochs} achieved best macro test RMSE "
        f"{history.best_macro_rmse:.2f} at round {history.best_round}. "
    )
    if per_subset_summary:
        for s, v in per_subset_summary.items():
            cent_ref = (p6_per_subset or {}).get(s)
            if cent_ref:
                interpretation += (
                    f"Per-subset: {s} macro RMSE {v['macro_rmse']:.2f} "
                    f"(centralized ref {cent_ref:.2f}). "
                )
    if gap_closed_pct is not None:
        interpretation += (
            f"Implied combined-RMSE gap closed: {gap_closed_pct:+.1f}% "
            f"(treating macro RMSE as a stand-in; see note in summary). "
        )

    payload = PhaseMetrics(
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        subset=display,
        interpretation=interpretation,
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
            "per_client_rmse_png": f"results/{PHASE_ID}/{figures['per_client'].name}",
            "per_subset_breakdown_png": f"results/{PHASE_ID}/{figures['per_subset'].name}",
        },
    )
    out_path = dump_phase_metrics(payload, args.out_dir)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
