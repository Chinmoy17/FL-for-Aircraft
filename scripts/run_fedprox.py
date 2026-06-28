"""FedProx — sweep mu on the P6 Non-IID partition.

This is the follow-up experiment RQ2's negative finding pointed at. RQ2
showed that no choice of aggregation weights closes the centralized-vs-
federated RMSE gap under Non-IID. Li et al. (MLSys 2020) argued the right
intervention layer is the **local optimisation step**, not aggregation —
specifically, adding a proximal term mu/2 * ||W - W_global||^2 to each
client's local loss to penalise drift.

This script runs the **same** P6 Non-IID partition (FD001 + FD003, 4
clients, 50 rounds × 2 local epochs, same seed) with vanilla sample-count
FedAvg aggregation, but varying mu ∈ {0.0, 0.001, 0.01, 0.1}. mu=0.0 is
the control (identical to P6's FedAvg run) and lets the script self-check
its own backward compatibility.

Outputs land in ``results/rq2_fedprox/`` (grouped with RQ2 narratively):

    metrics.json                            structured for the frontend
    per_round_mu_<mu>.csv                   per-scheme trajectory
    headline_comparison_fd001_fd003.png     bar chart vs centralized / local / FedAvg
    per_round_rmse_fd001_fd003.png          RMSE trajectories per mu vs references
    per_subset_breakdown_fd001_fd003.png    FD001 vs FD003 per mu
    best_fedprox_state_mu_<mu>.pt           checkpoint (for the demo backend)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_aircraft.data import (  # noqa: E402
    CMAPSSWindowDataset,
    MultiSubsetConfig,
    Normalizer,
    SUBSET_COL,
    UNIT_ID_COL,
    load_multi_subset_bundle,
    make_test_windows,
    partition_by_subset_halves,
)
from fl_aircraft.eval import (  # noqa: E402
    compute_classification_metrics,
    compute_regression_metrics,
)
from fl_aircraft.fl import run_fedavg_from_bundle  # noqa: E402
from fl_aircraft.fl.simulation import FederatedHistory  # noqa: E402
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics  # noqa: E402

PHASE_ID = "rq2_fedprox"
PHASE_NAME = "RQ2 follow-up — FedProx (μ-sweep) on FD001 + FD003"

DEFAULT_MUS = (0.0, 0.001, 0.01, 0.1)
MU_COLOR = {
    0.0: "tab:red",
    0.001: "tab:blue",
    0.01: "tab:green",
    0.1: "tab:purple",
}


@dataclass
class PerSubsetMetrics:
    subset: str
    n_engines: int
    rmse: float
    nasa_score: float
    auprc: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "subset": self.subset,
            "n_engines": self.n_engines,
            "rmse": round(self.rmse, 4),
            "nasa_score": round(self.nasa_score, 4),
            "auprc": round(self.auprc, 4),
            "f1": round(self.f1, 4),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD003"])
    p.add_argument(
        "--mus", nargs="+", type=float, default=list(DEFAULT_MUS),
        help="Values of mu to sweep over. Default: 0.0 0.001 0.01 0.1",
    )
    p.add_argument("--n-clients-per-subset", type=int, default=2)
    p.add_argument("--n-rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=2)
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


# ---------------------------------------------------------------------------
# Per-subset evaluation (mirrors scripts/run_rq2.py.evaluate_state_per_subset)
# ---------------------------------------------------------------------------
def _eval_on_subset(
    model: MultiTaskCNN,
    bundle,
    subset: str,
    batch_size: int,
) -> PerSubsetMetrics:
    """Score the already-loaded model on one subset's test slice.

    ``bundle.test_rul`` is a numpy array indexed by position in the full
    sorted ``unit_id`` list, not a pandas Series — so we look up positions
    explicitly. Same pattern as ``scripts/run_rq2.py``.
    """
    normalizer = Normalizer.fit(bundle.train_df, bundle.feature_cols)
    test_df = normalizer.transform(bundle.test_raw_df)
    full_engines = sorted(bundle.test_raw_df[UNIT_ID_COL].unique())
    sub_test_df = test_df.loc[test_df[SUBSET_COL] == subset].copy()
    if sub_test_df.empty:
        raise ValueError(f"No test rows for subset {subset!r}.")
    engine_ids_in_subset = sorted(sub_test_df[UNIT_ID_COL].unique())
    sub_rul = bundle.test_rul[
        np.array([full_engines.index(u) for u in engine_ids_in_subset])
    ]
    sub_arrays = make_test_windows(
        sub_test_df, sub_rul, bundle.feature_cols,
        bundle.window_size, bundle.rul_cap, bundle.fault_threshold,
    )
    loader = DataLoader(
        CMAPSSWindowDataset(sub_arrays), batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    model.eval()
    rul_preds: list[np.ndarray] = []
    fault_scores: list[np.ndarray] = []
    with torch.no_grad():
        for x, _y_rul, _y_fault in loader:
            pred = model(x)
            rul_preds.append(pred.rul.numpy())
            fault_scores.append(pred.fault_probs().numpy())
    y_rul_pred = np.concatenate(rul_preds)
    y_fault_score = np.concatenate(fault_scores)
    rul_m = compute_regression_metrics(sub_arrays.y_rul, y_rul_pred)
    fault_m = compute_classification_metrics(sub_arrays.y_fault, y_fault_score)
    return PerSubsetMetrics(
        subset=subset, n_engines=sub_arrays.n_samples,
        rmse=rul_m.rmse, nasa_score=rul_m.nasa_score,
        auprc=fault_m.auprc, f1=fault_m.f1,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_headline(
    out_path: Path,
    histories: dict[float, FederatedHistory],
    references: dict[str, float] | None,
    display: str,
) -> None:
    mus = sorted(histories.keys())
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    metric_specs = [
        ("RMSE", "global_test_rmse", "lower is better"),
        ("NASA", "global_test_nasa_score", "lower is better"),
        ("AUPRC", "global_test_auprc", "higher is better"),
        ("F1", "global_test_f1", "higher is better"),
    ]
    bar_colors = [MU_COLOR.get(mu, "tab:gray") for mu in mus]
    xlabels = [_mu_label(mu) for mu in mus]
    for ax, (title, key, subtitle) in zip(axes, metric_specs):
        # Best-round value per mu (the headline number).
        best_values = [
            _best_value_for(histories[mu], key, lower_is_better=("lower" in subtitle))
            for mu in mus
        ]
        bars = ax.bar(xlabels, best_values, color=bar_colors)
        for bar, v in zip(bars, best_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}" if v < 100 else f"{v:.0f}",
                ha="center", va="bottom", fontsize=9,
            )
        if references is not None:
            if title == "RMSE":
                if "centralized_rmse" in references:
                    ax.axhline(
                        references["centralized_rmse"], color="black",
                        linestyle="--", linewidth=1,
                        label=f"centralized ({references['centralized_rmse']:.2f})",
                    )
                if "local_only_rmse" in references:
                    ax.axhline(
                        references["local_only_rmse"], color="gray",
                        linestyle=":", linewidth=1,
                        label=f"local-only ({references['local_only_rmse']:.2f})",
                    )
                ax.legend(fontsize=8, loc="best")
        ax.set_title(f"{title}\n({subtitle})", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"FedProx μ-sweep — best-round metrics on {display}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_round_rmse(
    out_path: Path,
    histories: dict[float, FederatedHistory],
    references: dict[str, float] | None,
    display: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for mu in sorted(histories.keys()):
        rmses = [r.global_test_rmse for r in histories[mu].rounds]
        rounds = [r.round for r in histories[mu].rounds]
        ax.plot(
            rounds, rmses, label=_mu_label(mu),
            color=MU_COLOR.get(mu, "tab:gray"), linewidth=1.6,
        )
    if references is not None:
        if "centralized_rmse" in references:
            ax.axhline(
                references["centralized_rmse"], color="black", linestyle="--",
                linewidth=1.2, label=f"centralized ({references['centralized_rmse']:.2f})",
            )
        if "local_only_rmse" in references:
            ax.axhline(
                references["local_only_rmse"], color="gray", linestyle=":",
                linewidth=1.2, label=f"local-only ({references['local_only_rmse']:.2f})",
            )
    ax.set_xlabel("round")
    ax.set_ylabel("global test RMSE (cycles)")
    ax.set_title(f"Per-round RMSE for each μ — {display}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_subset_breakdown(
    out_path: Path,
    per_subset_per_mu: dict[float, list[PerSubsetMetrics]],
    references_per_subset: dict[str, dict[str, float]] | None,
    display: str,
) -> None:
    mus = sorted(per_subset_per_mu.keys())
    # Collect the union of subsets — should be FD001 and FD003 for the
    # canonical Non-IID setup.
    all_subsets: list[str] = []
    for ps_list in per_subset_per_mu.values():
        for ps in ps_list:
            if ps.subset not in all_subsets:
                all_subsets.append(ps.subset)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    width = 0.8 / max(len(mus), 1)
    x = np.arange(len(all_subsets))
    for i, mu in enumerate(mus):
        per_subset_dict = {p.subset: p for p in per_subset_per_mu[mu]}
        rmses = [per_subset_dict[s].rmse for s in all_subsets]
        ax.bar(
            x + (i - (len(mus) - 1) / 2) * width, rmses, width,
            color=MU_COLOR.get(mu, "tab:gray"), label=_mu_label(mu),
        )
        for j, v in enumerate(rmses):
            ax.text(
                x[j] + (i - (len(mus) - 1) / 2) * width, v,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8,
            )
    if references_per_subset is not None:
        for j, subset in enumerate(all_subsets):
            ref = references_per_subset.get(subset, {})
            if "centralized_rmse" in ref:
                ax.hlines(
                    ref["centralized_rmse"], x[j] - 0.4, x[j] + 0.4,
                    colors="black", linestyles="--", linewidth=1,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(all_subsets)
    ax.set_ylabel("test RMSE (cycles)")
    ax.set_title(
        f"Per-subset breakdown across μ — {display}\n"
        "(dashed black = centralized RMSE on that subset, if available)",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _mu_label(mu: float) -> str:
    if mu == 0.0:
        return "μ=0 (FedAvg)"
    return f"μ={mu:g}"


def _best_value_for(
    history: FederatedHistory, key: str, *, lower_is_better: bool,
) -> float:
    # Best round was selected by NASA score; for display we use the value
    # of `key` at that round (consistent with the headline-metric convention
    # used throughout the project).
    rec = history.rounds[history.best_round - 1]
    return float(getattr(rec, key))


# ---------------------------------------------------------------------------
# Reference loading from prior phases
# ---------------------------------------------------------------------------
def _load_p6_references() -> dict[str, float] | None:
    """Pull the P6 reference numbers (centralized / FedAvg / local-only) out
    of ``results/06_non_iid/metrics.json``. P6 stashes its headline numbers
    under ``summary``, not under ``test`` — paths verified against the file
    on disk."""
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
        ("centralized_f1", "centralized_f1"),
        ("centralized_auprc", "centralized_auprc"),
    ):
        if k_in in summary:
            out[k_out] = float(summary[k_in])
    return out if out else None


def _load_p6_per_subset_references() -> dict[str, dict[str, float]] | None:
    """Per-subset centralized RMSEs from P6's ``test.centralized_per_subset``."""
    p6 = REPO_ROOT / "results" / "06_non_iid" / "metrics.json"
    if not p6.exists():
        return None
    with p6.open(encoding="utf-8") as fh:
        d = json.load(fh)
    out: dict[str, dict[str, float]] = {}
    for ps in d.get("test", {}).get("centralized_per_subset", []) or []:
        out[ps["subset"]] = {"centralized_rmse": float(ps["rmse"])}
    return out if out else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mus = sorted({float(m) for m in args.mus})
    if any(m < 0 for m in mus):
        raise SystemExit("All mu values must be >= 0.")

    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
    multi_cfg = MultiSubsetConfig(subsets=tuple(args.subsets), data_dir=data_dir)
    bundle = load_multi_subset_bundle(multi_cfg)
    display = bundle.display_name
    print(f"--- FedProx μ-sweep ({display}) ---")
    print(f"  mus to sweep: {mus}")
    print(f"  rounds × local-epochs: {args.n_rounds} × {args.local_epochs}")
    print(f"  seed: {args.seed}\n")

    shards = partition_by_subset_halves(
        bundle.train_df, subsets=tuple(args.subsets),
        n_clients_per_subset=args.n_clients_per_subset, seed=args.seed,
    )

    p6 = _load_p6_references()
    if p6:
        print(
            f"  P6 reference: centralized RMSE={p6.get('centralized_rmse', float('nan')):.2f}, "
            f"local-only RMSE={p6.get('local_only_rmse', float('nan')):.2f}, "
            f"vanilla FedAvg RMSE={p6.get('fedavg_rmse', float('nan')):.2f}\n"
        )
    p6_per_subset = _load_p6_per_subset_references()

    histories: dict[float, FederatedHistory] = {}
    per_subset_per_mu: dict[float, list[PerSubsetMetrics]] = {}
    total_start = time.perf_counter()

    for mu in mus:
        print(f"\n========= Running μ = {mu} =========")
        run_start = time.perf_counter()
        history = run_fedavg_from_bundle(
            bundle, shards,
            n_rounds=args.n_rounds, local_epochs=args.local_epochs,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, lambda_fault=args.lambda_fault,
            use_cosine_schedule=not args.no_cosine,
            seed=args.seed, log_every=10,
            mu=mu,
        )
        elapsed = time.perf_counter() - run_start
        histories[mu] = history

        best = history.rounds[history.best_round - 1]
        print(
            f"  done μ={mu} in {elapsed:.1f}s — "
            f"best round {history.best_round}: "
            f"RMSE={best.global_test_rmse:.2f}  "
            f"NASA={best.global_test_nasa_score:.0f}  "
            f"AUPRC={best.global_test_auprc:.3f}  "
            f"F1={best.global_test_f1:.3f}"
        )

        # Persist per-round CSV.
        df = pd.DataFrame([r.as_dict() for r in history.rounds])
        df.to_csv(args.out_dir / f"per_round_mu_{mu}.csv", index=False)

        # Per-subset breakdown — needs the best state-dict loaded into a model.
        eval_model = MultiTaskCNN(
            MultiTaskCNNConfig(
                n_features=bundle.n_features, window_size=bundle.window_size,
            )
        )
        eval_model.load_state_dict(history.best_state_dict)
        per_subset = [
            _eval_on_subset(eval_model, bundle, subset, args.batch_size)
            for subset in args.subsets
        ]
        per_subset_per_mu[mu] = per_subset

        # Save the checkpoint so the demo backend can load it later.
        torch.save(
            {
                "state_dict": history.best_state_dict,
                "config": {
                    "n_features": bundle.n_features,
                    "window_size": bundle.window_size,
                    "mu": mu,
                    "best_round": history.best_round,
                    "rounds": args.n_rounds,
                    "local_epochs": args.local_epochs,
                },
            },
            args.out_dir / f"best_fedprox_state_mu_{mu}.pt",
        )

    total_seconds = time.perf_counter() - total_start
    print(f"\nTotal wall-clock: {total_seconds:.1f}s for {len(mus)} runs.")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    base = args.out_dir
    safe_display = display.replace(" ", "").replace("+", "+")
    figures = {
        "headline": base / f"headline_comparison_{safe_display.lower()}.png",
        "per_round": base / f"per_round_rmse_{safe_display.lower()}.png",
        "per_subset": base / f"per_subset_breakdown_{safe_display.lower()}.png",
    }
    _plot_headline(figures["headline"], histories, p6, display)
    _plot_per_round_rmse(figures["per_round"], histories, p6, display)
    _plot_per_subset_breakdown(
        figures["per_subset"], per_subset_per_mu, p6_per_subset, display,
    )

    # -----------------------------------------------------------------------
    # metrics.json — for /api/summary and the frontend
    # -----------------------------------------------------------------------
    best_per_mu: dict[str, dict[str, float]] = {}
    gap_closed_per_mu: dict[str, float] = {}
    central = p6.get("centralized_rmse") if p6 else None
    vanilla = p6.get("fedavg_rmse") if p6 else None
    headroom = (
        (vanilla - central) if central is not None and vanilla is not None else None
    )

    for mu in mus:
        rec = histories[mu].rounds[histories[mu].best_round - 1]
        best_per_mu[f"mu_{mu}"] = {
            "best_round": histories[mu].best_round,
            "rmse": round(rec.global_test_rmse, 4),
            "nasa_score": round(rec.global_test_nasa_score, 4),
            "auprc": round(rec.global_test_auprc, 4),
            "f1": round(rec.global_test_f1, 4),
            "wall_seconds": round(histories[mu].total_seconds, 1),
        }
        if headroom and headroom > 0 and vanilla is not None:
            # gap closed % = (vanilla - this) / (vanilla - centralized)
            gap_closed = (vanilla - rec.global_test_rmse) / headroom * 100.0
            gap_closed_per_mu[f"mu_{mu}"] = round(gap_closed, 2)

    best_mu = min(
        best_per_mu.items(), key=lambda kv: kv[1]["rmse"],
    )[0]
    headline_summary = {
        "best_mu": best_mu,
        "best_rmse": best_per_mu[best_mu]["rmse"],
        "best_gap_closed_pct": gap_closed_per_mu.get(best_mu),
        "vanilla_rmse_p6": vanilla,
        "centralized_rmse_p6": central,
        "rmse_gap_p6": round(headroom, 4) if headroom else None,
    }

    interpretation = (
        f"FedProx μ-sweep on the same P6 Non-IID partition that RQ2 ruled "
        f"out as unsalvageable by reweighting. Best result: μ={best_mu} → "
        f"RMSE {best_per_mu[best_mu]['rmse']}"
    )
    if headroom and "best_gap_closed_pct" in headline_summary and headline_summary["best_gap_closed_pct"]:
        interpretation += (
            f" (closing {headline_summary['best_gap_closed_pct']:.1f}% of the "
            f"local→centralized gap that vanilla FedAvg could not close)."
        )
    else:
        interpretation += "."

    payload = PhaseMetrics(
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        subset=display,
        interpretation=interpretation,
        config={
            "mus": mus,
            "n_clients_per_subset": args.n_clients_per_subset,
            "n_rounds": args.n_rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_fault": args.lambda_fault,
            "use_cosine_schedule": not args.no_cosine,
            "seed": args.seed,
            "subsets": list(args.subsets),
        },
        timing={
            "total_seconds": round(total_seconds, 1),
            "seconds_per_mu": round(total_seconds / max(len(mus), 1), 1),
        },
        summary=headline_summary,
        per_client=best_per_mu,
        per_subset={
            f"mu_{mu}": {
                ps.subset: ps.as_dict() for ps in per_subset_per_mu[mu]
            }
            for mu in mus
        },
        artifacts={
            "headline_comparison_png": (
                f"results/{PHASE_ID}/{figures['headline'].name}"
            ),
            "per_round_rmse_png": (
                f"results/{PHASE_ID}/{figures['per_round'].name}"
            ),
            "per_subset_breakdown_png": (
                f"results/{PHASE_ID}/{figures['per_subset'].name}"
            ),
        },
    )
    out_path = dump_phase_metrics(payload, args.out_dir)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
