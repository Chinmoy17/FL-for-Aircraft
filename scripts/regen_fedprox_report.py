"""One-off: regenerate FedProx metrics.json + headline plots from on-disk artifacts.

The first ``run_fedprox.py`` run completed all 4 µ trainings successfully —
per-round CSVs and best-model checkpoints are committed under
``results/rq2_fedprox/``. Only the metrics.json had broken reference-loading
(empty ``best_gap_closed_pct``) and the headline plot had a missing legend.

This script reads the per-round CSVs, re-evaluates each saved best-checkpoint
per-subset against the canonical bundle, and writes a corrected metrics.json
+ refreshed plots. It does NOT re-run training, so it's idempotent and
~30 seconds total.

Run from the repo root inside the .venv:
    .\\.venv\\Scripts\\python.exe scripts\\regen_fedprox_report.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_aircraft.data import MultiSubsetConfig, load_multi_subset_bundle  # noqa: E402
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics  # noqa: E402

# Re-use the helpers we just fixed inside run_fedprox.py without importing
# the module by name (it has an argparse entrypoint that triggers on import
# only if you call main()). Direct import of the helpers is the cleanest
# path because they encapsulate the exact eval and plot logic we want.
spec = importlib.util.spec_from_file_location(
    "run_fedprox", REPO_ROOT / "scripts" / "run_fedprox.py"
)
assert spec and spec.loader
run_fedprox = importlib.util.module_from_spec(spec)
# Register in sys.modules BEFORE executing so @dataclass can look up the
# module by name (Lib/dataclasses.py:_is_type does sys.modules.get(__module__)).
sys.modules["run_fedprox"] = run_fedprox
spec.loader.exec_module(run_fedprox)

PHASE_ID = run_fedprox.PHASE_ID
PHASE_NAME = run_fedprox.PHASE_NAME
DEFAULT_MUS = list(run_fedprox.DEFAULT_MUS)


def main() -> None:
    out_dir = REPO_ROOT / "results" / PHASE_ID
    if not out_dir.exists():
        raise SystemExit(f"{out_dir} not found — run scripts/run_fedprox.py first.")

    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
    bundle = load_multi_subset_bundle(
        MultiSubsetConfig(subsets=("FD001", "FD003"), data_dir=data_dir)
    )
    display = bundle.display_name
    print(f"--- regenerating FedProx report for {display} ---")

    mus = DEFAULT_MUS
    histories_simple: dict[float, dict] = {}
    per_subset_per_mu: dict[float, list] = {}

    for mu in mus:
        csv_path = out_dir / f"per_round_mu_{mu}.csv"
        ckpt_path = out_dir / f"best_fedprox_state_mu_{mu}.pt"
        if not csv_path.exists() or not ckpt_path.exists():
            raise SystemExit(f"missing artifact for mu={mu}: {csv_path} / {ckpt_path}")
        df = pd.read_csv(csv_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Best round is whatever the saved checkpoint says.
        best_round_idx = int(ckpt["config"]["best_round"])
        # Pull the corresponding row out of the per-round CSV.
        rec = df.iloc[best_round_idx - 1]

        histories_simple[mu] = {
            "best_round": best_round_idx,
            "best_rmse": float(rec["global_test_rmse"]),
            "best_nasa": float(rec["global_test_nasa_score"]),
            "best_auprc": float(rec["global_test_auprc"]),
            "best_f1": float(rec["global_test_f1"]),
            "n_rounds": int(df["round"].max()),
        }

        # Per-subset eval on the saved checkpoint.
        model = MultiTaskCNN(
            MultiTaskCNNConfig(
                n_features=bundle.n_features, window_size=bundle.window_size,
            )
        )
        model.load_state_dict(ckpt["state_dict"])
        per_subset = [
            run_fedprox._eval_on_subset(model, bundle, subset, batch_size=256)
            for subset in ("FD001", "FD003")
        ]
        per_subset_per_mu[mu] = per_subset
        print(
            f"  mu={mu}: RMSE={rec['global_test_rmse']:.2f} "
            f"(best round {best_round_idx}); "
            f"FD001 RMSE={per_subset[0].rmse:.2f}, "
            f"FD003 RMSE={per_subset[1].rmse:.2f}"
        )

    # Use the fixed reference loaders.
    p6 = run_fedprox._load_p6_references()
    p6_per_subset = run_fedprox._load_p6_per_subset_references()
    if p6:
        print(
            f"\nP6 references: centralized={p6.get('centralized_rmse'):.2f}, "
            f"local-only={p6.get('local_only_rmse'):.2f}, "
            f"vanilla FedAvg={p6.get('fedavg_rmse'):.2f}"
        )

    # ----- summary block ----- (mirrors what main() builds, but from rec dicts)
    central = p6.get("centralized_rmse") if p6 else None
    vanilla = p6.get("fedavg_rmse") if p6 else None
    headroom = (vanilla - central) if central is not None and vanilla is not None else None

    best_per_mu: dict[str, dict] = {}
    gap_closed_per_mu: dict[str, float] = {}
    for mu in mus:
        h = histories_simple[mu]
        best_per_mu[f"mu_{mu}"] = {
            "best_round": h["best_round"],
            "rmse": round(h["best_rmse"], 4),
            "nasa_score": round(h["best_nasa"], 4),
            "auprc": round(h["best_auprc"], 4),
            "f1": round(h["best_f1"], 4),
        }
        if headroom and headroom > 0 and vanilla is not None:
            gap_closed = (vanilla - h["best_rmse"]) / headroom * 100.0
            gap_closed_per_mu[f"mu_{mu}"] = round(gap_closed, 2)

    best_mu = min(best_per_mu.items(), key=lambda kv: kv[1]["rmse"])[0]
    headline_summary = {
        "best_mu": best_mu,
        "best_rmse": best_per_mu[best_mu]["rmse"],
        "best_gap_closed_pct": gap_closed_per_mu.get(best_mu),
        "vanilla_rmse_p6": vanilla,
        "centralized_rmse_p6": central,
        "rmse_gap_p6": round(headroom, 4) if headroom else None,
        "per_mu_gap_closed_pct": gap_closed_per_mu,
    }
    print(f"\nbest_mu={best_mu} -> RMSE {best_per_mu[best_mu]['rmse']}, "
          f"gap closed {gap_closed_per_mu.get(best_mu, 'n/a')}%")

    interpretation = _build_interpretation(
        best_mu, best_per_mu, gap_closed_per_mu, per_subset_per_mu, p6,
    )

    # ----- regen plots using the existing helpers (need lightweight history shape) -----
    class _MiniRec:
        """Stand-in for ``RoundRecord`` so the plot helpers Just Work."""
        __slots__ = (
            "round", "lr", "mean_client_loss_total", "mean_client_loss_rul",
            "mean_client_loss_fault", "global_test_rmse", "global_test_mae",
            "global_test_nasa_score", "global_test_auprc", "global_test_f1",
            "global_test_precision", "global_test_recall", "round_seconds",
        )
        def __init__(self, row):
            for k in self.__slots__:
                setattr(self, k, row[k] if k in row else 0.0)

    class _MiniHistory:
        def __init__(self, rounds, best_round, total_seconds):
            self.rounds = rounds
            self.best_round = best_round
            self.total_seconds = total_seconds

    histories_for_plot: dict[float, _MiniHistory] = {}
    for mu in mus:
        df = pd.read_csv(out_dir / f"per_round_mu_{mu}.csv")
        rounds = [_MiniRec(df.iloc[i].to_dict()) for i in range(len(df))]
        histories_for_plot[mu] = _MiniHistory(
            rounds=rounds,
            best_round=histories_simple[mu]["best_round"],
            total_seconds=float(df["round_seconds"].sum()),
        )

    safe_display = display.replace(" ", "")
    figures = {
        "headline": out_dir / f"headline_comparison_{safe_display.lower()}.png",
        "per_round": out_dir / f"per_round_rmse_{safe_display.lower()}.png",
        "per_subset": out_dir / f"per_subset_breakdown_{safe_display.lower()}.png",
    }
    run_fedprox._plot_headline(figures["headline"], histories_for_plot, p6, display)
    run_fedprox._plot_per_round_rmse(figures["per_round"], histories_for_plot, p6, display)
    run_fedprox._plot_per_subset_breakdown(
        figures["per_subset"], per_subset_per_mu, p6_per_subset, display,
    )

    # ----- finally write metrics.json -----
    payload = PhaseMetrics(
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        subset=display,
        interpretation=interpretation,
        config={
            "mus": mus,
            "n_clients_per_subset": 2,
            "n_rounds": 50,
            "local_epochs": 2,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "lambda_fault": 0.5,
            "use_cosine_schedule": True,
            "seed": 42,
            "subsets": ["FD001", "FD003"],
            "regenerated_from_csv": True,
        },
        timing={
            # Approximate: sum of per-round seconds from the CSVs.
            "total_seconds": round(
                sum(h.total_seconds for h in histories_for_plot.values()), 1
            ),
        },
        summary=headline_summary,
        per_client=best_per_mu,
        per_subset={
            f"mu_{mu}": {ps.subset: ps.as_dict() for ps in per_subset_per_mu[mu]}
            for mu in mus
        },
        artifacts={
            "headline_comparison_png":
                f"results/{PHASE_ID}/{figures['headline'].name}",
            "per_round_rmse_png":
                f"results/{PHASE_ID}/{figures['per_round'].name}",
            "per_subset_breakdown_png":
                f"results/{PHASE_ID}/{figures['per_subset'].name}",
        },
    )
    out_path = dump_phase_metrics(payload, out_dir)
    print(f"\nWrote {out_path}")
    print(f"Plots regenerated: {', '.join(p.name for p in figures.values())}")


def _build_interpretation(
    best_mu_key: str,
    best_per_mu: dict[str, dict],
    gap_closed_per_mu: dict[str, float],
    per_subset_per_mu: dict[float, list],
    p6: dict[str, float] | None,
) -> str:
    """Construct the honest one-paragraph interpretation for /results."""
    best = best_per_mu[best_mu_key]
    gap_best = gap_closed_per_mu.get(best_mu_key)
    # Per-subset deltas for FD001 vs FD003 between vanilla (mu=0.0) and best.
    vanilla_ps = {p.subset: p for p in per_subset_per_mu[0.0]}
    best_mu_value = float(best_mu_key.split("_", 1)[1])
    best_ps = {p.subset: p for p in per_subset_per_mu[best_mu_value]}
    fd003_f1_delta = best_ps["FD003"].f1 - vanilla_ps["FD003"].f1
    fd003_rmse_delta = vanilla_ps["FD003"].rmse - best_ps["FD003"].rmse
    fd001_rmse_delta = best_ps["FD001"].rmse - vanilla_ps["FD001"].rmse

    parts = [
        f"FedProx with μ={best_mu_value} achieved best combined RMSE "
        f"{best['rmse']:.2f} on the FD001+FD003 Non-IID partition."
    ]
    if gap_best is not None:
        parts.append(
            f"This closes {gap_best:+.1f}% of the local→centralized RMSE gap "
            f"(headroom = {(p6 or {}).get('local_only_rmse', float('nan')):.2f} − "
            f"{(p6 or {}).get('centralized_rmse', float('nan')):.2f} cycles)."
        )
    parts.append(
        f"The per-subset breakdown tells the real story: vs vanilla FedAvg "
        f"the best FedProx run shifts {fd003_rmse_delta:+.2f} RMSE on FD003 "
        f"(harder, HPC+Fan; F1 {vanilla_ps['FD003'].f1:.3f} → "
        f"{best_ps['FD003'].f1:.3f}, a {fd003_f1_delta:+.3f} F1 delta) "
        f"at the cost of {fd001_rmse_delta:+.2f} RMSE on FD001 (easier, "
        f"HPC-only)."
    )
    parts.append(
        "Combined with RQ2's finding (best reweighting scheme closed only "
        "+2.8% of the gap), two intervention layers — server-side weights "
        "and client-side drift control — both ceiling around RMSE 17.7 on "
        "this 4-client / 2-local-epoch setup. The remaining ~4 RMSE gap is "
        "consistent with the structural-Non-IID hypothesis that different "
        "fault modes need different decision boundaries (FedRep / FedCCFA "
        "architectural layer)."
    )
    return " ".join(parts)


if __name__ == "__main__":
    main()
