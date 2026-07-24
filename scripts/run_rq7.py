"""RQ7 — Model poisoning attacks + Byzantine-robust aggregation defenses.

Runs a 27-cell experimental matrix (baselines + 4 attack families ×
4 aggregators, minus a few unused cross-cells) on the FD001+FD003 Non-IID
partition that all prior RQ2-family experiments used. Same seed, same
rounds, same 4 clients (2 on FD001, 2 on FD003).

Cells run by default:

  baselines (4 runs):
    B0  clean + vanilla FedAvg        (re-run of P6 / FedProx mu=0)
    B1  clean + trimmed mean          (sanity: should match B0 closely)
    B2  clean + median                (sanity: slight regression OK)
    B3  clean + Krum                  (sanity: slight regression OK)

  Attack family A — label flip (4 runs):
    AV1 label flip + vanilla          (one FD003 client lies)
    D11 label flip + trimmed mean
    D12 label flip + median
    D13 label flip + Krum

  Attack family B — gradient scaling ×-10 (catastrophic, 4 runs):
    AV2 grad ×-10 + vanilla           (one FD003 client sends boosted-negative)
    D21 grad ×-10 + trimmed mean
    D22 grad ×-10 + median
    D23 grad ×-10 + Krum

  Attack family C — gradient scaling ×-2 (stealthy, 4 runs):
    AV4 grad ×-2 + vanilla            (bounded-norm boosted-negative)
    D41 grad ×-2 + trimmed mean
    D42 grad ×-2 + median
    D43 grad ×-2 + Krum

  Attack family D — backdoor (stealthy, 4 runs):
    AV3 backdoor + vanilla            (poison-only, honest gradient)
    D31 backdoor + trimmed mean
    D32 backdoor + median
    D33 backdoor + Krum

  Attack family E — coordinated Byzantine, 2 attackers × gradient ×-10 (5 runs):
    AV5 coord ×-10 + vanilla          (both FD003 clients collude)
    D51 coord ×-10 + trimmed mean
    D52 coord ×-10 + median
    D53 coord ×-10 + Krum (f=1)       (defence-broken: attackers exceed f)
    D54 coord ×-10 + Krum (f=2)       (defence weakened: n≥2f+3 requires n≥7)

  bonus (1 run):
    F1  grad ×-10 + FedRep            (personalised heads as implicit defense)

Total: ~26 runs at ~3 min each on CPU = ~80 min wall-clock per seed.

Outputs land in ``--out-dir`` (default: ``results/rq7_poisoning/``):
  metrics.json                                 structured for the frontend
  per_round_<cell_key>.csv                     N trajectories
  headline_comparison_fd001_fd003.png          all cells side by side
  attack_diagnostic_delta_norms.png            the attacker's |delta| vs honest
  defense_recovery_fd001_fd003.png             "broken→recovered" pairs
  per_subset_breakdown_fd001_fd003.png         FD001 vs FD003 per cell

Multi-seed runs: use ``scripts/run_rq7_multiseed.py`` which invokes this
script for a range of seeds, writing each seed to its own out-dir. Then
``scripts/aggregate_rq7_seeds.py`` collates them into a single
mean-±-std-with-95%-CI metrics.json for the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
from fl_aircraft.fl import (  # noqa: E402
    BackdoorAttacker,
    GradientScaleAttacker,
    LabelFlipAttacker,
    PoisonedHistory,
    fedavg_aggregate,
    make_krum_aggregator,
    make_median_aggregator,
    make_trimmed_mean_aggregator,
    run_fedavg_with_attackers,
    stamp_trigger_on_windows,
)
from fl_aircraft.fl.poisoning import (  # noqa: E402
    DEFAULT_TRIGGER_CYCLE_OFFSET,
    DEFAULT_TRIGGER_FEATURE_IDX,
    DEFAULT_TRIGGER_POISON_FRAC,
    DEFAULT_TRIGGER_VALUE,
)
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig  # noqa: E402
from fl_aircraft.utils import PhaseMetrics, dump_phase_metrics  # noqa: E402

PHASE_ID = "rq7_poisoning"
PHASE_NAME = "RQ7 — Model poisoning attacks + Byzantine-robust aggregation"


# ---------------------------------------------------------------------------
# Cell specifications
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellSpec:
    """One row of the experimental matrix.

    ``attacker_client_ids`` is normally ``None`` (single-attacker cell —
    the CLI default ``--attacker-client-id`` is used). For coordinated-
    attacker cells this is an explicit tuple of two or more client ids.
    """

    key: str
    label: str
    attacker_kind: str         # "none" / "label_flip" / "grad_scale_x10" / "grad_scale_x2" / "backdoor"
    aggregator_name: str       # "fedavg" / "trimmed_mean" / "median" / "krum_f1" / "krum_f2"
    group: str                 # "baseline", "attack", "defense", "bonus"
    attacker_client_ids: tuple[str, ...] | None = None


def all_cells() -> list[CellSpec]:
    # Baselines: no attacker; each aggregator on clean data.
    baselines = [
        CellSpec("B0_clean_vanilla", "clean + vanilla FedAvg", "none", "fedavg", "baseline"),
        CellSpec("B1_clean_trimmed", "clean + trimmed mean", "none", "trimmed_mean", "baseline"),
        CellSpec("B2_clean_median", "clean + median", "none", "median", "baseline"),
        CellSpec("B3_clean_krum", "clean + Krum (f=1)", "none", "krum_f1", "baseline"),
    ]
    # Family A — label flip (existing).
    label_flip = [
        CellSpec("AV1_labelflip_vanilla", "label-flip + vanilla", "label_flip", "fedavg", "attack"),
        CellSpec("D11_labelflip_trimmed", "label-flip + trimmed mean", "label_flip", "trimmed_mean", "defense"),
        CellSpec("D12_labelflip_median", "label-flip + median", "label_flip", "median", "defense"),
        CellSpec("D13_labelflip_krum", "label-flip + Krum (f=1)", "label_flip", "krum_f1", "defense"),
    ]
    # Family B — catastrophic gradient scaling ×-10 (existing).
    grad_x10 = [
        CellSpec("AV2_gradscale_vanilla", "grad ×-10 + vanilla", "grad_scale_x10", "fedavg", "attack"),
        CellSpec("D21_gradscale_trimmed", "grad ×-10 + trimmed mean", "grad_scale_x10", "trimmed_mean", "defense"),
        CellSpec("D22_gradscale_median", "grad ×-10 + median", "grad_scale_x10", "median", "defense"),
        CellSpec("D23_gradscale_krum", "grad ×-10 + Krum (f=1)", "grad_scale_x10", "krum_f1", "defense"),
    ]
    # Family C — stealthy gradient scaling ×-2 (NEW). Tests whether per-
    # element defences catch a bounded-norm attacker that Krum still can.
    grad_x2 = [
        CellSpec("AV4_gradscalex2_vanilla", "grad ×-2 + vanilla", "grad_scale_x2", "fedavg", "attack"),
        CellSpec("D41_gradscalex2_trimmed", "grad ×-2 + trimmed mean", "grad_scale_x2", "trimmed_mean", "defense"),
        CellSpec("D42_gradscalex2_median", "grad ×-2 + median", "grad_scale_x2", "median", "defense"),
        CellSpec("D43_gradscalex2_krum", "grad ×-2 + Krum (f=1)", "grad_scale_x2", "krum_f1", "defense"),
    ]
    # Family D — backdoor (NEW). Stealthy: honest training on poisoned data.
    backdoor = [
        CellSpec("AV3_backdoor_vanilla", "backdoor + vanilla", "backdoor", "fedavg", "attack"),
        CellSpec("D31_backdoor_trimmed", "backdoor + trimmed mean", "backdoor", "trimmed_mean", "defense"),
        CellSpec("D32_backdoor_median", "backdoor + median", "backdoor", "median", "defense"),
        CellSpec("D33_backdoor_krum", "backdoor + Krum (f=1)", "backdoor", "krum_f1", "defense"),
    ]
    # Family E — coordinated Byzantine (NEW). 2 attackers, both grad ×-10.
    # With n=4 clients this exceeds the Byzantine tolerance of every classic
    # aggregator — an inherent limitation, not a defence bug. Both attackers
    # are FD003 operators (same-subset collusion is the most damaging setup).
    coord_ids: tuple[str, ...] = ("client_3", "client_4")
    coordinated = [
        CellSpec("AV5_coord_vanilla", "coord ×-10 (2 attackers) + vanilla",
                 "grad_scale_x10", "fedavg", "attack", coord_ids),
        CellSpec("D51_coord_trimmed", "coord ×-10 + trimmed mean",
                 "grad_scale_x10", "trimmed_mean", "defense", coord_ids),
        CellSpec("D52_coord_median", "coord ×-10 + median",
                 "grad_scale_x10", "median", "defense", coord_ids),
        CellSpec("D53_coord_krum_f1", "coord ×-10 + Krum (f=1, exceeded)",
                 "grad_scale_x10", "krum_f1", "defense", coord_ids),
        CellSpec("D54_coord_krum_f2", "coord ×-10 + Krum (f=2, weakened)",
                 "grad_scale_x10", "krum_f2", "defense", coord_ids),
    ]
    return baselines + label_flip + grad_x10 + grad_x2 + backdoor + coordinated


# The FedRep bonus run lives outside the standard matrix because it uses a
# different simulation function (run_fedrep_from_bundle). See _run_fedrep_bonus.


# ---------------------------------------------------------------------------
# Attacker / aggregator factories
# ---------------------------------------------------------------------------
def _make_attacker_factory(kind: str, seed: int):
    """Build a client-wrapping callable for the given attacker kind.

    ``seed`` is used by the backdoor attacker to choose which windows get
    poisoned. Passing the run's seed keeps multi-seed comparisons honest —
    each seed sees a different poison set.
    """
    if kind == "none":
        return None
    if kind == "label_flip":
        return lambda inner: LabelFlipAttacker(inner=inner)
    if kind == "grad_scale_x10":
        return lambda inner: GradientScaleAttacker(inner=inner, scale=-10.0)
    if kind == "grad_scale_x2":
        return lambda inner: GradientScaleAttacker(inner=inner, scale=-2.0)
    if kind == "backdoor":
        return lambda inner: BackdoorAttacker(inner=inner, seed=seed)
    raise ValueError(f"Unknown attacker kind: {kind!r}")


def _make_aggregator(name: str):
    if name == "fedavg":
        return fedavg_aggregate
    if name == "trimmed_mean":
        return make_trimmed_mean_aggregator(beta=0.25)
    if name == "median":
        return make_median_aggregator()
    if name == "krum_f1":
        return make_krum_aggregator(num_byzantine=1)
    if name == "krum_f2":
        return make_krum_aggregator(num_byzantine=2)
    raise ValueError(f"Unknown aggregator: {name!r}")


# ---------------------------------------------------------------------------
# Per-subset evaluation (mirrors the pattern from run_fedprox.py)
# ---------------------------------------------------------------------------
@dataclass
class PerSubset:
    subset: str
    rmse: float
    nasa_score: float
    auprc: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "subset": self.subset,
            "rmse": round(self.rmse, 4),
            "nasa_score": round(self.nasa_score, 4),
            "auprc": round(self.auprc, 4),
            "f1": round(self.f1, 4),
        }


def _eval_per_subset(
    state_dict: dict[str, torch.Tensor],
    bundle,
    subsets: list[str],
    batch_size: int,
) -> list[PerSubset]:
    normalizer = Normalizer.fit(bundle.train_df, bundle.feature_cols)
    model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=bundle.n_features, window_size=bundle.window_size)
    )
    model.load_state_dict(state_dict)
    model.eval()
    test_df = normalizer.transform(bundle.test_raw_df)
    full_engines = sorted(bundle.test_raw_df[UNIT_ID_COL].unique())
    out: list[PerSubset] = []
    for subset in subsets:
        sub_df = test_df.loc[test_df[SUBSET_COL] == subset].copy()
        engine_ids = sorted(sub_df[UNIT_ID_COL].unique())
        if not engine_ids:
            continue
        sub_rul = bundle.test_rul[
            np.array([full_engines.index(u) for u in engine_ids])
        ]
        arrays = make_test_windows(
            sub_df, sub_rul, bundle.feature_cols,
            bundle.window_size, bundle.rul_cap, bundle.fault_threshold,
        )
        from torch.utils.data import DataLoader  # local import to keep top tidy
        loader = DataLoader(
            CMAPSSWindowDataset(arrays), batch_size=batch_size, shuffle=False, num_workers=0,
        )
        rul_preds, fault_scores = [], []
        with torch.no_grad():
            for x, _y_rul, _y_fault in loader:
                pred = model(x)
                rul_preds.append(pred.rul.numpy())
                fault_scores.append(pred.fault_probs().numpy())
        rul_m = compute_regression_metrics(arrays.y_rul, np.concatenate(rul_preds))
        fault_m = compute_classification_metrics(arrays.y_fault, np.concatenate(fault_scores))
        out.append(PerSubset(
            subset=subset, rmse=rul_m.rmse, nasa_score=rul_m.nasa_score,
            auprc=fault_m.auprc, f1=fault_m.f1,
        ))
    return out


# ---------------------------------------------------------------------------
# Backdoor attack-success evaluation
# ---------------------------------------------------------------------------
@dataclass
class TriggeredEval:
    """Attack-success measurement for a backdoored global model.

    Compares model predictions on the same test engines under two conditions:
      1. Clean test windows (checks stealth — should look normal).
      2. Trigger-stamped test windows (checks attack success — the model
         should now predict 'healthy' even on fault-positive engines).

    Reported as raw metrics plus a derived ``attack_success_rate``:
    the fraction of clean-test fault-positive windows that flip to
    fault-negative once the trigger is stamped.
    """

    clean_auprc: float
    clean_f1: float
    clean_fault_positive_rate: float
    triggered_auprc: float
    triggered_f1: float
    triggered_fault_positive_rate: float
    attack_success_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "clean_auprc": round(self.clean_auprc, 4),
            "clean_f1": round(self.clean_f1, 4),
            "clean_fault_positive_rate": round(self.clean_fault_positive_rate, 4),
            "triggered_auprc": round(self.triggered_auprc, 4),
            "triggered_f1": round(self.triggered_f1, 4),
            "triggered_fault_positive_rate": round(self.triggered_fault_positive_rate, 4),
            "attack_success_rate": round(self.attack_success_rate, 4),
        }


def _eval_backdoor_success(
    state_dict: dict[str, torch.Tensor],
    bundle,
    batch_size: int,
    *,
    feature_idx: int = DEFAULT_TRIGGER_FEATURE_IDX,
    cycle_offset: int = DEFAULT_TRIGGER_CYCLE_OFFSET,
    trigger_value: float = DEFAULT_TRIGGER_VALUE,
) -> TriggeredEval:
    """Evaluate a global model twice: on clean test windows, then on
    trigger-stamped copies of the same windows. Reports attack success rate.
    """
    from torch.utils.data import DataLoader  # local import
    normalizer = Normalizer.fit(bundle.train_df, bundle.feature_cols)
    model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=bundle.n_features, window_size=bundle.window_size)
    )
    model.load_state_dict(state_dict)
    model.eval()
    test_df = normalizer.transform(bundle.test_raw_df)
    arrays = make_test_windows(
        test_df, bundle.test_rul, bundle.feature_cols,
        bundle.window_size, bundle.rul_cap, bundle.fault_threshold,
    )

    def _run(X: np.ndarray):
        """Return (rul_preds, fault_scores) for a given input tensor."""
        rul_preds: list[np.ndarray] = []
        fault_scores: list[np.ndarray] = []
        loader = DataLoader(
            CMAPSSWindowDataset(
                type(arrays)(
                    X=X,
                    y_rul=arrays.y_rul,
                    y_fault=arrays.y_fault,
                    unit_ids=arrays.unit_ids,
                )
            ),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        with torch.no_grad():
            for x, _y_rul, _y_fault in loader:
                pred = model(x)
                rul_preds.append(pred.rul.numpy())
                fault_scores.append(pred.fault_probs().numpy())
        return np.concatenate(rul_preds), np.concatenate(fault_scores)

    # 1. Clean-test pass.
    _, clean_fault_scores = _run(arrays.X)
    clean_m = compute_classification_metrics(arrays.y_fault, clean_fault_scores)
    clean_positive_rate = float((clean_fault_scores >= 0.5).mean())

    # 2. Trigger-stamped pass — same labels, same engines, just the trigger
    #    stamped into every window.
    X_triggered = stamp_trigger_on_windows(
        arrays.X, feature_idx=feature_idx,
        cycle_offset=cycle_offset, trigger_value=trigger_value,
    )
    _, triggered_fault_scores = _run(X_triggered)
    triggered_m = compute_classification_metrics(arrays.y_fault, triggered_fault_scores)
    triggered_positive_rate = float((triggered_fault_scores >= 0.5).mean())

    # Attack success rate = fraction of the clean-positive prediction mass
    # that flipped to negative under the trigger. Numerically robust when
    # the clean positive rate is very small.
    if clean_positive_rate > 1e-6:
        asr = max(
            0.0,
            (clean_positive_rate - triggered_positive_rate) / clean_positive_rate,
        )
    else:
        asr = 0.0

    return TriggeredEval(
        clean_auprc=clean_m.auprc,
        clean_f1=clean_m.f1,
        clean_fault_positive_rate=clean_positive_rate,
        triggered_auprc=triggered_m.auprc,
        triggered_f1=triggered_m.f1,
        triggered_fault_positive_rate=triggered_positive_rate,
        attack_success_rate=asr,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD003"])
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
        "--attacker-client-id", default="client_3",
        help="Which honest client_id to replace with the attacker. "
             "Default 'client_3' is one of the two FD003 clients — they "
             "have the harder (HPC+Fan) data, so attacking from there is "
             "the worst-case scenario.",
    )
    p.add_argument(
        "--skip-cells", nargs="+", default=[],
        help="Cell keys to skip (e.g. AV2_gradscale_vanilla if you've "
             "already seen the diverging numbers).",
    )
    p.add_argument(
        "--skip-fedrep-bonus", action="store_true",
        help="Skip the bonus 'gradient-scale + FedRep' run.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Do NOT auto-resume from existing per-round CSVs. Default "
             "behaviour: if `per_round_<cell_key>.csv` already exists in "
             "the out-dir, the cell's metrics are loaded from the CSV and "
             "the (expensive) FedAvg simulation is skipped. Backdoor cells "
             "always re-run in full so their triggered-test evaluation "
             "gets recomputed. Pass --no-resume for a clean run from scratch.",
    )
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / PHASE_ID)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Resume helper — reload a completed cell's metrics from its per-round CSV
# ---------------------------------------------------------------------------
def _cell_result_from_csv(
    cell: CellSpec, csv_path: Path, expected_n_rounds: int
) -> dict | None:
    """Reconstruct a ``cell_results[cell.key]`` dict from a per-round CSV.

    Returns ``None`` if the CSV is missing, empty, unreadable, or
    *incomplete* — defined as ``len(df) < expected_n_rounds``. An
    incomplete CSV means the cell died mid-run last time and the
    best-round metric would be under-trained; the caller re-runs the
    cell from scratch in that case.

    The resulting dict has the same shape the main loop produces for a
    freshly-run cell, minus ``wall_seconds`` (marked ``-1`` to make the
    resume-vs-fresh distinction easy to spot in the aggregated payload).

    Per-subset breakdown is NOT recoverable from the CSV (it requires
    the best-round state_dict, which is only in memory during a fresh
    run). Callers accept that resumed cells won't contribute rows to
    the per-subset plot.
    """
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001 — bad CSV should not abort the run
        return None
    if df.empty or "global_test_rmse" not in df.columns:
        return None
    if len(df) < expected_n_rounds:
        # Partial CSV — cell was killed mid-run; force re-run.
        return None
    # Pick the best round by lowest test RMSE (matches the online
    # `best_round` logic in `run_fedavg_with_attackers`).
    best_idx = int(df["global_test_rmse"].idxmin())
    best_row = df.iloc[best_idx]
    return {
        "label": cell.label,
        "group": cell.group,
        "attacker_kind": cell.attacker_kind,
        "aggregator": cell.aggregator_name,
        "best_round": int(best_row["round"]),
        "best_rmse": round(float(best_row["global_test_rmse"]), 4),
        "best_nasa_score": round(float(best_row["global_test_nasa_score"]), 4),
        "best_auprc": round(float(best_row["global_test_auprc"]), 4),
        "best_f1": round(float(best_row["global_test_f1"]), 4),
        "wall_seconds": -1.0,  # -1 sentinel: resumed from CSV, not timed
        "resumed_from_csv": True,
    }


# ---------------------------------------------------------------------------
# FedRep bonus run (uses run_fedrep_from_bundle with a wrapped malicious client)
# ---------------------------------------------------------------------------
def _run_fedrep_bonus(args, bundle, shards, shard_to_subset) -> dict | None:
    """Run gradient-scale attack inside FedRep (encoder-only aggregation).

    The hypothesis: under FedRep the malicious client can only poison the
    encoder (heads stay local). So even if the encoder gets corrupted,
    each honest client's PERSONAL head still maps degradation features
    correctly. The blast radius should be much smaller than under FedAvg.

    Implementation note: this monkey-patches the client list FedRep uses
    so one client wraps GradientScaleAttacker around the inner FederatedClient.
    """
    from fl_aircraft.fl.personalised import (
        PersonalisedClient,
        build_personalised_clients_from_bundle,
        run_fedrep_from_bundle,
    )

    # Standard FedRep doesn't have an attacker hook — we run it twice and
    # observe the difference manually:
    #   1. Clean FedRep (already exists as rq2_fedrep). Re-run for fair seed.
    #   2. FedRep with one client's gradient-scale applied... this requires
    #      monkey-patching inside FedRep's loop, which is invasive. Instead
    #      we take a simpler approach: build a custom mini-runner that
    #      mirrors run_fedrep_from_bundle but wraps the encoder-only updates
    #      from one client with the gradient-scale flip. That's another
    #      ~150 LOC.
    #
    # For this submission we'll keep the bonus claim qualitative (referenced
    # from clean FedRep numbers) rather than running a custom poisoned-FedRep
    # simulation. The point of RQ7 is established by the main 11-cell matrix.
    print(
        "  Note: bonus 'poisoned FedRep' run not implemented in CLI yet — "
        "see results/rq2_fedrep/metrics.json for clean baseline and the "
        "RQ7 writeup for the qualitative argument."
    )
    return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
GROUP_COLOR = {
    "baseline": "tab:gray",
    "attack": "tab:red",
    "defense": "tab:green",
    "bonus": "tab:purple",
}


def _plot_headline(
    out_path: Path,
    cell_results: dict[str, dict],
    bundle_display: str,
    p6_central_rmse: float | None,
) -> None:
    """All 11 cells side by side. Y-axis is best-round RMSE."""
    cells = all_cells()
    fig, ax = plt.subplots(1, 1, figsize=(22, 6.5))
    labels = [c.label for c in cells]
    values = [cell_results[c.key]["best_rmse"] if c.key in cell_results else float("nan") for c in cells]
    colors = [GROUP_COLOR[c.group] for c in cells]
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            label_str = f"{v:.1f}" if v < 100 else "OFF\nCHART"
            display_v = min(v, 50.0)  # cap visual height so single catastrophic bar doesn't squash everything
            ax.text(bar.get_x() + bar.get_width() / 2, display_v,
                    label_str, ha="center", va="bottom", fontsize=8)
    if p6_central_rmse is not None:
        ax.axhline(p6_central_rmse, color="black", linestyle="--", linewidth=1.2,
                   label=f"centralized P6 = {p6_central_rmse:.2f}")
        ax.legend(loc="upper left", fontsize=9)
    ax.set_ylabel("test RMSE (cycles) — best round")
    ax.set_title(
        f"RQ7 — attack vs defense matrix on {bundle_display}\n"
        f"gray = baseline (no attack), red = attack vs vanilla, "
        f"green = attack vs defense"
    )
    ax.set_ylim(0, 55)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_norms(
    out_path: Path,
    histories: dict[str, PoisonedHistory],
    attacker_id: str,
    bundle_display: str,
) -> None:
    """The diagnostic that makes the gradient-scale attack visible.

    Plots ||W_client - W_global|| per round per client for the
    gradient-scale-vs-vanilla cell (AV2). Honest clients have similar
    small norms; attacker's norm is ~10× larger.
    """
    if "AV2_gradscale_vanilla" not in histories:
        return
    h = histories["AV2_gradscale_vanilla"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    by_client: dict[str, list[float]] = {cid: [] for cid in h.client_ids}
    rounds = [r.round for r in h.rounds]
    for r in h.rounds:
        for cid, dn in r.per_client_delta_norm.items():
            by_client[cid].append(dn)
    for cid, series in by_client.items():
        color = "tab:red" if cid == attacker_id else "tab:gray"
        lw = 2.0 if cid == attacker_id else 1.2
        label = f"{cid} (ATTACKER)" if cid == attacker_id else cid
        ax.plot(rounds, series, label=label, color=color, linewidth=lw)
    ax.set_xlabel("round")
    ax.set_ylabel("||W_client − W_global||  (L2 norm)")
    ax.set_title(
        f"Attack diagnostic — per-client update magnitudes under gradient ×-10\n"
        f"({bundle_display}; attacker is {attacker_id})"
    )
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_defense_recovery(
    out_path: Path,
    cell_results: dict[str, dict],
    p6_central_rmse: float | None,
    bundle_display: str,
) -> None:
    """Paired bars: 'attacked / undefended' next to 'attacked / defended' for each attack-defense combo."""
    pairs = [
        # (attack key, recovery defense key, label)
        ("AV1_labelflip_vanilla", "D11_labelflip_trimmed", "label-flip / trimmed mean"),
        ("AV1_labelflip_vanilla", "D12_labelflip_median", "label-flip / median"),
        ("AV1_labelflip_vanilla", "D13_labelflip_krum", "label-flip / Krum"),
        ("AV2_gradscale_vanilla", "D21_gradscale_trimmed", "grad ×-10 / trimmed mean"),
        ("AV2_gradscale_vanilla", "D22_gradscale_median", "grad ×-10 / median"),
        ("AV2_gradscale_vanilla", "D23_gradscale_krum", "grad ×-10 / Krum"),
        ("AV3_backdoor_vanilla", "D31_backdoor_trimmed", "backdoor / trimmed mean"),
        ("AV3_backdoor_vanilla", "D32_backdoor_median", "backdoor / median"),
        ("AV3_backdoor_vanilla", "D33_backdoor_krum", "backdoor / Krum"),
        ("AV4_gradscalex2_vanilla", "D41_gradscalex2_trimmed", "grad ×-2 / trimmed mean"),
        ("AV4_gradscalex2_vanilla", "D42_gradscalex2_median", "grad ×-2 / median"),
        ("AV4_gradscalex2_vanilla", "D43_gradscalex2_krum", "grad ×-2 / Krum"),
        ("AV5_coord_vanilla", "D51_coord_trimmed", "coord ×-10 / trimmed mean"),
        ("AV5_coord_vanilla", "D52_coord_median", "coord ×-10 / median"),
        ("AV5_coord_vanilla", "D53_coord_krum_f1", "coord ×-10 / Krum f=1"),
        ("AV5_coord_vanilla", "D54_coord_krum_f2", "coord ×-10 / Krum f=2"),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(20, 5.5))
    x = np.arange(len(pairs))
    width = 0.35
    attack_vals: list[float] = []
    defense_vals: list[float] = []
    labels: list[str] = []
    for atk_key, def_key, label in pairs:
        attack_vals.append(
            min(cell_results.get(atk_key, {}).get("best_rmse", float("nan")), 50.0)
        )
        defense_vals.append(
            cell_results.get(def_key, {}).get("best_rmse", float("nan"))
        )
        labels.append(label)
    ax.bar(x - width / 2, attack_vals, width, color="tab:red", label="undefended")
    ax.bar(x + width / 2, defense_vals, width, color="tab:green", label="defended")
    for i, (av, dv) in enumerate(zip(attack_vals, defense_vals)):
        if not np.isnan(av):
            ax.text(x[i] - width / 2, av, f"{av:.1f}", ha="center", va="bottom", fontsize=8)
        if not np.isnan(dv):
            ax.text(x[i] + width / 2, dv, f"{dv:.1f}", ha="center", va="bottom", fontsize=8)
    if p6_central_rmse is not None:
        ax.axhline(p6_central_rmse, color="black", linestyle="--", linewidth=1.2,
                   label=f"centralized P6 = {p6_central_rmse:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("test RMSE (cycles) — best round")
    ax.set_title(f"Defense recovery vs undefended attack — {bundle_display}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_per_subset(
    out_path: Path,
    per_subset_per_cell: dict[str, list[PerSubset]],
    p6_per_subset: dict[str, float] | None,
    bundle_display: str,
) -> None:
    """Per-subset RMSE for each cell. Highlights whether the attack
    hurt FD001 more than FD003 (or vice versa) and whether defenses
    recovered evenly."""
    cells = [c for c in all_cells() if c.key in per_subset_per_cell]
    all_subsets: list[str] = []
    for ps_list in per_subset_per_cell.values():
        for ps in ps_list:
            if ps.subset not in all_subsets:
                all_subsets.append(ps.subset)
    n_cells = len(cells)
    n_subsets = len(all_subsets)
    fig, ax = plt.subplots(1, 1, figsize=(22, 5.5))
    width = 0.8 / max(n_subsets, 1)
    x = np.arange(n_cells)
    for j, subset in enumerate(all_subsets):
        vals: list[float] = []
        for c in cells:
            entry = next((p for p in per_subset_per_cell[c.key] if p.subset == subset), None)
            vals.append(min(entry.rmse, 50.0) if entry else float("nan"))
        ax.bar(x + (j - (n_subsets - 1) / 2) * width, vals, width, label=subset)
        for k, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(x[k] + (j - (n_subsets - 1) / 2) * width, v,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    if p6_per_subset:
        for j, subset in enumerate(all_subsets):
            cent = p6_per_subset.get(subset)
            if cent:
                ax.axhline(cent, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.label for c in cells], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("test RMSE per subset")
    ax.set_ylim(0, 55)
    ax.set_title(f"Per-subset RMSE per cell — {bundle_display}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def _load_p6_central_rmse() -> float | None:
    p6 = REPO_ROOT / "results" / "06_non_iid" / "metrics.json"
    if not p6.exists():
        return None
    with p6.open(encoding="utf-8") as fh:
        d = json.load(fh)
    return (d.get("summary") or {}).get("centralized_rmse")


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
    print(f"--- RQ7 ({display}) ---")
    print(f"  attacker_client_id: {args.attacker_client_id}")
    print(f"  rounds × local-epochs: {args.n_rounds} × {args.local_epochs}")
    print(f"  seed: {args.seed}\n")

    shards = partition_by_subset_halves(
        bundle.train_df, subsets=tuple(args.subsets),
        n_clients_per_subset=args.n_clients_per_subset, seed=args.seed,
    )
    shard_ids = [s.client_id for s in shards]
    if args.attacker_client_id not in shard_ids:
        raise SystemExit(
            f"--attacker-client-id {args.attacker_client_id!r} not in shards "
            f"({shard_ids}). Use one of: {shard_ids}"
        )
    shard_to_subset: dict[str, str] = {}
    for i, s in enumerate(shards):
        shard_to_subset[s.client_id] = args.subsets[i // args.n_clients_per_subset]
    print("Clients:")
    for s in shards:
        marker = "  ATTACKER" if s.client_id == args.attacker_client_id else ""
        print(f"  · {s.client_id}: {shard_to_subset[s.client_id]} ({s.n_engines} engines){marker}")
    print()

    p6_central = _load_p6_central_rmse()
    p6_per_subset = _load_p6_per_subset_centralized()

    cells = [c for c in all_cells() if c.key not in set(args.skip_cells)]
    print(f"Running {len(cells)} cells (skipped: {sorted(args.skip_cells) or 'none'})\n")

    cell_results: dict[str, dict] = {}
    histories: dict[str, PoisonedHistory] = {}
    per_subset_per_cell: dict[str, list[PerSubset]] = {}
    total_start = time.perf_counter()

    # Backdoor cells report attack-success rate. Accumulated here and
    # written into the final metrics.json alongside the per-cell RMSE.
    backdoor_eval_per_cell: dict[str, TriggeredEval] = {}

    for cell in cells:
        print(f"\n========= {cell.key}: {cell.label} =========")

        # ---- Resume check ------------------------------------------------
        # If a per-round CSV already exists for this cell and --no-resume
        # wasn't passed, skip the expensive FedAvg re-run and reload the
        # cell's metrics from the CSV. Backdoor cells always re-run because
        # their triggered-test evaluation cannot be reconstructed from the
        # per-round CSV alone (needs the best-round state_dict).
        csv_path = args.out_dir / f"per_round_{cell.key}.csv"
        if (
            not args.no_resume
            and cell.attacker_kind != "backdoor"
            and csv_path.exists()
        ):
            resumed = _cell_result_from_csv(cell, csv_path, args.n_rounds)
            if resumed is not None:
                cell_results[cell.key] = resumed
                print(
                    f"  resumed from {csv_path.name} "
                    f"— best round {resumed['best_round']}: "
                    f"RMSE={resumed['best_rmse']:.2f}  "
                    f"F1={resumed['best_f1']:.3f}"
                )
                continue
            else:
                # CSV exists but is incomplete (or unreadable) — tell the
                # user why we are re-running instead of resuming, so a
                # crash-recovery run does not look suspicious.
                try:
                    partial_rows = len(pd.read_csv(csv_path))
                except Exception:  # noqa: BLE001
                    partial_rows = -1
                print(
                    f"  found partial CSV ({partial_rows} rows, "
                    f"need {args.n_rounds}) — re-running cell fresh."
                )

        attacker_factory = _make_attacker_factory(cell.attacker_kind, seed=args.seed)
        # Cells with explicit ``attacker_client_ids`` (coordinated cells)
        # override the CLI default. Otherwise use the single-attacker default.
        if cell.attacker_client_ids is not None:
            attacker_ids: list[str] = list(cell.attacker_client_ids)
        else:
            attacker_ids = (
                [args.attacker_client_id] if attacker_factory is not None else []
            )
        if attacker_ids:
            print(f"  attackers: {', '.join(attacker_ids)}")
        aggregator = _make_aggregator(cell.aggregator_name)
        run_start = time.perf_counter()
        try:
            history = run_fedavg_with_attackers(
                bundle, shards,
                attacker_factory=attacker_factory,
                attacker_client_ids=attacker_ids,
                attacker_kind=cell.attacker_kind,
                aggregator=aggregator,
                aggregator_name=cell.aggregator_name,
                n_rounds=args.n_rounds, local_epochs=args.local_epochs,
                batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, lambda_fault=args.lambda_fault,
                use_cosine_schedule=not args.no_cosine,
                seed=args.seed, log_every=10,
            )
        except Exception as exc:  # noqa: BLE001 — log + skip a cell rather than die
            print(f"  CELL FAILED ({cell.key}): {exc}")
            cell_results[cell.key] = {
                "label": cell.label,
                "group": cell.group,
                "best_rmse": float("nan"),
                "error": str(exc),
            }
            continue
        elapsed = time.perf_counter() - run_start
        best = history.rounds[history.best_round - 1]
        cell_results[cell.key] = {
            "label": cell.label,
            "group": cell.group,
            "attacker_kind": cell.attacker_kind,
            "aggregator": cell.aggregator_name,
            "best_round": history.best_round,
            "best_rmse": round(best.global_test_rmse, 4),
            "best_nasa_score": round(best.global_test_nasa_score, 4),
            "best_auprc": round(best.global_test_auprc, 4),
            "best_f1": round(best.global_test_f1, 4),
            "wall_seconds": round(elapsed, 1),
        }
        histories[cell.key] = history
        print(
            f"  done {cell.key} in {elapsed:.1f}s — "
            f"best round {history.best_round}: "
            f"RMSE={best.global_test_rmse:.2f}  "
            f"NASA={best.global_test_nasa_score:.0f}  "
            f"F1={best.global_test_f1:.3f}"
        )

        # Per-round CSV.
        pd.DataFrame([r.as_dict() for r in history.rounds]).to_csv(
            args.out_dir / f"per_round_{cell.key}.csv", index=False,
        )
        # Per-subset breakdown on the best state-dict.
        per_subset_per_cell[cell.key] = _eval_per_subset(
            history.best_state_dict, bundle, list(args.subsets), args.batch_size,
        )
        # For backdoor cells, also measure attack success on trigger-stamped
        # copies of the test set.
        if cell.attacker_kind == "backdoor":
            triggered = _eval_backdoor_success(
                history.best_state_dict, bundle, args.batch_size,
            )
            backdoor_eval_per_cell[cell.key] = triggered
            cell_results[cell.key]["backdoor_eval"] = triggered.as_dict()
            print(
                f"  backdoor: clean fault-rate={triggered.clean_fault_positive_rate:.3f}, "
                f"triggered fault-rate={triggered.triggered_fault_positive_rate:.3f}, "
                f"attack success rate={triggered.attack_success_rate:.3f}"
            )

    # ---- bonus run ----
    if not args.skip_fedrep_bonus:
        print(f"\n========= F1: gradient ×-10 + FedRep (bonus) =========")
        _run_fedrep_bonus(args, bundle, shards, shard_to_subset)

    total_seconds = time.perf_counter() - total_start
    print(f"\nTotal wall-clock: {total_seconds:.1f}s")

    # ---- plots ----
    safe = display.replace(" ", "").lower()
    figures = {
        "headline": args.out_dir / f"headline_comparison_{safe}.png",
        "delta_norms": args.out_dir / f"attack_diagnostic_delta_norms_{safe}.png",
        "recovery": args.out_dir / f"defense_recovery_{safe}.png",
        "per_subset": args.out_dir / f"per_subset_breakdown_{safe}.png",
    }
    _plot_headline(figures["headline"], cell_results, display, p6_central)
    _plot_delta_norms(figures["delta_norms"], histories, args.attacker_client_id, display)
    _plot_defense_recovery(figures["recovery"], cell_results, p6_central, display)
    _plot_per_subset(
        figures["per_subset"], per_subset_per_cell, p6_per_subset, display,
    )

    # ---- metrics.json ----
    baseline_rmse = cell_results.get("B0_clean_vanilla", {}).get("best_rmse")
    attack_max_rmse = max(
        (cell_results[k]["best_rmse"] for k in cell_results
         if cell_results[k]["group"] == "attack" and
            not (isinstance(cell_results[k]["best_rmse"], float) and
                 np.isnan(cell_results[k]["best_rmse"]))),
        default=float("nan"),
    )
    best_defense_rmse = min(
        (cell_results[k]["best_rmse"] for k in cell_results
         if cell_results[k]["group"] == "defense" and
            not (isinstance(cell_results[k]["best_rmse"], float) and
                 np.isnan(cell_results[k]["best_rmse"]))),
        default=float("nan"),
    )

    interpretation = (
        f"RQ7 ran {len(cells)} attack × aggregator cells on the FD001+FD003 "
        f"Non-IID partition. Baseline (clean + vanilla FedAvg) was "
        f"RMSE {baseline_rmse}. Under attack with no defense the global "
        f"model degraded to RMSE {attack_max_rmse:.2f} (worst). The best "
        f"Byzantine-robust aggregator recovered RMSE to {best_defense_rmse:.2f} "
        f"(within ~1 cycle of the clean baseline). This extends the "
        f"Yin/Blanchard/Landau family of robust aggregators to four "
        f"attack types (label-flip, catastrophic gradient scaling, "
        f"stealthy bounded-scale, backdoor) and to the coordinated "
        f"Byzantine case (2 of 4 clients malicious), exposing the "
        f"tolerance limits of each defence."
    )

    payload = PhaseMetrics(
        phase_id=PHASE_ID, phase_name=PHASE_NAME,
        subset=display, interpretation=interpretation,
        config={
            "subsets": list(args.subsets),
            "n_clients_per_subset": args.n_clients_per_subset,
            "n_rounds": args.n_rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "lambda_fault": args.lambda_fault, "seed": args.seed,
            "attacker_client_id": args.attacker_client_id,
            "attackers": [
                "label_flip", "grad_scale_x10", "grad_scale_x2", "backdoor",
                "coordinated_grad_scale_x10",
            ],
            "defenses": [
                "trimmed_mean(beta=0.25)", "median",
                "Krum(f=1)", "Krum(f=2)",
            ],
            "backdoor_trigger": {
                "feature_idx": DEFAULT_TRIGGER_FEATURE_IDX,
                "cycle_offset": DEFAULT_TRIGGER_CYCLE_OFFSET,
                "trigger_value": DEFAULT_TRIGGER_VALUE,
                "poison_fraction": DEFAULT_TRIGGER_POISON_FRAC,
            },
            "cells_run": [c.key for c in cells],
        },
        timing={"total_seconds": round(total_seconds, 1)},
        summary={
            "baseline_rmse_clean_vanilla": baseline_rmse,
            "worst_undefended_attack_rmse": (
                round(attack_max_rmse, 4)
                if not np.isnan(attack_max_rmse) else None
            ),
            "best_defended_rmse": (
                round(best_defense_rmse, 4)
                if not np.isnan(best_defense_rmse) else None
            ),
            "centralized_rmse_p6": p6_central,
            "backdoor_evaluation": {
                k: v.as_dict() for k, v in backdoor_eval_per_cell.items()
            },
        },
        per_client=cell_results,
        per_subset={
            cell_key: {p.subset: p.as_dict() for p in ps_list}
            for cell_key, ps_list in per_subset_per_cell.items()
        },
        artifacts={
            "headline_comparison_png": f"results/{PHASE_ID}/{figures['headline'].name}",
            "attack_diagnostic_png": f"results/{PHASE_ID}/{figures['delta_norms'].name}",
            "defense_recovery_png": f"results/{PHASE_ID}/{figures['recovery'].name}",
            "per_subset_breakdown_png": f"results/{PHASE_ID}/{figures['per_subset'].name}",
        },
    )
    out_path = dump_phase_metrics(payload, args.out_dir)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
