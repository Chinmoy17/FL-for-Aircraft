"""Stacked-defense bridge experiment: FedRep + Krum under sensor-value backdoor.

Question
--------
Does composing FedRep (Axis-1 winner --- personalization) with Krum-f=1
(Axis-2 winner --- Byzantine-robust aggregation) end-to-end deliver the
compositional defense the paper motivates?

Setup
-----
Identical to ``scripts/run_rq2_fedrep_under_backdoor.py`` (same FD001+FD003
partition, same trigger on sensor s_3 / T30, same client_3 attacker,
same 5-seed harness) with ONE change: the shared-encoder aggregation
step is Krum(f=1) instead of FedAvg. Concretely, we pass
``aggregator=make_krum_aggregator(num_byzantine=1)`` to
``run_fedrep_from_bundle``.

Per-client heads are still trained locally (unchanged from FedRep).
Only the encoder aggregation rule swaps. This matches the paper's
argument in Section 9.4 that Krum is drop-in compatible with FedRep
because Krum operates on the shared-encoder deltas in exactly the
same way it operates on FedAvg's whole-model deltas.

Expected outcome
----------------
If FedRep + Krum is a real compositional defense, the honest-mean ASR
should drop close to the standalone Krum baseline (0.064 +/- 0.100 from
Table 11) rather than staying at the FedRep-alone level
(0.633 +/- 0.320 from Table 12). If it's between the two, the
composition helps but is not additive.

Outputs (``results/rq_bridge_stacked/`` by default, or ``--out-dir``)
--------------------------------------------------------------------
    metrics.json           per-client backdoor eval + honest-avg ASR
    per_round.csv          macro RMSE trajectory across rounds

Run from the repo root inside the .venv::

    .\\.venv\\Scripts\\python.exe scripts\\run_rq2_fedrep_krum_bridge.py --seed 42

To run the full 5-seed matrix (~2 hours on CPU laptop)::

    for /L %s in (42,1,46) do (
        python scripts\\run_rq2_fedrep_krum_bridge.py --seed %s \\
            --out-dir results\\rq_bridge_stacked_seeds\\seed_%s
    )
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
    load_multi_subset_bundle,
    make_test_windows,
    partition_by_subset_halves,
)
from fl_aircraft.eval import compute_classification_metrics  # noqa: E402
from fl_aircraft.fl import (  # noqa: E402
    make_backdoor_poisoned_loader,
    make_krum_aggregator,
    run_fedrep_from_bundle,
    stamp_trigger_on_windows,
)
from fl_aircraft.fl.poisoning import (  # noqa: E402
    DEFAULT_TRIGGER_CYCLE_OFFSET,
    DEFAULT_TRIGGER_FEATURE_IDX,
    DEFAULT_TRIGGER_POISON_FRAC,
    DEFAULT_TRIGGER_VALUE,
)
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig  # noqa: E402

PHASE_ID = "rq_bridge_stacked"
PHASE_NAME = "Stacked bridge --- FedRep + Krum under backdoor (Axis 1 x Axis 2)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- FedRep base params (mirror scripts/run_rq2_fedrep_under_backdoor.py) ----
    p.add_argument("--subsets", nargs="+", default=["FD001", "FD003"])
    p.add_argument("--n-clients-per-subset", type=int, default=2)
    p.add_argument("--n-rounds", type=int, default=50)
    p.add_argument("--head-epochs", type=int, default=1)
    p.add_argument("--encoder-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-fault", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cosine", action="store_true")
    # ---- Krum aggregator params -----------------------------------------
    p.add_argument("--krum-f", type=int, default=1,
                   help="Krum Byzantine-tolerance parameter (default 1). "
                        "At N=4, only f=1 is well-defined "
                        "(n-f-2 >= 1 requires f <= 1).")
    # ---- Backdoor trigger params (mirror scripts/run_rq7.py defaults) ---
    p.add_argument("--attacker-client-id", type=str, default="client_3",
                   help="Client ID whose train_loader gets poisoned. "
                        "Default client_3 = first FD003 shard.")
    p.add_argument("--feature-idx", type=int, default=DEFAULT_TRIGGER_FEATURE_IDX)
    p.add_argument("--cycle-offset", type=int, default=DEFAULT_TRIGGER_CYCLE_OFFSET)
    p.add_argument("--trigger-value", type=float, default=DEFAULT_TRIGGER_VALUE)
    p.add_argument("--poison-frac", type=float, default=DEFAULT_TRIGGER_POISON_FRAC)
    # ---- Output ---------------------------------------------------------
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / PHASE_ID)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers (identical to run_rq2_fedrep_under_backdoor.py)
# ---------------------------------------------------------------------------
def _build_shard_to_subset(shards, subsets, n_per_subset) -> dict[str, str]:
    """First N shards -> subsets[0], next N -> subsets[1], etc."""
    if len(shards) != len(subsets) * n_per_subset:
        raise ValueError(
            f"shard count {len(shards)} != subsets x n_per_subset "
            f"({len(subsets)} x {n_per_subset})"
        )
    out: dict[str, str] = {}
    for i, shard in enumerate(shards):
        out[shard.client_id] = subsets[i // n_per_subset]
    return out


def eval_backdoor_per_client(
    state_dict: dict[str, torch.Tensor],
    bundle,
    batch_size: int,
    *,
    feature_idx: int,
    cycle_offset: int,
    trigger_value: float,
) -> dict[str, float]:
    """Evaluate one client's full state_dict on the POOLED test set.

    Identical to the eval used in ``run_rq2_fedrep_under_backdoor.py``.
    """
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

    def _run(X: np.ndarray) -> np.ndarray:
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
                fault_scores.append(pred.fault_probs().numpy())
        return np.concatenate(fault_scores)

    clean_scores = _run(arrays.X)
    clean_m = compute_classification_metrics(arrays.y_fault, clean_scores)
    clean_positive_rate = float((clean_scores >= 0.5).mean())

    X_triggered = stamp_trigger_on_windows(
        arrays.X, feature_idx=feature_idx,
        cycle_offset=cycle_offset, trigger_value=trigger_value,
    )
    triggered_scores = _run(X_triggered)
    triggered_m = compute_classification_metrics(arrays.y_fault, triggered_scores)
    triggered_positive_rate = float((triggered_scores >= 0.5).mean())

    if clean_positive_rate > 1e-6:
        asr = max(
            0.0,
            (clean_positive_rate - triggered_positive_rate) / clean_positive_rate,
        )
    else:
        asr = 0.0

    return {
        "clean_auprc": round(float(clean_m.auprc), 4),
        "clean_f1": round(float(clean_m.f1), 4),
        "clean_fault_positive_rate": round(clean_positive_rate, 4),
        "triggered_auprc": round(float(triggered_m.auprc), 4),
        "triggered_f1": round(float(triggered_m.f1), 4),
        "triggered_fault_positive_rate": round(triggered_positive_rate, 4),
        "attack_success_rate": round(asr, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- FedRep + Krum(f={args.krum_f}) bridge "
          f"({'+'.join(args.subsets)}) ---")
    print(f"  aggregator         : Krum(f={args.krum_f}) on shared-encoder deltas")
    print(f"  attacker_client_id : {args.attacker_client_id}")
    print(f"  trigger            : feature_idx={args.feature_idx}, "
          f"cycle_offset={args.cycle_offset}, value={args.trigger_value}, "
          f"poison_frac={args.poison_frac}")
    print(f"  rounds x (head+enc): {args.n_rounds} x "
          f"({args.head_epochs}+{args.encoder_epochs})")
    print(f"  seed               : {args.seed}\n")

    # ---- Load bundle + partition ---------------------------------------
    data_dir = REPO_ROOT / "Dataset" / "CMAPSS_NASA"
    bundle_config = MultiSubsetConfig(
        subsets=tuple(args.subsets),
        data_dir=data_dir,
    )
    bundle = load_multi_subset_bundle(bundle_config)
    shards = partition_by_subset_halves(
        bundle.train_df, subsets=tuple(args.subsets),
        n_clients_per_subset=args.n_clients_per_subset, seed=args.seed,
    )
    shard_to_subset = _build_shard_to_subset(
        shards, args.subsets, args.n_clients_per_subset,
    )

    print("Clients:")
    attacker_found = False
    for shard in shards:
        subset = shard_to_subset[shard.client_id]
        is_attacker = shard.client_id == args.attacker_client_id
        attacker_found = attacker_found or is_attacker
        marker = "  <- ATTACKER" if is_attacker else ""
        print(f"  - {shard.client_id}: {subset} "
              f"({shard.n_engines} engines){marker}")
    if not attacker_found:
        raise SystemExit(
            f"\nERROR: no client with id {args.attacker_client_id!r} found. "
            f"Available: {[s.client_id for s in shards]}"
        )
    print()

    # ---- Client hook: poison the attacker's train_loader ----------------
    def poison_client_hook(clients) -> None:
        n_poisoned_clients = 0
        for c in clients:
            if c.client_id == args.attacker_client_id:
                original = c.train_loader
                c.train_loader = make_backdoor_poisoned_loader(
                    original,
                    feature_idx=args.feature_idx,
                    cycle_offset=args.cycle_offset,
                    trigger_value=args.trigger_value,
                    poison_frac=args.poison_frac,
                    seed=args.seed,
                )
                n_windows = len(original.dataset)
                approx_n_poisoned = max(1, int(round(n_windows * args.poison_frac)))
                print(f"[hook] Poisoned {c.client_id}'s train_loader: "
                      f"~{approx_n_poisoned}/{n_windows} windows "
                      f"({args.poison_frac*100:.0f}%) carry the trigger.")
                n_poisoned_clients += 1
        if n_poisoned_clients != 1:
            raise SystemExit(
                f"[hook] Expected to poison exactly 1 client; "
                f"poisoned {n_poisoned_clients}."
            )
        print()

    # ---- Build the Krum aggregator (this is the ONLY change vs. the
    #      FedAvg bridge in scripts/run_rq2_fedrep_under_backdoor.py) ----
    aggregator = make_krum_aggregator(num_byzantine=args.krum_f)

    # ---- Run FedRep + Krum with the poison hook -------------------------
    start = time.perf_counter()
    history = run_fedrep_from_bundle(
        bundle=bundle,
        shards=shards,
        shard_to_subset=shard_to_subset,
        n_rounds=args.n_rounds,
        head_epochs=args.head_epochs,
        encoder_epochs=args.encoder_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_fault=args.lambda_fault,
        use_cosine_schedule=not args.no_cosine,
        seed=args.seed,
        log_every=10,
        client_hook=poison_client_hook,
        aggregator=aggregator,
    )
    total_seconds = time.perf_counter() - start

    print(f"\nWall clock            : {total_seconds:.1f}s "
          f"({total_seconds/args.n_rounds:.2f}s per round)")
    print(f"Best round            : {history.best_round} of {args.n_rounds}")
    print(f"Best macro test RMSE  : {history.best_macro_rmse:.4f}")
    print(f"Best macro NASA score : {history.best_macro_nasa_score:.4f}")

    # ---- Per-client triggered evaluation on POOLED test set -------------
    print("\n--- Triggered evaluation per client (pooled test set) ---")
    per_client_bd: dict[str, dict[str, float]] = {}
    for cid in sorted(history.best_state_dicts.keys()):
        sd = history.best_state_dicts[cid]
        bd = eval_backdoor_per_client(
            sd, bundle, args.batch_size,
            feature_idx=args.feature_idx,
            cycle_offset=args.cycle_offset,
            trigger_value=args.trigger_value,
        )
        per_client_bd[cid] = bd
        marker = " (attacker)" if cid == args.attacker_client_id else ""
        print(f"  {cid}{marker}: "
              f"clean_fpr={bd['clean_fault_positive_rate']:.3f}  "
              f"triggered_fpr={bd['triggered_fault_positive_rate']:.3f}  "
              f"ASR={bd['attack_success_rate']:.3f}  "
              f"(clean AUPRC={bd['clean_auprc']:.3f}, "
              f"trig AUPRC={bd['triggered_auprc']:.3f})")

    # ---- Aggregates: attacker vs honest-mean ASR ------------------------
    honest_ids = [c for c in per_client_bd if c != args.attacker_client_id]
    honest_asrs = [per_client_bd[c]["attack_success_rate"] for c in honest_ids]
    honest_mean_asr = float(np.mean(honest_asrs)) if honest_asrs else 0.0
    honest_max_asr = float(np.max(honest_asrs)) if honest_asrs else 0.0
    attacker_asr = per_client_bd[args.attacker_client_id]["attack_success_rate"]

    print(f"\nAttacker's own head ASR : {attacker_asr:.3f}")
    print(f"Honest clients' mean ASR: {honest_mean_asr:.3f} "
          f"(max {honest_max_asr:.3f}, n={len(honest_ids)})")
    print("Reference baselines (5-seed):")
    print("  Vanilla FedAvg + backdoor   : ASR = 0.949 +/- 0.079 (undefended)")
    print("  Krum + backdoor             : ASR = 0.064 +/- 0.100 (Krum alone)")
    print("  FedRep + backdoor (bridge)  : ASR = 0.633 +/- 0.320 (FedRep alone)")

    # ---- Write metrics.json --------------------------------------------
    metrics = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "config": {
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
            "attacker_client_id": args.attacker_client_id,
            "aggregator": f"krum_f{args.krum_f}",
            "krum_f": args.krum_f,
            "backdoor_trigger": {
                "feature_idx": args.feature_idx,
                "cycle_offset": args.cycle_offset,
                "trigger_value": args.trigger_value,
                "poison_fraction": args.poison_frac,
            },
        },
        "timing": {
            "total_seconds": round(total_seconds, 2),
            "seconds_per_round": round(total_seconds / args.n_rounds, 2),
        },
        "summary": {
            "best_round": history.best_round,
            "best_macro_rmse": round(float(history.best_macro_rmse), 4),
            "best_macro_nasa_score": round(float(history.best_macro_nasa_score), 4),
            "attacker_id": args.attacker_client_id,
            "attacker_asr": round(attacker_asr, 4),
            "honest_mean_asr": round(honest_mean_asr, 4),
            "honest_max_asr": round(honest_max_asr, 4),
            "n_honest_clients": len(honest_ids),
        },
        "per_client_backdoor_evaluation": per_client_bd,
        "per_client_clean_test_rmse_best_round": {
            m.client_id: round(float(m.rmse), 4)
            for m in history.rounds[history.best_round - 1].per_client_metrics
        },
        "per_client_clean_test_rmse_final_round": {
            m.client_id: round(float(m.rmse), 4)
            for m in history.rounds[-1].per_client_metrics
        },
        "reference_baselines": {
            "note": "5-seed aggregates from Table 11 (matrix) and Table 12 (bridge).",
            "AV3_backdoor_vanilla_asr_mean": 0.949,
            "AV3_backdoor_vanilla_asr_std": 0.079,
            "D33_backdoor_krum_asr_mean": 0.064,
            "D33_backdoor_krum_asr_std": 0.100,
            "FedRep_alone_bridge_honest_asr_mean": 0.633,
            "FedRep_alone_bridge_honest_asr_std": 0.320,
        },
    }
    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {metrics_path}")

    # ---- Per-round CSV (macro trajectory) -------------------------------
    rows = []
    for r in history.rounds:
        rows.append({
            "round": r.round,
            "lr": r.lr,
            "mean_client_loss_total": r.mean_client_loss_total,
            "mean_client_loss_rul": r.mean_client_loss_rul,
            "mean_client_loss_fault": r.mean_client_loss_fault,
            "macro_rmse": r.macro_rmse,
            "macro_nasa_score": r.macro_nasa_score,
            "macro_auprc": r.macro_auprc,
            "macro_f1": r.macro_f1,
            "round_seconds": r.round_seconds,
        })
    csv_path = args.out_dir / "per_round.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
