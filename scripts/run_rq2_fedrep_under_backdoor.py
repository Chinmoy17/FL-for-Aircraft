"""FedRep-under-backdoor bridge experiment (Axis 1 × Axis 2 cross-cut).

Question
--------
Does FedRep's per-client head shield honest clients from a backdoor that
one malicious client injects via poisoned training data?

Setup
-----
FedRep runs on the exact same FD001+FD003 structural non-IID partition as
``run_fedrep.py`` (4 clients: 2 FD001 + 2 FD003, 50 engines each,
:func:`partition_by_subset_halves`). Before training starts, we swap
``--attacker-client-id``'s (default ``client_3`` → FD003 side) train_loader
with a poisoned version that stamps the same RQ7 backdoor trigger
(sensor s_3 / T30 = -3.5 z-score at the last cycle) on ``--poison-frac``
of its windows and rewrites those labels to healthy (RUL=cap, fault=0).

After training completes, each client's full model (shared encoder + own
head) is evaluated for **Attack Success Rate** on the **pooled** test set
(all FD001+FD003 engines), using the SAME triggered-eval logic as
``scripts/run_rq7.py``'s ``_eval_backdoor_success``. This yields per-client
ASRs comparable directly to Table 10's vanilla-FedAvg baseline.

Compare against RQ7 baselines
-----------------------------
    AV3_backdoor_vanilla : ASR = 94.9 ± 7.9 %   (undefended FedAvg)
    D33_backdoor_krum    : ASR =  6.4 ± 10.0 %  (Krum-defended)

If FedRep's mean honest-client ASR is:
    * near  0 %  → personalization is a natural, structural backdoor defense
                   (the shared encoder does absorb the poison, but honest
                    heads reject it → new positive contribution)
    * 20-80 %   → partial protection; stacking with Krum still needed
    * near 95 % → personalization alone does not defend; encoder poisoning
                   transfers through heads unchanged

Outputs (``results/rq_bridge/`` by default, or ``--out-dir``)
-------------------------------------------------------------
    metrics.json           per-client backdoor eval + honest-avg ASR
    per_round.csv          macro RMSE trajectory across rounds (informative)

Run from the repo root inside the .venv::

    .\\.venv\\Scripts\\python.exe scripts\\run_rq2_fedrep_under_backdoor.py --seed 42
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

PHASE_ID = "rq_bridge"
PHASE_NAME = "Bridge — FedRep under backdoor (Axis 1 × Axis 2 cross-cut)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- FedRep base params (mirror scripts/run_fedrep.py) --------------
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
# Helpers
# ---------------------------------------------------------------------------
def _build_shard_to_subset(shards, subsets, n_per_subset) -> dict[str, str]:
    """Same helper as scripts/run_fedrep.py — first N shards -> subsets[0],
    next N -> subsets[1], etc. ``partition_by_subset_halves`` lays them out
    in subset order, so this contiguous split is correct."""
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

    Adapted from ``_eval_backdoor_success`` in scripts/run_rq7.py. Uses a
    GLOBAL normalizer fitted on ``bundle.train_df`` (not any per-client
    normalizer) so the ASRs across clients are directly comparable to
    Table 10's vanilla-FedAvg numbers.

    Returns a plain dict with rounded scalars, safe for JSON.
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
        """Return fault-probability scores for a given input tensor."""
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

    # 1. Clean pass
    clean_scores = _run(arrays.X)
    clean_m = compute_classification_metrics(arrays.y_fault, clean_scores)
    clean_positive_rate = float((clean_scores >= 0.5).mean())

    # 2. Trigger-stamped pass — same labels, same engines
    X_triggered = stamp_trigger_on_windows(
        arrays.X, feature_idx=feature_idx,
        cycle_offset=cycle_offset, trigger_value=trigger_value,
    )
    triggered_scores = _run(X_triggered)
    triggered_m = compute_classification_metrics(arrays.y_fault, triggered_scores)
    triggered_positive_rate = float((triggered_scores >= 0.5).mean())

    # ASR = fraction of clean-positive prediction mass that flipped to
    # negative under the trigger. Clamped >= 0 for numeric safety.
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

    print(f"--- FedRep-under-backdoor bridge ({'+'.join(args.subsets)}) ---")
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
                # _BackdoorPoisonedDataset exposes n_poisoned via its len()
                # of the poison mask; the DataLoader hides that. Print
                # only for the user's sanity check.
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

    # ---- Run FedRep with the poison hook --------------------------------
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
    print("Compare vs RQ7 vanilla-FedAvg-under-backdoor baseline: "
          "ASR = 94.9 +/- 7.9 %")

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
            "note": "From results/rq7_poisoning/metrics_aggregated.json "
                    "(5-seed mean+/-std) for direct comparison.",
            "AV3_backdoor_vanilla_asr_mean": 0.949,
            "AV3_backdoor_vanilla_asr_std": 0.079,
            "D33_backdoor_krum_asr_mean": 0.064,
            "D33_backdoor_krum_asr_std": 0.100,
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
