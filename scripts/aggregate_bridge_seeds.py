"""Aggregate bridge-experiment multi-seed runs into paper-ready mean +/- std +/- 95% CI.

Reads every ``metrics.json`` under ``--seeds-root/seed_*/`` and produces
one aggregated ``metrics.json`` where each headline number (best_rmse,
attacker_asr, honest_mean_asr, per-client backdoor eval) is expressed
as ``{mean, std, ci_lo, ci_hi, n, values}``.

The bridge experiment's schema differs from RQ7's — each seed produces
per-client backdoor evaluations (one MultiTaskCNN per client, one ASR
number per client on the pooled test set), plus per-client clean RMSE.
The aggregator handles both.

Usage::

    python scripts/aggregate_bridge_seeds.py                 # defaults
    python scripts/aggregate_bridge_seeds.py --alpha 0.05    # 95% CI (default)

Output lands in ``--out-file`` (default:
``results/rq_bridge/metrics_aggregated.json``).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS_ROOT = REPO_ROOT / "results" / "rq_bridge"
DEFAULT_OUT_FILE = REPO_ROOT / "results" / "rq_bridge" / "metrics_aggregated.json"


# ---------------------------------------------------------------------------
# Statistics (copied verbatim from aggregate_rq7_seeds.py for consistency)
# ---------------------------------------------------------------------------
def _t_critical(alpha: float, df: int) -> float:
    """Two-sided Student-t critical value for the given df and alpha."""
    try:
        from scipy import stats  # type: ignore
        return float(stats.t.ppf(1.0 - alpha / 2.0, df))
    except ImportError:
        table_95 = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        }
        if abs(alpha - 0.05) < 1e-6 and df in table_95:
            return table_95[df]
        return 1.96 if abs(alpha - 0.05) < 1e-6 else 1.645


def _summarize(values: Iterable[float], alpha: float = 0.05) -> dict[str, Any]:
    """Compute mean/std/CI for a list of scalar samples. Skips NaN and None."""
    xs = np.array(
        [
            v for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ],
        dtype=np.float64,
    )
    n = int(xs.size)
    if n == 0:
        return {"mean": None, "std": None, "ci_lo": None, "ci_hi": None,
                "n": 0, "values": []}
    mean = float(xs.mean())
    if n == 1:
        return {"mean": round(mean, 4), "std": 0.0, "ci_lo": round(mean, 4),
                "ci_hi": round(mean, 4), "n": 1, "values": [round(mean, 4)]}
    std = float(xs.std(ddof=1))
    t = _t_critical(alpha, df=n - 1)
    half_width = t * std / math.sqrt(n)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_lo": round(mean - half_width, 4),
        "ci_hi": round(mean + half_width, 4),
        "n": n,
        "values": [round(float(x), 4) for x in xs],
    }


# ---------------------------------------------------------------------------
# Load per-seed metrics.json files
# ---------------------------------------------------------------------------
def _load_seed_dirs(seeds_root: Path) -> list[tuple[int, dict]]:
    dirs = sorted(seeds_root.glob("seed_*"))
    if not dirs:
        raise SystemExit(
            f"No seed_* subdirectories under {seeds_root}. "
            "Run scripts/run_rq2_fedrep_under_backdoor.py --seed <n> first."
        )
    out: list[tuple[int, dict]] = []
    for d in dirs:
        seed_str = d.name.removeprefix("seed_")
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            print(f"  WARN: {metrics_path} not found; skipping")
            continue
        with metrics_path.open(encoding="utf-8") as fh:
            out.append((seed, json.load(fh)))
    if not out:
        raise SystemExit(f"No valid seed metrics found under {seeds_root}.")
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
_SUMMARY_FIELDS = (
    "best_round",
    "best_macro_rmse",
    "best_macro_nasa_score",
    "attacker_asr",
    "honest_mean_asr",
    "honest_max_asr",
)
_PER_CLIENT_BD_FIELDS = (
    "clean_auprc",
    "clean_f1",
    "clean_fault_positive_rate",
    "triggered_auprc",
    "triggered_f1",
    "triggered_fault_positive_rate",
    "attack_success_rate",
)


def _aggregate(payloads: list[tuple[int, dict]], alpha: float) -> dict[str, Any]:
    seeds = [s for s, _ in payloads]

    # ---- summary-level scalars ----
    summary_agg: dict[str, dict] = {}
    for field in _SUMMARY_FIELDS:
        vals = [p["summary"].get(field) for _, p in payloads]
        summary_agg[field] = _summarize(vals, alpha)

    # ---- per-client backdoor evaluation ----
    # Collect the union of client IDs across all seeds (should be the same
    # set for a given experiment configuration).
    all_client_ids: set[str] = set()
    for _, p in payloads:
        all_client_ids.update((p.get("per_client_backdoor_evaluation") or {}).keys())

    per_client_bd_agg: dict[str, dict[str, dict]] = {}
    for cid in sorted(all_client_ids):
        per_client_bd_agg[cid] = {}
        for field in _PER_CLIENT_BD_FIELDS:
            vals = [
                (p.get("per_client_backdoor_evaluation", {})
                  .get(cid, {})
                  .get(field))
                for _, p in payloads
            ]
            per_client_bd_agg[cid][field] = _summarize(vals, alpha)

    # ---- per-client clean-test RMSE (best round + final round) ----
    per_client_rmse_best: dict[str, dict] = {}
    per_client_rmse_final: dict[str, dict] = {}
    for cid in sorted(all_client_ids):
        vals_best = [
            (p.get("per_client_clean_test_rmse_best_round", {}).get(cid))
            for _, p in payloads
        ]
        vals_final = [
            (p.get("per_client_clean_test_rmse_final_round", {}).get(cid))
            for _, p in payloads
        ]
        per_client_rmse_best[cid] = _summarize(vals_best, alpha)
        per_client_rmse_final[cid] = _summarize(vals_final, alpha)

    # ---- derived: attacker minus honest-mean ASR per seed ----
    # If personalization shielded honest clients, we'd expect this delta
    # to be large and positive (attacker's own head learns the backdoor
    # but honest heads don't). If it's near zero, personalization does
    # NOT shield — which is the paper's finding.
    attacker_minus_honest = [
        p["summary"]["attacker_asr"] - p["summary"]["honest_mean_asr"]
        for _, p in payloads
    ]
    delta_agg = _summarize(attacker_minus_honest, alpha)

    # ---- pull common config from the first seed (they should all match) ----
    common_config = dict(payloads[0][1].get("config", {}))
    common_config.pop("seed", None)  # varies by definition

    return {
        "phase_id": "rq_bridge",
        "phase_name": "Bridge - FedRep under backdoor (aggregated over "
                       f"{len(seeds)} seeds)",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "alpha": alpha,
        "config_common": common_config,
        "summary": summary_agg,
        "attacker_minus_honest_asr": delta_agg,
        "per_client_backdoor_evaluation": per_client_bd_agg,
        "per_client_clean_test_rmse_best_round": per_client_rmse_best,
        "per_client_clean_test_rmse_final_round": per_client_rmse_final,
        "per_seed_summary": {
            str(seed): p["summary"] for seed, p in payloads
        },
        "reference_baselines": payloads[0][1].get("reference_baselines", {}),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds-root", type=Path, default=DEFAULT_SEEDS_ROOT,
                   help="Root dir containing seed_*/metrics.json files.")
    p.add_argument("--out-file", type=Path, default=DEFAULT_OUT_FILE,
                   help="Where to write the aggregated metrics.json.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Two-sided CI level (default 0.05 = 95% CI).")
    return p.parse_args()


def _print_report(agg: dict[str, Any]) -> None:
    print(f"Aggregated {agg['n_seeds']} seeds: {agg['seeds']}\n")

    # Per-seed table (concise)
    print(f"{'seed':>4} {'best_rd':>7} {'best_rmse':>10} "
          f"{'attacker_asr':>13} {'honest_asr':>11}")
    for seed_str, s in agg["per_seed_summary"].items():
        print(f"{seed_str:>4} {s['best_round']:>7} "
              f"{s['best_macro_rmse']:>10.3f} "
              f"{s['attacker_asr']:>13.3f} "
              f"{s['honest_mean_asr']:>11.3f}")

    print()
    for name, key in [
        ("attacker ASR", "attacker_asr"),
        ("honest mean ASR", "honest_mean_asr"),
        ("honest max ASR", "honest_max_asr"),
        ("best macro RMSE", "best_macro_rmse"),
        ("best round", "best_round"),
    ]:
        s = agg["summary"][key]
        print(f"  {name:22s}: {s['mean']:.3f} +/- {s['std']:.3f} "
              f"(95% CI [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}], n={s['n']})")

    d = agg["attacker_minus_honest_asr"]
    print(f"\n  attacker - honest ASR (delta): "
          f"{d['mean']:.3f} +/- {d['std']:.3f} "
          f"(95% CI [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}])")
    print("  --> if this is near 0, personalization does NOT shield honest "
          "clients from the backdoor.")

    refs = agg.get("reference_baselines", {})
    if refs:
        print("\nReference baselines from RQ7:")
        print(f"  vanilla FedAvg backdoor: ASR = "
              f"{refs.get('AV3_backdoor_vanilla_asr_mean', 0):.3f} +/- "
              f"{refs.get('AV3_backdoor_vanilla_asr_std', 0):.3f}")
        print(f"  Krum-defended backdoor : ASR = "
              f"{refs.get('D33_backdoor_krum_asr_mean', 0):.3f} +/- "
              f"{refs.get('D33_backdoor_krum_asr_std', 0):.3f}")


def main() -> None:
    args = parse_args()
    print(f"Loading from: {args.seeds_root}")
    payloads = _load_seed_dirs(args.seeds_root)
    print(f"Found {len(payloads)} seed(s)\n")

    agg = _aggregate(payloads, args.alpha)
    _print_report(agg)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2)
    print(f"\nWrote {args.out_file}")


if __name__ == "__main__":
    main()
