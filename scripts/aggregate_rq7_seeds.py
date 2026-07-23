"""Aggregate RQ7 multi-seed runs into paper-ready mean ± std ± 95% CI.

Reads every ``metrics.json`` under ``--seeds-root/seed_*/`` and produces
one aggregated ``metrics.json`` where each headline number (best_rmse,
best_f1, backdoor attack success rate, etc.) is expressed as a small
dict::

    {"mean": ..., "std": ..., "ci_lo": ..., "ci_hi": ..., "n": ...}

This is what the paper reports; it's what the reader wants to see; and
it's what turns "17.95 vs 17.92" into a defensible statistical claim
or an admission of tied results.

Usage::

    python scripts/aggregate_rq7_seeds.py                # defaults
    python scripts/aggregate_rq7_seeds.py --alpha 0.05   # 95% CI (default)
    python scripts/aggregate_rq7_seeds.py --alpha 0.10   # 90% CI

Output lands in ``--out-file`` (default:
``results/rq7_poisoning/metrics_aggregated.json``) so it can sit
alongside the single-seed ``metrics.json`` without overwriting it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS_ROOT = REPO_ROOT / "results" / "rq7_poisoning_seeds"
DEFAULT_OUT_FILE = REPO_ROOT / "results" / "rq7_poisoning" / "metrics_aggregated.json"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _t_critical(alpha: float, df: int) -> float:
    """Two-sided Student-t critical value for the given df and alpha.

    Uses scipy when available (accurate); falls back to a small lookup
    table (df 1..30) plus a normal-approximation tail for larger df.
    The paper's typical case is df=n_seeds-1 = 4 for 5 seeds, so the
    exact table entry (2.776) is used.
    """
    try:
        from scipy import stats  # type: ignore
        return float(stats.t.ppf(1.0 - alpha / 2.0, df))
    except ImportError:
        # Two-sided 95% t-critical values, df in [1..30].
        table_95 = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
            13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
            19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
            25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        }
        if abs(alpha - 0.05) < 1e-6 and df in table_95:
            return table_95[df]
        # Normal-approximation fallback (df > 30 or non-95% alpha).
        return 1.96 if abs(alpha - 0.05) < 1e-6 else 1.645


def _summarize(values: Iterable[float], alpha: float = 0.05) -> dict[str, Any]:
    """Compute mean/std/ci for a list of scalar samples.

    Skips NaN entries silently (a failed cell in one seed shouldn't
    contaminate the summary of the others).
    """
    xs = np.array([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    n = int(xs.size)
    if n == 0:
        return {"mean": None, "std": None, "ci_lo": None, "ci_hi": None, "n": 0}
    mean = float(xs.mean())
    if n == 1:
        return {"mean": round(mean, 4), "std": 0.0, "ci_lo": round(mean, 4), "ci_hi": round(mean, 4), "n": 1}
    std = float(xs.std(ddof=1))
    t = _t_critical(alpha, df=n - 1)
    half_width = t * std / math.sqrt(n)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_lo": round(mean - half_width, 4),
        "ci_hi": round(mean + half_width, 4),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Load N per-seed metrics.json files
# ---------------------------------------------------------------------------
def _load_seed_dirs(seeds_root: Path) -> list[tuple[int, dict]]:
    dirs = sorted(seeds_root.glob("seed_*"))
    if not dirs:
        raise SystemExit(
            f"No seed_* subdirectories under {seeds_root}. "
            "Run scripts/run_rq7_multiseed.py first."
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
_CELL_METRICS = ("best_rmse", "best_nasa_score", "best_auprc", "best_f1")
_BACKDOOR_METRICS = (
    "clean_auprc", "clean_f1", "clean_fault_positive_rate",
    "triggered_auprc", "triggered_f1", "triggered_fault_positive_rate",
    "attack_success_rate",
)


def _aggregate(payloads: list[tuple[int, dict]], alpha: float) -> dict[str, Any]:
    """Aggregate multi-seed payloads into paper-ready statistics."""
    seeds = [s for s, _ in payloads]
    # ---- cell-level metrics (best_rmse etc.) ----
    all_cell_keys: set[str] = set()
    for _, p in payloads:
        all_cell_keys.update((p.get("per_client") or {}).keys())

    per_cell: dict[str, dict] = {}
    for cell_key in sorted(all_cell_keys):
        entry: dict[str, Any] = {}
        for metric in _CELL_METRICS:
            values = [
                (p.get("per_client") or {}).get(cell_key, {}).get(metric)
                for _, p in payloads
            ]
            entry[metric] = _summarize(values, alpha=alpha)
        # Nominal per-cell attributes are stable across seeds; take the first.
        first = next(
            ((p.get("per_client") or {})[cell_key] for _, p in payloads
             if cell_key in (p.get("per_client") or {})),
            None,
        )
        if first is not None:
            entry["label"] = first.get("label")
            entry["group"] = first.get("group")
            entry["attacker_kind"] = first.get("attacker_kind")
            entry["aggregator"] = first.get("aggregator")
        per_cell[cell_key] = entry

    # ---- backdoor cells: attack success + clean/triggered fault rates ----
    backdoor_summary: dict[str, dict] = {}
    all_backdoor_keys: set[str] = set()
    for _, p in payloads:
        be = ((p.get("summary") or {}).get("backdoor_evaluation") or {})
        all_backdoor_keys.update(be.keys())
    for k in sorted(all_backdoor_keys):
        entry = {}
        for metric in _BACKDOOR_METRICS:
            values = [
                ((p.get("summary") or {}).get("backdoor_evaluation") or {})
                .get(k, {}).get(metric)
                for _, p in payloads
            ]
            entry[metric] = _summarize(values, alpha=alpha)
        backdoor_summary[k] = entry

    return {
        "n_seeds": len(payloads),
        "seeds": seeds,
        "alpha": alpha,
        "per_cell": per_cell,
        "backdoor_summary": backdoor_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seeds-root", type=Path, default=DEFAULT_SEEDS_ROOT,
        help=f"Directory containing seed_<N>/metrics.json subdirs. "
             f"Default: {DEFAULT_SEEDS_ROOT.relative_to(REPO_ROOT)}",
    )
    p.add_argument(
        "--out-file", type=Path, default=DEFAULT_OUT_FILE,
        help=f"Where to write the aggregated JSON. "
             f"Default: {DEFAULT_OUT_FILE.relative_to(REPO_ROOT)}",
    )
    p.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for the two-sided confidence interval. "
             "Default 0.05 (=> 95%% CI).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payloads = _load_seed_dirs(args.seeds_root)
    print(f"Loaded {len(payloads)} seeds: {[s for s, _ in payloads]}")

    aggregated = _aggregate(payloads, alpha=args.alpha)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8") as fh:
        json.dump(aggregated, fh, indent=2)
    print(f"Wrote {args.out_file}")

    # Print a compact table for immediate eyeballing.
    print("\nHeadline RMSE (mean ± std, 95% CI):")
    for k, entry in aggregated["per_cell"].items():
        r = entry["best_rmse"]
        if r["mean"] is None:
            continue
        print(
            f"  {k:32s}  {r['mean']:6.2f} ± {r['std']:5.2f}  "
            f"[{r['ci_lo']:6.2f}, {r['ci_hi']:6.2f}]  (n={r['n']})"
        )
    if aggregated["backdoor_summary"]:
        print("\nBackdoor attack success rate (mean ± std, 95% CI):")
        for k, entry in aggregated["backdoor_summary"].items():
            asr = entry["attack_success_rate"]
            if asr["mean"] is None:
                continue
            print(
                f"  {k:32s}  {asr['mean']:5.3f} ± {asr['std']:5.3f}  "
                f"[{asr['ci_lo']:5.3f}, {asr['ci_hi']:5.3f}]  (n={asr['n']})"
            )


if __name__ == "__main__":
    main()
