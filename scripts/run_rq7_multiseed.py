"""RQ7 multi-seed runner — invoke ``run_rq7.py`` for each seed in a range.

For the paper we need mean ± std + 95 % CI on every headline number.
This wrapper drives ``scripts/run_rq7.py`` across 5 seeds sequentially,
writing each seed's output to ``results/rq7_poisoning_seeds/seed_<N>/``.

Then ``scripts/aggregate_rq7_seeds.py`` consumes those per-seed
``metrics.json`` files and produces one aggregated ``metrics.json``
with statistics per cell.

Usage (defaults are what the paper submission uses)::

    python scripts/run_rq7_multiseed.py                 # seeds 42..46
    python scripts/run_rq7_multiseed.py --seeds 100 200 # custom list
    python scripts/run_rq7_multiseed.py --dry-run       # print commands only

    # N=6 on FD001+FD003, pinned to the first GPU:
    python scripts/run_rq7_multiseed.py --subsets FD001 FD003 \
        --n-clients-per-subset 3 --device cuda:0 --out-root results/rq7_n6_fd13

    # FD002+FD004 (N=4); Krum-f2 is undefined at N=4 so skip that cell:
    python scripts/run_rq7_multiseed.py --subsets FD002 FD004 \
        --device cuda --skip-cells D54_coord_krum_f2 --out-root results/rq7_fd24

Expected wall-clock: ~80 min per seed × 5 seeds = ~6-7 hours on CPU
(substantially faster on a GPU).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "rq7_poisoning_seeds"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help=f"Seeds to run. Default: {list(DEFAULT_SEEDS)}",
    )
    p.add_argument(
        "--out-root", type=Path, default=DEFAULT_OUT_ROOT,
        help="Parent directory for per-seed output folders. "
             "Each seed writes to <out-root>/seed_<N>/.",
    )
    p.add_argument(
        "--subsets", nargs="+", default=["FD001", "FD003"],
        help="CMAPSS subsets to federate (forwarded to run_rq7.py). "
             "Default: FD001 FD003.",
    )
    p.add_argument(
        "--n-clients-per-subset", type=int, default=2,
        help="Clients per subset (forwarded to run_rq7.py). 2 -> N=4, "
             "3 -> N=6.",
    )
    p.add_argument(
        "--device", default=None,
        help="Torch device forwarded to each child run (e.g. 'cuda', "
             "'cuda:0', 'cuda:1'). Pin cuda:0/cuda:1 in two separate "
             "invocations to spread seeds across both GPUs.",
    )
    p.add_argument(
        "--skip-cells", nargs="+", default=[],
        help="Cell keys to skip, forwarded to run_rq7.py (e.g. "
             "D54_coord_krum_f2 which is undefined at N=4).",
    )
    p.add_argument(
        "--skip-fedrep-bonus", action="store_true",
        help="Pass --skip-fedrep-bonus to every child run.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would be run and exit.",
    )
    p.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter to use for the child processes.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"RQ7 multi-seed runner")
    print(f"  seeds:    {args.seeds}")
    print(f"  out root: {args.out_root}")
    print()

    start = time.perf_counter()
    for i, seed in enumerate(args.seeds, start=1):
        seed_dir = args.out_root / f"seed_{seed}"
        cmd = [
            args.python,
            str(REPO_ROOT / "scripts" / "run_rq7.py"),
            "--seed", str(seed),
            "--out-dir", str(seed_dir),
            "--subsets", *args.subsets,
            "--n-clients-per-subset", str(args.n_clients_per_subset),
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.skip_cells:
            cmd += ["--skip-cells", *args.skip_cells]
        if args.skip_fedrep_bonus:
            cmd.append("--skip-fedrep-bonus")

        print(f"[{i}/{len(args.seeds)}] seed={seed} -> {seed_dir}")
        print(f"    {' '.join(cmd)}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise SystemExit(
                f"seed {seed} failed with exit code {result.returncode}; "
                "aborting the multi-seed sweep. Fix the cell error and "
                "re-run with the same --seeds to resume."
            )

    total = time.perf_counter() - start
    print(f"\nAll {len(args.seeds)} seeds complete in {total / 60:.1f} min.")
    print(
        f"Next: python scripts/aggregate_rq7_seeds.py --seeds-root {args.out_root}"
    )


if __name__ == "__main__":
    main()
