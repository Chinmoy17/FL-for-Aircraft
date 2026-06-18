"""Aggregate every ``results/<NN_phase>/metrics.json`` into ``results/summary.json``.

The summary is the **single source of truth** the React frontend will fetch.
Re-run this after any phase script completes::

    python scripts/build_results_summary.py

Idempotent: it always re-scans the ``results/`` directory and overwrites
``results/summary.json``. No caching, no incremental builds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fl_aircraft.utils import build_summary, dump_summary  # noqa: E402


def _git_head_commit() -> str | None:
    """Best-effort capture of the current git HEAD sha (short)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results",
        help="Folder containing the per-phase NN_<name> sub-folders.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "summary.json",
        help="Path of the aggregated summary JSON to write.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args.results_root, git_commit=_git_head_commit())
    dump_summary(summary, args.out)
    phases = list(summary["phases"].keys())
    print(f"Wrote {args.out}")
    print(f"  project    : {summary['project']}")
    print(f"  generated  : {summary['generated_at']}")
    print(f"  git_commit : {summary['git_commit']}")
    print(f"  phases ({len(phases)}): {', '.join(phases) or '(none)'}")
    # Pretty-print which artifacts each phase declares (handy for spotting missing PNGs).
    for pid, payload in summary["phases"].items():
        artifacts = payload.get("artifacts", {})
        if not artifacts:
            continue
        print(f"  - {pid} artifacts:")
        for name, path in artifacts.items():
            print(f"      {name}: {path}")


if __name__ == "__main__":
    main()
