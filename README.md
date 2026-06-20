# Federated Learning for Aircraft Engine PHM

> A PhD research project — federated training for Remaining Useful Life
> (RUL) estimation and early fault detection on NASA C-MAPSS turbofan
> engines, with three answered research questions, three follow-up
> experiments, an interpretability pipeline, and a security study.

| Stat | Value |
| --- | --- |
| Experimental phases | **11** |
| Tests passing | **216 / 216** |
| Model parameters | **30,018** (GroupNorm-only, FL-safe) |
| Simulated airline clients | **4** |
| Total CPU wall-clock to reproduce | **~5 hours** |

---

## Headline findings

Of the seven research questions in the project brief:

| RQ | Topic | Verdict | Headline |
| --- | --- | --- | --- |
| **RQ2** | Imbalance-aware aggregation | **Negative** | Three reweighting schemes closed `<14%` of the Non-IID gap. Cure lives at a different layer. |
| **RQ3** | Sensor attribution + ontology | **Positive** | 12 cross-model attributions surface an `os_2` subset-proxy failure that RMSE alone hid. |
| **RQ7** | Model poisoning + Byzantine defense | **Positive** | Gradient-scaling attack pushes RMSE from 17.95 → **84.03**; Krum recovers to **19.80** (within 1.85 of clean). |
| RQ4 / RQ5 | Concept drift / cross-client eval | **Synthesised** | Substantial cross-cutting evidence in existing phases — written up as a synthesis page, not a new experiment. |
| RQ1 / RQ6 | Sensor heterogeneity / privacy | **Open** | Scoped honestly as follow-up work. |

The **RQ2 follow-up trilogy** (FedProx / FedRep / FedCCFA) closes
the loop on the negative RQ2 finding:

- **FedProx (μ-sweep):** +6% gap closed — small but positive.
- **FedRep (per-client heads):** **+73% gap closed** — the project's
  largest positive result.
- **FedCCFA (clustered heads):** null on this dataset; the heads
  collapse to a single cluster.

Empirical layer hierarchy: `aggregation < drift-control < per-client architecture`.

---

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12 (uv will
download it if missing).

```powershell
# 1. Create and populate the virtual environment from the lockfile
uv sync                       # runtime + dev (lean install)
uv sync --group eda           # add Jupyter for the EDA notebook

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# 3. Run the full test suite (~30 s on CPU)
pytest
```

`uv sync` always installs the **exact** versions captured in `uv.lock`,
so the environment is fully reproducible across machines and in Docker.

### Run one experiment

Every phase has its own script under `scripts/`. Each writes a
per-phase `metrics.json` + figures into `results/<phase>/`:

```powershell
# Baseline (each script is independent — pick one)
python scripts/check_data_pipeline.py        # Phase 01 — partition sanity
python scripts/smoke_train.py                # Phase 02 — wire-test
python scripts/run_centralized.py            # Phase 03 — upper bound
python scripts/run_local_only.py             # Phase 04 — lower bound
python scripts/run_fedavg.py                 # Phase 05 — FedAvg IID
python scripts/run_non_iid.py                # Phase 06 — Non-IID

# Research questions
python scripts/run_rq2.py                    # 4-scheme aggregation sweep
python scripts/run_rq3.py --no-llm           # 4 checkpoints × N engines
python scripts/run_rq7.py                    # 11-cell attack × defense matrix

# RQ2 follow-ups
python scripts/run_fedprox.py                # μ-sweep
python scripts/run_fedrep.py                 # per-client heads
python scripts/run_fedccfa.py                # clustered heads

# Refresh the aggregate manifest for the frontend
python scripts/build_results_summary.py
```

Each script accepts `--help`. Default configs match what every figure /
table in the reports was generated with (seed 42, batch 256, lr 1e-3
cosine, GroupNorm, 50 rounds × 2 local epochs).

---

## Where to read

| You want… | Open… |
| --- | --- |
| **Per-RQ technical reports** (closest to a chapter draft) | [`rq2_report.md`](rq2_report.md) · [`rq3_report.md`](rq3_report.md) · [`rq7_report.md`](rq7_report.md) |
| **Per-phase narrative** (numbers + interpretation) | [`results.md`](results.md) |
| **Engineering history** (decisions, challenges, what was built) | [`progress.md`](progress.md) |
| **Original brief** | [`Project for PhD Applicants.pdf`](Project%20for%20PhD%20Applicants.pdf) · [`explanation.txt`](explanation.txt) |
| **EDA notebook** (rendered on GitHub) | [`notebooks/01_eda_cmapss.ipynb`](notebooks/01_eda_cmapss.ipynb) |
| **Machine-readable phase metrics** | [`results/summary.json`](results/summary.json) and individual `results/<phase>/metrics.json` |
| **Interactive academic frontend** | See the `p7_demo` branch — `frontend/` Vite + React 19 app with sidebar TOC, per-phase explanations, interactive demo, and the three RQ story pages. |

### The three technical reports

Each follows the same 8-section template (problem, previous work,
dataset, methods, experiment, mechanism, future directions, caveats)
plus a 10-bullet TL;DR and an artifact-pointer appendix. Total:
**132 KB / 2,134 lines** of long-form writing.

| Report | Lines | Verdict | Focus |
| --- | --- | --- | --- |
| [`rq2_report.md`](rq2_report.md) | 586 | Negative | Why aggregation alone can't fix structural Non-IID |
| [`rq3_report.md`](rq3_report.md) | 779 | Positive | Integrated Gradients + ontology + cross-model comparison |
| [`rq7_report.md`](rq7_report.md) | 769 | Positive | Boosted-Byzantine attacks vs Krum, trimmed mean, median |

---

## Project layout

```
FL-for-Aircraft/
├── Dataset/CMAPSS_NASA/       # NASA C-MAPSS turbofan dataset
├── src/fl_aircraft/           # importable package
│   ├── data/                  # loaders, windowing, client partitioning
│   ├── models/                # multi-task CNN (shared encoder + 2 heads)
│   ├── fl/                    # client, server, aggregators, simulation
│   │                          # (including FedProx, FedRep, FedCCFA,
│   │                          #  poisoning attacks, robust aggregators)
│   ├── explain/               # RQ3: ontology + IG attribution + narrative
│   ├── train/                 # centralized + local-only training loops
│   ├── eval/                  # RMSE / NASA score / AUPRC / plots
│   └── utils/                 # seeding, PhaseMetrics writer, config helpers
├── tests/                     # 216 pytest unit + integration tests
├── scripts/                   # 14 CLI entrypoints (one per phase + helpers)
├── notebooks/                 # EDA notebook (rebuilt by _build_eda.py)
├── results/                   # one folder per phase + summary.json
│   ├── 00_eda/  01_data/  02_smoke/  03_centralized/
│   ├── 04_local_only/  05_fedavg/  06_non_iid/
│   ├── rq2_imbalance_aware/  rq2_fedprox/  rq2_fedrep/  rq2_fedccfa/
│   ├── rq3_explanations/  rq7_poisoning/
│   └── summary.json           # aggregated by build_results_summary.py
├── rq2_report.md  rq3_report.md  rq7_report.md
├── results.md  progress.md  baseline_report.md  paper_summary.md
├── pyproject.toml             # project metadata + dependencies
└── uv.lock                    # pinned, reproducible dependency graph
```

---

## Branches

| Branch | What's on it |
| --- | --- |
| `main` | Stable releases (PRs from `dev`). |
| `dev` | Integration branch. All science + reports + tests live here. **You are reading this README from `dev`.** |
| `p7_demo` | Interactive academic frontend (Vite + React 19 + Tailwind v4). Light slate-blue theme, full sidebar TOC, per-phase pages with explained figures, three RQ story pages, RQ4/RQ5 synthesis, live `/demo` route backed by FastAPI. |
| `rq2`, `rq3`, `rq7` | Per-RQ feature branches — merged to `dev`, kept for archaeology. |
| `fedprox`, `fedrep`, `fedccfa` | RQ2 follow-up branches — merged to `dev`, kept for archaeology. |

---

## Key design decisions

- **GroupNorm, not BatchNorm.** BatchNorm's running buffers would have
  to be averaged across heterogeneous clients — mathematically wrong
  under FedAvg. A regression test (`test_no_batchnorm_layers_present`)
  ensures this never regresses.
- **Tiny model (30K params).** Small enough that 50 rounds × 2 local
  epochs × 4 clients fit comfortably on CPU. Bigger models would
  overfit on FD001's 17,731 training windows.
- **Pluggable aggregator pattern.** `FedAvgServer(aggregator=...)`
  accepts any `Sequence[ClientUpdate] -> dict[str, torch.Tensor]`. RQ2
  reweighting schemes, FedProx, and the RQ7 robust aggregators all
  plug into the same slot.
- **All metrics reported both combined AND per-subset.** The RQ5
  synthesis explains why this is non-negotiable for honest FL
  reporting.

---

## License

MIT — see [`pyproject.toml`](pyproject.toml).

## Author

Chinmoy Mitra · [chinmoy17.github.io](https://chinmoy17.github.io) ·
[github.com/Chinmoy17](https://github.com/Chinmoy17)

