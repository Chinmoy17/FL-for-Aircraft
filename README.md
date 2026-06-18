# Federated Learning for Aircraft Engine PHM

Federated Learning baseline and research extensions for **Remaining Useful Life (RUL)
estimation** and **early fault detection** on the NASA **C-MAPSS** turbofan engine dataset.

This work is structured around two deliverables from the project brief:

1. **Task 1 — Baseline**: a federated PHM pipeline with 4 simulated airline clients and a
   central FedAvg aggregator, producing joint RUL + fault-detection predictions.
2. **Task 2 — Research extensions**: targeted work on **RQ2** (imbalance-aware
   aggregation), **RQ5** (Non-IID validation bias) and **RQ3** (sensor-level
   interpretability via SHAP + maintenance ontology).

See [`explanation.txt`](explanation.txt) and `Project for PhD Applicants.pdf` for the
underlying research context.

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12 (uv will download it if
missing).

```powershell
# 1. Create and populate the virtual environment from the lockfile
uv sync                       # runtime + dev (lean install, used by Docker / CI)
uv sync --group eda           # add Jupyter for the EDA notebook

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# 3. Run the smoke tests
pytest
```

`uv sync` always installs the **exact** versions captured in `uv.lock`, so the
environment is fully reproducible across machines and in Docker.

## Exploratory data analysis

Phase 0 EDA lives in [`notebooks/01_eda_cmapss.ipynb`](notebooks/01_eda_cmapss.ipynb).
It is committed **with embedded outputs and figures** so GitHub renders it directly
without requiring a Python install. Figures are also saved as standalone PNGs under
[`results/eda/`](results/eda/) for inclusion in the report.

The notebook is built programmatically from [`notebooks/_build_eda.py`](notebooks/_build_eda.py)
— regenerate with:

```powershell
python notebooks/_build_eda.py
python -m nbconvert --to notebook --execute --inplace notebooks/01_eda_cmapss.ipynb
```

## Data pipeline sanity check

Phase 1 pipeline (loader → labels → per-client normalizer → sliding windows →
client partition) can be exercised end-to-end with:

```powershell
python scripts/check_data_pipeline.py            # default: FD001, 4 clients
python scripts/check_data_pipeline.py --subset FD003 --n-clients 4
```

Outputs land in [`results/data/`](results/data/):

- `p1_client_summary_<subset>.csv` — per-client engine count, # windows, RUL stats, fault positive rate
- `p1_client_fault_imbalance_<subset>.png` — bar chart of per-client positive rates vs. global

Tests covering every preprocessing invariant live in [`tests/test_data.py`](tests/test_data.py)
(28 tests, ~7 s on CPU).

## Project layout

```
FL-for-Aircraft/
├── Dataset/CMAPSS_NASA/       # NASA C-MAPSS dataset (tracked)
├── src/fl_aircraft/           # importable package
│   ├── data/                  # loaders, windowing, client partitioning
│   ├── models/                # multi-task CNN (shared encoder + RUL & fault heads)
│   ├── fl/                    # FL client, server, aggregation, simulation loop
│   ├── train/                 # centralized & local-only training entrypoints
│   ├── eval/                  # RMSE / NASA score / AUPRC / plots
│   └── utils/                 # seeding, logging, config helpers
├── tests/                     # pytest smoke + unit tests
├── configs/                   # YAML experiment configs (added in P1+)
├── scripts/                   # CLI entrypoints (added in P3+)
├── results/                   # CSV logs and figures
├── pyproject.toml             # project metadata + dependencies
└── uv.lock                    # pinned, reproducible dependency graph
```

## Roadmap

| Phase | Scope |
| --- | --- |
| **P0** | Environment scaffolding (this commit). |
| **P1** | Data pipeline: loaders, sensor selection, windowing, client partitioning. |
| **P2** | Multi-task CNN + losses + metrics + centralized smoke run. |
| **P3** | Centralized baseline (FD001). |
| **P4** | Local-only baseline (4 clients, FD001). |
| **P5** | FedAvg baseline (4 clients, FD001) + 3-way comparison. |
| **P6** | Non-IID baseline on FD001 + FD003. |
| **P7** | `run_all.py` + reproducibility pass. |
| **RQ work** | RQ2 → RQ5 → RQ3, each on its own feature branch. |

## License

MIT — see [`pyproject.toml`](pyproject.toml).
