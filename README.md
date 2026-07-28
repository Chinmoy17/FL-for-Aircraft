# Federated Learning for Aircraft Engine PHM

Turbofan engines are monitored continuously to estimate their **Remaining
Useful Life (RUL)** — how long before maintenance is due. Several operators
would each benefit from a shared prediction model, yet none can pool their raw
flight data. **Federated learning** resolves that tension: every operator
trains locally and shares only model updates, which a server merges into one
global model. This repository studies the two obstacles that separate that
promise from a system trustworthy enough to fly behind — operators whose
engines fail in **different ways**, and operators who may be **dishonest** —
and a defense that addresses both.

The code and experiments accompany a manuscript on robust,
heterogeneity-aware federated prognostics. Everything runs on the public NASA
C-MAPSS dataset and is fully reproducible from a pinned lockfile.

| Stat | Value |
| --- | --- |
| Tests passing | **233 / 233** |
| Model parameters | **30,018** (GroupNorm-only, FL-safe) |
| Dataset | **NASA C-MAPSS** FD001–FD004 |
| Federated clients | **4** (baseline) · **6** (scale test) |
| Seeds per reported result | **5** `{42–46}` (mean ± std) |
| Reproduce | pinned `uv.lock` + one script per result |

> **A note on the `RQ*` labels.** This work began as a structured research
> project whose questions were numbered `RQ1…RQ7`. The manuscript reorganizes the
> material around **two axes**, but the original numbers are kept throughout the
> code and reports as stable identifiers. The mapping is: **RQ2 →
> heterogeneity / personalization**, **RQ3 → explainability**, **RQ7 →
> robustness / poisoning**. `RQ1 / RQ4 / RQ5 / RQ6` are scoped as synthesis or
> future work.

---

## What this repository demonstrates

The project is built around two questions and a bridge between them.

**Heterogeneity — can a shared model cope when operators are different?** When
each client sees only some fault modes, plain federated averaging degrades
badly: RMSE rises to 17.95 against a centralized upper bound of 14.0.
Reweighting the average barely moves that number. What works is
*personalization* — FedRep keeps one shared backbone but gives every client its
own prediction head, recovering to 14.91, within touching distance of the
centralized model. The effect survives on the harder six-condition data
(FD002+FD004), though the margin narrows, and the reports explain where and why.

**Robustness — can the model survive a dishonest operator, and would we even
notice?** This is the sharper result. A stealthy *failure-masking backdoor* can
push the attack-success rate to between 0.95 and 0.999 while the model's
everyday accuracy stays perfectly normal — the poisoning is invisible to RMSE,
and surfaces only when attack success is measured directly. Among the defenses,
Krum is the consistent survivor (attack success 0.03–0.16); trimmed-mean and
median aggregation both collapse the moment two attackers collude.

**The bridge — can we have both at once?** Personalization on its own is not
protection: FedRep under attack still reaches 0.63. Stacking a robust aggregator
on the shared backbone — a combination we call **SHARP** — brings robustness
back to 0.028 without surrendering personalization's accuracy gain, and the
result holds as the federation grows from four to six clients and moves onto the
harder dataset.

**At a glance** — RMSE is RUL error (lower is better); ASR is attack-success
rate (lower is better); every figure is a mean over five random seeds:

| Setting | Key number | Reading |
| --- | --- | --- |
| Non-IID FedAvg vs centralized | 17.95 vs 14.0 RMSE | plain averaging leaves a large gap |
| FedRep (personalization) | 14.91 RMSE | closes almost all of it |
| Backdoor, undefended | 0.95–0.999 ASR at normal RMSE | attack invisible to accuracy |
| Backdoor + Krum | 0.03–0.16 ASR | robust aggregation holds |
| FedRep alone, under attack | 0.63 ASR | personalization is not protection |
| SHARP (FedRep + Krum) | 0.028 ASR | both properties at once |

**How far it generalizes.** The ordering of defenses is stable as the setting
gets harder. With six clients a stronger variant of Krum becomes applicable and
turns back even two coordinated attackers; on the six-condition FD002+FD004 data
every defense weakens — the undefended backdoor reaches 0.999 and Krum still
leads at 0.16 — but the ranking never changes. Where results soften, the reports
say so rather than round them away.

**Why it predicts what it predicts.** An Integrated-Gradients pipeline over a
maintenance ontology shows that a heterogeneous model can base its RUL estimate
on *operating settings* rather than genuine degradation signals — a flaw a
healthy-looking RMSE would quietly hide. It is the backdoor lesson from another
direction: a good headline number can sit on top of an unhealthy model.

**On novelty.** The building blocks — FedRep, FedProx, Krum, trimmed-mean and
median aggregation, Integrated Gradients — are established methods. The
contribution is bringing them together for aeroengine prognostics under a safety
lens: showing that a failure-masking backdoor is invisible to accuracy, and
mapping systematically which defenses withstand which attacks. Where a result
echoes a known general finding — that personalization alone does not confer
robustness, for instance — it is presented as confirmation in a new domain
rather than a new claim.

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

# Axis 1 — heterogeneity / personalization
python scripts/run_rq2.py                    # 4-scheme aggregation sweep
python scripts/run_fedprox.py                # FedProx μ-sweep
python scripts/run_fedrep.py                 # FedRep per-client heads (Axis-1 winner)
python scripts/run_fedccfa.py                # FedCCFA clustered heads

# Explainability
python scripts/run_rq3.py --no-llm           # attributions, 4 checkpoints × N engines

# Axis 2 — robustness / poisoning (5 attacks × 4 aggregators)
python scripts/run_rq7.py                    # single-seed attack × defense matrix
python scripts/run_rq7_multiseed.py          # 5-seed campaign (--subsets, --n-clients-per-subset, --device)

# Bridge — SHARP (FedRep + Krum, under backdoor)
python scripts/run_rq2_fedrep_krum_bridge.py # add --subsets / --n-clients-per-subset / --feature-idx for N=6 / FD002+4

# Refresh the aggregate manifest for the frontend
python scripts/build_results_summary.py
```

Each script accepts `--help`. Default configs match what every figure /
table in the reports was generated with (seed 42, batch 256, lr 1e-3
cosine, GroupNorm, 50 rounds × 2 local epochs). Reported headline numbers
are mean ± std over **5 seeds `{42–46}`**. Pass `--device cuda` (or set
`FL_DEVICE=cuda`) to train on a GPU; the default auto-detects CUDA and
falls back to CPU, so the same code reproduces on either.

---

## Where to read

The repository doubles as its own write-up. The three per-topic reports read
like paper sections and are the best entry point; the per-phase narrative and
the engineering log go deeper still.

| You want… | Open… |
| --- | --- |
| **Per-topic technical reports** (closest to paper sections) | [`rq2_report.md`](rq2_report.md) (heterogeneity) · [`rq3_report.md`](rq3_report.md) (explainability) · [`rq7_report.md`](rq7_report.md) (robustness) |
| **Per-phase narrative** (numbers + interpretation) | [`results.md`](results.md) |
| **Engineering history** (decisions, what was built) | [`progress.md`](progress.md) |
| **EDA notebook** (rendered on GitHub) | [`notebooks/01_eda_cmapss.ipynb`](notebooks/01_eda_cmapss.ipynb) |
| **Machine-readable metrics** | [`results/summary.json`](results/summary.json) + per-experiment `results/<name>/metrics*.json` |
| **Interactive frontend** | [`frontend/`](frontend/) — Vite + React 19 + Tailwind v4 app with per-phase pages, story pages, and a live `/demo` route backed by FastAPI (run below). |
| **Original project brief** (background) | [`explanation.txt`](explanation.txt) |

### The three technical reports

Each report follows the same structure — problem, prior work, dataset, methods,
experiment, mechanism, future directions, and caveats — with a TL;DR at the top
and an appendix pointing back to the artifacts that produced every figure.

| Report | Topic | Focus |
| --- | --- | --- |
| [`rq2_report.md`](rq2_report.md) | Heterogeneity | Why reweighting the average can't fix structural non-IID, and what can |
| [`rq3_report.md`](rq3_report.md) | Explainability | Integrated Gradients over a maintenance ontology, compared across models |
| [`rq7_report.md`](rq7_report.md) | Robustness | Boosted-Byzantine and backdoor attacks against Krum, trimmed-mean, and median |

---

## Project layout

```
FL-for-Aircraft/
├── Dataset/CMAPSS_NASA/       # NASA C-MAPSS turbofan dataset
├── src/fl_aircraft/           # importable package
│   ├── data/                  # loaders, windowing, client partitioning
│   ├── models/                # multi-task CNN (shared encoder + 2 heads)
│   ├── fl/                    # client, server, aggregators, simulation
│   │                          # (FedProx, FedRep, FedCCFA, poisoning
│   │                          #  attacks, robust aggregators, bridge)
│   ├── explain/               # explainability: ontology + IG attribution
│   ├── train/                 # centralized + local-only training loops
│   ├── eval/                  # RMSE / NASA score / AUPRC / plots
│   └── utils/                 # seeding, device (CPU/GPU) resolver, metrics, config
├── tests/                     # 233 pytest unit + integration tests
├── scripts/                   # CLI entrypoints (one per experiment + helpers)
├── notebooks/                 # EDA notebook (rebuilt by _build_eda.py)
├── kaggle/                    # dual-GPU campaign notebook (N=6 / FD002+4)
├── results/                   # one folder per experiment + summary.json
│   ├── 00_eda/ 01_data/ 02_smoke/ 03_centralized/ 04_local_only/
│   ├── 05_fedavg/ 06_non_iid/                          # baselines
│   ├── rq2_imbalance_aware/ rq2_fedprox/ rq2_fedrep/ rq2_fedccfa/   # Axis 1
│   ├── rq2_fd24_fedrep/                                # Axis 1 on hard data
│   ├── rq3_explanations/                               # explainability
│   ├── rq7_poisoning/ rq7_n6_fd13/ rq7_fd24/           # Axis 2 (N=4, N=6, hard)
│   ├── rq_bridge_stacked_seeds/ bridge_n6_fd13/ bridge_fd24/   # SHARP bridge
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
| `main` | Stable releases. Includes the full science stack + reports + tests + the React/FastAPI frontend. |
| `dev` | Integration branch. PR target for new work before promoting to `main`. |
| `p7_demo` | Original feature branch where the frontend was built. Merged into `dev` → `main`. Kept for archaeology. |
| `rq2`, `rq3`, `rq7` | Per-RQ feature branches — merged to `dev`, kept for archaeology. |
| `fedprox`, `fedrep`, `fedccfa` | RQ2 follow-up branches — merged to `dev`, kept for archaeology. |
| `multiseed` | Multi-seed campaigns, N=6 / FD002+FD004 generalization, and the SHARP bridge. Current integration branch. |

---

## Run the demo locally (FastAPI + React)

Two terminals from the repo root:

```powershell
# Terminal 1 — FastAPI backend on :8000
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Vite dev server on :5173 (proxies /api → :8000)
cd frontend
npm install     # first time only
npm run dev
```

Then open `http://localhost:5173`. The Vite dev server proxies every
`/api/*` call to the FastAPI server, so the figures, summary, and live
predict endpoints work transparently.

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
- **All metrics reported both combined AND per-subset.** Honest FL
  reporting requires per-client breakdowns, not just a global average.
- **Device-agnostic training.** Every loop resolves its device from
  `--device` / `FL_DEVICE` (else auto-CUDA), so identical code reproduces
  on CPU or GPU; the multi-seed campaigns ran on dual-T4 GPUs and were
  cross-checked against CPU smokes.
- **Deterministic, multi-seed reporting.** Headline numbers are mean ± std
  over 5 seeds `{42–46}`, so nothing rests on a single lucky run.

---

## License

MIT — see [`pyproject.toml`](pyproject.toml).

## Author

Chinmoy Mitra · [chinmoy17.github.io](https://chinmoy17.github.io) ·
[github.com/Chinmoy17](https://github.com/Chinmoy17)

