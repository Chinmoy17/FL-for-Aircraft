# `backend/` — FastAPI demo server

The science layer of this project produces JSON + PNG artefacts under
[`results/`](../results/). This package wraps those artefacts (and the
ability to run a fresh attribution on demand) into a small HTTP API the
React frontend consumes.

## Run locally

```powershell
# from the repo root, with the venv already set up via `uv sync --group backend`
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

Then visit:

- <http://localhost:8000/docs> — interactive Swagger UI (auto-generated).
- <http://localhost:8000/api/health> — sanity check.

## Endpoints

| Method + path | Purpose |
| --- | --- |
| `GET  /api/health` | `{"status":"ok"}` — liveness probe for Docker / Azure. |
| `GET  /api/summary` | The full `results/summary.json` — drives the "results" page of the frontend. |
| `GET  /api/checkpoints` | The 4 discoverable trained checkpoints with display names, training subsets, and test-engine counts. |
| `GET  /api/checkpoints/{key}/engines` | Engine ids + true RUL for each test engine the checkpoint can be queried with. |
| `POST /api/predict` | Body `{checkpoint_key, engine_id, top_k=5, use_llm=false}` → on-the-fly attribution + ontology-grounded narrative. Returns the same payload shape `EngineExplanation.to_dict()` produces in `scripts/run_rq3.py`. |
| `GET  /api/figures/{rel_path}` | Streams a PNG from `results/` (path-traversal guarded). The frontend uses this to embed P3/P5/P6/RQ2/RQ3 figures without copying them. |

## Design notes

- **No retraining at request time.** Every endpoint operates on already-saved
  checkpoints (`results/03_centralized/*.pt`, `results/05_fedavg/*.pt`,
  `results/06_non_iid/*.pt`).
- **Checkpoint catalogue lives in `src/fl_aircraft/explain/checkpoint_catalog.py`** so the
  RQ3 CLI and the backend stay in sync — exact same windows, exact same models.
- **Lazy model + bundle loading**, cached per-process in
  [`backend/services.py`](services.py). First request per checkpoint warms the
  cache; subsequent requests are ~5 s (dominated by Integrated Gradients).
- **LLM rewrite is opt-in per request** via the `use_llm` flag in
  `/api/predict` and falls back transparently to the deterministic narrative
  if `OPENAI_API_KEY` is unset.
- **CORS is permissive** for `http://localhost:5173` (Vite dev server default).
  Tighten before any production deployment.
