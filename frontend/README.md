# `frontend/` — React live-demo dashboard

Vite + React 19 + TypeScript. Talks to the FastAPI backend
in [`../backend/`](../backend/). The original "contract" document for what
the dashboard should eventually visualise is preserved at
[`README.contract.md`](README.contract.md).

## Run locally

Two terminals.

**Terminal 1 — backend:**

```powershell
cd ..
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000 --reload
```

**Terminal 2 — frontend:**

```powershell
cd frontend
npm install        # one-off
npm run dev
```

Then open <http://localhost:5173>. Vite proxies `/api/*` to the FastAPI
server on port 8000, so no CORS plumbing is needed in dev.

## What's here so far

| File | Purpose |
| --- | --- |
| `package.json` | React 19 + Vite 7 + TypeScript 5.7. |
| `vite.config.ts` | Dev-server proxy for `/api/*` → `http://localhost:8000`. |
| `tsconfig*.json` | Strict TS config (composite project, `noUnusedLocals`, etc.). |
| `index.html` | Root HTML — single `#root` mount point. |
| `src/types.ts` | TypeScript mirrors of `backend/schemas.py`. |
| `src/api.ts` | Thin typed `fetch()` wrapper exposing 3 endpoints. |
| `src/main.tsx` | React root; mounts `<App/>` in `StrictMode`. |
| `src/App.tsx` | Page shell — header + `<LiveDemo/>`. |
| `src/components/LiveDemo.tsx` | The live demo. Two dropdowns (checkpoint, engine) + Predict button + rendered explanation. |
| `src/index.css` | Minimal dark-mode styling, no design framework. |

## What's *not* yet here

- Results page (consumes `/api/summary`, renders the per-phase tables and
  embeds existing P3 / P5 / P6 / RQ2 PNGs).
- RQ3 cross-model comparison view.
- Routing — currently single-page. Will add `react-router-dom` when the
  results page lands.
- Tailwind / shadcn — kept out for now. Pick up later only if a screenshot
  needs to look polished beyond what plain CSS gives us.

## Build for production

```powershell
npm run build
```

Outputs to `dist/`. The Docker image will serve this static bundle alongside
the FastAPI backend on the same Azure free-tier container.
