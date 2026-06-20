"""FastAPI app for the demo backend.

Run locally::

    .\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend import services
from backend.schemas import (
    CheckpointSummary,
    CheckpointsResponse,
    EngineSummary,
    EnginesResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SummaryResponse,
)

logger = logging.getLogger("backend.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model cache on startup so the first /api/predict call is fast."""
    logger.info("Warming model cache from available checkpoints…")
    try:
        counts = services.warm_cache()
        for key, n in counts.items():
            logger.info("  · %s: %d test engines loaded", key, n)
    except Exception:  # noqa: BLE001
        # Don't crash the server if a checkpoint is missing — endpoints still
        # work for those that exist, and `/api/health` will report the count.
        logger.exception("warm_cache failed (server will still start)")
    yield


app = FastAPI(
    title="FL-Aircraft demo backend",
    version="0.1.0",
    description=(
        "Demo backend for the Federated Learning Aircraft Engine PHM project. "
        "Wraps the trained P3/P5/P6 checkpoints and the RQ3 attribution + "
        "ontology pipeline into a small HTTP API for the React frontend."
    ),
    lifespan=lifespan,
)

# CORS — allow the Vite dev server (5173) and the typical Vite preview port (4173).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        rq3_checkpoints_available=len(services.list_specs()),
    )


@app.get("/api/summary", response_model=SummaryResponse, tags=["results"])
def summary() -> SummaryResponse:
    try:
        return SummaryResponse(summary=services.get_summary())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/checkpoints", response_model=CheckpointsResponse, tags=["demo"])
def checkpoints() -> CheckpointsResponse:
    specs = services.list_specs()
    items: list[CheckpointSummary] = []
    for spec in specs:
        engines = services.list_engines(spec)
        items.append(
            CheckpointSummary(
                key=spec.key,
                display_name=spec.display_name,
                training_subsets=list(spec.training_subsets),
                test_engine_count=len(engines),
                checkpoint_file=str(
                    spec.checkpoint_path.relative_to(REPO_ROOT)
                ).replace("\\", "/"),
            )
        )
    return CheckpointsResponse(checkpoints=items)


@app.get(
    "/api/checkpoints/{key}/engines",
    response_model=EnginesResponse,
    tags=["demo"],
)
def engines(key: str) -> EnginesResponse:
    try:
        spec = services.get_spec(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return EnginesResponse(
        checkpoint_key=key,
        engines=[EngineSummary(**e) for e in services.list_engines(spec)],
    )


@app.post("/api/predict", response_model=PredictResponse, tags=["demo"])
def predict(req: PredictRequest) -> PredictResponse:
    try:
        spec = services.get_spec(req.checkpoint_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        payload = services.predict(
            spec,
            req.engine_id,
            top_k=req.top_k,
            use_llm=req.use_llm,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PredictResponse(**payload)


@app.get("/api/figures/{rel_path:path}", tags=["results"])
def figure(rel_path: str) -> FileResponse:
    """Stream a PNG / JPG / SVG from ``results/`` (path-traversal guarded)."""
    try:
        path = services.resolve_figure_path(rel_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return FileResponse(path)
