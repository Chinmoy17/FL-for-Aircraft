# syntax=docker/dockerfile:1.6
#
# Single-image build for the FL Aircraft PHM demo (frontend + backend).
#
# Stage 1 (node):    builds the Vite + React 19 frontend → frontend/dist/
# Stage 2 (python):  installs the FastAPI backend with uv, copies the
#                    built frontend, and serves both from one uvicorn
#                    process on $PORT (defaults to 8000).
#
# Tuned for Azure App Service for Containers free tier (F1):
#   • Listens on $PORT (Azure sets WEBSITES_PORT=8000 → maps to 80/443).
#   • CPU-only torch wheel index already declared in pyproject.toml, so
#     the runtime layer stays small.
#   • Single process — no nginx, no supervisord — keeps the F1's 1 GB
#     RAM ceiling comfortable.
#
# Build locally:
#   docker build -t fl-aircraft-phm .
#   docker run --rm -p 8000:8000 fl-aircraft-phm
#   open http://localhost:8000
# ============================================================================

# ----------------------------------------------------------------------------
# Stage 1 — Frontend build
# ----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Layer-cache: install dependencies first, then copy sources.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund --loglevel=error

COPY frontend/ ./
RUN npm run build

# Output: /app/frontend/dist/


# ----------------------------------------------------------------------------
# Stage 2 — Python runtime
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Small set of OS packages we actually need at runtime.
#  - ca-certificates  for HTTPS to PyPI / pytorch index (during build only)
#  - libgomp1         OpenMP runtime used by numpy / scikit-learn / torch CPU
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv from its lightweight official image — avoids a stale pip cache
# layer and gives us the same lockfile-driven install we use locally.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /usr/local/bin/

# Skip uv's automatic Python download; use the slim image's interpreter.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first (long layer, cache-friendly). uv resolves from
# uv.lock so we get the exact versions tested in CI and locally.
#
# `--no-dev` skips the dev group (pytest etc.); `--group backend` adds
# fastapi / uvicorn / httpx2. Runtime deps (torch, numpy, captum, …) are
# always installed because they're in [project.dependencies].
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group backend --no-install-project

# Now copy the project source. Hatchling needs src/fl_aircraft/ to exist
# before we install the project itself (last layer).
COPY src/                 ./src/
COPY backend/             ./backend/
COPY results/             ./results/
COPY Dataset/             ./Dataset/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Install the project itself (no-deps — every transitive was just installed).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group backend --no-editable

# Make the venv binaries the default PATH so `uvicorn` resolves directly.
ENV PATH="/app/.venv/bin:$PATH"

# Azure App Service for Containers sets PORT for us; default to 8000 for
# local `docker run`. Bind to 0.0.0.0 so the container is reachable.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, os, sys; \
        sys.exit(0 if urllib.request.urlopen( \
            f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health' \
        ).status == 200 else 1)" || exit 1

# Single uvicorn process; one worker is correct for the F1 tier (1 GB RAM,
# shared-core CPU). The frontend is served statically by the same process.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
