"""Pydantic request / response schemas for the backend API.

Kept in their own module so they can be imported by both the route handlers
and by the React-end type generator (e.g. ``openapi-typescript``) if we add
one later.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Liveness flag.")
    rq3_checkpoints_available: int = Field(
        ..., description="Number of trained checkpoints discoverable on disk."
    )


# ---------------------------------------------------------------------------
# Checkpoint catalogue
# ---------------------------------------------------------------------------
class CheckpointSummary(BaseModel):
    """One trained checkpoint the frontend can let the user pick from."""

    key: str = Field(..., description="Stable identifier — use in /api/predict.")
    display_name: str = Field(..., description="Human-friendly label for dropdowns.")
    training_subsets: list[str] = Field(
        ..., description="Which CMAPSS subsets this checkpoint was trained on."
    )
    test_engine_count: int = Field(
        ..., description="Number of test engines the checkpoint can predict on."
    )
    checkpoint_file: str = Field(
        ..., description="Repo-relative path to the .pt file (for debugging only)."
    )


class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointSummary]


class EngineSummary(BaseModel):
    """One test engine the user can ask the model to predict on."""

    engine_id: int
    subset: str = Field(
        ..., description="Origin CMAPSS subset for this test engine (FD001 / FD003 / …)."
    )
    true_rul: float = Field(..., description="Ground-truth RUL from the test set RUL file.")
    true_fault: int = Field(..., description="Binary fault label from the RUL threshold.")


class EnginesResponse(BaseModel):
    checkpoint_key: str
    engines: list[EngineSummary]


# ---------------------------------------------------------------------------
# Prediction + explanation
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    checkpoint_key: str = Field(..., description="One of the keys from /api/checkpoints.")
    engine_id: int = Field(..., description="Test engine to explain.")
    top_k: int = Field(
        default=5, ge=1, le=17,
        description="How many sensors to include in the top-K narrative.",
    )
    use_llm: bool = Field(
        default=False,
        description="If true and OPENAI_API_KEY is set, polish the narrative via GPT.",
    )


class PredictResponse(BaseModel):
    """Mirror of ``EngineExplanation.to_dict()`` + the request context."""

    checkpoint_key: str
    checkpoint_display_name: str
    engine_id: int
    subset: str
    true_rul: float
    true_fault: int
    # The actual explanation payload, kept as a dict so the frontend gets the
    # same shape ``scripts/run_rq3.py`` writes to disk under
    # ``results/rq3_explanations/explanations_*.json``.
    explanation: dict[str, Any]


# ---------------------------------------------------------------------------
# Summary passthrough
# ---------------------------------------------------------------------------
class SummaryResponse(BaseModel):
    """Wraps the contents of ``results/summary.json`` as-is."""

    summary: dict[str, Any]
