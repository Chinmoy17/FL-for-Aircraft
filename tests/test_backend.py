"""Smoke tests for the FastAPI backend.

Uses FastAPI's TestClient (which spins up an in-process app) so the tests
don't need a real network port. The heavy lifting — loading 4 checkpoints
and running Integrated Gradients — is exercised end-to-end exactly once
to keep the suite fast.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Import inside the fixture so collection doesn't pay the FastAPI startup
    # cost when only other tests are being run.
    from backend.main import app

    with TestClient(app) as c:
        yield c


def test_health_lists_checkpoints(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    # We have 4 checkpoints committed under results/ — should always be there.
    assert payload["rq3_checkpoints_available"] >= 1


def test_summary_passthrough(client: TestClient) -> None:
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    summary = resp.json()["summary"]
    # Spot-check the shape produced by scripts/build_results_summary.py.
    assert "phases" in summary
    assert "rq3_explanations" in summary["phases"]


def test_checkpoints_includes_p3(client: TestClient) -> None:
    resp = client.get("/api/checkpoints")
    assert resp.status_code == 200
    keys = {c["key"] for c in resp.json()["checkpoints"]}
    assert "p3_centralized_fd001" in keys


def test_engines_for_p3_has_100(client: TestClient) -> None:
    resp = client.get("/api/checkpoints/p3_centralized_fd001/engines")
    assert resp.status_code == 200
    engines = resp.json()["engines"]
    assert len(engines) == 100
    # First engine in FD001 test set is unit 1.
    first = engines[0]
    assert first["engine_id"] == 1
    assert first["subset"] == "FD001"


def test_engines_unknown_checkpoint_returns_404(client: TestClient) -> None:
    resp = client.get("/api/checkpoints/does_not_exist/engines")
    assert resp.status_code == 404


def test_predict_round_trip(client: TestClient) -> None:
    resp = client.post(
        "/api/predict",
        json={
            "checkpoint_key": "p3_centralized_fd001",
            "engine_id": 25,
            "top_k": 5,
            "use_llm": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["checkpoint_key"] == "p3_centralized_fd001"
    assert body["engine_id"] == 25
    assert body["subset"] == "FD001"
    explanation = body["explanation"]
    assert "predicted_rul" in explanation
    assert "top_sensors" in explanation
    assert len(explanation["top_sensors"]) == 5
    assert "narrative" in explanation
    assert isinstance(explanation["narrative"], str)
    # LLM was not requested, so narrative_llm must be None.
    assert explanation.get("narrative_llm") is None


def test_predict_unknown_engine_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/api/predict",
        json={
            "checkpoint_key": "p3_centralized_fd001",
            "engine_id": 9999,
            "top_k": 5,
            "use_llm": False,
        },
    )
    assert resp.status_code == 404


def test_figure_traversal_blocked(client: TestClient) -> None:
    # Naive path-traversal attempt must be rejected.
    resp = client.get("/api/figures/..%2F..%2Fpyproject.toml")
    assert resp.status_code in (403, 404)


def test_figure_serves_real_png(client: TestClient) -> None:
    # The cross-model comparison engine 25 PNG is committed to the repo.
    resp = client.get("/api/figures/rq3_explanations/cross_model_comparison_engine_25.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 1000  # not an empty placeholder
