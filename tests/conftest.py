"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def data_dir(repo_root: Path) -> Path:
    """Path to the C-MAPSS dataset directory; skips the test if the data is missing."""
    p = repo_root / "Dataset" / "CMAPSS_NASA"
    if not (p / "train_FD001.txt").exists():
        pytest.skip(f"C-MAPSS dataset not available at {p}")
    return p
