"""Smoke tests that verify the Python environment is wired up correctly.

These tests exist to catch broken installs early — before we spend time debugging
training code. They should run in under a second on any machine.
"""

from __future__ import annotations


def test_torch_imports_and_cpu_tensor_ops() -> None:
    """Torch must be installed and able to perform a CPU matmul."""
    import torch

    x = torch.randn(4, 8)
    y = torch.randn(8, 2)
    z = x @ y
    assert z.shape == (4, 2)
    # We deliberately target CPU for the entire baseline; just confirm the CPU device works.
    assert z.device.type == "cpu"


def test_core_scientific_stack_imports() -> None:
    """All runtime dependencies declared in pyproject.toml must import cleanly."""
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import tqdm  # noqa: F401
    import yaml  # noqa: F401


def test_package_importable() -> None:
    """The `fl_aircraft` package itself must be importable from the venv."""
    import fl_aircraft

    assert fl_aircraft.__version__
