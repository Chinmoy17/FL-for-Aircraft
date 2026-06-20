"""C-MAPSS dataset loading, RUL/fault labelling, and per-feature z-score normalization.

The pipeline is built from small composable functions so the same primitives serve
the centralized, local-only, and federated training entry points:

    1. ``load_and_label_train`` — raw text -> labeled DataFrame (RUL + fault).
    2. ``Normalizer.fit``        — fit per-feature z-score on **one** client's data.
    3. ``Normalizer.transform``  — apply to any DataFrame with the same columns.
    4. (separately) windowing + partitioning live in their own modules.

Why per-client normalization rather than a single global one? In a real FL deployment
each airline already preprocesses its own data; sharing the global mean/std with the
server would itself leak statistics. The centralized baseline reuses these primitives
by treating all engines as one "client" — see ``scripts/check_data_pipeline.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    COLUMNS,
    CYCLE_COL,
    DEFAULT_FAULT_THRESHOLD,
    DEFAULT_RUL_CAP,
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    OP_SETTING_COLS,
    SUBSETS,
    UNIT_ID_COL,
    informative_sensors,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CMAPSSConfig:
    """Preprocessing configuration for one C-MAPSS subset.

    Attributes:
        subset: One of ``FD001`` / ``FD002`` / ``FD003`` / ``FD004``.
        data_dir: Path to the directory holding ``train_FD00X.txt`` etc.
        window_size: Sliding window length in cycles.
        stride: Sliding window stride in cycles (defaults to 1).
        rul_cap: Piecewise-linear cap applied to RUL labels.
        fault_threshold: A row's binary fault label is 1 iff its raw RUL <= this.
        include_op_settings: If ``True``, prepend the 3 operational settings to
            the feature vector. Defaults to ``True`` — the model can learn to
            ignore them for single-regime subsets, and they are informative for
            FD002/FD004.
    """

    subset: str
    data_dir: Path
    window_size: int = DEFAULT_WINDOW_SIZE
    stride: int = DEFAULT_STRIDE
    rul_cap: int = DEFAULT_RUL_CAP
    fault_threshold: int = DEFAULT_FAULT_THRESHOLD
    include_op_settings: bool = True

    def __post_init__(self) -> None:
        if self.subset not in SUBSETS:
            raise ValueError(
                f"Unknown subset {self.subset!r}; expected one of {SUBSETS}."
            )
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}.")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}.")
        if self.rul_cap < 1:
            raise ValueError(f"rul_cap must be >= 1, got {self.rul_cap}.")
        if self.fault_threshold < 0:
            raise ValueError(
                f"fault_threshold must be >= 0, got {self.fault_threshold}."
            )

    @property
    def feature_cols(self) -> list[str]:
        """Columns fed into the model (op settings + informative sensors)."""
        sensors = informative_sensors(self.subset)
        return OP_SETTING_COLS + sensors if self.include_op_settings else sensors

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


# ---------------------------------------------------------------------------
# Raw I/O
# ---------------------------------------------------------------------------
def load_raw(subset: str, split: str, data_dir: Path) -> pd.DataFrame:
    """Read a raw ``{split}_{subset}.txt`` table into a named-column DataFrame."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}.")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
    fp = Path(data_dir) / f"{split}_{subset}.txt"
    if not fp.exists():
        raise FileNotFoundError(f"CMAPSS file not found: {fp}")
    return pd.read_csv(fp, sep=r"\s+", header=None, names=COLUMNS)


def load_test_rul(subset: str, data_dir: Path) -> np.ndarray:
    """Per-engine ground-truth RUL for a test subset (``RUL_FD00X.txt``)."""
    fp = Path(data_dir) / f"RUL_{subset}.txt"
    if not fp.exists():
        raise FileNotFoundError(f"CMAPSS RUL file not found: {fp}")
    return pd.read_csv(fp, header=None).iloc[:, 0].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
def compute_rul_labels(
    train_df: pd.DataFrame, cap: int = DEFAULT_RUL_CAP
) -> pd.DataFrame:
    """Add ``RUL_raw`` and ``RUL_capped`` columns.

    For training data the RUL at cycle ``c`` of engine ``e`` is
    ``max_cycle(e) - c``. The piecewise-linear cap concentrates loss on the
    informative end-of-life region (notebook section 8).
    """
    df = train_df.copy()
    max_cycle = df.groupby(UNIT_ID_COL)[CYCLE_COL].transform("max")
    df["RUL_raw"] = (max_cycle - df[CYCLE_COL]).astype(np.float32)
    df["RUL_capped"] = df["RUL_raw"].clip(upper=cap).astype(np.float32)
    return df


def compute_fault_labels(
    df: pd.DataFrame, threshold: int = DEFAULT_FAULT_THRESHOLD
) -> pd.DataFrame:
    """Add a binary ``fault`` column = 1 iff ``RUL_raw <= threshold``.

    Requires :func:`compute_rul_labels` to have been called first.
    """
    if "RUL_raw" not in df.columns:
        raise ValueError("compute_rul_labels(df) must be called before compute_fault_labels(df).")
    df = df.copy()
    df["fault"] = (df["RUL_raw"] <= threshold).astype(np.int8)
    return df


def load_and_label_train(config: CMAPSSConfig) -> pd.DataFrame:
    """Load the training table and attach RUL / fault columns. No normalization."""
    df = load_raw(config.subset, "train", config.data_dir)
    df = compute_rul_labels(df, config.rul_cap)
    df = compute_fault_labels(df, config.fault_threshold)
    return df


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Normalizer:
    """Per-feature z-score normalizer.

    Designed to be fit on **one** client's data and stored alongside that
    client's model — never shared across clients in the FL setting.
    """

    feature_cols: tuple[str, ...]
    mean: np.ndarray  # shape (n_features,), float32
    std: np.ndarray  # shape (n_features,), float32 — clipped to >= 1e-7

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_cols: list[str]) -> "Normalizer":
        """Fit per-feature z-score; near-constant columns get std=1 (no scaling)."""
        x = df[feature_cols].to_numpy(dtype=np.float64)
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        # Avoid div-by-zero for constant columns (e.g. op_settings on FD001).
        std = np.where(std < 1e-7, np.float32(1.0), std)
        return cls(feature_cols=tuple(feature_cols), mean=mean, std=std)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with ``feature_cols`` z-scored.

        The subtraction is performed in float64 to avoid catastrophic cancellation
        when raw sensor values (e.g. ~8000 K turbine outlet temperature) are close
        in magnitude to the fitted mean. The result is cast back to float32 — the
        precision the rest of the pipeline (PyTorch) uses anyway.
        """
        cols = list(self.feature_cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Normalizer columns missing from DataFrame: {missing}")
        df = df.copy()
        x = df[cols].to_numpy(dtype=np.float64)
        x = (x - self.mean.astype(np.float64)) / self.std.astype(np.float64)
        df[cols] = x.astype(np.float32)
        return df
