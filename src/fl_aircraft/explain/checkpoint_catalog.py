"""Checkpoint catalogue + per-engine test-window preparation.

Shared between ``scripts/run_rq3.py`` and the FastAPI backend. Both consumers
need the same answer to two questions:

    1. Which trained checkpoints exist on disk right now?
    2. For a given (checkpoint, engine_id), what is the *normalized* sliding
       window the model was trained to consume?

Keeping the answer in one module guarantees the demo backend predicts on
exactly the same windows the RQ3 figures were generated from.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from fl_aircraft.data import (
    CMAPSSConfig,
    MultiSubsetConfig,
    Normalizer,
    TrainTestBundle,
    bundle_from_config,
    load_multi_subset_bundle,
    make_test_windows,
)
from fl_aircraft.models import MultiTaskCNN, MultiTaskCNNConfig


@dataclass(frozen=True)
class CheckpointSpec:
    """One trained-model checkpoint that the demo / RQ3 will load."""

    key: str
    display_name: str
    checkpoint_path: Path
    bundle_kind: str  # "single" or "multi"
    subset: str | None = None
    subsets: tuple[str, ...] | None = None

    @property
    def training_subsets(self) -> tuple[str, ...]:
        """Tuple form regardless of single- vs multi-subset bundle kind."""
        if self.bundle_kind == "single":
            assert self.subset is not None
            return (self.subset,)
        assert self.subsets is not None
        return self.subsets


@dataclass
class WindowPair:
    """One test engine's window in the bundle's normalised feature space."""

    engine_id: int
    subset: str
    window_normalized: np.ndarray  # (T, F), z-scored
    rul_true: float
    fault_true: int


def candidate_checkpoints(repo_root: Path) -> list[CheckpointSpec]:
    """Return every checkpoint the project might expose to the demo.

    Caller is expected to filter to those whose ``checkpoint_path`` exists,
    so that the demo gracefully degrades if a phase has not been re-run yet.
    """
    return [
        CheckpointSpec(
            key="p3_centralized_fd001",
            display_name="P3 Centralized (FD001, IID)",
            checkpoint_path=repo_root / "results" / "03_centralized" / "best_model_fd001.pt",
            bundle_kind="single", subset="FD001",
        ),
        CheckpointSpec(
            key="p5_fedavg_iid_fd001",
            display_name="P5 FedAvg IID (FD001)",
            checkpoint_path=repo_root / "results" / "05_fedavg" / "best_global_model_fd001.pt",
            bundle_kind="single", subset="FD001",
        ),
        CheckpointSpec(
            key="p6_centralized_combined",
            display_name="P6 Centralized (FD001+FD003)",
            checkpoint_path=repo_root / "results" / "06_non_iid" / "best_centralized_fd001_fd003.pt",
            bundle_kind="multi", subsets=("FD001", "FD003"),
        ),
        CheckpointSpec(
            key="p6_fedavg_non_iid",
            display_name="P6 FedAvg Non-IID (FD001+FD003)",
            checkpoint_path=repo_root / "results" / "06_non_iid" / "best_fedavg_fd001_fd003.pt",
            bundle_kind="multi", subsets=("FD001", "FD003"),
        ),
        # ---- RQ2 follow-up: FedProx μ-sweep (Non-IID FD001+FD003) ----
        # μ=0.0 is omitted because it is bit-equivalent to vanilla FedAvg
        # (the P6 FedAvg row above). Exposing all three non-zero μ values
        # lets a reviewer compare how drift-control changes the attribution
        # of the same engine across the sweep.
        CheckpointSpec(
            key="rq2_fedprox_mu0_001",
            display_name="FedProx μ=0.001 (FD001+FD003)",
            checkpoint_path=repo_root / "results" / "rq2_fedprox" / "best_fedprox_state_mu_0.001.pt",
            bundle_kind="multi", subsets=("FD001", "FD003"),
        ),
        CheckpointSpec(
            key="rq2_fedprox_mu0_01",
            display_name="FedProx μ=0.01 (FD001+FD003)",
            checkpoint_path=repo_root / "results" / "rq2_fedprox" / "best_fedprox_state_mu_0.01.pt",
            bundle_kind="multi", subsets=("FD001", "FD003"),
        ),
        CheckpointSpec(
            key="rq2_fedprox_mu0_1",
            display_name="FedProx μ=0.1 (FD001+FD003)",
            checkpoint_path=repo_root / "results" / "rq2_fedprox" / "best_fedprox_state_mu_0.1.pt",
            bundle_kind="multi", subsets=("FD001", "FD003"),
        ),
    ]


def available_checkpoints(repo_root: Path) -> list[CheckpointSpec]:
    """Filter ``candidate_checkpoints`` to entries whose file exists on disk."""
    return [s for s in candidate_checkpoints(repo_root) if s.checkpoint_path.exists()]


def load_bundle(spec: CheckpointSpec, data_dir: Path) -> TrainTestBundle:
    """Build the bundle the checkpoint was trained on."""
    if spec.bundle_kind == "single":
        cfg = CMAPSSConfig(subset=spec.subset, data_dir=data_dir)
        return bundle_from_config(cfg)
    if spec.bundle_kind == "multi":
        cfg = MultiSubsetConfig(subsets=spec.subsets, data_dir=data_dir)
        return load_multi_subset_bundle(cfg)
    raise ValueError(f"Unknown bundle_kind {spec.bundle_kind!r}")


def load_model(spec: CheckpointSpec, bundle: TrainTestBundle) -> MultiTaskCNN:
    """Reconstruct the MultiTaskCNN from a saved state-dict + config payload."""
    state = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
    cfg_payload = state.get("config", {})
    n_features = cfg_payload.get("n_features", bundle.n_features)
    window_size = cfg_payload.get("window_size", bundle.window_size)
    model = MultiTaskCNN(
        MultiTaskCNNConfig(n_features=n_features, window_size=window_size)
    )
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def prepare_test_windows(bundle: TrainTestBundle) -> list[WindowPair]:
    """One ``WindowPair`` per test engine, in the bundle's z-score space."""
    normalizer = Normalizer.fit(bundle.train_df, bundle.feature_cols)
    arrays = make_test_windows(
        normalizer.transform(bundle.test_raw_df),
        bundle.test_rul,
        bundle.feature_cols,
        bundle.window_size,
        bundle.rul_cap,
        bundle.fault_threshold,
    )
    pairs: list[WindowPair] = []
    sorted_unit_ids = sorted(bundle.test_raw_df["unit_id"].unique())
    for i, uid in enumerate(sorted_unit_ids):
        if "source_subset" in bundle.test_raw_df.columns:
            origin = str(
                bundle.test_raw_df.loc[
                    bundle.test_raw_df["unit_id"] == uid, "source_subset"
                ].iloc[0]
            )
        else:
            origin = bundle.subsets[0]
        pairs.append(
            WindowPair(
                engine_id=int(uid),
                subset=origin,
                window_normalized=arrays.X[i],
                rul_true=float(arrays.y_rul[i]),
                fault_true=int(arrays.y_fault[i]),
            )
        )
    return pairs


def find_engine(pairs: Iterable[WindowPair], engine_id: int) -> WindowPair | None:
    """Linear lookup by ``engine_id`` (test sets are small — < 200 rows)."""
    for p in pairs:
        if p.engine_id == engine_id:
            return p
    return None


__all__ = [
    "CheckpointSpec",
    "WindowPair",
    "available_checkpoints",
    "candidate_checkpoints",
    "find_engine",
    "load_bundle",
    "load_model",
    "prepare_test_windows",
]
