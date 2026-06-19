"""Plot helpers for RQ3 explanations.

Three figures per explained engine:

- :func:`plot_attribution_heatmap` — 30×17 attribution grid, colored by
  signed contribution (red = lowers RUL = "fault-like", blue = raises RUL
  = "healthy-like").
- :func:`plot_top_sensor_bar` — horizontal bar chart of the top-k sensor
  contributions, with sensor names + CMAPSS short names.
- :func:`plot_sensor_trajectory_with_attribution` — single sensor's raw
  (or normalized) trajectory across the window, with an attribution-tinted
  background highlighting which cycles drove the prediction.

These are deliberately small composable helpers: the CLI in
``scripts/run_rq3.py`` calls each one and stitches the panels together.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .attribution import AttributionResult
from .ontology import SENSOR_ONTOLOGY


def plot_attribution_heatmap(
    attr: AttributionResult,
    path: Path,
    *,
    title: str | None = None,
    annotate_top_sensors: int = 5,
) -> None:
    """Save a heatmap of ``attr.attribution`` (shape T × F) to ``path``."""
    a = attr.attribution
    feature_cols = list(attr.feature_cols)
    fig, ax = plt.subplots(figsize=(11, 6))
    vmax = float(np.abs(a).max()) or 1e-6
    # imshow expects (rows, cols); we want sensors on y-axis and cycles on x-axis.
    im = ax.imshow(
        a.T, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax,
        origin="lower",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(
        "contribution (cycles)" if attr.target_head == "rul" else "contribution (logit units)"
    )
    ax.set_xlabel("cycle within window")
    ax.set_yticks(range(len(feature_cols)))
    # Use CMAPSS short names where available; fall back to column name.
    pretty_labels = [
        f"{c}\n({SENSOR_ONTOLOGY[c].cmapss_name})"
        if c in SENSOR_ONTOLOGY else c
        for c in feature_cols
    ]
    ax.set_yticklabels(pretty_labels, fontsize=8)
    ax.set_xticks(range(0, attr.window_size, 5))
    ax.set_title(
        title or
        f"Attribution heatmap — predicted {attr.target_head}={attr.predicted_value:.2f}",
    )

    # Highlight top-K sensor rows with a thin border so the reader's eye lands there.
    if annotate_top_sensors > 0:
        top = attr.top_sensors(k=annotate_top_sensors)
        for col, _ in top:
            if col not in feature_cols:
                continue
            row = feature_cols.index(col)
            ax.axhline(row - 0.5, color="black", linewidth=0.3, alpha=0.4)
            ax.axhline(row + 0.5, color="black", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_top_sensor_bar(
    attr: AttributionResult,
    path: Path,
    *,
    top_k: int = 8,
    title: str | None = None,
) -> None:
    """Save a horizontal bar chart of the top-k sensor contributions."""
    top = attr.top_sensors(k=top_k)
    if not top:
        return
    cols, scores = zip(*top)
    pretty = [
        f"{SENSOR_ONTOLOGY[c].cmapss_name} ({c})" if c in SENSOR_ONTOLOGY else c
        for c in cols
    ]
    colors = ["crimson" if s < 0 else "steelblue" for s in scores]

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.45 * len(top) + 1.5)))
    y = np.arange(len(top))
    ax.barh(y, scores, color=colors, edgecolor="white")
    for i, s in enumerate(scores):
        ax.text(
            s, i, f"  {s:+.2f}  ",
            va="center", ha="left" if s >= 0 else "right",
            fontsize=9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(pretty)
    ax.invert_yaxis()  # top contributor at the top
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(
        "contribution to predicted RUL (cycles)"
        if attr.target_head == "rul"
        else "contribution to fault logit"
    )
    ax.set_title(title or f"Top-{top_k} contributing features")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sensor_trajectory_with_attribution(
    attr: AttributionResult,
    sensor_col: str,
    path: Path,
    *,
    title: str | None = None,
) -> None:
    """Plot a single sensor's normalized trajectory plus its attribution overlay."""
    if sensor_col not in attr.feature_cols:
        raise ValueError(
            f"Sensor {sensor_col!r} not in window features: "
            f"{list(attr.feature_cols)}"
        )
    f_idx = list(attr.feature_cols).index(sensor_col)
    trajectory = attr.window[:, f_idx]
    contributions = attr.attribution[:, f_idx]
    cycles = np.arange(attr.window_size)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Color the background by per-cycle contribution.
    vmax = float(np.abs(contributions).max()) or 1e-6
    cmap = plt.colormaps["RdBu"]
    for c in cycles:
        normed = 0.5 + 0.5 * (-contributions[c] / vmax)  # red where contribution lowers RUL
        ax.axvspan(c - 0.5, c + 0.5, color=cmap(normed), alpha=0.35, lw=0)

    ax.plot(cycles, trajectory, color="black", marker="o", linewidth=1.5, markersize=3)
    pretty = (
        f"{SENSOR_ONTOLOGY[sensor_col].cmapss_name} ({sensor_col}) — "
        f"{SENSOR_ONTOLOGY[sensor_col].description}"
        if sensor_col in SENSOR_ONTOLOGY else sensor_col
    )
    ax.set_xlabel("cycle within window")
    ax.set_ylabel("normalized reading (z-score)")
    ax.set_title(title or f"Trajectory + attribution: {pretty}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
