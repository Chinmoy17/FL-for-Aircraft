"""Combine numeric attribution + ontology into engineer-facing explanations.

Three layers:

1. :func:`build_explanation` — fully deterministic. Takes an
   :class:`AttributionResult` and produces an :class:`EngineExplanation`
   dataclass with structured fields (top sensors, inferred fault mode,
   recommended action) plus a plain-English ``narrative`` string built by
   string templates.

2. :func:`rewrite_with_llm` — optional. If ``OPENAI_API_KEY`` is set in the
   environment, calls a chat-completion API with a strict prompt to rewrite
   the deterministic narrative in a more natural tone. The LLM is told
   explicitly to **not introduce new facts**. Falls back to the deterministic
   text on any error.

3. :func:`explain_window` — convenience top-level entry point that does
   attribution → build_explanation in one call. Skipped by default; the
   pipeline in ``scripts/run_rq3.py`` calls the pieces separately so we can
   control which checkpoint is being explained for each step.

The LLM is **never the source of truth**. The deterministic narrative is
what gets used in the React frontend's default view; the LLM-polished
version is a side-by-side option labelled clearly as such.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models import MultiTaskCNN
from .attribution import AttributionResult, attribute_window
from .ontology import (
    FaultModeRule,
    SensorMeta,
    lookup_sensor,
    match_fault_mode,
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------
@dataclass
class EngineExplanation:
    """Structured, engineer-facing explanation of one prediction.

    Attributes:
        predicted_rul: Model's RUL prediction in cycles, ≥ 0.
        fault_probability: Model's sigmoid(fault_logit), in [0, 1].
        target_head: Which head the attribution targets (``"rul"`` is the
            default; setting ``"fault"`` re-frames the explanation around
            failure probability instead of cycles-remaining).
        top_sensors: Up to ``k`` entries of ``(column, sensor_meta, signed_score)``,
            ordered by ``|score|`` desc.
        inferred_fault_mode: The :class:`FaultModeRule` matched by the top
            sensors, or ``None`` if no rule applies.
        narrative: Deterministic English explanation produced by string
            templates. **The scientific source of truth.**
        narrative_llm: Optional LLM-polished rewrite of ``narrative``.
            Present only when :func:`rewrite_with_llm` was called and the
            ``OPENAI_API_KEY`` environment variable was set.
        convergence_delta: Pass-through of the IG completeness check; values
            close to zero indicate a trustworthy attribution map.
    """

    predicted_rul: float
    fault_probability: float
    target_head: str
    top_sensors: list[tuple[str, SensorMeta, float]]
    inferred_fault_mode: FaultModeRule | None
    narrative: str
    convergence_delta: float
    narrative_llm: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Frontend-friendly dict (no torch / numpy / dataclasses)."""
        return {
            "predicted_rul": round(self.predicted_rul, 4),
            "fault_probability": round(self.fault_probability, 4),
            "target_head": self.target_head,
            "top_sensors": [
                {
                    "column": col,
                    "name": meta.cmapss_name,
                    "description": meta.description,
                    "subsystem": meta.subsystem,
                    "contribution": round(score, 4),
                }
                for col, meta, score in self.top_sensors
            ],
            "inferred_fault_mode": (
                {
                    "fault_mode": self.inferred_fault_mode.fault_mode,
                    "affected_components": list(self.inferred_fault_mode.affected_components),
                    "recommended_action": self.inferred_fault_mode.recommended_action,
                    "confidence_label": self.inferred_fault_mode.confidence_label,
                }
                if self.inferred_fault_mode else None
            ),
            "narrative": self.narrative,
            "narrative_llm": self.narrative_llm,
            "convergence_delta": round(self.convergence_delta, 6),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Deterministic builder
# ---------------------------------------------------------------------------
def build_explanation(
    attr: AttributionResult,
    *,
    predicted_rul: float,
    fault_probability: float,
    top_k: int = 5,
) -> EngineExplanation:
    """Turn an :class:`AttributionResult` into an :class:`EngineExplanation`.

    Args:
        attr: Output of :func:`attribute_window`.
        predicted_rul: Predicted RUL in cycles (typically the same number
            as ``attr.predicted_value`` when ``attr.target_head == "rul"``,
            but passed explicitly so callers can mix heads).
        fault_probability: Predicted fault probability (sigmoid of the
            fault logit).
        top_k: How many sensors to surface.
    """
    top_pairs = attr.top_sensors(k=top_k)
    enriched: list[tuple[str, SensorMeta, float]] = []
    for col, score in top_pairs:
        try:
            meta = lookup_sensor(col)
        except KeyError:
            continue
        enriched.append((col, meta, score))

    sensor_columns_only = [
        col for col, meta, _ in enriched
        if meta.subsystem != "Operational"
    ]
    inferred = match_fault_mode(sensor_columns_only)

    notes: list[str] = []
    if abs(attr.convergence_delta) > 0.5:
        notes.append(
            f"Integrated-Gradients convergence_delta = {attr.convergence_delta:+.2f}; "
            "attribution map may be less reliable for this engine."
        )
    if not sensor_columns_only:
        notes.append(
            "Top contributors are all operational settings, not degradation "
            "sensors. The prediction may be driven by inputs the ontology "
            "cannot interpret."
        )

    narrative = _render_narrative(
        predicted_rul=predicted_rul,
        fault_probability=fault_probability,
        target_head=attr.target_head,
        top_sensors=enriched,
        inferred=inferred,
    )
    return EngineExplanation(
        predicted_rul=float(predicted_rul),
        fault_probability=float(fault_probability),
        target_head=attr.target_head,
        top_sensors=enriched,
        inferred_fault_mode=inferred,
        narrative=narrative,
        convergence_delta=attr.convergence_delta,
        notes=notes,
    )


def _render_narrative(
    *,
    predicted_rul: float,
    fault_probability: float,
    target_head: str,
    top_sensors: list[tuple[str, SensorMeta, float]],
    inferred: FaultModeRule | None,
) -> str:
    """Deterministic template-driven English."""
    fault_pct = fault_probability * 100.0
    head_focus = "RUL regression" if target_head == "rul" else "fault classification"

    lines: list[str] = []
    lines.append(
        f"Predicted RUL: {predicted_rul:.1f} cycles · "
        f"Fault probability: {fault_pct:.1f}%  (explained head: {head_focus})"
    )

    if top_sensors:
        lines.append("")
        lines.append("Most influential inputs (top contributors):")
        for col, meta, score in top_sensors:
            direction = "raises RUL" if score > 0 else "lowers RUL"
            magnitude = abs(score)
            lines.append(
                f"  • {meta.cmapss_name:<8} ({col:>4})  {direction} by {magnitude:.2f} — "
                f"{meta.description}; subsystem: {meta.subsystem}"
            )

    if inferred is not None:
        lines.append("")
        lines.append(f"Inferred fault mode: {inferred.fault_mode}")
        lines.append("Affected components:")
        for comp in inferred.affected_components:
            lines.append(f"  - {comp}")
        lines.append(f"Recommended action: {inferred.recommended_action}")
    else:
        lines.append("")
        lines.append(
            "No fault-mode rule matched the top contributors. Treat the "
            "prediction as a soft estimate and cross-check against the raw "
            "sensor trajectory."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional LLM rewrite layer
# ---------------------------------------------------------------------------
LLM_PROMPT_TEMPLATE = """You are an aviation maintenance assistant.
Rewrite the following deterministic explanation for a ground engineer in a
clear, professional tone of voice. **You must not add facts that are not in
the input**: do not invent new sensors, components, root causes, or actions.
Keep the predicted RUL, fault probability, and recommended action numerically
unchanged. Output plain prose (no bullet lists, no markdown).

---

{narrative}

---"""


def rewrite_with_llm(narrative: str, *, model: str = "gpt-4o-mini") -> str | None:
    """Optionally rewrite ``narrative`` using an LLM.

    Activated only when ``OPENAI_API_KEY`` is set in the environment.
    Returns ``None`` if disabled, the model dependency is missing, or the
    API call fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        # Local import so the module has no hard dependency on `openai`.
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        prompt = LLM_PROMPT_TEMPLATE.format(narrative=narrative)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Any failure (missing package, network, auth, rate limit) falls back
        # silently — the deterministic narrative is still authoritative.
        return None


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------
def explain_window(
    model: MultiTaskCNN,
    window: np.ndarray,
    feature_cols: list[str],
    *,
    target_head: str = "rul",
    top_k: int = 5,
    use_llm: bool = False,
    n_steps: int = 50,
) -> tuple[AttributionResult, EngineExplanation]:
    """One-call attribution + explanation for a single window.

    Returns both objects so the caller can plot ``attr.attribution`` and
    log/display ``explanation.narrative``.
    """
    attr = attribute_window(
        model, window, feature_cols,
        target_head=target_head, n_steps=n_steps,  # type: ignore[arg-type]
    )

    # Predicted values from both heads, even though attribution targets one.
    import torch

    was_training = model.training
    model.eval()
    try:
        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            pred = model(x)
        predicted_rul = float(pred.rul.item())
        fault_prob = float(pred.fault_probs().item())
    finally:
        if was_training:
            model.train()

    explanation = build_explanation(
        attr,
        predicted_rul=predicted_rul,
        fault_probability=fault_prob,
        top_k=top_k,
    )
    if use_llm:
        polished = rewrite_with_llm(explanation.narrative)
        if polished:
            explanation.narrative_llm = polished
    return attr, explanation
