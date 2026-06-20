import { useState } from "react";
import { FigureGrid } from "./Figure";
import { MetricCard } from "./MetricCard";
import { MetricsTable, rowsFromRecord } from "./MetricsTable";
import type { PhaseMetrics } from "../summaryTypes";

/**
 * Renders one phase entry from results/summary.json with a consistent layout:
 *
 *   1. Interpretation TL;DR (one paragraph)
 *   2. Headline metric tiles auto-picked from test.{rul,fault} when present
 *   3. Per-subset breakdown card (when test entries exist per subset)
 *   4. Figures grid (artifacts map, lazy-loaded)
 *   5. Collapsible: configuration (config block) + timing + per-client raw
 */
export function PhaseDetail({ phase }: { phase: PhaseMetrics }) {
  const headlineCards = pickHeadlineMetrics(phase);
  const configRows = rowsFromRecord(phase.config);
  const timingRows = rowsFromRecord(phase.timing);
  const summaryRows = rowsFromRecord(phase.summary);
  const perSubset = phase.per_subset
    ? Object.entries(phase.per_subset)
    : [];

  return (
    <article className="space-y-6">
      <header>
        <p className="text-xs uppercase tracking-wider text-text-dim font-medium">
          {phase.phase_id}
        </p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-text">
          {phase.phase_name}
        </h1>
      </header>

      {phase.interpretation && (
        <p className="text-[15px] leading-relaxed text-text max-w-4xl">
          {phase.interpretation}
        </p>
      )}

      {headlineCards.length > 0 && (
        <section>
          <h2 className="text-xs uppercase tracking-wider text-text-dim font-medium mb-3">
            Headline metrics
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {headlineCards.map((m) => (
              <MetricCard
                key={m.label}
                label={m.label}
                value={m.value}
                unit={m.unit}
                tone={m.tone}
              />
            ))}
          </div>
        </section>
      )}

      {summaryRows.length > 0 && (
        <details className="group rounded-md border border-border bg-bg-subtle">
          <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium text-text-dim hover:text-text">
            Summary block ({summaryRows.length} keys)
          </summary>
          <div className="px-4 pb-4">
            <MetricsTable rows={summaryRows} />
          </div>
        </details>
      )}

      {perSubset.length > 0 && (
        <section>
          <h2 className="text-xs uppercase tracking-wider text-text-dim font-medium mb-3">
            Per-subset breakdown
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {perSubset.map(([subset, block]) => {
              const rul = (block as { rul?: Record<string, number> }).rul;
              const fault = (block as { fault?: Record<string, number> })
                .fault;
              return (
                <div key={subset} className="space-y-2">
                  <h3 className="text-sm font-semibold text-text">{subset}</h3>
                  {rul && (
                    <MetricsTable
                      caption="RUL"
                      rows={rowsFromRecord(rul)}
                    />
                  )}
                  {fault && (
                    <MetricsTable
                      caption="Fault"
                      rows={rowsFromRecord(fault)}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {phase.artifacts && Object.keys(phase.artifacts).length > 0 && (
        <section>
          <h2 className="text-xs uppercase tracking-wider text-text-dim font-medium mb-3">
            Figures
          </h2>
          <FigureGrid artifacts={phase.artifacts} />
        </section>
      )}

      {(configRows.length > 0 || timingRows.length > 0) && (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {configRows.length > 0 && (
            <CollapsibleTable title="Configuration" rows={configRows} />
          )}
          {timingRows.length > 0 && (
            <CollapsibleTable title="Timing" rows={timingRows} />
          )}
        </section>
      )}
    </article>
  );
}

function CollapsibleTable({
  title,
  rows,
}: {
  title: string;
  rows: Array<[string, unknown]>;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-md border border-border bg-bg-subtle overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((b) => !b)}
        className="w-full text-left px-4 py-2.5 text-sm font-medium text-text-dim hover:text-text flex items-center justify-between"
      >
        <span>
          {title}{" "}
          <span className="text-text-muted text-xs">({rows.length})</span>
        </span>
        <span aria-hidden className="text-text-muted text-xs">
          {open ? "▾" : "▸"}
        </span>
      </button>
      {open && (
        <div className="px-4 pb-3">
          <MetricsTable rows={rows} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Headline metric picker — knows how to pull the most-quoted numbers from
// each phase's metrics.json shape.
// ---------------------------------------------------------------------------
type Card = {
  label: string;
  value: number | string;
  unit?: string;
  tone?: "neutral" | "good" | "bad" | "accent";
};

function pickHeadlineMetrics(phase: PhaseMetrics): Card[] {
  const cards: Card[] = [];
  const test = phase.test ?? null;
  if (test?.rul) {
    const rmse = (test.rul as Record<string, number>).rmse;
    const nasa = (test.rul as Record<string, number>).nasa_score;
    const mae = (test.rul as Record<string, number>).mae;
    if (typeof rmse === "number") {
      cards.push({ label: "RMSE", value: rmse, unit: "cycles" });
    }
    if (typeof nasa === "number") {
      cards.push({
        label: "NASA score",
        value: nasa,
        unit: "lower is better",
      });
    }
    if (typeof mae === "number") {
      cards.push({ label: "MAE", value: mae, unit: "cycles" });
    }
  }
  if (test?.fault) {
    const auprc = (test.fault as Record<string, number>).auprc;
    const f1 = (test.fault as Record<string, number>).f1;
    if (typeof auprc === "number") {
      cards.push({ label: "AUPRC", value: auprc });
    }
    if (typeof f1 === "number") {
      cards.push({ label: "F1", value: f1 });
    }
  }
  // Fall back to picking notable summary keys for phases without a test
  // block (00_eda, 01_data).
  if (cards.length === 0 && phase.summary) {
    for (const [k, v] of Object.entries(phase.summary).slice(0, 4)) {
      if (typeof v === "number" || typeof v === "string") {
        cards.push({ label: humanize(k), value: v });
      }
    }
  }
  return cards;
}

function humanize(k: string): string {
  return k
    .replace(/_/g, " ")
    .replace(/^./, (c) => c.toUpperCase());
}
