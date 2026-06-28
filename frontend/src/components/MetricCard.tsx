/**
 * Generic metric tile — used by the long-form RQ story pages.
 *
 * Sits inside a `bg-bg-subtle` card (so it must be `bg-bg` per UI-craft's
 * "don't nest bg-surface in bg-surface" rule). Renders a label, a big
 * tabular-numerics value, an optional unit / delta, and an optional
 * tone (good / bad / accent).
 */
type Tone = "neutral" | "good" | "bad" | "accent";

const toneClass: Record<Tone, string> = {
  neutral: "text-text",
  good: "text-good",
  bad: "text-bad",
  accent: "text-accent",
};

export function MetricCard({
  label,
  value,
  unit,
  tone = "neutral",
}: {
  label: string;
  value: string | number;
  unit?: string;
  tone?: Tone;
}) {
  const renderedValue =
    typeof value === "number"
      ? Number.isFinite(value)
        ? value.toLocaleString(undefined, { maximumFractionDigits: 4 })
        : "—"
      : value;
  return (
    <div className="rounded-md border border-border bg-bg px-3 py-2.5">
      <div className="text-[10.5px] uppercase tracking-wider text-text-dim font-medium">
        {label}
      </div>
      <div
        className={`mt-0.5 text-2xl font-semibold font-mono-num ${toneClass[tone]}`}
      >
        {renderedValue}
      </div>
      {unit && <div className="text-[11px] text-text-muted mt-0.5">{unit}</div>}
    </div>
  );
}
