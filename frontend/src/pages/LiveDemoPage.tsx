import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type {
  CheckpointSummary,
  EngineSummary,
  PredictResponse,
} from "../types";

type LoadState<T> =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; value: T }
  | { kind: "err"; message: string };

const initialIdle: LoadState<never> = { kind: "idle" };

export function LiveDemoPage() {
  const [checkpoints, setCheckpoints] = useState<LoadState<CheckpointSummary[]>>(
    { kind: "loading" },
  );
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
  const [engines, setEngines] = useState<LoadState<EngineSummary[]>>(initialIdle);
  const [selectedEngineId, setSelectedEngineId] = useState<number | null>(null);
  const [topK, setTopK] = useState<number>(5);
  const [useLlm, setUseLlm] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<LoadState<PredictResponse>>(
    initialIdle,
  );

  useEffect(() => {
    let cancelled = false;
    api
      .listCheckpoints()
      .then((res) => {
        if (cancelled) return;
        setCheckpoints({ kind: "ok", value: res.checkpoints });
        if (res.checkpoints.length > 0) {
          setSelectedCheckpoint(res.checkpoints[0].key);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setCheckpoints({ kind: "err", message: err.message });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedCheckpoint) return;
    let cancelled = false;
    setEngines({ kind: "loading" });
    setSelectedEngineId(null);
    setPrediction(initialIdle);
    api
      .listEngines(selectedCheckpoint)
      .then((res) => {
        if (cancelled) return;
        setEngines({ kind: "ok", value: res.engines });
        if (res.engines.length > 0) {
          setSelectedEngineId(res.engines[0].engine_id);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setEngines({ kind: "err", message: err.message });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedCheckpoint]);

  const checkpointDisplayName = useMemo(() => {
    if (checkpoints.kind !== "ok") return null;
    return (
      checkpoints.value.find((c) => c.key === selectedCheckpoint)
        ?.display_name ?? null
    );
  }, [checkpoints, selectedCheckpoint]);

  const canPredict =
    !!selectedCheckpoint &&
    selectedEngineId !== null &&
    prediction.kind !== "loading";

  const runPredict = async () => {
    if (selectedEngineId === null) return;
    setPrediction({ kind: "loading" });
    try {
      const res = await api.predict({
        checkpoint_key: selectedCheckpoint,
        engine_id: selectedEngineId,
        top_k: topK,
        use_llm: useLlm,
      });
      setPrediction({ kind: "ok", value: res });
    } catch (err) {
      setPrediction({
        kind: "err",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  };

  return (
    <article className="w-full">
      <header className="px-10 md:px-16 lg:px-24 pt-16 pb-10 border-b border-border">
        <div className="eyebrow">Interactive · ~5 seconds per prediction</div>
        <h1 className="font-display text-[44px] sm:text-[52px] leading-[1.05] tracking-tight text-text mt-4 max-w-[24ch]">
          Live <em className="not-italic text-accent">prediction</em> +
          attribution
        </h1>
        <p className="mt-6 text-lg text-text-dim">
          Pick a trained checkpoint and a test engine. The backend runs
          Integrated Gradients on demand and returns a sensor-level
          explanation grounded in the project&apos;s 17-entry maintenance
          ontology — the same pipeline described on the{" "}
          <a href="/rq3-story" className="text-accent">
            RQ3 story page
          </a>
          , exposed as an interactive API.
        </p>
        <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-text-muted">
          <span>4 checkpoints · 200-engine test set · CPU-only inference</span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">FastAPI /api/predict</span>
        </div>
      </header>

      <div className="px-10 md:px-16 lg:px-24 py-10">
        <ControlsPanel
          checkpoints={checkpoints}
          selectedCheckpoint={selectedCheckpoint}
          onCheckpointChange={setSelectedCheckpoint}
          engines={engines}
          selectedEngineId={selectedEngineId}
          onEngineChange={setSelectedEngineId}
          topK={topK}
          onTopKChange={setTopK}
          useLlm={useLlm}
          onUseLlmChange={setUseLlm}
          canPredict={canPredict}
          isPredicting={prediction.kind === "loading"}
          onPredict={runPredict}
          currentCheckpointName={checkpointDisplayName}
        />
        <ResultsPanel prediction={prediction} />
      </div>
    </article>
  );
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------
type ControlsPanelProps = {
  checkpoints: LoadState<CheckpointSummary[]>;
  selectedCheckpoint: string;
  onCheckpointChange: (key: string) => void;
  engines: LoadState<EngineSummary[]>;
  selectedEngineId: number | null;
  onEngineChange: (id: number) => void;
  topK: number;
  onTopKChange: (n: number) => void;
  useLlm: boolean;
  onUseLlmChange: (b: boolean) => void;
  canPredict: boolean;
  isPredicting: boolean;
  onPredict: () => void;
  currentCheckpointName: string | null;
};

function ControlsPanel(props: ControlsPanelProps) {
  return (
    <section className="rounded-lg border border-border bg-bg-subtle p-6 mb-6">
      <h2 className="text-base font-semibold text-text mb-4">
        1 · Pick a trained model and a test engine
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 items-end">
        <Field id="ckpt" label="Checkpoint">
          <select
            id="ckpt"
            className={selectClass}
            value={props.selectedCheckpoint}
            disabled={props.checkpoints.kind !== "ok"}
            onChange={(e) => props.onCheckpointChange(e.target.value)}
          >
            {props.checkpoints.kind === "ok" &&
              props.checkpoints.value.map((c) => (
                <option key={c.key} value={c.key}>
                  {c.display_name}
                </option>
              ))}
            {props.checkpoints.kind === "loading" && (
              <option>Loading…</option>
            )}
            {props.checkpoints.kind === "err" && (
              <option>(failed to load)</option>
            )}
          </select>
        </Field>
        <Field id="engine" label="Test engine">
          <select
            id="engine"
            className={selectClass}
            value={props.selectedEngineId ?? ""}
            disabled={props.engines.kind !== "ok"}
            onChange={(e) => props.onEngineChange(Number(e.target.value))}
          >
            {props.engines.kind === "ok" &&
              props.engines.value.map((eng) => (
                <option key={eng.engine_id} value={eng.engine_id}>
                  #{eng.engine_id} — {eng.subset} (true RUL ={" "}
                  {eng.true_rul.toFixed(0)})
                </option>
              ))}
            {props.engines.kind === "loading" && (
              <option>Loading engines…</option>
            )}
            {props.engines.kind === "err" && (
              <option>(failed to load)</option>
            )}
          </select>
        </Field>
        <Field id="topk" label="Top sensors">
          <input
            id="topk"
            type="number"
            min={1}
            max={17}
            className={selectClass}
            value={props.topK}
            onChange={(e) =>
              props.onTopKChange(
                Math.min(17, Math.max(1, Number(e.target.value) || 5)),
              )
            }
          />
        </Field>
        <div className="flex flex-col gap-2">
          <button
            onClick={props.onPredict}
            disabled={!props.canPredict}
            type="button"
            className="bg-accent text-[#fafaf7] font-semibold rounded-md px-4 py-2 text-sm hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-2 focus-visible:ring-offset-bg-subtle transition"
          >
            {props.isPredicting ? "Predicting…" : "Predict + Explain"}
          </button>
          <label className="flex items-center gap-2 text-xs text-text-dim">
            <input
              type="checkbox"
              checked={props.useLlm}
              onChange={(e) => props.onUseLlmChange(e.target.checked)}
              className="accent-accent"
            />
            Polish narrative with LLM (requires OPENAI_API_KEY)
          </label>
        </div>
      </div>
      {props.checkpoints.kind === "err" && (
        <ErrorBox className="mt-4">
          Failed to load checkpoints: {props.checkpoints.message}. Is the
          FastAPI backend running on port 8000?
        </ErrorBox>
      )}
      {props.currentCheckpointName && (
        <p className="mt-3 text-xs text-text-dim">
          Currently selected:{" "}
          <span className="text-text font-medium">
            {props.currentCheckpointName}
          </span>
        </p>
      )}
    </section>
  );
}

const selectClass =
  "w-full bg-bg text-text border border-border rounded-md px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:border-accent/60 disabled:opacity-60";

function Field({
  id,
  label,
  children,
}: {
  id: string;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label
        htmlFor={id}
        className="block text-xs text-text-dim mb-1.5 font-medium"
      >
        {label}
      </label>
      {children}
    </div>
  );
}

function ErrorBox({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={`rounded-md border border-bad bg-bad/10 px-4 py-3 text-sm text-bad ${className}`}
    >
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Results panel
// ---------------------------------------------------------------------------
function ResultsPanel({
  prediction,
}: {
  prediction: LoadState<PredictResponse>;
}) {
  if (prediction.kind === "idle") {
    return (
      <SectionShell title="2 · Prediction + explanation">
        <p className="text-sm text-text-dim">
          Choose a checkpoint and a test engine above, then click{" "}
          <em>Predict + Explain</em>. The backend will run Integrated
          Gradients and return a sensor-level explanation in ~5 seconds.
        </p>
      </SectionShell>
    );
  }
  if (prediction.kind === "loading") {
    return (
      <SectionShell title="2 · Prediction + explanation">
        <p className="text-sm text-text-dim">
          <span className="spinner" />
          Running Integrated Gradients (this takes ~5 s on CPU)…
        </p>
      </SectionShell>
    );
  }
  if (prediction.kind === "err") {
    return (
      <SectionShell title="2 · Prediction + explanation">
        <ErrorBox>Prediction failed: {prediction.message}</ErrorBox>
      </SectionShell>
    );
  }

  const res = prediction.value;
  const e = res.explanation;
  const error = e.predicted_rul - res.true_rul;
  const isGood = Math.abs(error) <= 10;
  const maxAbsContribution = Math.max(
    ...e.top_sensors.map((s) => Math.abs(s.contribution)),
    1,
  );

  return (
    <SectionShell
      title="2 · Prediction + explanation"
      subtitle={`${res.checkpoint_display_name} · engine #${res.engine_id} (${res.subset})`}
    >
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
        <MetricCard
          label="Predicted RUL"
          value={e.predicted_rul.toFixed(1)}
          unit="cycles"
          tone={isGood ? "good" : "bad"}
        />
        <MetricCard
          label="True RUL"
          value={res.true_rul.toFixed(1)}
          unit="cycles"
        />
        <MetricCard
          label="Error"
          value={`${error >= 0 ? "+" : ""}${error.toFixed(1)}`}
          unit="predicted − true"
          tone={isGood ? "good" : "bad"}
        />
        <MetricCard
          label="Fault probability"
          value={`${(e.fault_probability * 100).toFixed(1)}%`}
          unit={`ground truth: ${res.true_fault === 1 ? "faulty" : "healthy"}`}
        />
        <MetricCard
          label="IG completeness gap"
          value={e.convergence_delta.toFixed(3)}
          unit="should be ≪ |predicted − baseline|"
        />
      </div>

      <h3 className="text-sm font-semibold text-text mb-3">
        Top {e.top_sensors.length} contributing sensors
      </h3>
      <ul className="divide-y divide-border rounded-md border border-border bg-bg overflow-hidden">
        {e.top_sensors.map((s) => {
          const pct = (Math.abs(s.contribution) / maxAbsContribution) * 100;
          const isPos = s.contribution >= 0;
          return (
            <li
              key={s.column}
              className="grid grid-cols-[120px_1fr_140px] gap-3 items-center px-4 py-2.5"
            >
              <div className="font-mono-num text-sm">
                <span className="text-accent font-semibold">{s.name}</span>
                <span className="text-text-dim ml-1.5">({s.column})</span>
              </div>
              <div className="text-sm text-text-dim">
                {s.description}
                {s.subsystem && (
                  <>
                    {" · "}
                    <em>{s.subsystem}</em>
                  </>
                )}
              </div>
              <div className="flex items-center justify-end gap-2 font-mono-num text-sm font-semibold">
                <div
                  className={`h-1.5 rounded-full ${
                    isPos ? "bg-good" : "bg-bad"
                  }`}
                  style={{ width: `${Math.max(pct, 6)}%` }}
                />
                <span className={isPos ? "text-good" : "text-bad"}>
                  {isPos ? "+" : ""}
                  {s.contribution.toFixed(2)}
                </span>
              </div>
            </li>
          );
        })}
      </ul>

      {e.inferred_fault_mode && (
        <div className="mt-5 rounded-r-md border-l-4 border-accent bg-bg px-4 py-3">
          <div className="text-xs uppercase tracking-wider text-text-dim">
            Inferred fault mode
          </div>
          <div className="text-base font-semibold mt-1">
            {e.inferred_fault_mode.fault_mode}
          </div>
          <div className="text-sm mt-2">
            <span className="font-semibold">Recommended action: </span>
            {e.inferred_fault_mode.recommended_action}
          </div>
          {e.inferred_fault_mode.affected_components.length > 0 && (
            <ul className="mt-2 list-disc list-inside text-xs text-text-dim space-y-0.5">
              {e.inferred_fault_mode.affected_components.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      <pre className="mt-5 rounded-md border border-border bg-bg p-4 text-xs font-mono-num whitespace-pre-wrap text-text">
        {e.narrative_llm ?? e.narrative}
      </pre>
      {e.narrative_llm && (
        <p className="mt-2 text-xs text-text-dim">
          ✦ Narrative polished by LLM. Underlying numbers are unchanged.
        </p>
      )}
      {e.notes.length > 0 && (
        <ul className="mt-3 text-xs text-text-dim list-disc list-inside space-y-0.5">
          {e.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}
    </SectionShell>
  );
}

function SectionShell({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border bg-bg-subtle p-6">
      <div className="mb-4">
        <h2 className="text-base font-semibold text-text">{title}</h2>
        {subtitle && (
          <p className="text-xs text-text-dim mt-0.5">{subtitle}</p>
        )}
      </div>
      {children}
    </section>
  );
}

function MetricCard({
  label,
  value,
  unit,
  tone,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: "good" | "bad";
}) {
  const toneClass =
    tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : "text-text";
  return (
    <div className="rounded-md border border-border bg-bg px-3 py-2.5">
      <div className="text-[10.5px] uppercase tracking-wider text-text-dim font-medium">
        {label}
      </div>
      <div className={`mt-0.5 text-2xl font-semibold font-mono-num ${toneClass}`}>
        {value}
      </div>
      {unit && (
        <div className="text-[11px] text-text-muted mt-0.5">{unit}</div>
      )}
    </div>
  );
}
