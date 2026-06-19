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

export function LiveDemo() {
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

  // Load checkpoints on mount.
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

  // When the selected checkpoint changes, load its engine list.
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
    <>
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
    </>
  );
}

// ---------------------------------------------------------------------------
// Controls panel — two dropdowns + knobs + predict button
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
    <section className="panel">
      <h2>1 · Pick a trained model and a test engine</h2>
      <div className="controls">
        <div>
          <label htmlFor="ckpt">Checkpoint</label>
          <select
            id="ckpt"
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
        </div>
        <div>
          <label htmlFor="engine">Test engine</label>
          <select
            id="engine"
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
        </div>
        <div>
          <label htmlFor="topk">Top sensors</label>
          <input
            id="topk"
            type="number"
            min={1}
            max={17}
            value={props.topK}
            onChange={(e) =>
              props.onTopKChange(
                Math.min(17, Math.max(1, Number(e.target.value) || 5)),
              )
            }
          />
        </div>
        <div>
          <button
            onClick={props.onPredict}
            disabled={!props.canPredict}
            type="button"
          >
            {props.isPredicting ? "Predicting…" : "Predict + Explain"}
          </button>
          <div className="checkbox-row">
            <input
              id="llm"
              type="checkbox"
              checked={props.useLlm}
              onChange={(e) => props.onUseLlmChange(e.target.checked)}
            />
            <label htmlFor="llm" style={{ margin: 0 }}>
              Polish narrative with LLM (requires OPENAI_API_KEY)
            </label>
          </div>
        </div>
      </div>
      {props.checkpoints.kind === "err" && (
        <div className="error-box" style={{ marginTop: 12 }}>
          Failed to load checkpoints: {props.checkpoints.message}. Is the
          FastAPI backend running on port 8000?
        </div>
      )}
      {props.currentCheckpointName && (
        <p className="muted" style={{ marginTop: 12, marginBottom: 0 }}>
          Currently selected: <strong>{props.currentCheckpointName}</strong>
        </p>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Results panel — headline metrics, top sensors, fault mode, narrative
// ---------------------------------------------------------------------------
function ResultsPanel({
  prediction,
}: {
  prediction: LoadState<PredictResponse>;
}) {
  if (prediction.kind === "idle") {
    return (
      <section className="panel">
        <h2>2 · Prediction + explanation</h2>
        <p className="muted">
          Choose a checkpoint and a test engine above, then click{" "}
          <em>Predict + Explain</em>. The backend will run Integrated
          Gradients and return a sensor-level explanation in ~5 seconds.
        </p>
      </section>
    );
  }
  if (prediction.kind === "loading") {
    return (
      <section className="panel">
        <h2>2 · Prediction + explanation</h2>
        <p className="muted">
          <span className="spinner" />
          Running Integrated Gradients (this takes ~5 s on CPU)…
        </p>
      </section>
    );
  }
  if (prediction.kind === "err") {
    return (
      <section className="panel">
        <h2>2 · Prediction + explanation</h2>
        <div className="error-box">Prediction failed: {prediction.message}</div>
      </section>
    );
  }

  const res = prediction.value;
  const e = res.explanation;
  const error = e.predicted_rul - res.true_rul;
  const errorClass = Math.abs(error) <= 10 ? "good" : "bad";
  const maxAbsContribution = Math.max(
    ...e.top_sensors.map((s) => Math.abs(s.contribution)),
    1,
  );

  return (
    <section className="panel">
      <h2>2 · Prediction + explanation</h2>

      <div className="headline-row">
        <div className="metric-card">
          <div className="label">Predicted RUL</div>
          <div className={`value ${errorClass}`}>
            {e.predicted_rul.toFixed(1)}
          </div>
          <div className="delta">cycles</div>
        </div>
        <div className="metric-card">
          <div className="label">True RUL</div>
          <div className="value">{res.true_rul.toFixed(1)}</div>
          <div className="delta">cycles</div>
        </div>
        <div className="metric-card">
          <div className="label">Error</div>
          <div className={`value ${errorClass}`}>
            {error >= 0 ? "+" : ""}
            {error.toFixed(1)}
          </div>
          <div className="delta">predicted − true</div>
        </div>
        <div className="metric-card">
          <div className="label">Fault probability</div>
          <div className="value">{(e.fault_probability * 100).toFixed(1)}%</div>
          <div className="delta">
            ground truth: {res.true_fault === 1 ? "faulty" : "healthy"}
          </div>
        </div>
        <div className="metric-card">
          <div className="label">IG completeness gap</div>
          <div className="value">{e.convergence_delta.toFixed(3)}</div>
          <div className="delta">
            should be ≪ |predicted − baseline|
          </div>
        </div>
      </div>

      <h3 style={{ marginTop: 0, marginBottom: 12, fontSize: 15 }}>
        Top {e.top_sensors.length} contributing sensors
      </h3>
      <ul className="sensor-list">
        {e.top_sensors.map((s) => {
          const pct = (Math.abs(s.contribution) / maxAbsContribution) * 100;
          const sign = s.contribution >= 0 ? "pos" : "neg";
          return (
            <li key={s.column} className="sensor-row">
              <div className="col-name">
                <span className="short">{s.name}</span>
                <span className="raw">({s.column})</span>
              </div>
              <div className="col-desc">
                {s.description}
                {s.subsystem && (
                  <>
                    {" "}
                    · <em>{s.subsystem}</em>
                  </>
                )}
              </div>
              <div className="col-bar">
                <div
                  className={`contrib-bar ${sign}`}
                  style={{ width: `${pct}%`, minWidth: 4 }}
                />
                {s.contribution >= 0 ? "+" : ""}
                {s.contribution.toFixed(2)}
              </div>
            </li>
          );
        })}
      </ul>

      {e.inferred_fault_mode && (
        <div className="fault-mode-box">
          <div className="label">Inferred fault mode</div>
          <div className="name">{e.inferred_fault_mode.fault_mode}</div>
          <div className="recommendation">
            <strong>Recommended action: </strong>
            {e.inferred_fault_mode.recommended_action}
          </div>
          {e.inferred_fault_mode.affected_components.length > 0 && (
            <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 18 }}>
              {e.inferred_fault_mode.affected_components.map((c) => (
                <li key={c} style={{ fontSize: 13 }}>
                  {c}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="narrative-box">{e.narrative_llm ?? e.narrative}</div>
      {e.narrative_llm && (
        <p className="muted" style={{ marginTop: 8, marginBottom: 0 }}>
          ✦ Narrative polished by LLM. Underlying numbers are unchanged.
        </p>
      )}
      {e.notes.length > 0 && (
        <ul className="muted" style={{ marginTop: 12, paddingLeft: 18 }}>
          {e.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
