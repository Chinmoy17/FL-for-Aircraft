import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { PhaseDetail } from "../components/PhaseDetail";
import type { PhaseMetrics, ProjectSummary } from "../summaryTypes";

type LoadState<T> =
  | { kind: "loading" }
  | { kind: "ok"; value: T }
  | { kind: "err"; message: string };

// Hard-coded display order — matches the order in results.md and the
// chronological order of the experiments.
const PHASE_ORDER: string[] = [
  "00_eda",
  "01_data",
  "02_smoke",
  "03_centralized",
  "04_local_only",
  "05_fedavg",
  "06_non_iid",
];

// RQ2 and RQ3 each have their own dedicated long-form story page; they
// are excluded from the generic results list so the dedicated framing is
// not in tension with the metrics-dump treatment.
const EXCLUDED_FROM_RESULTS = new Set<string>([
  "rq2_imbalance_aware",
  "rq3_explanations",
]);

export function ResultsPage() {
  const [summary, setSummary] = useState<LoadState<ProjectSummary>>({
    kind: "loading",
  });
  const [selectedPhaseId, setSelectedPhaseId] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    api
      .getSummary()
      .then((res) => {
        if (cancelled) return;
        setSummary({ kind: "ok", value: res.summary });
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setSummary({ kind: "err", message: err.message });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const orderedPhases: PhaseMetrics[] = useMemo(() => {
    if (summary.kind !== "ok") return [];
    const phases = summary.value.phases;
    const visible: PhaseMetrics[] = [];
    for (const id of PHASE_ORDER) {
      const p = phases[id];
      if (p && !EXCLUDED_FROM_RESULTS.has(id)) visible.push(p);
    }
    // Add anything not in PHASE_ORDER at the end, alphabetically.
    const ordered = new Set(PHASE_ORDER);
    const extras = Object.values(phases)
      .filter(
        (p) =>
          !ordered.has(p.phase_id) && !EXCLUDED_FROM_RESULTS.has(p.phase_id),
      )
      .sort((a, b) => a.phase_id.localeCompare(b.phase_id));
    return [...visible, ...extras];
  }, [summary]);

  // Pick a default phase once loaded.
  useEffect(() => {
    if (!selectedPhaseId && orderedPhases.length > 0) {
      setSelectedPhaseId(orderedPhases[0].phase_id);
    }
  }, [orderedPhases, selectedPhaseId]);

  if (summary.kind === "loading") {
    return (
      <div className="max-w-6xl mx-auto px-6 py-12 text-text-dim">
        <span className="spinner" /> Loading results…
      </div>
    );
  }
  if (summary.kind === "err") {
    return (
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="rounded-md border border-bad bg-bad/10 px-4 py-3 text-sm text-bad">
          Could not load /api/summary: {summary.message}. Is the FastAPI
          backend running?
        </div>
      </div>
    );
  }

  const selectedPhase =
    orderedPhases.find((p) => p.phase_id === selectedPhaseId) ??
    orderedPhases[0];

  return (
    <div className="w-full px-6 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight text-text">
          Results
        </h1>
        <p className="mt-1 text-sm text-text-dim">
          {summary.value.project} ·{" "}
          <span className="font-mono-num">{summary.value.git_commit}</span> ·
          generated {new Date(summary.value.generated_at).toLocaleString()}
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-[220px_1fr] gap-6">
        <aside>
          <nav aria-label="Phases" className="space-y-0.5">
            {orderedPhases.map((p) => {
              const isActive = p.phase_id === selectedPhaseId;
              return (
                <button
                  key={p.phase_id}
                  type="button"
                  onClick={() => setSelectedPhaseId(p.phase_id)}
                  className={[
                    "w-full text-left px-3 py-2 rounded-md text-sm transition-colors",
                    isActive
                      ? "bg-bg-subtle text-text font-medium border-l-2 border-accent pl-[10px]"
                      : "text-text-dim hover:text-text hover:bg-bg-subtle/60",
                  ].join(" ")}
                >
                  <div className="font-mono-num text-[11px] text-text-muted leading-tight">
                    {p.phase_id}
                  </div>
                  <div className="leading-snug">
                    {shortenPhaseName(p.phase_name)}
                  </div>
                </button>
              );
            })}
          </nav>

          <div className="mt-6 rounded-md border border-border bg-bg-subtle p-3 text-xs text-text-dim leading-relaxed space-y-3">
            <div>
              <p className="font-semibold text-text mb-1">
                Research questions have their own pages
              </p>
              <p>
                RQ2 and RQ3 are presented as long-form stories with
                dedicated framing for their findings.
              </p>
            </div>
            <ul className="space-y-1">
              <li>
                <a href="/rq2-story" className="text-accent">
                  → RQ2 story (negative finding)
                </a>
              </li>
              <li>
                <a href="/rq3-story" className="text-accent">
                  → RQ3 story (cross-model interpretability)
                </a>
              </li>
            </ul>
          </div>
        </aside>

        <div>
          {selectedPhase ? (
            <PhaseDetail phase={selectedPhase} />
          ) : (
            <p className="text-text-dim">No phases to show.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function shortenPhaseName(name: string): string {
  // "Phase 3 — Centralized baseline (FD001, 50 epochs)" → "Centralized baseline"
  // Cuts the leading "Phase N —" and the trailing parenthetical.
  return name
    .replace(/^Phase\s+\d+\s*[-—]\s*/i, "")
    .replace(/\s*\([^)]*\)\s*$/, "")
    .trim();
}
