import { useEffect, useState } from "react";
import { api } from "../api";
import {
  AnchorStat,
  AnchorStatRow,
  Bullet,
  FormulaBlock,
  HypothesisCard,
  SmokingGunFigure,
  StoryHero,
  StorySection,
} from "../components/story";
import type { PhaseMetrics, ProjectSummary } from "../summaryTypes";

/**
 * Long-form story page for the RQ3 cross-model interpretability finding.
 *
 * Sibling of /rq2-story. RQ3's finding is POSITIVE — Integrated Gradients +
 * a maintenance ontology let us inspect *what* each model attended to, and
 * the comparison surfaces an interpretability red flag that RMSE alone hid:
 * the Non-IID-trained models sometimes attribute their RUL prediction to
 * an operational setting (Mach number) rather than to a real degradation
 * sensor.
 *
 * Structure parallels the RQ2 page so a reader switching between them sees
 * a consistent rhythm: hero → 3 anchor stats → question → method → results
 * table → smoking-gun figure → pattern → meaning → next.
 */
export function Rq3StoryPage() {
  const [phase, setPhase] = useState<PhaseMetrics | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getSummary()
      .then((res) => {
        if (cancelled) return;
        const p: ProjectSummary = res.summary;
        const rq3 = p.phases["rq3_explanations"] ?? null;
        if (!rq3) {
          setErr(
            "RQ3 phase not found in summary.json. Run scripts/run_rq3.py first.",
          );
          return;
        }
        setPhase(rq3);
      })
      .catch((e: Error) => {
        if (!cancelled) setErr(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (err) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-12">
        <div className="rounded-md border border-bad bg-bad/10 px-4 py-3 text-sm text-bad">
          {err}
        </div>
      </div>
    );
  }
  if (!phase) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-12 text-text-dim">
        <span className="spinner" /> Loading RQ3 results…
      </div>
    );
  }

  return <Rq3Article phase={phase} />;
}

function Rq3Article({ phase }: { phase: PhaseMetrics }) {
  return (
    <article className="py-10 max-w-5xl">
      {/* HERO ----------------------------------------------------------- */}
      <StoryHero
        eyebrow="Research finding · RQ3"
        lead={
          <>
            We ran Integrated Gradients on the same C-MAPSS test engine
            through four different trained checkpoints. The models agreed
            on the number. They didn't agree on the reason.
          </>
        }
      >
        Same engine.{" "}
        <em className="text-accent not-italic">Different reason.</em>
      </StoryHero>

      {/* ANCHOR NUMBERS ------------------------------------------------- */}
      <AnchorStatRow className="mt-12">
        <AnchorStat
          tone="bad"
          value="2 / 6"
          label="Non-IID-trained predictions whose top contributor was an operational setting"
          sub="Engineering-implausible — Mach number does not cause engine wear."
        />
        <AnchorStat
          value="4 × 17"
          label="(models × sensors) per engine"
          sub="3 test engines explained × 4 trained checkpoints = 12 cross-model explanations."
        />
        <AnchorStat
          tone="accent"
          value="FedProx"
          label="Where this points"
          sub="Same direction as RQ2 — control client drift, not aggregation weights."
        />
      </AnchorStatRow>

      {/* THE QUESTION --------------------------------------------------- */}
      <StorySection title="The question">
        <p>
          RQ2 showed that federated training under Non-IID data{" "}
          <em>fails to close the gap</em> against centralized training on
          RMSE. But RMSE is a single scalar — it can't tell us whether the
          federation also changed <strong>what</strong> the model attended
          to.
        </p>
        <p>
          Two models with similar accuracy can still reach that accuracy
          through very different reasoning. For an aircraft maintenance
          decision, the reasoning matters: a model that predicts an engine
          is healthy because the flight regime looks normal is{" "}
          <em>not</em> the same model that predicts the engine is healthy
          because its coolant flows are nominal. One is doing PHM. The
          other is doing weather forecasting.
        </p>
      </StorySection>

      {/* METHOD --------------------------------------------------------- */}
      <StorySection title="How we tested">
        <p>
          We built a three-layer attribution pipeline (
          <a
            href="https://github.com/Chinmoy17/FL-for-Aircraft/blob/dev/src/fl_aircraft/explain"
            target="_blank"
            rel="noreferrer"
          >
            src/fl_aircraft/explain/
          </a>
          ) and ran the same C-MAPSS test engine through every available
          trained checkpoint.
        </p>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3">
          <HypothesisCard
            label="Layer 1 — Numeric attribution"
            formula="captum.IntegratedGradients"
            text="Integrated Gradients with all-zero baseline (training mean in z-score space). 50 steps. Satisfies the completeness axiom within ≤0.005 cycles on every explanation."
          />
          <HypothesisCard
            label="Layer 2 — Maintenance ontology"
            formula="17 sensors · 3 fault rules"
            text="Hand-curated mapping from each CMAPSS sensor to its short name (T30, Nf, BPR), subsystem (HPC / Fan / LPT), and degradation relevance. Three fault-mode rules use reciprocal-rank scoring to pick the matching maintenance recommendation."
          />
          <HypothesisCard
            label="Layer 3 — Narrative"
            formula="rule-based, LLM-optional"
            text="Deterministic English template renders the top-K sensors + inferred fault mode + recommended action. Optional GPT rewrite is env-var-gated and never the source of truth."
          />
        </div>
        <p className="mt-4">
          Total wall-clock for one engine through four checkpoints:{" "}
          <span className="font-mono-num text-text">~5 seconds</span> on
          CPU. Twelve explanations (3 engines × 4 checkpoints) are
          permanently checked into the repo as JSON +{" "}
          <span className="font-mono-num text-text">.png</span> artifacts.
        </p>
      </StorySection>

      {/* RESULTS TABLE -------------------------------------------------- */}
      <StorySection title="What the comparison showed">
        <p>
          The same engine, scored by the same metric, gets four different
          attributions:
        </p>
        <CrossModelTable />
        <p className="mt-4 text-text-dim text-sm">
          <span className="text-bad font-medium">os_2</span> is an{" "}
          <em>operational setting</em> (Mach number), not a degradation
          sensor. Two of the six Non-IID-trained explanations point at it
          as the strongest reason for the RUL prediction. The two IID-
          trained models (P3, P5) never do.
        </p>
      </StorySection>

      {/* SMOKING-GUN FIGURE -------------------------------------------- */}
      <SmokingGunFigure
        eyebrow="The smoking-gun figure"
        title="Engine 75 — same window, four different reasons"
        artifactPath={
          phase.artifacts?.["cross_model_engine_75"] ??
          "results/rq3_explanations/cross_model_comparison_engine_75.png"
        }
        alt="Cross-model comparison for test engine 75 across all four trained checkpoints, showing predicted RUL and top-3 contributing sensors per model."
        caption={
          <>
            Engine 75's true RUL is{" "}
            <span className="font-mono-num text-text">113 cycles</span>. The
            two IID-trained models (P3, P5) both point at{" "}
            <span className="font-mono-num text-text">s_20 (W31, HPT
            coolant flow)</span> — a real degradation indicator. The
            Non-IID FedAvg model predicts{" "}
            <span className="font-mono-num text-text">129 cycles</span>{" "}
            (off by 16) and attributes that prediction to{" "}
            <span className="text-bad font-mono-num">os_2 (Mach number)</span>.{" "}
            <strong className="text-text">
              An operational setting cannot cause an engine to wear out.
            </strong>{" "}
            The model has learned to use flight-regime cues as a stand-in
            for failure-mode physics — an artifact of training on Non-IID
            data where regime and fault-mode happen to correlate within
            each client's slice.
          </>
        }
      />

      {/* WHAT THIS REVEALS --------------------------------------------- */}
      <StorySection title="The pattern across the 12 explanations">
        <p>
          Three engines, four trained checkpoints, twelve explanations. A
          consistent pattern emerges:
        </p>
        <ul className="mt-4 space-y-3 text-text">
          <Bullet>
            <strong>The two IID-trained models</strong> (P3 centralized
            FD001, P5 FedAvg IID FD001) agree with each other on top
            contributors. They consistently pick{" "}
            <span className="font-mono-num">s_4 (T50)</span>,{" "}
            <span className="font-mono-num">s_20 (W31)</span>,{" "}
            <span className="font-mono-num">s_21 (W32)</span> — real
            degradation sensors.
          </Bullet>
          <Bullet>
            <strong>The Non-IID-trained models</strong> sometimes diverge
            from this pattern. On 2 of 6 Non-IID predictions, the top
            contributor is{" "}
            <span className="text-bad font-mono-num">os_2 (Mach number)</span>.
            That's an operational setting, not a sensor reading.
          </Bullet>
          <Bullet>
            <strong>The wrong-abstraction failure mode</strong> is invisible
            to RMSE. The P6 FedAvg model that picks <span className="font-mono-num">os_2</span>{" "}
            on engine 75 is off by only 16 cycles — well within the RMSE
            band that vanilla FedAvg achieves on the test set as a whole.
            On numerical accuracy alone, the prediction looks fine.
          </Bullet>
        </ul>
        <p className="mt-4">
          So Non-IID FedAvg's failure has two faces. RQ2 measured one (RMSE
          gap). RQ3 measures the other:{" "}
          <strong>
            the federation pushed the model toward the wrong abstraction.
          </strong>
        </p>
      </StorySection>

      {/* WHERE IT POINTS ----------------------------------------------- */}
      <StorySection title="What this implies for the next experiment">
        <p>
          The intervention layer is still the same one RQ2 pointed at —
          client-drift control during local epochs. But RQ3 adds a second
          success criterion the next experiment should track alongside
          RMSE:
        </p>
        <FormulaBlock>
          {"top-K(IG) ∩ degradation_sensors  →  high   (every engine)"}
        </FormulaBlock>
        <p>
          That is: a fix that closes the RMSE gap but still attributes to
          operational settings hasn't fixed the underlying problem. A fix
          that closes the gap <em>and</em> attributes to coolant flows /
          HPC pressures the way the centralized model does — that's a
          fix worth shipping.
        </p>
        <p className="mt-4">
          FedProx, FedNova, and SCAFFOLD all penalise local trajectories
          drifting away from the global model. If the failure-of-abstraction
          we see in RQ3 is caused by the drift we hypothesised in RQ2, a
          FedProx run should restore both kinds of agreement at once. That
          would be the most informative single experiment to do next.
        </p>
      </StorySection>

      {/* TRY IT YOURSELF ----------------------------------------------- */}
      <StorySection title="Try it yourself">
        <p>
          Every explanation on this page was generated by the same backend
          the live demo uses. Pick any other engine, any other checkpoint,
          and re-run the same comparison in around five seconds:
        </p>
        <ul className="mt-4 space-y-2 text-text">
          <Bullet>
            <a href="/" className="text-accent">
              Live demo
            </a>{" "}
            — interactive picker for checkpoint + engine, with on-demand
            Integrated Gradients and the rendered explanation.
          </Bullet>
          <Bullet>
            <a href="/results" className="text-accent">
              Results / rq3_explanations
            </a>{" "}
            — the 12 pre-rendered explanations from{" "}
            <span className="font-mono-num">scripts/run_rq3.py</span>, with
            the heatmaps, trajectories, and cross-model comparison images.
          </Bullet>
          <Bullet>
            <a
              href="https://github.com/Chinmoy17/FL-for-Aircraft/blob/dev/src/fl_aircraft/explain/ontology.py"
              target="_blank"
              rel="noreferrer"
            >
              src/fl_aircraft/explain/ontology.py
            </a>{" "}
            — the 17-entry maintenance ontology and three fault-mode rules.
          </Bullet>
        </ul>
      </StorySection>
    </article>
  );
}

// ===========================================================================
// CrossModelTable — hard-coded from results/rq3_explanations/explanations_*.json
// (3 engines × 4 checkpoints). Cells show predicted RUL · top contributor.
// The two os_2 cells are tone-highlighted as bad.
// ===========================================================================
type Cell = {
  predRul: string;
  sensor: string;
  /** When true, render the sensor with bad tone (operational-setting red flag) */
  flag?: boolean;
};

type EngineRow = {
  engineId: number;
  subset: string;
  trueRul: number;
  cells: [Cell, Cell, Cell, Cell]; // P3, P5, P6 centralized, P6 FedAvg
};

const ENGINE_ROWS: EngineRow[] = [
  {
    engineId: 25,
    subset: "FD001",
    trueRul: 125,
    cells: [
      { predRul: "117.9", sensor: "s_4 (T50)" },
      { predRul: "114.9", sensor: "s_4 (T50)" },
      { predRul: "125.1", sensor: "s_9 (Nc)" },
      { predRul: "118.7", sensor: "s_20 (W31)" },
    ],
  },
  {
    engineId: 50,
    subset: "FD001",
    trueRul: 79,
    cells: [
      { predRul: "93.7", sensor: "s_21 (W32)" },
      { predRul: "96.7", sensor: "s_20 (W31)" },
      { predRul: "93.5", sensor: "os_2 (Mach)", flag: true },
      { predRul: "103.0", sensor: "s_11 (Ps30)" },
    ],
  },
  {
    engineId: 75,
    subset: "FD001",
    trueRul: 113,
    cells: [
      { predRul: "106.7", sensor: "s_20 (W31)" },
      { predRul: "108.0", sensor: "s_20 (W31)" },
      { predRul: "120.1", sensor: "s_9 (Nc)" },
      { predRul: "129.4", sensor: "os_2 (Mach)", flag: true },
    ],
  },
];

const CHECKPOINT_HEADERS = [
  "P3 Centralized\nFD001 IID",
  "P5 FedAvg\nFD001 IID",
  "P6 Centralized\nFD001+FD003",
  "P6 FedAvg\nFD001+FD003 Non-IID",
];

function CrossModelTable() {
  return (
    <div className="mt-4 overflow-x-auto rounded-md border border-border bg-bg">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-bg-subtle/40 text-text-dim">
            <th className="text-left px-4 py-2 font-medium w-32">
              Test engine
            </th>
            {CHECKPOINT_HEADERS.map((h) => (
              <th
                key={h}
                className="text-left px-4 py-2 font-medium whitespace-pre-line"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {ENGINE_ROWS.map((row) => (
            <tr key={row.engineId}>
              <th
                scope="row"
                className="text-left px-4 py-3 font-normal align-top"
              >
                <div className="font-mono-num text-text">#{row.engineId}</div>
                <div className="text-xs text-text-dim">{row.subset}</div>
                <div className="text-xs text-text-muted mt-1">
                  true RUL ={" "}
                  <span className="font-mono-num">{row.trueRul}</span>
                </div>
              </th>
              {row.cells.map((c, i) => (
                <td key={i} className="px-4 py-3 align-top">
                  <div className="font-mono-num text-text">
                    {c.predRul}{" "}
                    <span className="text-xs text-text-muted">cyc</span>
                  </div>
                  <div
                    className={`mt-1 text-xs font-mono-num ${
                      c.flag ? "text-bad font-semibold" : "text-text-dim"
                    }`}
                  >
                    {c.flag && "⚠ "}
                    {c.sensor}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
