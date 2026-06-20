import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 01 — Data pipeline sanity.
 *
 * The phase that wires raw CMAPSS → 4-client federated training
 * shape, with a single figure showing the per-client fault imbalance.
 */
export function ExperimentDataPage() {
  return (
    <ExperimentLayout
      phaseId="Phase 01 · Data"
      title="Data pipeline sanity"
      lede={
        <>
          Build the 4-client partition we will use for the IID baseline
          phases, verify the sliding-window totals match the analytical
          formula, and confirm the per-client fault imbalance stays
          balanced. This is the &quot;does the FL framing wire up
          correctly?&quot; check before any model is trained.
        </>
      }
      metaRow={
        <>
          <span>FD001 · 4 clients · seed 42 · stratified-by-lifetime</span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">
            scripts/check_data_pipeline.py
          </span>
        </>
      }
      prev={{ id: "00", title: "EDA", to: "/experiments/00-eda" }}
      next={{ id: "02", title: "Smoke run", to: "/experiments/02-smoke" }}
    >
      <ExperimentSection
        eyebrow="At a glance"
        title="Per-client breakdown across the stratified-by-lifetime split."
        intro={
          <p>
            Each client gets 25 of FD001's 100 training engines, chosen so
            that lifetime histograms match across clients. The resulting
            per-client fault-positive rate sits within{" "}
            <strong>0.13 percentage points</strong> of the global rate —
            deliberately near-IID, because this baseline isolates
            &quot;does FedAvg converge?&quot; from &quot;does FedAvg
            handle Non-IID?&quot;. The interesting Non-IID setting arrives
            later, at Phase 06.
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "100 / 4", label: "FD001 train engines / clients" },
            { value: "17,731", label: "Total sliding windows" },
            { value: "30 × 17", label: "Window shape (cycles × features)" },
            { value: "17.5 %", label: "Mean per-client fault rate" },
            { value: "0.13 pp", label: "Inter-client fault-rate spread" },
            { value: "0 / 100 %", label: "NaNs / shape check" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection eyebrow="Figure" title="Per-client fault imbalance">
        <ExplainedFigure
          artifactPath="results/01_data/client_fault_imbalance_fd001.png"
          caption="Fault-positive rate per client on the stratified-by-lifetime split"
          takeaway="All four clients sit at ~17.5% fault rate. This is the IID baseline; Non-IID structure arrives in Phase 06."
          explanation={
            <>
              <p>
                Four bars, one per client, each showing the fraction of
                training windows whose true RUL is ≤ 30 cycles. All four
                land within{" "}
                <strong>0.13 percentage points</strong> of each other (17.43%
                to 17.56%). This is the dataset-side analogue of
                &quot;controlled experiment design&quot; — we removed every
                cross-client heterogeneity dimension we could so that any
                later FedAvg convergence problems can be cleanly attributed
                to either local-step drift or the federated protocol itself,
                not to data skew.
              </p>
              <p>
                The test-set fault rate is higher than train (≈25% vs ≈17%)
                because CMAPSS test trajectories are deliberately truncated
                near end-of-life by the benchmark authors. That asymmetry is
                consistent across all 4 clients (the test set is common),
                so it does not bias the per-client comparison.
              </p>
            </>
          }
        />
      </ExperimentSection>
    </ExperimentLayout>
  );
}
