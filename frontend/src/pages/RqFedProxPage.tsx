import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * RQ2 follow-up: FedProx.
 *
 * Adds a proximal term to each client's local loss:
 *
 *   L_local = L_task + (μ/2) · ‖W_local − W_global‖²
 *
 * The μ-sweep tests four values (0, 0.001, 0.01, 0.1) on the same
 * FD001+FD003 Non-IID partition as P6 / RQ2. The headline finding:
 * +6% gap closed and a more balanced per-subset profile.
 */
export function RqFedProxPage() {
  return (
    <ExperimentLayout
      phaseId="RQ2 follow-up · FedProx"
      title="FedProx — proximal regularisation"
      lede={
        <>
          RQ2 ruled out the aggregation layer. FedProx tests the{" "}
          <strong>client-optimisation layer</strong>: add a proximal
          penalty <em>(μ/2)·‖W<sub>local</sub> − W<sub>global</sub>‖²</em>{" "}
          to each client&apos;s loss so local SGD can&apos;t drift too far
          from the round-start global weights. Drift control, not weighting.
          The expected result was a moderate gain; the actual result was a{" "}
          <strong>small but positive +6%</strong> gap closed.
        </>
      }
      metaRow={
        <>
          <span>FD001 + FD003 · 4 clients · 50 rounds · μ ∈ {`{`}0, 0.001, 0.01, 0.1{`}`}</span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">scripts/run_fedprox.py</span>
        </>
      }
      prev={{ id: "RQ2", title: "Aggregation negative finding", to: "/rq2-story" }}
      next={{
        id: "FedRep",
        title: "Per-client heads",
        to: "/rq2-followups/fedrep",
      }}
    >
      <ExperimentSection
        eyebrow="At a glance"
        title="Four μ values, four runs, one small positive shift."
        intro={
          <p>
            Best result lands at <strong>μ = 0.1</strong> with combined
            RMSE 17.70 — 0.25 RMSE better than vanilla FedAvg&apos;s 17.95
            and 6.0% of the way from FedAvg to centralized&apos;s 13.77.
            μ = 0 reproduces vanilla FedAvg bit-exactly (the regression
            test).
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "17.95", label: "Vanilla FedAvg RMSE (μ=0)" },
            { value: "17.70", label: "FedProx best RMSE (μ=0.1)", tone: "good" },
            { value: "+6.0 %", label: "Gap closed vs centralized", tone: "good" },
            { value: "0.25", label: "RMSE absolute improvement" },
            { value: "0.895", label: "FedProx F1 (vs vanilla 0.871)" },
            { value: "4 / 4", label: "μ-sweep cells completed" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="The headline figure"
        title="μ = 0.1 wins on combined RMSE, but the story is per-subset."
      >
        <ExplainedFigure
          artifactPath="results/rq2_fedprox/headline_comparison_fd001_fd003.png"
          caption="Combined-test-set RMSE across μ values + P6 references"
          eyebrow="Figure 01"
          takeaway="Combined RMSE moves from 17.95 to 17.70. Small, but in the right direction — and consistent with the FedProx literature on similar Non-IID benchmarks."
          explanation={
            <>
              <p>
                Five bars: the four μ values plus the centralized + local-
                only references from P6. The leftmost (μ=0) reproduces
                vanilla FedAvg exactly — by construction, the proximal
                term vanishes when μ=0, so this is the regression check
                that proves the FedProx implementation doesn&apos;t alter
                anything when disabled.
              </p>
              <p>
                As μ increases, combined RMSE drops modestly. The
                literature suggests larger μ values (1, 10) would help
                further on truly extreme Non-IID, but on our 4-client
                FD001+FD003 partition the gains plateau quickly. The real
                story is in the per-subset breakdown.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="The per-subset story"
        title="FedProx rebalances the per-subset profile."
      >
        <ExplainedFigure
          artifactPath="results/rq2_fedprox/per_subset_breakdown_fd001_fd003.png"
          caption="Per-subset RMSE across the μ sweep"
          eyebrow="Figure 02"
          takeaway="Vanilla FedAvg biases toward FD001 (17.0 vs FD003's 18.9). FedProx flattens this — both subsets converge to ~17.7. Same combined number, much better distribution."
          explanation={
            <>
              <p>
                Two bar groups (FD001 + FD003) per μ value. Vanilla FedAvg
                (μ=0) is visibly asymmetric: FD001 RMSE 17.0 vs FD003 RMSE
                18.9 — a 1.9 RMSE gap between the two subsets. FedProx at
                μ=0.1 closes this gap: FD001 RMSE 17.7, FD003 RMSE 17.7,
                roughly equal performance on both halves of the world.
              </p>
              <p>
                For a maintenance operator deploying the global model
                across both engine types, this matters: the FedProx model
                gives <em>operationally comparable accuracy</em> on both
                fault modes, while vanilla FedAvg is noticeably worse on
                the harder one. Even when combined RMSE is statistically
                tied, the per-subset balance is real.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Per-round dynamics"
        title="Convergence is smoother — drift control working as advertised."
      >
        <ExplainedFigure
          artifactPath="results/rq2_fedprox/per_round_rmse_fd001_fd003.png"
          caption="Combined RMSE across 50 communication rounds, one curve per μ"
          eyebrow="Figure 03"
          takeaway="Higher μ produces a smoother trajectory with less round-to-round oscillation. The proximal term IS bounding drift, even when its effect on the final number is modest."
          explanation={
            <>
              <p>
                Four traces, one per μ. μ=0 oscillates noisily across
                rounds 10-50. μ=0.1 produces a much smoother monotonic
                descent. This is the visual confirmation that the
                proximal term is doing its job — local updates can&apos;t
                drift far enough from the round-start weights to swing
                the aggregate sharply between rounds.
              </p>
              <p>
                Operationally, smoother convergence also means the
                &quot;best round&quot; checkpoint is more stable — less
                sensitive to which exact round happens to land at the
                local minimum. In a long-running federation this is
                non-trivial.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Verdict"
        title="Positive but small — the cure does not fully live at this layer either."
        intro={
          <>
            <p>
              FedProx is the canonical drift-control method and it
              behaves exactly as the literature predicts: small combined-
              RMSE improvement, real per-subset rebalancing, smoother
              optimisation. <strong>This is a partial fix, not a
              complete one.</strong>
            </p>
            <p>
              The remaining gap to centralized (still ~4 RMSE) is what
              FedRep — the architectural follow-up — addresses next.
              Together with the negative RQ2 result, the empirical
              hierarchy of intervention layers becomes clear:{" "}
              <span className="font-mono-num text-text">
                aggregation &lt; drift-control &lt; per-client architecture
              </span>
              .
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>
    </ExperimentLayout>
  );
}
