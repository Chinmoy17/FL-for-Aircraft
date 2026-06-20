import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 04 — Local-only baseline (4 clients, FD001).
 *
 * Each client trains alone on its 25 engines. Quantifies the
 * federation's lower bound under IID partitioning.
 */
export function ExperimentLocalOnlyPage() {
  return (
    <ExperimentLayout
      phaseId="Phase 04 · Local-only"
      title="Isolated per-client training"
      lede={
        <>
          What happens if every airline trains alone? Four independent
          50-epoch runs, same recipe as the centralized baseline, but each
          model sees only its 25-engine slice. Evaluated on the common
          test set so per-client numbers are directly comparable to P3.
          The mean of the four sets the federation's <em>lower bound</em>{" "}
          — FedAvg must beat this for the federation to have a point.
        </>
      }
      metaRow={
        <>
          <span>FD001 · 4 clients × 50 epochs · stratified-by-lifetime</span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">scripts/run_local_only.py</span>
        </>
      }
      prev={{
        id: "03",
        title: "Centralized baseline",
        to: "/experiments/03-centralized",
      }}
      next={{
        id: "05",
        title: "FedAvg IID baseline",
        to: "/experiments/05-fedavg",
      }}
    >
      <ExperimentSection
        eyebrow="At a glance"
        title="Penalty for isolation is real but small — about +1 RMSE on IID FD001."
        intro={
          <p>
            With balanced data and the same training recipe, isolated
            clients land at <strong>RMSE 15.02 ± 0.29</strong> versus
            centralized&apos;s 14.02. The 1-RMSE penalty is the federation's
            target — FedAvg should close most of it under IID. The
            interesting case (where this gap becomes huge) is the Non-IID
            partition at Phase 06.
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "15.02 ± 0.29", label: "Mean per-client test RMSE" },
            { value: "+1.00", label: "Penalty vs centralized (14.02)" },
            { value: "0.973", label: "Mean AUPRC (down 0.014)" },
            { value: "0.923", label: "Mean F1 (down 0.039)" },
            { value: "client_3", label: "Weakest client (RMSE 15.50)" },
            { value: "82.1 s", label: "Wall-clock total" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 01"
        title="Centralized vs local-only — the federation target"
      >
        <ExplainedFigure
          artifactPath="results/04_local_only/centralized_vs_local_fd001.png"
          caption="P3 centralized vs P4 local-only mean across RMSE / NASA / AUPRC / F1"
          takeaway="The federation has 1 RMSE / 50 NASA of gap to close. It will close ~85% of it in Phase 05."
          explanation={
            <>
              <p>
                Side-by-side bars for each metric (lower is better for
                RMSE/NASA; higher is better for AUPRC/F1). Centralized wins
                on every metric — that&apos;s the upper bound — but the
                margins are modest. RMSE differs by 1.0 cycle, NASA by ~50
                points, AUPRC by 0.014, F1 by 0.039.
              </p>
              <p>
                Why so small? Because the P1 partition is deliberately
                near-IID: every client&apos;s 25 engines were stratified by
                lifetime so the fault-positive rates and engine-life
                distributions match. With 4.4K windows per client (a
                quarter of the centralized total), each local model is
                statistically saturated for this architecture; pooling the
                clients into one helps only marginally.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 02"
        title="Per-client best-epoch metrics"
      >
        <ExplainedFigure
          artifactPath="results/04_local_only/per_client_metrics_fd001.png"
          caption="Bar chart of best-epoch RMSE / NASA / AUPRC / F1 for each of the 4 clients"
          takeaway="client_3 lags slightly (RMSE 15.50) because it received the shortest-lifetime engines — less degradation signal per engine."
          explanation={
            <>
              <p>
                Four groups of four bars, one group per client. Clients 1,
                2, and 4 land within RMSE 14.7–15.0 — essentially
                equivalent. Client 3 is the visible outlier at RMSE 15.50.
                The reason is in the partition: the stratified-by-lifetime
                split gave client_3 the engines with the shortest mean
                lifetime (205.5 cycles vs 206–207 elsewhere), so fewer
                cycles of degradation signal per engine and slightly worse
                generalisation.
              </p>
              <p>
                In a real-world deployment this asymmetry would be much
                worse — small airlines, regional carriers, and operators
                with newer fleets have orders of magnitude less failure
                data than larger players. The natural unfairness of FL is
                why server-side reweighting is so often proposed (and why
                RQ2 tests it).
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 03"
        title="Per-client training loss curves"
      >
        <ExplainedFigure
          artifactPath="results/04_local_only/loss_curves_fd001.png"
          caption="Training loss across 50 epochs, one curve per client (log scale)"
          takeaway="All four clients converge to similar minima at similar paces — by design, since the partition is near-IID."
          explanation={
            <>
              <p>
                Four loss curves are virtually overlapping. Each client
                hits its own best epoch between 15 and 28 — later than
                centralized&apos;s epoch 5 because each model sees only a
                quarter of the data. The 4×-data argument for centralized
                shows up here as 4× earlier convergence.
              </p>
              <p>
                The visual overlap is itself a finding: under near-IID
                partitioning, every client&apos;s local trajectory is
                essentially the same. This is exactly the setting where
                FedAvg should excel — averaging weights from similar
                gradient updates is close to averaging the gradients
                centrally. The contrasting Phase 06 picture (where local
                curves diverge sharply) is what makes the Non-IID
                challenge concrete.
              </p>
            </>
          }
        />
      </ExperimentSection>
    </ExperimentLayout>
  );
}
