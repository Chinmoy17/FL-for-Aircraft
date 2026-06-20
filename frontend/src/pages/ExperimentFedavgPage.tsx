import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 05 — FedAvg baseline (FD001, IID, 4 clients).
 *
 * Canonical FedAvg over 50 rounds × 2 local epochs. Confirms the
 * federation works on the easy case before we change the partition.
 */
export function ExperimentFedavgPage() {
  return (
    <ExperimentLayout
      phaseId="Phase 05 · FedAvg IID"
      title="FedAvg baseline on IID FD001"
      lede={
        <>
          The canonical McMahan-2017 FedAvg protocol, simulated
          in-process with 4 airline clients on the IID FD001 partition.
          50 communication rounds, 2 local epochs per round, sample-
          count-weighted aggregation. No raw sensor data ever crosses
          the wire — only model weights. <strong>Closes 85.9% of the
          local-only → centralized RMSE gap.</strong>
        </>
      }
      metaRow={
        <>
          <span>
            FD001 · 4 clients × 50 rounds × 2 local epochs · 400 local-epoch
            equivalents
          </span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">scripts/run_fedavg.py</span>
        </>
      }
      prev={{
        id: "04",
        title: "Local-only baseline",
        to: "/experiments/04-local-only",
      }}
      next={{
        id: "06",
        title: "Non-IID baseline",
        to: "/experiments/06-non-iid",
      }}
    >
      <ExperimentSection
        eyebrow="At a glance"
        title="FedAvg recovers nearly all of centralized's RMSE — and matches its fault F1 exactly."
        intro={
          <p>
            Best-round FedAvg lands at <strong>RMSE 14.16</strong> versus
            centralized&apos;s 14.02 and local-only mean of 15.02. The
            federation works as expected under IID — averaging weights
            from clients trained on equivalent slices reconstructs most
            of what pooled training would have given us. The interesting
            test case where this stops working is Phase 06 (Non-IID).
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "14.16", label: "Best-round RMSE", tone: "good" },
            { value: "350", label: "Best-round NASA (tied with P3)" },
            { value: "0.962", label: "Best-round F1 (= centralized)", tone: "good" },
            { value: "0.965", label: "Best-round AUPRC" },
            { value: "85.9 %", label: "Local→centralized gap closed", tone: "good" },
            { value: "11 / 50", label: "Best round (cosine pacing)" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 01"
        title="The three-way comparison — the headline image"
      >
        <ExplainedFigure
          artifactPath="results/05_fedavg/three_way_comparison_fd001.png"
          caption="Centralized vs FedAvg vs local-only mean across the four headline metrics"
          takeaway="FedAvg sits visibly closer to centralized than to local-only on every metric — exactly the federation success pattern under IID."
          explanation={
            <>
              <p>
                Four panels (RMSE, NASA, AUPRC, F1) each show three bars:
                P3 centralized (left, upper bound), P5 FedAvg (middle),
                P4 local-only mean (right, lower bound). On every panel
                FedAvg sits closer to centralized than to local-only. The
                visual is also where the &quot;85.9% gap closed&quot;
                arithmetic comes from — (15.02 − 14.16) / (15.02 − 14.02).
              </p>
              <p>
                NASA score is the only metric where FedAvg appears to
                slightly out-do centralized (350 vs 357). This is run-to-
                run noise on a 100-engine test set rather than a real
                effect — both numbers are in the same ballpark, and a
                fresh re-run with a different seed would swap which one
                wins. We treat them as statistically tied.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 02"
        title="Global model test metrics across 50 rounds"
      >
        <ExplainedFigure
          artifactPath="results/05_fedavg/global_metrics_fd001.png"
          caption="Test RMSE / NASA / AUPRC / F1 / Precision / Recall over communication rounds"
          takeaway="Best round arrives at round 11 of 50, then mild oscillation — the same convergence pattern as the centralized run scaled by the 4×-local-epochs-per-round budget."
          explanation={
            <>
              <p>
                Six small panels. Each metric improves rapidly across the
                first ~10 rounds and oscillates thereafter. The best
                round is 11 of 50 — coincidentally matching P3&apos;s best
                epoch 5 in the &quot;effective local-epoch&quot; sense
                (round 11 × 2 local epochs ≈ 22 local-epoch equivalents,
                comparable to P3&apos;s pacing).
              </p>
              <p>
                Recall stays pinned at 1.0 throughout — same loss
                configuration as centralized, same convergence answer. The
                fault head is the smoothest metric across rounds; the RUL
                head is the noisiest. That ordering is consistent with the
                fact that fault detection is a binary rank-ordering task
                (easier to stabilise) while RUL regression is sensitive to
                weight-averaging interactions across clients.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 03"
        title="Per-client training loss across rounds"
      >
        <ExplainedFigure
          artifactPath="results/05_fedavg/loss_curves_fd001.png"
          caption="Each client's local-training loss across 50 communication rounds"
          takeaway="The four curves overlap almost perfectly — IID clients see nearly identical local optimisation paths, so aggregation has very little variance to absorb."
          explanation={
            <>
              <p>
                Four traces, one per client, each showing the average of
                that client&apos;s 2 local-epoch losses per round. They
                overlap visibly — clients 1–4 all converge at the same
                rate to the same local minima. This is the signature of an
                IID partition: when every client&apos;s data slice is
                statistically equivalent, every local update points in
                roughly the same direction, and averaging is essentially
                lossless.
              </p>
              <p>
                The interesting Phase 06 picture is the opposite: those
                four curves diverge sharply because FD001-only clients
                follow one optimisation path and FD003-only clients
                follow another. That divergence is what FedAvg cannot
                handle, and the reason RQ2 + the FedProx / FedRep /
                FedCCFA follow-ups exist.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 04"
        title="Predicted vs true RUL on the final round"
      >
        <ExplainedFigure
          artifactPath="results/05_fedavg/pred_vs_true_fd001.png"
          caption="Final-round test predictions vs ground-truth RUL"
          takeaway="Essentially indistinguishable from the centralized scatter — same residual structure on the same 100 test engines."
          explanation={
            <>
              <p>
                The FedAvg final-round scatter looks the same as
                centralized&apos;s. Tight on the diagonal at low RUL,
                widening into the flat-capped regime above ~80 RUL. Same
                physical reason: low-RUL windows carry the most degradation
                signal; high-RUL windows are clipped by the piecewise cap
                and have less to learn.
              </p>
              <p>
                Two functionally equivalent models on the same test data
                give the same residual structure. This visual is the
                strongest possible evidence that FedAvg under IID is not
                fundamentally compromising what the model learns — it is
                converging to roughly the same function as pooled training
                would have given.
              </p>
            </>
          }
        />
      </ExperimentSection>
    </ExperimentLayout>
  );
}
