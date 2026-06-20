import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 02 — Centralized smoke run (1 epoch).
 *
 * One epoch on FD001, confirming the data → model → loss → metrics
 * wiring is correct. Not a benchmark; a wire-test.
 */
export function ExperimentSmokePage() {
  return (
    <ExperimentLayout
      phaseId="Phase 02 · Smoke"
      title="Centralized smoke run (1 epoch)"
      lede={
        <>
          One epoch of the multi-task 1-D CNN (30,018 parameters) on pooled
          FD001. The goal is not benchmark quality but{" "}
          <em>end-to-end correctness</em>: optimizer steps, loss
          monotonicity, no NaNs, all shapes match. Pass conditions are
          design-spec sanity checks; numerical performance is expected to
          be terrible.
        </>
      }
      metaRow={
        <>
          <span>FD001 · 1 epoch · 256 batch · Adam lr=1e-3 · GroupNorm</span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">scripts/smoke_train.py</span>
        </>
      }
      prev={{ id: "01", title: "Data pipeline sanity", to: "/experiments/01-data" }}
      next={{
        id: "03",
        title: "Full centralized baseline",
        to: "/experiments/03-centralized",
      }}
    >
      <ExperimentSection
        eyebrow="At a glance"
        title="Sanity numbers after one epoch — what passes, what intentionally fails."
        intro={
          <p>
            The fault head's AUPRC = 0.845 after one epoch is the &quot;the
            architecture works&quot; signal — random baseline on a 25%-
            positive test set would give 0.25. The RUL head&apos;s RMSE
            ≈ 62.7 is &quot;the head is untrained but the optimizer is
            moving&quot;. Both expected after a single epoch.
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "1.5 s", label: "Wall-clock on CPU" },
            { value: "770 → 530", label: "First → last batch loss" },
            { value: "62.7", label: "Test RMSE (untrained, expected)", tone: "bad" },
            { value: "0.845", label: "Test AUPRC (real signal already)", tone: "good" },
            { value: "1.0 / 0.25", label: "Recall / Precision (mis-calibrated)" },
            { value: "0", label: "NaNs / shape errors" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection eyebrow="Figure" title="Per-batch loss trace">
        <ExplainedFigure
          artifactPath="results/02_smoke/loss_curve_fd001.png"
          caption="Total / Huber / BCE loss across 70 mini-batches of one epoch"
          takeaway="Loss falls monotonically across the whole epoch. The optimizer works; the architecture is sound."
          explanation={
            <>
              <p>
                Three traces share one panel: total combined loss, RUL-head
                Huber loss, and fault-head BCE-with-logits loss (the latter
                weighted by 0.5). Total loss starts at ~770 and ends at ~530
                — a smooth monotone descent across the 70 mini-batches that
                make up one epoch. There are no spikes, no plateaus, no
                NaNs.
              </p>
              <p>
                The RUL component dominates the total because Huber is in
                cycle-units (predictions in ~60s, residuals in tens) while
                BCE is in nats (typically &lt; 1 per sample). This is
                expected and is the reason we use a fixed 0.5 weight on the
                fault head rather than equal mixing — it keeps the gradient
                magnitudes proportional.
              </p>
              <p>
                The fault head&apos;s Precision = 0.25 / Recall = 1.0 mis-
                calibration is the one finding from this phase to carry
                forward: <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">pos_weight=4.72</code> is correct for a converged
                head, but it over-corrects an untrained one. Phase 03 logs
                per-epoch P/R curves explicitly to watch the self-
                calibration arrive around epoch 5.
              </p>
            </>
          }
        />
      </ExperimentSection>
    </ExperimentLayout>
  );
}
