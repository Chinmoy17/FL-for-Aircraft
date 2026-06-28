import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 02 — Centralized smoke run (1 epoch on FD001).
 *
 * Polished into the same question-driven structure as Phase 00/01:
 * a "why this phase exists" opener, then each question section
 * follows WHAT WE NEED TO KNOW → THE EVIDENCE → WHAT WE FOUND.
 *
 * Only one figure exists for this phase (loss_curve), so the page
 * is bulked out with a hand-written PassConditions checklist that
 * makes the wire-test pass criteria auditable in one glance.
 */
export function ExperimentSmokePage() {
  return (
    <ExperimentLayout
      phaseId="Phase 02 · Smoke"
      title="Centralized smoke run (1 epoch)"
      lede={
        <>
          A single epoch of the multi-task 1-D CNN (30 018 parameters)
          on pooled FD001. Not a benchmark — a wire test. The deliverable
          is the proof that the pipeline assembled in Phase 01 actually
          produces gradients, runs them through the optimizer, and emits
          test metrics in the expected shape. Numerical performance is
          expected to be terrible; the pass criteria are structural.
        </>
      }
      metaRow={
        <>
          <span>FD001 · 1 epoch · 256 batch · Adam lr = 1e-3 · GroupNorm</span>
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
        eyebrow="Why this phase exists"
        title="What does a 1-epoch smoke run actually deliver?"
        intro={
          <>
            <p>
              The previous phase proved the <em>data</em> wires up. This
              phase proves the <em>model and training loop</em> wire up.
              A 50-epoch centralized baseline (Phase 03) is an
              expensive way to discover that your loss returns NaN on
              batch 12 or that your fault head has the wrong output
              shape — much cheaper to discover those bugs in 1.5
              seconds of CPU time first.
            </p>
            <p>
              The contract is narrow on purpose. We do not ask &quot;is
              the model accurate?&quot; — we ask &quot;does every layer
              of the pipeline produce something a downstream layer can
              consume?&quot;. The pass conditions below are written as
              structural checks, not as performance bars.
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>

      <ExperimentSection
        eyebrow="At a glance"
        title="The wire-test numbers after one epoch."
        intro={
          <p>
            Fault-head AUPRC of <strong>0.845</strong> after one epoch
            is the &quot;the encoder is learning real signal&quot;
            indicator — random performance on the 25 %-positive test
            set would give 0.25. The RUL-head RMSE of{" "}
            <strong>62.7</strong> is &quot;the head is untrained but
            the optimizer is moving the right way.&quot; Both are
            expected, neither is a finding worth carrying forward.
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "1.4 s", label: "Wall-clock on CPU" },
            { value: "766 → 530", label: "First → last batch loss" },
            { value: "62.7", label: "Test RMSE (untrained, expected)", tone: "bad" },
            { value: "0.845", label: "Test AUPRC (real signal already)", tone: "good" },
            { value: "1.0 / 0.25", label: "Recall / Precision (mis-calibrated)" },
            { value: "0", label: "NaNs / shape errors" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 1"
        title="Do the optimizer, losses, and two heads all wire up correctly?"
        intro={
          <p>
            <strong>What we need to know:</strong> before we burn 50
            epochs on the real baseline, we have to verify each piece
            of the pipeline produces what the next piece expects.
            Six structural checks have to pass simultaneously — any
            single failure means the Phase 03 run is going to crash
            in some confusing way.
          </p>
        }
      >
        <PassConditionsChecklist />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 2"
        title="Does the loss actually fall across the very first epoch?"
        intro={
          <p>
            <strong>What we need to know:</strong> a passing
            pipeline can still have a broken optimizer. The cleanest
            signal that the gradient step is doing useful work is a
            monotone downward trajectory of the per-batch loss within
            the first epoch — no NaN spikes, no oscillation that
            doesn&apos;t decay, no plateau at the initial value.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/02_smoke/loss_curve_fd001.png"
          caption="Total / Huber / BCE loss across 70 mini-batches of one epoch"
          eyebrow="Figure 01 · Findings"
          takeaway="Loss falls smoothly from 766 to 530 across all 70 mini-batches. No spikes, no plateau, no NaNs. The optimizer is working."
          explanation={
            <>
              <p>
                Three traces share one panel: total combined loss,
                RUL-head Huber loss, and fault-head BCE-with-logits
                loss (the latter weighted by 0.5). Total loss starts
                at ~766 and ends at ~530 — a smooth monotone descent
                across the 70 mini-batches that make up one epoch.
                There are no spikes, no plateaus, no NaNs.
              </p>
              <p>
                The RUL component dominates the total because Huber is
                in cycle-units (predictions in ~60s, residuals in
                tens) while BCE is in nats (typically &lt; 1 per
                sample). This is expected and is the reason we use a
                fixed 0.5 weight on the fault head rather than equal
                mixing — it keeps the gradient magnitudes proportional.
              </p>
              <p>
                The fault head&apos;s Precision = 0.25 / Recall = 1.0
                mis-calibration is the one finding from this phase to
                carry forward:{" "}
                <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
                  pos_weight = 4.72
                </code>{" "}
                is correct for a converged head, but it over-corrects
                an untrained one. Phase 03 logs per-epoch P/R curves
                explicitly to watch the self-calibration arrive around
                epoch 5.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="What this phase decided"
        title="Cleared for Phase 03 — 50-epoch centralized training is safe to launch."
        intro={
          <p>
            Six structural checks passed, one expected numerical
            artefact (the untrained fault-head mis-calibration) noted
            for Phase 03 to watch. The pipeline is allowed to spend
            real CPU time on real training. Phase 03 runs the same
            model for 50 epochs with the cosine schedule that
            actually lets the metrics arrive.
          </p>
        }
      >
        <></>
      </ExperimentSection>
    </ExperimentLayout>
  );
}

// ---------------------------------------------------------------------------
// Inline single-use content blocks
// ---------------------------------------------------------------------------

/**
 * Six-row checklist of structural pass conditions for the smoke
 * run. Each row is a check + a measured outcome + a verdict pill.
 */
function PassConditionsChecklist() {
  type Row = { check: string; outcome: string; passed: boolean };
  const rows: Row[] = [
    {
      check: "Data loader yields windows of shape [256, 30, 17]",
      outcome: "shape match across all 70 batches",
      passed: true,
    },
    {
      check: "Two heads emit (RUL ∈ ℝ₊, fault_logit ∈ ℝ) per window",
      outcome: "output tensor shape [256, 2] verified",
      passed: true,
    },
    {
      check: "Combined loss is finite for every batch",
      outcome: "no NaN / Inf observed across 70 batches",
      passed: true,
    },
    {
      check: "Loss decreases monotonically over the epoch",
      outcome: "first batch 765.8 → last batch 530.1 (−31 %)",
      passed: true,
    },
    {
      check: "Fault head learns something measurable (AUPRC > 0.5)",
      outcome: "test AUPRC = 0.845 (random baseline = 0.25)",
      passed: true,
    },
    {
      check: "RUL head learns something measurable (RMSE < initial)",
      outcome: "test RMSE 62.7 (initial ≈ 90, mean-prediction ≈ 70)",
      passed: true,
    },
  ];

  return (
    <div className="my-8 border border-border rounded-lg overflow-hidden">
      <table className="w-full text-[14.5px]">
        <thead>
          <tr className="bg-bg-subtle text-text-muted text-[11.5px] uppercase tracking-[0.12em]">
            <th className="text-left px-5 py-3 font-medium w-[42%]">Structural check</th>
            <th className="text-left px-5 py-3 font-medium">Measured outcome</th>
            <th className="text-left px-5 py-3 font-medium w-[100px]">Verdict</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((r) => (
            <tr key={r.check} className="align-top">
              <td className="px-5 py-4 text-text font-medium">{r.check}</td>
              <td className="px-5 py-4 text-text-dim leading-[1.55]">{r.outcome}</td>
              <td className="px-5 py-4">
                <span
                  className={
                    r.passed
                      ? "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-good/10 text-good text-[11.5px] font-semibold uppercase tracking-[0.1em]"
                      : "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-bad/10 text-bad text-[11.5px] font-semibold uppercase tracking-[0.1em]"
                  }
                >
                  {r.passed ? "✓ Pass" : "✗ Fail"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
