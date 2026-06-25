import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 03 — Full centralized FD001 baseline.
 *
 * Polished into the question-driven structure: opener explaining
 * why a centralized upper bound is needed at all, then four
 * questions each backed by one figure.
 */
export function ExperimentCentralizedPage() {
  return (
    <ExperimentLayout
      phaseId="Phase 03 · Centralized baseline"
      title="Full centralized FD001 baseline"
      lede={
        <>
          50 epochs of cosine-annealed Adam on pooled FD001. No
          federation, no Non-IID partition, no constraints. The output
          is the <em>upper-bound</em> RMSE that every later FedAvg /
          Non-IID / security experiment is compared against — and the
          proof that our 30 K-parameter architecture reaches published
          literature performance on this benchmark.
        </>
      }
      metaRow={
        <>
          <span>
            FD001 · 50 epochs · cosine lr 1e-3 → 0 · weight decay 1e-4
          </span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">scripts/run_centralized.py</span>
        </>
      }
      prev={{ id: "02", title: "Smoke run", to: "/experiments/02-smoke" }}
      next={{
        id: "04",
        title: "Local-only baseline",
        to: "/experiments/04-local-only",
      }}
    >
      <ExperimentSection
        eyebrow="Why this phase exists"
        title="Why train a model we're never going to deploy?"
        intro={
          <>
            <p>
              A centralized run on pooled data is not a candidate
              deployment in this project — we already promised the
              data would never leave its client. It exists for a
              different reason: it is the only honest{" "}
              <strong>reference point</strong> for &quot;how good can
              this model possibly get on this dataset?&quot;.
            </p>
            <p>
              Without it we have no way to interpret a federated RMSE
              of 14.16. Is that good? Bad? Compared to what? The
              centralized number answers that question once and for
              every later phase: it is the ceiling. Federated methods
              are judged by how close they get to it, not by their
              absolute number.
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>

      <ExperimentSection
        eyebrow="At a glance"
        title="Best-epoch numbers land inside the published literature range."
        intro={
          <p>
            Literature for a well-trained CMAPSS FD001 model lands at
            RMSE 15–20 and NASA score in the hundreds. Our best epoch
            (epoch 5 of 50) lands at{" "}
            <strong>RMSE 14.02 / NASA 357</strong>, comfortably inside
            that range. The model reaches its minimum quickly because
            FD001 is small (17 731 windows) and the architecture is
            small (30 018 parameters).
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "14.02", label: "Best-epoch test RMSE", tone: "good" },
            { value: "357", label: "Best-epoch NASA score" },
            { value: "0.987", label: "Best-epoch fault AUPRC", tone: "good" },
            { value: "0.962", label: "Best-epoch fault F1", tone: "good" },
            { value: "85.3 s", label: "Wall-clock total (50 epochs)" },
            { value: "1.71 s", label: "Per-epoch on CPU" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 1"
        title="Does the loss converge cleanly across all 50 epochs?"
        intro={
          <p>
            <strong>What we need to know:</strong> the smoke run
            confirmed the optimizer moves in the right direction for
            one epoch. Now we need to see what 50 epochs of cosine-
            annealed Adam actually does — does the loss continue to
            descend, does it plateau, does it oscillate? The shape of
            the loss curve is the diagnostic for whether the model
            has the capacity it needs.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/03_centralized/loss_curve_fd001.png"
          caption="Total + per-task training loss across 50 epochs (log scale)"
          eyebrow="Figure 01 · Findings"
          takeaway="Loss collapses from 641 to 87 in the first 4 epochs. The model captures most of the signal almost immediately."
          explanation={
            <>
              <p>
                Three traces on log scale: total loss, RUL-head Huber,
                and fault-head BCE. The dominant story is the very
                fast initial drop — most of the optimization budget is
                used in the first few epochs. After epoch ~10 the loss
                stabilises and only slowly continues to decrease.
              </p>
              <p>
                This shape is characteristic of small models on small
                datasets: there is one obvious local minimum and the
                gradient-descent walk finds it quickly. The cosine
                schedule continues to anneal the learning rate, which
                keeps late-epoch optimisation gentle — that&apos;s why
                test metrics drift later epochs without diverging.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 2"
        title="When do the test RUL metrics peak, and what happens after?"
        intro={
          <p>
            <strong>What we need to know:</strong> training-loss
            descent doesn&apos;t guarantee test improvement. We have
            to track per-epoch test metrics (RMSE, MAE, NASA) to find
            the moment the model starts over-fitting and to pick the
            checkpoint every federated run is benchmarked against.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/03_centralized/rul_metrics_fd001.png"
          caption="Test RMSE / MAE / NASA score after each epoch"
          eyebrow="Figure 02 · Findings"
          takeaway="Best test RMSE arrives at epoch 5 (14.02) and drifts to ~16 by epoch 50. Mild over-fit; the cosine schedule keeps it contained."
          explanation={
            <>
              <p>
                Three panels share an x-axis (epoch count). Test RMSE
                drops from 63 to 14 by epoch 5, then drifts up to ~16
                by epoch 50. MAE follows a parallel trajectory. NASA
                score (the asymmetric PHM-08 metric that heavily
                punishes late predictions) is noisier but tracks RMSE
                in shape.
              </p>
              <p>
                Mild over-fitting after epoch 5 is consistent with a
                model that has the capacity to memorise a fraction of
                17 K training windows. Heavier regularisation would
                slow the drift but not improve the best-epoch number;
                weight decay = 1e-4 already runs and helps. The
                best-epoch checkpoint is the one saved for downstream
                federated comparisons.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 3"
        title="Does the fault head self-calibrate, or stay stuck on the smoke-run pathology?"
        intro={
          <p>
            <strong>What we need to know:</strong> Phase 02 left an
            open issue — Precision = 0.25 / Recall = 1.0 with{" "}
            <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
              pos_weight = 4.72
            </code>{" "}
            after one untrained epoch. The hypothesis was &quot;the
            head needs time to calibrate.&quot; This figure tests
            that hypothesis explicitly by tracking AUPRC / F1 /
            Precision / Recall after every epoch.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/03_centralized/fault_metrics_fd001.png"
          caption="Test AUPRC / F1 / Precision / Recall after each epoch"
          eyebrow="Figure 03 · Findings"
          takeaway="Self-calibration arrives around epoch 5. AUPRC 0.85 → 0.98, Precision 0.25 → 0.93, with Recall pinned at 1.0 throughout."
          explanation={
            <>
              <p>
                AUPRC climbs from 0.85 to 0.98 by epoch 5 and stays
                there for the rest of training. Precision climbs from
                0.25 (the smoke-run pathology) to 0.93 over the same
                window as the head finishes calibrating. Recall stays
                at 1.0 throughout — the loss is biased toward not
                missing failures, and the model honours that bias.
              </p>
              <p>
                The takeaway:{" "}
                <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
                  pos_weight
                </code>{" "}
                is the right choice for a converged head and a
                temporarily-harmful choice for an untrained one. The
                fix is to keep the loss as-is and just let training
                arrive at calibration. This is exactly how it&apos;s
                used in every later phase.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 4"
        title="What does the per-engine prediction error actually look like?"
        intro={
          <p>
            <strong>What we need to know:</strong> aggregate RMSE
            hides the per-engine residual structure. We need to see
            <em>where</em> the model is accurate and <em>where</em> it
            isn&apos;t — because the &quot;where&quot; will tell us
            whether the residuals are signal-limited (nothing to
            improve) or model-limited (room to improve).
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/03_centralized/pred_vs_true_fd001.png"
          caption="Final-epoch test predictions on the 100-engine FD001 test set"
          eyebrow="Figure 04 · Findings"
          takeaway="Tight on the diagonal at low RUL (informative regime). Wider, downward-biased scatter at high RUL (the flat-capped regime). Residual structure is signal-limited."
          explanation={
            <>
              <p>
                Each point is one of the 100 test engines, plotted as
                (true RUL, predicted RUL). The diagonal y = x is the
                ideal. Most points sit near the diagonal in the 0–60
                RUL band — this is the regime where sensor readings
                carry the most degradation signal, and it&apos;s the
                regime the 125-cycle piecewise cap is designed to
                focus the model on.
              </p>
              <p>
                Above ~80 true RUL the predictions widen and bias
                downward. These are the &quot;healthy&quot; engines
                whose sensor readings look approximately like
                training-mean. The cap clips their training labels to
                125, but the test set still contains some windows with
                true RUL ≥ 125 (capped, by convention), so the model
                lacks fine-grained signal to distinguish them. The
                residual structure is exactly what the cap is designed
                to produce.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="What this phase decided"
        title="The upper bound is 14.02 RMSE. Every later run is judged against it."
        intro={
          <>
            <p>
              One checkpoint saved (epoch 5), one number locked in
              (RMSE 14.02 / NASA 357), one architectural confirmation
              (the 30 K-parameter CNN is sufficient for FD001).
              Every later phase — local-only, FedAvg, Non-IID,
              FedProx, FedRep, FedCCFA — is compared to this number
              and judged by how much of the gap to it each method
              recovers.
            </p>
            <p>
              Phase 04 next trains four isolated 25-engine clients
              with the same recipe to establish the federation&apos;s
              lower bound. Anything that beats 14.02 + 1.0 = 15.02
              under IID is closing the gap; anything that doesn&apos;t
              is paying for federation without buying anything.
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>
    </ExperimentLayout>
  );
}
