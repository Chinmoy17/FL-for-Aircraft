import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 00 — Exploratory data analysis.
 *
 * First experiment page built on the new ExperimentLayout +
 * ExplainedFigure contract. Every figure on this page is accompanied
 * by a 2–3 paragraph explanation that answers two questions: what
 * does this figure show, and what does it tell us about the dataset?
 *
 * This is the "EDA was just images" gap the user called out, fixed.
 */
export function ExperimentEdaPage() {
  return (
    <ExperimentLayout
      phaseId="Phase 00 · EDA"
      title="Exploratory data analysis"
      lede={
        <>
          Before any model is trained, we have to know the dataset.
          Six figures that establish C-MAPSS's structural properties:
          per-engine lifetime distribution, sensor correlations,
          operational regimes, sensor trajectories, RUL cap behaviour,
          and the fault-class imbalance every later phase has to
          contend with.
        </>
      }
      metaRow={
        <>
          <span>4 CMAPSS subsets · 709 engines · 160,359 windows · 0 NaN</span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">
            notebooks/01_eda_cmapss.ipynb
          </span>
        </>
      }
      next={{
        id: "01",
        title: "Data pipeline sanity",
        to: "/experiments/01-data",
      }}
    >
      {/* HEADLINE NUMBERS ------------------------------------------------ */}
      <ExperimentSection
        eyebrow="At a glance"
        title="Headline numbers across the four CMAPSS subsets."
        intro={
          <p>
            The columns we care most about for the federated experiments are
            the <em>fault rate</em> (which sets the natural class imbalance)
            and the <em>operational regime count</em> (which determines whether
            any one client carries multiple operating conditions). FD001 and
            FD003 — the pair this project uses end-to-end — are both
            single-regime, sea-level. They differ only in fault-mode count
            (1 vs 2), which is the controlled variable for the Non-IID study.
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "709", label: "Engines across 4 subsets" },
            { value: "160,359", label: "Total training rows" },
            { value: "14 / 16", label: "Informative sensors (FD001-3 / FD002-4)" },
            { value: "1 vs 6", label: "Operating regimes (FD001/3 vs FD002/4)" },
            { value: "15.0 %", label: "FD001 fault rate (RUL ≤ 30)" },
            { value: "0", label: "Missing values" },
          ]}
        />
      </ExperimentSection>

      {/* FIG 01 ENGINE LIFETIMES ---------------------------------------- */}
      <ExperimentSection
        eyebrow="Figure 01"
        title="Engine lifetime distribution"
        intro={
          <p>
            How long does an engine run before failure? The answer determines
            the project's window-size budget. Any sliding window must comfortably
            fit inside even the shortest engine's lifetime, otherwise the
            earliest cycles get dropped and the training set shrinks.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/01_engine_lifetimes.png"
          caption="Engine lifetime histogram per subset"
          takeaway="The shortest engine runs 128 cycles. Our window size of 30 fits comfortably with a wide safety margin."
          explanation={
            <>
              <p>
                Each subset is a separate histogram. The minimum lifetime is{" "}
                <strong>128 cycles</strong> on FD001/FD002/FD004 and{" "}
                <strong>145 cycles</strong> on FD003. The bulk of FD001 and
                FD003 engines (single-regime) cluster around 200 cycles; the
                multi-regime subsets (FD002, FD004) skew slightly longer
                because they experience more diverse operating conditions
                that can mask early degradation.
              </p>
              <p>
                The practical implication: a 30-cycle sliding window is safe
                across every subset. The number of training windows per
                engine is then{" "}
                <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
                  lifetime − window + 1
                </code>{" "}
                — for a 200-cycle engine that's 171 windows. Pooling these
                across 100 FD001 engines gives the 17,731 training-window
                figure used in every later phase.
              </p>
            </>
          }
        />
      </ExperimentSection>

      {/* FIG 02 SENSOR CORRELATION -------------------------------------- */}
      <ExperimentSection
        eyebrow="Figure 02"
        title="Sensor correlation heatmaps"
      >
        <ExplainedFigure
          artifactPath="results/00_eda/02_sensor_correlation.png"
          caption="Pairwise Pearson correlation, FD001 vs FD004"
          takeaway="The 14 informative FD001 sensors form a tight tri-block — HPC, Fan, and core dynamics. FD004's multi-regime contamination breaks that block structure entirely."
          explanation={
            <>
              <p>
                Two side-by-side heatmaps. FD001 (left, single regime) shows{" "}
                <strong>three visible blocks of correlated sensors</strong>:
                HPC-flavoured temperatures and pressures (T30 / Ps30 / P30 /
                phi), Fan-flavoured speeds (Nf / NRf / BPR), and core
                stability (Nc / NRc). This block structure is the same
                physical-subsystem grouping our maintenance ontology in
                RQ3 encodes by hand.
              </p>
              <p>
                FD004 (right, 6 operating regimes) has the same physical
                sensors but the regime variation dominates the correlation
                signal. Per-regime degradation is buried; a model trained on
                FD004 raw data must first learn to subtract the operating
                regime before degradation is visible. This is the central
                reason FD002 / FD004 are out of scope for this project's
                core experiments — the regime-disentanglement problem is
                its own research thread (RQ1 territory).
              </p>
            </>
          }
        />
      </ExperimentSection>

      {/* FIG 03 OPERATIONAL REGIMES ------------------------------------- */}
      <ExperimentSection
        eyebrow="Figure 03"
        title="Operational regime clustering"
      >
        <ExplainedFigure
          artifactPath="results/00_eda/03_operational_regimes.png"
          caption="3-D KMeans clustering of operational settings (alt × Mach × TRA)"
          takeaway="KMeans cleanly separates FD002/FD004's six regimes; FD001/FD003 collapse to a single tight cloud — confirming the literature's regime count for both subsets."
          explanation={
            <>
              <p>
                Each panel scatter-plots a subset's three operational settings
                (altitude, Mach, throttle resolver angle) coloured by a
                KMeans cluster assignment. For FD001 and FD003 (top row),{" "}
                <strong>every point sits in a single tight cluster at
                altitude ≈ 0, Mach ≈ 0, TRA ≈ 100</strong> — confirming the
                single-regime sea-level operation the literature describes.
                For FD002 and FD004 (bottom row), KMeans finds six clearly
                separated clusters along the altitude axis.
              </p>
              <p>
                This figure is the empirical justification for picking
                FD001 + FD003 as the project's Non-IID pair. By holding
                regime constant, we isolate <em>fault-mode heterogeneity</em>{" "}
                as the single Non-IID dimension. Mixing FD001 with FD002
                would force a model to disentangle two confounded things at
                once (regime and fault mode), and we wouldn't be able to
                claim our Non-IID gap is structural.
              </p>
            </>
          }
        />
      </ExperimentSection>

      {/* FIG 04 SENSOR TRAJECTORIES ------------------------------------- */}
      <ExperimentSection
        eyebrow="Figure 04"
        title="Per-engine sensor trajectories"
      >
        <ExplainedFigure
          artifactPath="results/00_eda/04_sensor_trajectories.png"
          caption="Selected sensor trajectories for one median-lifetime engine per subset"
          takeaway="Degradation signal is real but small. T30 drifts upward, Ps30 drifts downward — the typical HPC-degradation fingerprint our model has to detect."
          explanation={
            <>
              <p>
                One engine per subset (median lifetime) is selected and a
                small set of physically meaningful sensors is plotted against
                cycle. The clear visual story:{" "}
                <strong>most sensors are noisy but stationary for most of
                the engine's life</strong>, with a small but detectable drift
                in the final ~40 cycles before failure.
              </p>
              <p>
                T30 (HPC outlet temperature) drifts <em>upward</em> as the
                compressor degrades — efficiency loss requires more
                throttling, more throttling raises post-compressor
                temperature. Ps30 (HPC outlet static pressure) drifts{" "}
                <em>downward</em> — same compressor inefficiency reduces
                pressure ratio. The model has to learn to read this
                signature in a 30-cycle window that ends mid-degradation,
                not at the obvious moment of failure.
              </p>
              <p>
                This figure is also the visual anchor for the RQ3
                interpretability work: when Integrated Gradients later
                attributes a prediction primarily to T30 with positive
                contribution, the figure is the ground truth that says
                "yes, T30 rising near end-of-life is the right signal to
                weight on."
              </p>
            </>
          }
        />
      </ExperimentSection>

      {/* FIG 05 RUL DISTRIBUTION ---------------------------------------- */}
      <ExperimentSection
        eyebrow="Figure 05"
        title="Raw RUL vs piecewise-capped RUL"
      >
        <ExplainedFigure
          artifactPath="results/00_eda/05_rul_distribution.png"
          caption="RUL label distribution before and after the 125-cycle piecewise cap"
          takeaway="The 125-cycle cap (standard CMAPSS practice) clips the long flat-healthy tail and forces the model to focus on the degradation-informative final cycles."
          explanation={
            <>
              <p>
                Without intervention the raw RUL label runs from ~360 down to
                0 cycles, with most of the distribution concentrated above
                100 — the long flat-healthy plateau of a typical engine.
                Asking the model to predict raw RUL is asking it to fit a
                regression where most of the training signal carries no
                actual degradation information, which biases predictions
                toward the mean and makes the small differences near failure
                statistically invisible.
              </p>
              <p>
                The piecewise cap at 125 cycles is the standard CMAPSS
                solution: any window with true RUL ≥ 125 gets its label
                clipped to 125; windows below 125 keep their true RUL. The
                resulting distribution (right panel) is bimodal in shape but
                concentrates the regression's gradient on the part of the
                input space where sensor readings actually carry
                information.
              </p>
              <p>
                Every downstream model in this project is trained against
                capped RUL. The cap is also why our model's softplus head
                outputs are bounded — the model never has to predict beyond
                125, even though the test set occasionally has windows where
                true RUL = 125.0 (a "healthy" engine).
              </p>
            </>
          }
        />
      </ExperimentSection>

      {/* FIG 06 FAULT IMBALANCE ----------------------------------------- */}
      <ExperimentSection
        eyebrow="Figure 06"
        title="Fault-class imbalance"
      >
        <ExplainedFigure
          artifactPath="results/00_eda/06_fault_imbalance.png"
          caption="Fault-positive rate (RUL ≤ 30) per subset, training data"
          takeaway="The natural fault positive rate is 15% on FD001 and 13% on FD003 — a ~6:1 negative:positive imbalance. Every model since uses pos_weight ≈ 4.7 to compensate."
          explanation={
            <>
              <p>
                Fault rate (RUL ≤ 30 cycles, project convention) is{" "}
                <strong>15.03 %</strong> on FD001 and{" "}
                <strong>12.54 %</strong> on FD003. Roughly 1 in 7 training
                windows is positive — most windows are healthy engine
                operation. This is the imbalance that motivates the entire
                RQ2 research question: rare failure examples are easy to
                lose when client weights get averaged together.
              </p>
              <p>
                Concretely, every downstream model uses{" "}
                <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
                  BCEWithLogitsLoss(pos_weight=n_neg/n_pos)
                </code>{" "}
                with a per-client positive weight near 4.7. This is the
                cheap, well-understood baseline for class imbalance; RQ2
                then asks whether a smarter <em>aggregation</em> rule on top
                can preserve rare failure signal further. (Spoiler — the
                answer is no, and the mechanism is captured in detail on the{" "}
                RQ2 story page.)
              </p>
              <p>
                FD003 is slightly less imbalanced than FD001 because FD003
                engines have a second fault mode (Fan degradation) that
                shortens some lifetimes, raising the fraction of windows
                close to failure. The 2.5-percentage-point difference is
                small but is itself a feature of the structural Non-IID
                between the two subsets.
              </p>
            </>
          }
        />
      </ExperimentSection>
    </ExperimentLayout>
  );
}
