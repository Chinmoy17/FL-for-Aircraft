import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 00 — Exploratory data analysis.
 *
 * Reorganized as a question-driven narrative:
 *
 *   - Why EDA exists at all (motivation, before any figure)
 *   - Dataset at a glance (headline numbers)
 *   - Six questions, one per section:
 *
 *       Q1  How long does an engine run before failure?     Fig 01
 *       Q2  What does the model actually predict?           Fig 02
 *       Q3  How rare is the positive class?                 Fig 03
 *       Q4  Why FD001 + FD003 specifically?                 Fig 04
 *       Q5  What does the 17-feature input contain?         Fig 05
 *       Q6  Is degradation actually visible in the traces?  Fig 06
 *
 *   - Closing: every dataset-prep decision that flows from EDA
 *
 * Each question section follows the same three-beat structure the
 * user asked for: WHAT WE NEED TO KNOW (heading + intro) →
 * THE IMAGE (full-width, sharp) → WHAT WE FOUND (findings block
 * inside ExplainedFigure: eyebrow + title + takeaway + body).
 */
export function ExperimentEdaPage() {
  return (
    <ExperimentLayout
      phaseId="Phase 00 · EDA"
      title="Exploratory data analysis"
      lede={
        <>
          Every model in the rest of this project trains on{" "}
          <strong>NASA C-MAPSS</strong>. Before the first checkpoint, six
          concrete questions had to be answered — each one determines a
          choice the rest of the pipeline depends on. This page walks
          those six questions in order, paired with the figure that
          settles each one.
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
      <ExperimentSection
        eyebrow="Why this phase exists"
        title="What is EDA for, and why is it the very first step?"
        intro={
          <>
            <p>
              Exploratory data analysis is the phase where the dataset
              stops being an abstract download and starts being a thing
              with shape, gaps, and quirks the rest of the pipeline has
              to handle. No model is trained here. No metric is reported.
              The deliverable is a small set of figures that <em>force
              decisions</em> about how the data is going to be prepared,
              labelled, and split for everything that comes after.
            </p>
            <p>
              Concretely, EDA exists to answer the questions that drive
              every downstream design choice: what window size to use,
              how to define the label, which class-balance correction to
              apply, which subsets to combine, which sensors to keep,
              and whether the signal we want to model is even present
              in the raw traces. Every choice in Phases 01 through 07
              traces back to a figure on this page.
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>

      <ExperimentSection
        eyebrow="At a glance"
        title="The dataset, in one line."
        intro={
          <p>
            Four CMAPSS subsets, two orthogonal axes of variation. FD001
            and FD003 — the pair this project uses end-to-end — are
            both single-regime, sea-level. They differ only in
            fault-mode count (1 vs 2), which is the controlled variable
            for every later Non-IID study. Everything else (sensor
            count, fault rate, window budget) is comparable across the
            two.
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

      <ExperimentSection
        eyebrow="Question 1"
        title="How long does an engine run before failure?"
        intro={
          <p>
            <strong>What we need to know:</strong> the shortest engine
            in the dataset sets a hard upper bound on the model&apos;s
            window length. If any engine runs for fewer cycles than our
            window, those engines drop out of training entirely. Before
            we pick a window size, we have to look at the lifetime
            distribution across all four subsets.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/01_engine_lifetimes.png"
          caption="Engine lifetime histogram per subset"
          eyebrow="Figure 01 · Findings"
          takeaway="Shortest engine runs 128 cycles. A window size of 30 fits comfortably with a 4× safety margin — and is what every model in this project uses."
          explanation={
            <>
              <p>
                Each subset is a separate histogram. The minimum lifetime
                is <strong>128 cycles</strong> on FD001 / FD002 / FD004
                and <strong>145 cycles</strong> on FD003. The bulk of
                FD001 and FD003 engines cluster around 200 cycles; the
                multi-regime subsets (FD002, FD004) skew slightly longer
                because they experience more diverse operating
                conditions that can mask early degradation.
              </p>
              <p>
                With a 30-cycle window the per-engine sample count is{" "}
                <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
                  lifetime − window + 1
                </code>{" "}
                — for a 200-cycle engine that&apos;s 171 windows.
                Pooling across 100 FD001 engines gives the 17,731
                training windows used in every later phase.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 2"
        title="What does the model actually predict?"
        intro={
          <p>
            <strong>What we need to know:</strong> the raw label is
            Remaining Useful Life — the number of cycles until that
            engine fails. But raw RUL ranges from 0 to ~360 cycles, and
            most engines spend most of their life in a flat-healthy
            regime where the sensors aren&apos;t telling us anything
            interesting yet. Before we train, we have to decide what
            target the regression head actually optimizes against.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/05_rul_distribution.png"
          caption="Raw RUL vs piecewise-capped RUL"
          eyebrow="Figure 02 · Findings"
          takeaway="A 125-cycle piecewise cap clips the flat-healthy tail and concentrates the regression on the degradation-informative cycles."
          explanation={
            <>
              <p>
                Without intervention (left panel) the distribution is
                dominated by the long high-RUL plateau every engine
                spends most of its life in. The 125-cycle piecewise cap
                (right panel) clips any window with true RUL ≥ 125 to
                exactly 125 while leaving windows below 125 unchanged.
                The resulting distribution is bimodal but concentrates
                the regression&apos;s gradient on the part of the input
                space where sensor readings actually carry information.
              </p>
              <p>
                Every downstream model in this project trains against
                capped RUL. The cap is also why the model&apos;s softplus
                head is effectively bounded — it never has to predict
                beyond 125 cycles.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 3"
        title="How rare is the positive class?"
        intro={
          <p>
            <strong>What we need to know:</strong> our model also has a
            binary head — <em>fault = 1 if RUL ≤ 30</em> — that reframes
            the regression as a near-failure detector. But because real
            engine failure is rare, positive examples are heavily
            outnumbered by healthy windows. Before we touch the loss
            function we have to measure exactly how imbalanced the
            classes are, per subset.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/06_fault_imbalance.png"
          caption="Fault-positive rate (RUL ≤ 30) per subset"
          eyebrow="Figure 03 · Findings"
          takeaway="15 % positive on FD001, 13 % on FD003 — roughly 1 in 7 windows. Every model compensates with pos_weight ≈ 4.7 on BCE."
          explanation={
            <>
              <p>
                The fault rate is <strong>15.03 %</strong> on FD001 and{" "}
                <strong>12.54 %</strong> on FD003. About 1 in 7 training
                windows is positive — most windows are healthy engine
                operation. This is the imbalance that motivates the
                entire RQ2 research question: rare failure examples are
                easy to lose when client weights get averaged together
                under federated learning.
              </p>
              <p>
                Concretely, every downstream model uses{" "}
                <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
                  BCEWithLogitsLoss(pos_weight = n_neg / n_pos)
                </code>{" "}
                with a per-client positive weight near 4.7. That is the
                cheap, well-understood baseline. RQ2 then asks whether
                a smarter <em>aggregation</em> rule on top can preserve
                rare-failure signal further. (Spoiler: it cannot, and
                the RQ2 story explains why.)
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 4"
        title="Why FD001 + FD003 specifically?"
        intro={
          <p>
            <strong>What we need to know:</strong> the controlled
            Non-IID variable in this project is{" "}
            <strong>fault-mode heterogeneity</strong>. To isolate that
            cleanly, we need a pair of subsets where everything else
            (operating regime, sensor count) is held constant. Before
            we pick the pair, we have to verify that the operational
            settings actually collapse to one regime on the candidates
            and not on the others.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/03_operational_regimes.png"
          caption="3-D KMeans clustering of operational settings"
          eyebrow="Figure 04 · Findings"
          takeaway="FD001 / FD003 collapse to a single tight cluster (sea-level operation). FD002 / FD004 split cleanly into 6 regimes — a confound we cannot afford in a Non-IID study."
          explanation={
            <>
              <p>
                Each panel scatter-plots a subset&apos;s three
                operational settings (altitude, Mach, throttle resolver
                angle) coloured by a KMeans cluster assignment. For
                FD001 and FD003 (top row), every point sits in a single
                tight cluster at altitude ~0, Mach ~0, TRA ~100 —
                confirming the single-regime sea-level operation the
                literature describes. For FD002 and FD004 (bottom row),
                KMeans finds six clearly separated clusters along the
                altitude axis.
              </p>
              <p>
                Mixing FD001 with FD002 would force a model to
                disentangle two confounded things at once (regime{" "}
                <em>and</em> fault mode) — we could not claim our
                Non-IID gap was structural. By choosing FD001 + FD003
                we lock the regime axis and isolate fault-mode
                heterogeneity as the single Non-IID dimension.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 5"
        title="What does the 17-feature input vector actually contain?"
        intro={
          <p>
            <strong>What we need to know:</strong> the literature drops
            7 of the 21 raw CMAPSS sensors because they are constant
            (or near-constant) within single-regime subsets. That
            leaves 14 informative sensors + 3 operational settings =
            the 17-feature input every model in this project consumes.
            Before we commit to that feature set, we have to verify
            the correlation structure makes physical sense — and
            understand why the multi-regime subsets break it.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/02_sensor_correlation.png"
          caption="Pairwise Pearson correlation, FD001 vs FD004"
          eyebrow="Figure 05 · Findings"
          takeaway="FD001 shows 3 visible blocks (HPC temps + pressures, Fan speeds, core stability). FD004's multi-regime contamination obliterates that block structure."
          explanation={
            <>
              <p>
                Two side-by-side heatmaps. FD001 (left, single regime)
                shows three visible blocks of correlated sensors:
                HPC-flavoured temperatures and pressures (T30 / Ps30 /
                P30 / phi), Fan-flavoured speeds (Nf / NRf / BPR), and
                core stability (Nc / NRc). This block structure is the
                same physical-subsystem grouping our RQ3 maintenance
                ontology later encodes by hand.
              </p>
              <p>
                FD004 (right, 6 operating regimes) has the same
                physical sensors, but the regime variation dominates
                the correlation signal. Per-regime degradation is
                buried. This is the second reason FD002 / FD004 are out
                of scope for the project&apos;s core experiments —
                even the sensor correlation structure isn&apos;t
                directly visible.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 6"
        title="Is degradation actually visible in the raw traces?"
        intro={
          <p>
            <strong>What we need to know:</strong> we have engines,
            labels, an imbalanced positive class, and a chosen sensor
            structure. One question remains before training is even
            meaningful — <em>is there a learnable signal at all?</em>{" "}
            If sensors look the same in healthy and near-failure
            windows, no architecture will help. We have to inspect the
            raw per-engine trajectories.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/04_sensor_trajectories.png"
          caption="Selected sensor trajectories for one median-lifetime engine per subset"
          eyebrow="Figure 06 · Findings"
          takeaway="T30 drifts upward, Ps30 drifts downward in the final ~40 cycles. Real degradation signal — small but detectable, and exactly what the model has to read from a 30-cycle window."
          explanation={
            <>
              <p>
                One engine per subset (median lifetime) is selected and
                a small set of physically meaningful sensors is plotted
                against cycle. The clear visual story: most sensors are
                noisy but stationary for most of the engine&apos;s
                life, with a small but detectable drift in the final
                ~40 cycles before failure.
              </p>
              <p>
                <strong className="text-text">T30</strong> (HPC outlet
                temperature) drifts <em>upward</em> as the compressor
                degrades — efficiency loss requires more throttling,
                more throttling raises post-compressor temperature.{" "}
                <strong className="text-text">Ps30</strong> (HPC outlet
                static pressure) drifts <em>downward</em> — the same
                compressor inefficiency reduces pressure ratio. The
                model has to learn to read this signature in a 30-cycle
                window that ends mid-degradation, not at the obvious
                moment of failure.
              </p>
              <p>
                This figure is also the visual anchor for the RQ3
                interpretability work: when Integrated Gradients later
                attributes a prediction primarily to T30 with positive
                contribution, the trajectory above is the ground truth
                that says &quot;yes, T30 rising near end-of-life is the
                right signal to weight on.&quot;
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="What EDA decided"
        title="Every dataset-prep choice for Phases 01–07 traces back here."
        intro={
          <>
            <p>
              Six questions, six answers, six concrete decisions baked
              into the rest of the project. The window size is{" "}
              <strong>30 cycles</strong> (Q1). The label is{" "}
              <strong>piecewise-capped RUL</strong> at 125 with a
              paired binary fault head at RUL ≤ 30 (Q2). The class
              imbalance is handled with{" "}
              <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
                pos_weight ≈ 4.7
              </code>{" "}
              on BCE (Q3). The subset pair is{" "}
              <strong>FD001 + FD003</strong> — sea-level,
              fault-mode-heterogeneous (Q4). The 17 input features are
              3 operational settings + 14 informative sensors,
              structured around HPC / Fan / core sub-systems (Q5).
              Degradation is detectable in T30 and Ps30 in the final
              ~40 cycles (Q6).
            </p>
            <p>
              Phase 01 next splits FD001 into 4 simulated airline
              clients and verifies the partition is balanced before
              the first model trains in Phase 02.
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>
    </ExperimentLayout>
  );
}
