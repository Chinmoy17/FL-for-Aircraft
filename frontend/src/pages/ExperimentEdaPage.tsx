import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 00 — Exploratory data analysis.
 *
 * Rewritten as a narrative pass over the dataset, with figures
 * reordered to follow the story arc (not numbered file order):
 *
 *   1. Engines run to failure — but how long? (lifetimes)         Fig 01
 *   2. What does the model predict?            (RUL distribution) Fig 05
 *   3. And as a binary fault, it's rare        (class imbalance)  Fig 06
 *   4. Why these two subsets?                  (op regimes)       Fig 03
 *   5. What do the sensors look like?          (correlation)      Fig 02
 *   6. Is degradation actually visible?        (trajectories)     Fig 04
 *
 * Each section is preceded by a short transition paragraph so the
 * reader is told *why* the next figure matters before they see it.
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
          questions had to be answered: how long does an engine run, what
          is the label, how rare are positive examples, why these subsets,
          what is the sensor structure, and is degradation actually
          detectable in the raw traces. The six figures below walk those
          questions in order.
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
        eyebrow="At a glance"
        title="The dataset at a glance, before any figure."
        intro={
          <p>
            Four CMAPSS subsets, two orthogonal axes of variation. FD001 and
            FD003 — the pair this project uses end-to-end — are both
            single-regime, sea-level. They differ only in fault-mode count
            (1 vs 2), which is the controlled variable for every later
            Non-IID study. Everything else (sensor count, fault rate,
            window budget) is roughly comparable across the two.
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
        eyebrow="Section 1 — The engines"
        title="How long does an engine run before failure?"
        intro={
          <p>
            The first thing any sliding-window CNN needs to know:{" "}
            <em>does the window fit?</em> If the shortest engine in the
            dataset runs for fewer cycles than our window length, those
            engines drop out of training entirely. We need a window size
            that fits comfortably inside even the rarest short-lived
            engine.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/01_engine_lifetimes.png"
          caption="Engine lifetime histogram per subset"
          eyebrow="Figure 01"
          takeaway="Shortest engine runs 128 cycles. A window size of 30 fits comfortably with ~4x safety margin."
          explanation={
            <>
              <p>
                Each subset is a separate histogram. The minimum lifetime is{" "}
                <strong>128 cycles</strong> on FD001/FD002/FD004 and{" "}
                <strong>145 cycles</strong> on FD003. The bulk of FD001 and
                FD003 engines cluster around 200 cycles; the multi-regime
                subsets (FD002, FD004) skew slightly longer because they
                experience more diverse operating conditions that can mask
                early degradation.
              </p>
              <p>
                With a 30-cycle window the per-engine sample count is{" "}
                <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
                  lifetime − window + 1
                </code>{" "}
                — for a 200-cycle engine that&apos;s 171 windows. Pooling
                across 100 FD001 engines gives the 17,731 training windows
                used in every later phase.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Section 2 — The label"
        title="What does the model actually predict?"
        intro={
          <p>
            Each window&apos;s label is the Remaining Useful Life (RUL) —
            the number of cycles until that engine fails. Raw RUL ranges
            from 0 to ~360 cycles, but most engines spend most of their
            life in a <em>flat-healthy</em> regime where sensor readings
            don&apos;t change much. Asking the model to predict raw RUL
            wastes most of its gradient on a region where there is no
            signal to learn from.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/05_rul_distribution.png"
          caption="Raw RUL vs piecewise-capped RUL"
          eyebrow="Figure 02"
          takeaway="The 125-cycle piecewise cap clips the flat-healthy tail and concentrates the regression on the degradation-informative cycles."
          explanation={
            <>
              <p>
                Without intervention the distribution (left panel) is
                dominated by the long high-RUL plateau every engine spends
                most of its life in. The 125-cycle piecewise cap (right
                panel) clips any window with true RUL ≥ 125 to exactly 125
                while leaving windows below 125 unchanged. The resulting
                distribution is bimodal but concentrates the
                regression&apos;s gradient on the part of the input space
                where sensor readings actually carry information.
              </p>
              <p>
                Every downstream model in this project is trained against
                capped RUL. The cap is also why the model&apos;s softplus
                head outputs are effectively bounded — it never has to
                predict beyond 125 cycles.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Section 3 — The rare event"
        title="As a binary fault, positive examples are rare."
        intro={
          <p>
            Our model also has a binary head: <em>fault = 1 if RUL ≤ 30</em>.
            This reframes the regression as a near-failure detector — the
            framing operations cares about. But the rare-event nature of
            engine failure means the positive class is dominated by
            healthy windows. The class imbalance has to be handled
            explicitly or the model will collapse to predicting
            &quot;healthy&quot;.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/06_fault_imbalance.png"
          caption="Fault-positive rate (RUL ≤ 30) per subset"
          eyebrow="Figure 03"
          takeaway="15% positive on FD001, 13% on FD003. Roughly 1-in-7 windows is positive — every model uses pos_weight ~ 4.7 to compensate."
          explanation={
            <>
              <p>
                The fault rate is <strong>15.03%</strong> on FD001 and{" "}
                <strong>12.54%</strong> on FD003. About 1 in 7 training
                windows is positive — most windows are healthy engine
                operation. This is the imbalance that motivates the entire
                RQ2 research question: rare failure examples are easy to
                lose when client weights get averaged together under
                federated learning.
              </p>
              <p>
                Concretely, every downstream model uses{" "}
                <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
                  BCEWithLogitsLoss(pos_weight = n_neg / n_pos)
                </code>{" "}
                with a per-client positive weight near 4.7. This is the
                cheap, well-understood baseline. RQ2 then asks whether a
                smarter <em>aggregation</em> rule on top can preserve rare
                failure signal further. (Spoiler: it cannot, and the RQ2
                story explains why.)
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Section 4 — The subset choice"
        title="Why FD001 + FD003 specifically?"
        intro={
          <p>
            The project&apos;s controlled Non-IID variable is{" "}
            <strong>fault-mode heterogeneity</strong>. To isolate that, we
            need a pair of subsets where everything else (operating
            regime, sensor count) is identical. The operational-regime
            clustering below confirms that FD001 and FD003 sit in a
            single tight cluster — a luxury FD002 and FD004 do not have.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/03_operational_regimes.png"
          caption="3-D KMeans clustering of operational settings"
          eyebrow="Figure 04"
          takeaway="FD001/FD003 collapse to a single tight cluster (sea-level operation). FD002/FD004 split cleanly into 6 regimes — confounded with degradation signal."
          explanation={
            <>
              <p>
                Each panel scatter-plots a subset&apos;s three operational
                settings (altitude, Mach, throttle resolver angle) coloured
                by a KMeans cluster assignment. For FD001 and FD003 (top
                row), every point sits in a single tight cluster at
                altitude ~ 0, Mach ~ 0, TRA ~ 100 — confirming the
                single-regime sea-level operation the literature describes.
                For FD002 and FD004 (bottom row), KMeans finds six clearly
                separated clusters along the altitude axis.
              </p>
              <p>
                Mixing FD001 with FD002 would force a model to disentangle
                two confounded things at once (regime and fault mode) —
                we couldn&apos;t claim our Non-IID gap is structural. By
                choosing FD001+FD003 we lock the regime axis and isolate
                fault-mode heterogeneity as the single Non-IID dimension.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Section 5 — The sensors"
        title="What does the 17-feature input vector actually contain?"
        intro={
          <p>
            The literature drops 7 of the 21 raw CMAPSS sensors because
            they are constant (or near-constant) within the single-regime
            subsets. That leaves 14 informative sensors + 3 operational
            settings = the 17-feature input every model in this project
            consumes. The correlation structure validates that decision
            and previews the physical sub-systems the RQ3 ontology will
            later encode by name.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/02_sensor_correlation.png"
          caption="Pairwise Pearson correlation, FD001 vs FD004"
          eyebrow="Figure 05"
          takeaway="FD001 shows 3 visible blocks (HPC temps/pressures, Fan speeds, core stability). FD004's multi-regime contamination obliterates that block structure."
          explanation={
            <>
              <p>
                Two side-by-side heatmaps. FD001 (left, single regime)
                shows three visible blocks of correlated sensors:
                HPC-flavoured temperatures and pressures (T30 / Ps30 / P30
                / phi), Fan-flavoured speeds (Nf / NRf / BPR), and core
                stability (Nc / NRc). This block structure is the same
                physical-subsystem grouping our RQ3 maintenance ontology
                later encodes by hand.
              </p>
              <p>
                FD004 (right, 6 operating regimes) has the same physical
                sensors but the regime variation dominates the correlation
                signal. Per-regime degradation is buried. This is the
                second reason FD002 / FD004 are out of scope for the
                project&apos;s core experiments — even the sensor
                correlation structure isn&apos;t directly visible.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Section 6 — The signal"
        title="Is degradation actually visible in the raw traces?"
        intro={
          <p>
            We have engines, labels, an imbalanced positive class, and a
            sensor structure. One question remains before training is
            meaningful: <em>is there a learnable signal at all?</em> If
            sensors look the same in healthy and near-failure windows,
            no architecture will help. The per-engine trajectories below
            show the canonical HPC-degradation fingerprint the model
            must learn to read.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/00_eda/04_sensor_trajectories.png"
          caption="Selected sensor trajectories for one median-lifetime engine per subset"
          eyebrow="Figure 06"
          takeaway="T30 drifts upward, Ps30 drifts downward in the final ~40 cycles. Real degradation signal, small but detectable. The model has to read it from a 30-cycle window."
          explanation={
            <>
              <p>
                One engine per subset (median lifetime) is selected and a
                small set of physically meaningful sensors is plotted
                against cycle. The clear visual story: most sensors are
                noisy but stationary for most of the engine&apos;s life,
                with a small but detectable drift in the final ~40 cycles
                before failure.
              </p>
              <p>
                <strong className="text-text">T30</strong> (HPC outlet
                temperature) drifts <em>upward</em> as the compressor
                degrades — efficiency loss requires more throttling, more
                throttling raises post-compressor temperature.{" "}
                <strong className="text-text">Ps30</strong> (HPC outlet
                static pressure) drifts <em>downward</em> — same
                compressor inefficiency reduces pressure ratio. The model
                has to learn to read this signature in a 30-cycle window
                that ends mid-degradation, not at the obvious moment of
                failure.
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
        eyebrow="Closing"
        title="Now the model can be trained."
        intro={
          <>
            <p>
              Six questions, six answers. The window size is 30 cycles.
              The label is piecewise-capped RUL with a paired binary
              fault head. The class imbalance is handled by{" "}
              <code className="font-mono-num text-text bg-bg-subtle px-1 rounded">
                pos_weight ~ 4.7
              </code>
              . The subset pair is FD001 + FD003 (sea-level, fault-mode
              heterogeneous). The 17 input features are 3 operational
              settings + 14 informative sensors, structured around HPC /
              Fan / core sub-systems. Degradation is detectable in T30
              and Ps30 in the final ~40 cycles.
            </p>
            <p>
              Phase 01 next splits FD001 into 4 simulated airline
              clients to verify the partition is balanced before the
              first model trains in Phase 02.
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>
    </ExperimentLayout>
  );
}
