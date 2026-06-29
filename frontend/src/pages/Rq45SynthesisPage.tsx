import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Rq45SynthesisPage — `/rq4-rq5-synthesis`.
 *
 * Synthesis page (not a new experiment). The project originally
 * scoped only RQ2 / RQ3 / RQ7 for dedicated study, but while
 * analysing those phases enough cross-cutting evidence accumulated
 * to draw minor conclusions for RQ4 (concept drift) and RQ5
 * (cross-client evaluation). This page surfaces that evidence
 * explicitly so a reviewer can audit the claim without reading
 * every phase end-to-end.
 *
 * Page rewritten for clarity:
 *  1. Lede + meta row
 *  2. "What the two questions ask" — plain-English definition
 *     of RQ4 and RQ5 with 'what it asks / why it matters'
 *  3. "At a glance" — concrete headline numbers tied to findings
 *  4. RQ5 evidence — 5 scannable cards (replaces a dense <ol>)
 *  5. The figure that anchors RQ5
 *  6. RQ4 evidence — "have / don't have" two-column with caveats
 */
export function Rq45SynthesisPage() {
  return (
    <ExperimentLayout
      phaseId="RQ4 / RQ5 · Synthesis"
      title={
        <>
          Concept drift &amp;{" "}
          <em className="not-italic text-accent">cross-client evaluation</em>
        </>
      }
      lede={
        <>
          The project originally scoped RQ2, RQ3, and RQ7 for
          dedicated study. While analysing those results we noticed
          enough cross-cutting evidence to draw minor but defensible
          conclusions about <strong>RQ4</strong> (concept drift over
          time) and <strong>RQ5</strong> (cross-client evaluation).
          This page collects that evidence — no new code, no new
          experiments, just an honest re-read of what the other
          phases already showed.
        </>
      }
      metaRow={
        <>
          <span>
            Cross-cutting evidence from P6 · RQ2 · RQ3 · FedRep · RQ7
          </span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">no new experiments</span>
        </>
      }
      prev={{ id: "RQ7", title: "Security", to: "/rq7-story" }}
    >
      {/* WHAT THE QUESTIONS ASK ------------------------------------------ */}
      <ExperimentSection
        eyebrow="The two questions, in plain English"
        title="What RQ4 and RQ5 actually ask."
        intro={
          <p>
            Before walking through evidence, the two questions in
            one paragraph each — what each one asks, and why a real
            federated PHM deployment cares about the answer.
          </p>
        }
      >
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-lg border border-border bg-bg p-6">
            <div className="eyebrow !text-accent mb-2">RQ4</div>
            <h3 className="font-display text-xl text-text mb-3">
              Concept drift over time
            </h3>
            <p className="text-[14.5px] text-text-dim leading-relaxed mb-3">
              <strong className="text-text">What it asks: </strong>
              when a federated model is in production, the input
              distribution drifts. Engines age, sensors are
              recalibrated between maintenance cycles, seasonal
              effects change operating conditions. Can a federated
              model detect that drift mid-deployment and recover
              within a small number of rounds?
            </p>
            <p className="text-[14.5px] text-text-dim leading-relaxed">
              <strong className="text-text">Why it matters: </strong>
              static-data benchmarks freeze the world. Real fleets do
              not. A federation that cannot adapt to drift is a
              federation that quietly gets worse the longer it runs.
            </p>
          </div>

          <div className="rounded-lg border border-border bg-bg p-6">
            <div className="eyebrow !text-accent mb-2">RQ5</div>
            <h3 className="font-display text-xl text-text mb-3">
              Cross-client evaluation under Non-IID
            </h3>
            <p className="text-[14.5px] text-text-dim leading-relaxed mb-3">
              <strong className="text-text">What it asks: </strong>
              when each client owns a structurally different slice
              of the world, does reporting a single combined-test-set
              metric hide what the model actually does per client?
              How much per-client truth does the standard combined
              RMSE bury?
            </p>
            <p className="text-[14.5px] text-text-dim leading-relaxed">
              <strong className="text-text">Why it matters: </strong>
              federation is sold to operators on &quot;every client
              benefits&quot;. If the combined metric averages out a
              per-subset asymmetry of several RMSE points, only the
              cleanest clients actually benefit — the harder ones
              get a worse-than-isolated model.
            </p>
          </div>
        </div>
      </ExperimentSection>

      {/* AT A GLANCE ----------------------------------------------------- */}
      <ExperimentSection
        eyebrow="At a glance"
        title="One question is substantially answered, the other partially."
        intro={
          <p>
            <strong>RQ5</strong> is supported by five independent
            observations across the project — closer to{" "}
            <em>&quot;addressed across five phases&quot;</em> than{" "}
            <em>&quot;skipped&quot;</em>.{" "}
            <strong>RQ4</strong> is more limited: we observe
            training-side drift behaviour, but the CMAPSS partition is
            structurally static, so input-side drift over deployment
            time is honestly out of scope for this round of work.
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "Substantial", label: "RQ5 verdict", tone: "good" },
            { value: "Partial", label: "RQ4 verdict — with caveats" },
            { value: "5", label: "Independent RQ5 observations" },
            { value: "1.9", label: "FD001 vs FD003 RMSE gap hidden by combined metric" },
            { value: "0", label: "New experiments needed" },
            { value: "Open", label: "Input-side drift (RQ4)" },
          ]}
        />
      </ExperimentSection>

      {/* RQ5 EVIDENCE ---------------------------------------------------- */}
      <ExperimentSection
        eyebrow="RQ5 · Five observations"
        title="The combined-test metric hides per-client truth that matters operationally."
        intro={
          <p>
            Five independent phases of the project surface a
            different facet of the same fact: reporting only a
            combined-test RMSE on Non-IID data buries the per-client
            behaviour that an operator actually cares about. Each
            card below names its source phase, the observation, and
            the operational implication.
          </p>
        }
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mt-6">
          <EvidenceCard
            source="P6 — Non-IID baseline"
            heading="Non-IID FedAvg flips the FD001 / FD003 difficulty ordering."
            body={
              <>
                Centralized scores FD003 (RMSE 12.7) <em>easier</em>{" "}
                than FD001 (14.8). After Non-IID training, FedAvg
                inverts this — FD001 (17.0) is now easier than FD003
                (18.9). The Non-IID damage is{" "}
                <strong className="text-text">asymmetric</strong> and
                changes <em>which</em> subset is hard. Invisible from
                a single combined number.
              </>
            }
          />
          <EvidenceCard
            source="RQ2 — Aggregation sweep"
            heading="Different reweighting schemes help different subsets."
            body={
              <>
                Scheme B (validation-F1 softmax) is the only RQ2
                scheme that improves FD003. Schemes A and C improve
                FD001 at FD003&apos;s expense. A combined-RMSE
                comparison would call Scheme A &quot;neutral&quot;
                (+0.29 combined) — per-subset reveals it&apos;s
                actively harmful on the hard subset (FD003 RMSE
                +0.9 vs vanilla).
              </>
            }
          />
          <EvidenceCard
            source="RQ3 — Cross-model attribution"
            heading="Same RMSE, qualitatively different reasoning."
            body={
              <>
                Combined-data models attribute predictions to{" "}
                <code className="font-mono-num bg-bg-subtle px-1 rounded">
                  os_2
                </code>{" "}
                (Mach number) as a subset-identity proxy — FD001-only
                models never do. The same combined RMSE is being
                reached through a different decision mechanism.
                Per-client evaluation catches this; combined buries it.
              </>
            }
          />
          <EvidenceCard
            source="FedRep follow-up"
            heading="FedRep can't be reported without choosing a framing."
            body={
              <>
                FedRep clients each see only their own subset&apos;s
                test slice. We had to pick: report macro-mean
                (operationally honest, one number per client) or
                pooled combined RMSE (cross-phase comparable). The
                two numbers differ. The project literally cannot
                tell its FedRep story without committing to a
                cross-client evaluation framing.
              </>
            }
          />
          <EvidenceCard
            source="RQ7 — Adversarial robustness"
            heading="Catastrophic attack flattens the per-subset asymmetry."
            body={
              <>
                The AV2 gradient-scale attack pushes FD001 to RMSE
                84.5 and FD003 to 83.5 — the per-subset asymmetry{" "}
                <em>vanishes</em> when the model collapses to a
                near-constant prediction. Per-client evaluation shows
                the catastrophe is isotropic across subsets; combined
                would have just shown &quot;model broken&quot;.
              </>
            }
          />
          {/* Conclusion card sits in the 6th cell of the 2-column grid */}
          <div className="rounded-lg border border-good/40 bg-good/5 p-6 flex flex-col">
            <div className="eyebrow !text-good mb-2">RQ5 conclusion</div>
            <h3 className="font-display text-xl text-text mb-3">
              The substantial answer.
            </h3>
            <p className="text-[14.5px] text-text leading-relaxed">
              Five independent observations, each from a different
              phase, point at the same fact:{" "}
              <strong>
                any FL paper that reports only a combined-test metric
                on Non-IID data is hiding what the model actually
                does to individual airlines.
              </strong>{" "}
              The project&apos;s default reporting style — combined{" "}
              <em>plus</em> per-subset breakdown — is the right
              framing.
            </p>
          </div>
        </div>
      </ExperimentSection>

      {/* RQ5 ANCHOR FIGURE ----------------------------------------------- */}
      <ExperimentSection
        eyebrow="The anchor figure"
        title="What the asymmetry looks like."
        intro={
          <p>
            The Phase 06 per-subset breakdown is the cleanest visual
            of RQ5&apos;s argument. Each method&apos;s bar is split
            across FD001 and FD003 — read horizontally to see how
            each model behaves on each half of the world.
          </p>
        }
        indent
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/per_subset_breakdown_fd001_fd003.png"
          caption="P6 per-subset breakdown — what RQ5 is talking about"
          takeaway="Local models excel on their own subset, fail on the other. FedAvg doesn't catastrophically fail on either — but is best on neither. The combined RMSE hides all of this."
          explanation={
            <>
              <p>
                The headline message: each local-only client is good
                on the subset it trained on and visibly worse on the
                other one. The combined-test-set RMSE on FedAvg
                (17.95) sits between them and tells the reader
                nothing about that split.
              </p>
              <p>
                Every method tested in the rest of the project lands
                somewhere on this chart. RQ2&apos;s reweighting
                schemes shift these bars without closing the gap.
                FedRep shrinks them dramatically. RQ7&apos;s
                defended-Krum row is only 1–2 cycles worse than
                vanilla on FD001 and <em>better</em> on FD003 — a
                per-subset story the combined number doesn&apos;t
                tell.
              </p>
            </>
          }
        />
      </ExperimentSection>

      {/* RQ4 SECTION ---------------------------------------------------- */}
      <ExperimentSection
        eyebrow="RQ4 · Drift we can measure, drift we can't"
        title="Training-side drift is observable; input-side drift is honestly out of scope."
        intro={
          <>
            <p>
              The honest framing on RQ4 is that we can speak to it
              only partially. CMAPSS engines run to failure in a
              single regime — there is no temporal axis spanning
              multiple training rounds. Our partition is static
              across all 50 rounds, by construction.
            </p>
            <p>
              What we <em>can</em> observe and have observed is{" "}
              <strong>training-side drift</strong>: the per-round
              optimisation trajectory drifts under FedAvg-on-Non-IID
              because each client&apos;s local step pulls toward a
              different optimum. FedProx exists specifically to bound
              that drift; its μ-sweep is a quantitative answer to
              the &quot;does drift-control help?&quot; sub-question.
            </p>
          </>
        }
      >
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-lg border border-good/40 bg-good/5 p-6">
            <div className="eyebrow !text-good mb-3">
              What we have (training-side drift)
            </div>
            <ul className="space-y-3 text-[14.5px] text-text-dim leading-relaxed">
              <li>
                <strong className="text-text">FedProx μ-sweep — </strong>
                quantitative answer to &quot;does bounding local-step
                drift help?&quot; Yes, by ~6 % gap closed. Small but
                positive and consistent with the literature.
              </li>
              <li>
                <strong className="text-text">Per-round trajectories — </strong>
                every phase logs per-round loss and metrics. The
                FedAvg-vs-FedProx-vs-FedRep optimisation drift over
                50 rounds is fully visible.
              </li>
              <li>
                <strong className="text-text">RQ3 cross-checkpoint drift — </strong>
                the function the model learns drifts based on
                training-data scope (FD001-only vs combined). This
                is the architectural dual of input drift: same data
                fed to differently-trained models gets different
                answers.
              </li>
              <li>
                <strong className="text-text">RQ7 attacker norm drift — </strong>
                the RQ7 diagnostic plot shows update-magnitude drift
                across the 50 rounds caused by an adversarial client.
                Anomalous over-time drift in the FL protocol is
                detectable.
              </li>
            </ul>
          </div>

          <div className="rounded-lg border border-border bg-bg p-6">
            <div className="eyebrow !text-text-muted mb-3">
              What we don&apos;t (input-side drift)
            </div>
            <ul className="space-y-3 text-[14.5px] text-text-dim leading-relaxed">
              <li>
                <strong className="text-text">Temporal partition of clients — </strong>
                we would need to split each client&apos;s engines by
                a temporal axis (e.g. engines 1–25 train rounds 1–25,
                engines 26–50 train rounds 26–50) to simulate
                &quot;fleet ageing mid-training&quot;. Not done; our
                partition is static.
              </li>
              <li>
                <strong className="text-text">Mid-training distribution shift — </strong>
                the canonical RQ4 experiment injects a distribution
                shift at round 25 (e.g. one client&apos;s sensor
                calibration changes) and measures recovery. Not done.
              </li>
              <li>
                <strong className="text-text">Input concept drift natively — </strong>
                CMAPSS engines don&apos;t age across federated rounds;
                each engine is a complete run-to-failure trace. The
                dataset itself doesn&apos;t support this experiment
                without synthetic augmentation.
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-10 rounded-lg border border-accent/40 bg-accent-subtle p-6">
          <div className="eyebrow !text-accent mb-2">RQ4 conclusion</div>
          <p className="text-[15.5px] text-text leading-relaxed">
            <strong>Honest framing</strong>: training-side drift is
            measurable on this project and we have evidence on it.
            Input-side concept drift over deployment time is out of
            scope for this round — a complete RQ4 study would need
            either a temporal partition of CMAPSS engines or a
            different dataset with natural temporal structure (e.g.
            a real run-to-failure log spanning multiple calendar
            quarters).
          </p>
        </div>
      </ExperimentSection>
    </ExperimentLayout>
  );
}

// ---------------------------------------------------------------------------
// EvidenceCard — single-source observation tile used in the RQ5 grid.
// ---------------------------------------------------------------------------
function EvidenceCard({
  source,
  heading,
  body,
}: {
  source: string;
  heading: string;
  body: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-bg p-6 flex flex-col">
      <div className="eyebrow !text-accent mb-2">{source}</div>
      <h3 className="font-display text-[19px] text-text leading-snug mb-3">
        {heading}
      </h3>
      <p className="text-[14.5px] text-text-dim leading-relaxed">{body}</p>
    </div>
  );
}
