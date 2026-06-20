import { NavLink } from "react-router-dom";
import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Rq45SynthesisPage — `/rq4-rq5-synthesis`.
 *
 * A synthesis page (not a new experiment). The point of this page
 * is to make explicit what evidence already in the project says
 * about RQ4 (concept drift over time) and RQ5 (cross-client
 * evaluation), drawn from the 11 experiments already shipped.
 *
 * The page is honest about which conclusions are "substantial"
 * (RQ5 — five independent lines of evidence) vs "partial"
 * (RQ4 — observable in training-side metrics but our partition
 * is static so input-side drift is out of scope).
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
          A synthesis — not a new experiment. We did not run dedicated
          studies for RQ4 (concept drift over time) and RQ5 (cross-client
          evaluation), but evidence about both has been accumulating
          across the 11 phases. This page makes that evidence explicit:
          RQ5 is substantially answered from cross-cutting data; RQ4 is
          partially answered with documented caveats about what our
          static partition cannot reach.
        </>
      }
      metaRow={
        <>
          <span>
            Cross-cutting evidence drawn from P6 / RQ2 / RQ3 / FedRep / RQ7
          </span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">no new code, no new experiments</span>
        </>
      }
      prev={{ id: "RQ7", title: "Security", to: "/rq7-story" }}
    >
      {/* HEADLINE -------------------------------------------------------- */}
      <ExperimentSection
        eyebrow="Verdict"
        title="One question is substantially answered, the other partially."
        intro={
          <p>
            RQ5 (cross-client evaluation) has five independent lines of
            evidence across the project — it is closer to{" "}
            <em>"addressed across 5 phases"</em> than{" "}
            <em>"skipped"</em>. RQ4 (concept drift) has training-side and
            architectural-side evidence but our partition is structurally
            static; input-side drift is honestly out of scope for this
            round of work.
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "5 phases", label: "Independent RQ5 evidence lines" },
            { value: "Substantial", label: "RQ5 verdict", tone: "good" },
            { value: "Partial", label: "RQ4 verdict — with caveats" },
            { value: "0", label: "New experiments needed" },
            { value: "5", label: "Cross-cutting figures cited" },
            { value: "Open", label: "Input-side concept drift" },
          ]}
        />
      </ExperimentSection>

      {/* RQ5 SECTION ---------------------------------------------------- */}
      <ExperimentSection
        eyebrow="RQ5 · Cross-client evaluation"
        title="The combined-test-set metric hides per-client asymmetries that matter operationally."
        intro={
          <>
            <p>
              RQ5 asks whether reporting one combined-test-set RMSE
              obscures real performance differences between client
              segments. The five evidence lines below all answer{" "}
              <strong>yes</strong>, and three of them quantify the
              direction of the bias.
            </p>
          </>
        }
      >
        <ol className="list-decimal list-inside space-y-6 mt-6 max-w-[78ch]">
          <li>
            <strong>P6 baseline per-subset table.</strong>{" "}
            Centralized: FD003 (RMSE 12.7) is <em>easier</em> than FD001
            (14.8). Non-IID FedAvg <em>inverts</em> this: FD001 (17.0) is
            now easier than FD003 (18.9). The Non-IID damage is{" "}
            <strong>asymmetric AND flips the difficulty ordering</strong> —
            invisible from a single combined RMSE number.
          </li>
          <li>
            <strong>RQ2 per-subset table (4 aggregator schemes).</strong>{" "}
            Different aggregators help different subsets. Scheme B (val-F1
            softmax) is the <em>only</em> scheme that improves on FD003 (the
            hard subset). Schemes A and C improve FD001 at FD003&apos;s
            expense. A combined RMSE comparison would call Scheme A
            &quot;neutral&quot; (+0.29 combined) — per-subset shows it&apos;s{" "}
            <strong>actively harmful on the hard subset</strong> (FD003 RMSE
            +0.9 vs vanilla).
          </li>
          <li>
            <strong>RQ3 cross-model attribution.</strong>{" "}
            Combined-data models attribute predictions to `os_2` (Mach
            number) as a subset-identity proxy; FD001-only models never do.
            The same combined RMSE is being achieved through{" "}
            <strong>a qualitatively different mechanism</strong>. Per-client
            evaluation would catch this; combined evaluation buries it.
          </li>
          <li>
            <strong>FedRep macro-RMSE vs combined-RMSE choice.</strong>{" "}
            FedRep clients each see only their own subset&apos;s test slice,
            so we had to pick: report a macro mean (operationally honest) or
            a combined RMSE (cross-phase comparable). Both numbers
            differ — the project literally cannot tell its FedRep story
            without committing to a cross-client evaluation framing.
          </li>
          <li>
            <strong>RQ7 per-subset breakdown under attack.</strong>{" "}
            The AV2 gradient-scale attack pushes FD001 to RMSE 84.5 and
            FD003 to RMSE 83.5 — the per-subset asymmetry{" "}
            <em>vanishes</em> when the model collapses to a near-constant
            prediction. Per-client evaluation shows the catastrophe is
            isotropic across subsets — combined would have just shown
            &quot;model broken&quot;.
          </li>
        </ol>

        <div className="mt-10 rounded-lg border border-good/40 bg-good/5 p-6 max-w-[78ch]">
          <div className="eyebrow !text-good mb-2">RQ5 conclusion</div>
          <p className="text-[15.5px] text-text leading-relaxed">
            Five independent phases, each surface a different aspect of
            why per-client/per-subset evaluation is necessary. This is the
            substantial answer: <strong>any FL paper that reports only a
            combined test-set metric is hiding what the model actually
            does to individual airlines.</strong> The project&apos;s
            default reporting style — combined plus per-subset breakdown —
            is the right framing.
          </p>
        </div>
      </ExperimentSection>

      {/* RQ5 SUPPORTING FIGURE ------------------------------------------ */}
      <ExperimentSection
        eyebrow="Supporting figure"
        title="The figure that makes RQ5's substantial answer visually obvious."
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/per_subset_breakdown_fd001+fd003.png"
          caption="P6 per-subset breakdown — the visual that motivated RQ5's framing"
          takeaway="Local models excel on their own subset and fail on the other. FedAvg is the only model that doesn't catastrophically fail on either — but is best on neither."
          explanation={
            <>
              <p>
                This Phase 06 figure (also embedded on the Non-IID
                experiment page) is the cleanest demonstration of why
                cross-client evaluation matters. Reading the bars
                horizontally per subset reveals that{" "}
                <strong>each local model is good on one half of the
                world and bad on the other</strong>. The combined-test-set
                RMSE on FedAvg (17.95) hides this entirely.
              </p>
              <p>
                Every RQ in the project that involves the FD001+FD003
                Non-IID partition shows up here. RQ2&apos;s reweighting
                schemes shift these bars but don&apos;t close the gap. FedRep
                shrinks them dramatically. RQ7&apos;s defended-Krum row
                would be only 1-2 cycles worse than vanilla on FD001 and{" "}
                <em>better</em> on FD003 — a per-subset story Krum&apos;s
                combined number doesn&apos;t tell.
              </p>
            </>
          }
        />
      </ExperimentSection>

      {/* RQ4 SECTION ---------------------------------------------------- */}
      <ExperimentSection
        eyebrow="RQ4 · Concept drift"
        title="Training-side drift is observable; input-side drift is honestly out of scope."
        intro={
          <>
            <p>
              RQ4 asks how a federated model handles input-distribution
              shift over time — sensor calibration drifting between
              quarters, fleet ageing, seasonal operating-regime changes.
              CMAPSS does not naturally have this property: every engine
              runs to failure in a single regime, with no temporal axis
              spanning multiple training rounds. Our partition is static.
            </p>
            <p>
              What we <em>can</em> observe — and have observed — is{" "}
              <strong>training-side drift</strong>: the optimisation
              trajectory drifts under FedAvg-on-Non-IID because each
              client&apos;s local step pulls toward a different optimum.
              FedProx is designed specifically to bound this drift, and
              its μ-sweep is a quantitative answer to the
              &quot;does drift-control help?&quot; sub-question.
            </p>
          </>
        }
      >
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-lg border border-border bg-bg p-6">
            <div className="eyebrow !text-good mb-2">What we have</div>
            <ul className="space-y-3 text-[14.5px] text-text-dim leading-relaxed">
              <li>
                <strong className="text-text">FedProx μ-sweep:</strong>{" "}
                quantitative answer to &quot;does drift-control help?&quot;
                Yes, by ~6% gap-closed. Small but positive.
              </li>
              <li>
                <strong className="text-text">Per-round trajectories:</strong>{" "}
                every phase logs per-round loss + metrics, so the FedAvg-
                vs-FedProx-vs-FedRep difference across 50 rounds is fully
                visible.
              </li>
              <li>
                <strong className="text-text">
                  RQ3 cross-checkpoint drift:
                </strong>{" "}
                the function the model learns drifts depending on training-
                data scope (FD001-only vs combined). This is the dual of
                input concept drift — same data fed to differently-trained
                models gets different answers.
              </li>
              <li>
                <strong className="text-text">RQ7 attacker norm drift:</strong>{" "}
                the smoking-gun figure on the RQ7 page shows attacker-
                induced drift in update magnitudes over the 50 rounds.
                Anomalous over-time drift in the FL protocol is detectable.
              </li>
            </ul>
          </div>
          <div className="rounded-lg border border-border bg-bg p-6">
            <div className="eyebrow !text-text-muted mb-2">What we don&apos;t</div>
            <ul className="space-y-3 text-[14.5px] text-text-dim leading-relaxed">
              <li>
                <strong className="text-text">
                  Temporal partition of clients:
                </strong>{" "}
                we would need to split each client&apos;s engines by some
                temporal axis (e.g. engines 1–25 train rounds 1–25, engines
                26–50 train rounds 26–50) to simulate &quot;fleet ageing
                mid-training&quot;. We don&apos;t do this — our partition
                is static across all 50 rounds.
              </li>
              <li>
                <strong className="text-text">
                  Mid-training distribution shift:
                </strong>{" "}
                the canonical RQ4 experiment injects a distribution shift
                at round 25 (e.g. sensor calibration changes for one
                client) and measures recovery. Not done.
              </li>
              <li>
                <strong className="text-text">
                  Input concept drift natively:
                </strong>{" "}
                CMAPSS engines don&apos;t age across the federated
                rounds — each engine is a complete run-to-failure trace.
                The dataset itself doesn&apos;t support this experiment
                without synthetic augmentation.
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-10 rounded-lg border border-accent/40 bg-accent-subtle p-6 max-w-[78ch]">
          <div className="eyebrow !text-accent mb-2">RQ4 conclusion</div>
          <p className="text-[15.5px] text-text leading-relaxed">
            <strong>Honest framing</strong>: we have training-side drift
            evidence (FedProx, per-round trajectories, cross-checkpoint
            comparison, RQ7 attacker norms) but our partition is static so
            input-side concept drift is out of scope for this round. The
            full RQ4 study requires either a temporal partition of CMAPSS
            engines or a different dataset with natural temporal structure.
          </p>
        </div>
      </ExperimentSection>

      {/* WHY THIS MATTERS ----------------------------------------------- */}
      <ExperimentSection
        eyebrow="Why this synthesis exists"
        title="Honest scope is a feature, not a bug."
        intro={
          <>
            <p>
              The original project brief listed seven research questions.
              The temptation when wrapping up is to claim more than was
              actually done. The opposite move — committing transparently
              to which RQs are answered, which are synthesised, which are
              open — produces a stronger thesis defence than the
              alternative.
            </p>
            <p>
              For the writeup, the framing is:{" "}
              <em>
                &quot;Three RQs answered directly (RQ2, RQ3, RQ7), two
                synthesised from cross-cutting evidence (RQ4 partially,
                RQ5 substantially), two scoped as open follow-ups
                (RQ1 sensor heterogeneity, RQ6 privacy).&quot;
              </em>{" "}
              That covers all seven questions in the brief without
              over-claiming any of them.
            </p>
          </>
        }
      >
        <div className="mt-6 flex flex-wrap gap-4">
          <NavLink
            to="/abstract"
            className="
              inline-flex items-center gap-2 px-4 py-2.5 rounded-md
              border border-border text-text font-medium text-sm
              hover:border-border-strong hover:bg-bg-subtle/40 transition
            "
          >
            See the full verdict matrix on /abstract <span aria-hidden>→</span>
          </NavLink>
          <NavLink
            to="/experiments/06-non-iid"
            className="
              inline-flex items-center gap-2 px-4 py-2.5 rounded-md
              text-accent font-medium text-sm hover:underline
            "
          >
            Re-read the P6 baseline that grounds RQ5{" "}
            <span aria-hidden>→</span>
          </NavLink>
        </div>
      </ExperimentSection>
    </ExperimentLayout>
  );
}
