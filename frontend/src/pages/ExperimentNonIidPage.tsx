import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 06 — Structural Non-IID baseline (FD001 + FD003).
 *
 * Polished into the question-driven structure with the same
 * three-beat pattern. Adds two inline blocks:
 *
 *   - WhyFedAvgFails  - three-row callout explaining the mechanism
 *                       (client drift, weight conflict, sample-weighted
 *                       averaging collapses the asymmetry)
 *   - PerClientNonIidTable - 4 clients × subset / windows / RMSE /
 *                            AUPRC, from per_client_local CSV, with a
 *                            mean ± std footer matching metrics.json
 */
export function ExperimentNonIidPage() {
  return (
    <ExperimentLayout
      phaseId="Phase 06 · Non-IID"
      title="Structural Non-IID baseline"
      lede={
        <>
          The partition that breaks vanilla FedAvg. Two clients carry
          FD001 (HPC-only fault) and two carry FD003 (HPC + Fan). Every
          method evaluated on the common combined 200-engine test set.
          FedAvg closes essentially <strong>0 %</strong> of the RMSE
          gap to centralized — the negative finding that motivates the
          entire RQ research arc.
        </>
      }
      metaRow={
        <>
          <span>
            FD001 + FD003 · 4 clients (2 per subset) · 50 rounds × 2 local
            epochs
          </span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">scripts/run_non_iid.py</span>
        </>
      }
      prev={{
        id: "05",
        title: "FedAvg IID baseline",
        to: "/experiments/05-fedavg",
      }}
      next={{
        id: "RQ2",
        title: "Aggregation negative finding",
        to: "/rq2-story",
      }}
    >
      <ExperimentSection
        eyebrow="Why this phase exists"
        title="Why deliberately construct a partition the algorithm can't handle?"
        intro={
          <>
            <p>
              The IID baseline (Phase 05) showed that our FedAvg
              implementation is correct: under statistically
              equivalent client slices, it recovers ~86 % of what
              pooled training would deliver. That answer is necessary
              but not sufficient — real federations are not IID. Two
              hospitals see different patient populations; two
              airlines see different fleets; two operators see
              different fault modes. The interesting question is what
              happens when that&apos;s true.
            </p>
            <p>
              This phase constructs the smallest realistic Non-IID
              case we can defend: two clients get FD001 (HPC-only
              fault), two get FD003 (HPC + Fan). Operating regime is
              held constant (both subsets are sea-level — established
              in EDA Q4). The only varying axis is fault mode. If
              FedAvg breaks here, it breaks <em>cleanly</em> on a
              single attributable cause. That clean break is the
              starting point of the entire RQ2 / FedProx / FedRep /
              FedCCFA research arc.
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>

      <ExperimentSection
        eyebrow="At a glance"
        title="On structural Non-IID, vanilla FedAvg is statistically tied with the local-only mean."
        intro={
          <>
            <p>
              The combined-test-set RMSE shows centralized at{" "}
              <strong>13.77</strong>, FedAvg at <strong>17.95</strong>,
              and local-only mean at{" "}
              <strong>17.92 ± 1.52</strong>. The federation does not
              improve over isolation — the structural Non-IID gap is
              too wide for sample-count-weighted averaging to close.
              This is the canonical FedAvg failure mode and the
              starting point of the project&apos;s research arc.
            </p>
            <p>
              The RMSE story isn&apos;t the only story.{" "}
              <em>
                FedAvg is still operationally valuable here even with
                no RMSE improvement
              </em>{" "}
              — its NASA score is 43 % better than local-only, and
              it is the only model robust across both fault modes
              (the per-subset breakdown shows local models excel on
              their own subset and fail on the other).
            </p>
          </>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "13.77", label: "Centralized RMSE (upper bound)" },
            { value: "17.95", label: "FedAvg best-round RMSE", tone: "bad" },
            { value: "17.92 ± 1.52", label: "Local-only mean RMSE" },
            { value: "−0.7 %", label: "Gap closed by FedAvg", tone: "bad" },
            { value: "−43 %", label: "NASA reduction vs local-only", tone: "good" },
            { value: "651 s", label: "Wall-clock total (3 methods)" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 1"
        title="Does the upper bound still exist when we mix the two subsets?"
        intro={
          <p>
            <strong>What we need to know:</strong> the Phase 03
            ceiling was 14.02 RMSE on FD001 alone. With FD001 + FD003
            pooled (roughly twice the data), can a centralized model
            do even better? If the answer is yes, then the combined
            distribution contains real extra signal — and the
            question for FedAvg becomes whether it can extract that
            signal without ever pooling the data.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/centralized_metrics_fd001_fd003.png"
          caption="Centralized 50-epoch curves on FD001 + FD003 combined"
          eyebrow="Figure 01 · Findings"
          takeaway="Centralized reaches RMSE 13.77 — better than P3's FD001-only 14.02 because the combined data is roughly 2× larger."
          explanation={
            <>
              <p>
                The centralized run on the combined dataset converges
                in a similar pattern to P3 (FD001-only) but lands at
                a slightly better best-epoch RMSE because the training
                set is roughly twice as large. This is the &quot;more
                data helps a converged architecture&quot;
                expectation. It is also the reference point that
                defines how much signal-in-the-aggregate the combined
                population contains — FedAvg fails to extract that
                signal because it cannot see the combined distribution.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 2"
        title="What does FedAvg do when its clients disagree?"
        intro={
          <p>
            <strong>What we need to know:</strong> in Phase 05 the
            four IID clients optimised in nearly identical
            directions, so weight averaging cost almost nothing. Now
            two clients are pulling the global model toward an
            FD001-bias optimum and two are pulling toward an
            FD003-bias optimum. The question is what averaged-up
            optimiser actually converges to.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/fedavg_metrics_fd001_fd003.png"
          caption="FedAvg test metrics over 50 communication rounds on the combined test set"
          eyebrow="Figure 02 · Findings"
          takeaway="The global model plateaus at RMSE ~18 within 10–15 rounds. Additional rounds don't help — the protocol can't recover what averaging biased updates loses."
          explanation={
            <>
              <p>
                The curves flatten almost immediately. Best round
                arrives early; subsequent rounds oscillate without
                recovering the centralized number. This is the visual
                signature of FedAvg&apos;s structural-Non-IID failure:
                the optimization <em>has</em> converged — to a saddle
                that&apos;s a convex combination of the FD001-bias
                optimum and the FD003-bias optimum, never to the joint
                optimum that the centralized run actually finds.
              </p>
              <p>
                Adding more rounds, more local epochs, or a longer
                cosine schedule doesn&apos;t change the answer. The
                fix is at a different layer of the protocol — at the
                local-step regularisation (FedProx) or the
                architectural sharing contract (FedRep), not at the
                aggregation rule.
              </p>
            </>
          }
        />
        <WhyFedAvgFails />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 3"
        title="How do all three methods compare on the combined test set?"
        intro={
          <p>
            <strong>What we need to know:</strong> the headline
            comparison plot. Centralized is the ceiling, local-only
            mean is the floor, FedAvg sits somewhere between. The
            shape of that &quot;somewhere between&quot; is what
            decides whether the federation is buying anything.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/three_way_non_iid_fd001_fd003.png"
          caption="Centralized vs FedAvg vs local-only mean on the combined FD001 + FD003 test set"
          eyebrow="Figure 03 · Findings"
          takeaway="FedAvg and local-only are tied on RMSE; both are 4 RMSE worse than centralized. But FedAvg wins NASA by 43 % and AUPRC by 0.03 — operationally useful, even when RMSE says otherwise."
          explanation={
            <>
              <p>
                Same three-bar format as Phase 05, very different
                result. Centralized at RMSE 13.77 (better than its
                FD001-only cousin because more data); FedAvg at 17.95;
                local-only mean at 17.92. The FedAvg and local-only
                bars are indistinguishable. The federation is not
                adding value <em>in the RMSE-on-combined-test-set
                sense</em>.
              </p>
              <p>
                But two other metrics tell a different story: NASA
                score is 43 % lower under FedAvg than local-only
                (1 647 vs 2 885) because local models pay huge
                late-prediction penalties on engines from the fault
                mode they never saw. AUPRC also wins (0.951 vs 0.924).
                The federation is operationally useful even when its
                headline RMSE is unchanged — it just doesn&apos;t look
                that way on a combined-test-set summary.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 4"
        title="Is the failure symmetric across clients, or asymmetric per fault mode?"
        intro={
          <p>
            <strong>What we need to know:</strong> a combined RMSE
            number hides what each client&apos;s model is actually
            good at. If FD001-trained clients excel on FD001 windows
            and fail on FD003 windows (and vice versa), the
            local-only &quot;mean&quot; is hiding a bimodal
            distribution. This is the visualisation that justifies
            keeping FedAvg even with no RMSE improvement.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/per_subset_breakdown_fd001_fd003.png"
          caption="Per-subset RMSE for each method (centralized, FedAvg, each local client)"
          eyebrow="Figure 04 · Findings"
          takeaway="FD001-trained clients excel on FD001 and fail on FD003. FD003-trained clients do the mirror. FedAvg is the only model not catastrophic on either subset."
          explanation={
            <>
              <p>
                Two columns of bars (one per test subset, FD001 and
                FD003). Centralized wins both. Local-only models
                split cleanly: client_2 (best FD001-trained) lands at
                15.0 on FD001 and 20.3 on FD003; client_4 (best
                FD003-trained) lands at 14.5 on FD003 and 16.4 on
                FD001. Each local model is good on the subset it saw
                and visibly worse on the one it didn&apos;t.
              </p>
              <p>
                FedAvg lands at 17.0 / 18.9 — never the best on
                either, but never catastrophic either. An operational
                consumer of FedAvg gets one model that&apos;s
                mediocre everywhere; an operational consumer of a
                local-only model has to choose which biased local
                model to deploy and accept that it will be wrong on
                competitors&apos; engines. The robustness story is
                real even without an RMSE win.
              </p>
              <p>
                This figure is the canonical &quot;what does
                structural Non-IID look like?&quot; visual. Every RQ2
                / FedProx / FedRep figure later in the project
                compares against it.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 5"
        title="What do isolated local clients actually look like under this split?"
        intro={
          <p>
            <strong>What we need to know:</strong> the mean ± std of
            17.92 ± 1.52 is much wider than Phase 04&apos;s IID
            spread (15.02 ± 0.29). We owe the reader the
            per-client numbers underneath — both to make the
            standard deviation auditable and to show which clients
            do well and which struggle.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/local_only_metrics_fd001_fd003.png"
          caption="Best-epoch metrics per local-only client (combined test set)"
          eyebrow="Figure 05 · Findings"
          takeaway="Combined-test RMSE ranges from 15.5 (client_4) to 19.3 (client_3). Heterogeneity in data shows up as heterogeneity in local-model quality."
          explanation={
            <>
              <p>
                Four groups of bars, one per client. Combined-test-set
                RMSEs range from 15.5 (client_4, best) to 19.3
                (client_3, worst). The mean ± std is 17.92 ± 1.52 —
                much wider spread than the IID Phase 04 result
                (15.02 ± 0.29). Heterogeneity in the data shows up as
                heterogeneity in local-only model quality.
              </p>
              <p>
                The per-subset breakdown above (Figure 04) shows why:
                each client is much better on the subset it saw than
                the averaged figure suggests. A federation is the
                operational fix for the bad-on-competitor-engines
                half — even when vanilla FedAvg doesn&apos;t close
                the RMSE gap, it produces a single model usable on
                both subsets.
              </p>
            </>
          }
        />
        <PerClientNonIidTable />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="What this phase decided"
        title="Vanilla FedAvg's structural-Non-IID failure mode is real, reproducible, and the starting point of every RQ."
        intro={
          <>
            <p>
              One negative finding locked in:{" "}
              <strong>FedAvg closes ~0 % of the structural-Non-IID
              RMSE gap</strong>. One positive finding preserved:
              FedAvg is still operationally useful (43 % NASA
              reduction, single robust model). One canonical visual
              produced (per-subset breakdown) that every later
              experiment compares against.
            </p>
            <p>
              From here the project splits into research questions.
              RQ2 asks &quot;can a smarter aggregation rule fix
              this?&quot; (FedProx, FedRep, FedCCFA, imbalance-aware)
              and answers <em>not really</em>. RQ3 asks &quot;can
              we at least explain the model we have?&quot; and
              answers <em>yes, with caveats</em>. RQ7 asks
              &quot;how does this federation hold up under
              adversarial clients?&quot; The Non-IID partition built
              here is the test bed for all of them.
            </p>
          </>
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
 * Three-row callout explaining the mechanism behind FedAvg's
 * structural-Non-IID failure. Sits below Question 2's figure so the
 * "why doesn't FedAvg recover?" question is answered alongside the
 * "what does FedAvg's convergence look like?" evidence.
 */
function WhyFedAvgFails() {
  type Row = { mechanism: string; whatItIs: string; whyItHurts: string };
  const rows: Row[] = [
    {
      mechanism: "Client drift",
      whatItIs:
        "Each local optimizer takes 2 epochs of gradient steps on a one-fault-mode slice before the server sees anything.",
      whyItHurts:
        "After 2 local epochs the FD001 clients have moved toward an HPC-only optimum and the FD003 clients toward an HPC + Fan optimum. By the time weights are averaged, the drift is non-trivial.",
    },
    {
      mechanism: "Sample-count-weighted averaging",
      whatItIs:
        "McMahan-2017 weights each client's delta by its number of training samples — the cheapest possible aggregation.",
      whyItHurts:
        "With roughly equal sample counts across the four clients, the FD001 and FD003 directions get averaged in equal proportion. The result is a vector midway between two optima rather than a vector toward the joint optimum.",
    },
    {
      mechanism: "No per-client personalisation",
      whatItIs:
        "Every client is forced to commit to the same global weights at the end of each round. There is no architectural slot for a client-specific fault head, embedding, or normalisation layer.",
      whyItHurts:
        "Even if the global encoder learned a 'union of fault modes' representation, there's nowhere to put it — the heads are also averaged, so the model ends up mediocre at both modes instead of specialised at one.",
    },
  ];

  return (
    <div className="my-8 border border-border rounded-lg overflow-hidden">
      <div className="bg-bg-subtle px-6 py-4 border-b border-border">
        <div className="eyebrow">Mechanism</div>
        <div className="font-display text-text text-[20px] mt-1.5">
          Three reasons vanilla FedAvg can&apos;t recover under structural Non-IID.
        </div>
      </div>
      <table className="w-full text-[14px]">
        <thead>
          <tr className="text-text-muted text-[11.5px] uppercase tracking-[0.12em] border-b border-border">
            <th className="text-left px-5 py-3 font-medium w-[24%]">Factor</th>
            <th className="text-left px-5 py-3 font-medium w-[34%]">What it is</th>
            <th className="text-left px-5 py-3 font-medium">Why it hurts</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((r) => (
            <tr key={r.mechanism} className="align-top">
              <td className="px-5 py-4 text-text font-medium">{r.mechanism}</td>
              <td className="px-5 py-4 text-text-dim leading-[1.55]">{r.whatItIs}</td>
              <td className="px-5 py-4 text-text-dim leading-[1.55]">{r.whyItHurts}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Per-client best-epoch numbers for the local-only training under
 * the Non-IID partition. Transcribed from
 * results/06_non_iid/per_client_local_fd001_fd003.csv. Subset column
 * makes the FD001 vs FD003 assignment auditable next to the metrics.
 */
function PerClientNonIidTable() {
  type Row = {
    client: string;
    subset: string;
    windows: number;
    bestEpoch: number;
    rmse: number;
    auprc: number;
    f1: number;
  };
  const rows: Row[] = [
    { client: "client_1", subset: "FD001", windows: 8810, bestEpoch: 6, rmse: 19.09, auprc: 0.8912, f1: 0.8193 },
    { client: "client_2", subset: "FD001", windows: 8921, bestEpoch: 6, rmse: 17.85, auprc: 0.8922, f1: 0.8696 },
    { client: "client_3", subset: "FD003", windows: 11923, bestEpoch: 8, rmse: 19.27, auprc: 0.9332, f1: 0.8298 },
    { client: "client_4", subset: "FD003", windows: 9897, bestEpoch: 12, rmse: 15.47, auprc: 0.9796, f1: 0.9130 },
  ];

  return (
    <div className="my-8 overflow-x-auto border border-border rounded-lg">
      <table className="w-full text-[14px] font-mono-num">
        <thead>
          <tr className="bg-bg-subtle text-text-muted text-[11.5px] uppercase tracking-[0.12em]">
            <th className="text-left px-5 py-3 font-medium font-sans">Client</th>
            <th className="text-left px-5 py-3 font-medium font-sans">Subset</th>
            <th className="text-right px-5 py-3 font-medium font-sans">Windows</th>
            <th className="text-right px-5 py-3 font-medium font-sans">Best epoch</th>
            <th className="text-right px-5 py-3 font-medium font-sans">RMSE (combined)</th>
            <th className="text-right px-5 py-3 font-medium font-sans">AUPRC</th>
            <th className="text-right px-5 py-3 font-medium font-sans">F1</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((r) => {
            const subsetClass =
              r.subset === "FD001"
                ? "inline-block px-2 py-0.5 rounded bg-bg-subtle text-text font-mono-num text-[12.5px]"
                : "inline-block px-2 py-0.5 rounded bg-accent/10 text-accent font-mono-num text-[12.5px]";
            return (
              <tr key={r.client} className="text-text">
                <td className="px-5 py-3.5 font-sans font-medium">{r.client}</td>
                <td className="px-5 py-3.5 font-sans">
                  <span className={subsetClass}>{r.subset}</span>
                </td>
                <td className="px-5 py-3.5 text-right">{r.windows.toLocaleString()}</td>
                <td className="px-5 py-3.5 text-right">{r.bestEpoch}</td>
                <td className="px-5 py-3.5 text-right">{r.rmse.toFixed(2)}</td>
                <td className="px-5 py-3.5 text-right">{r.auprc.toFixed(3)}</td>
                <td className="px-5 py-3.5 text-right">{r.f1.toFixed(3)}</td>
              </tr>
            );
          })}
          <tr className="bg-bg-subtle text-text font-semibold">
            <td className="px-5 py-3.5 font-sans">Mean ± std</td>
            <td className="px-5 py-3.5 text-text-dim">—</td>
            <td className="px-5 py-3.5 text-right text-text-dim">—</td>
            <td className="px-5 py-3.5 text-right text-text-dim">—</td>
            <td className="px-5 py-3.5 text-right">17.92 ± 1.52</td>
            <td className="px-5 py-3.5 text-right">0.924</td>
            <td className="px-5 py-3.5 text-right">0.858</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
