import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * RQ2 follow-up: FedCCFA.
 *
 * Clustered classifier-head federation. Built on top of FedRep:
 * pairwise cosine similarity of client heads + connectivity-based
 * clustering + per-cluster head averaging. The hypothesis was that
 * grouping similar clients would close even more of the gap.
 *
 * The actual result is a null finding — heads collapse to a single
 * cluster on this dataset. The page documents WHY, which is itself
 * publishable.
 */
export function RqFedCcfaPage() {
  return (
    <ExperimentLayout
      phaseId="RQ2 follow-up · FedCCFA"
      title="FedCCFA — clustered heads (null result)"
      lede={
        <>
          The third RQ2 follow-up. Build on FedRep&apos;s per-client heads
          by clustering similar heads (pairwise cosine + connectivity) and
          averaging within clusters. The natural expected structure was
          two clusters: {`{`}client_1, client_2{`}`} on FD001 and
          {` {`}client_3, client_4{`}`} on FD003. <strong>What actually
          happens: all four heads collapse to a single cluster.</strong>{" "}
          The algorithm works correctly — it is the heads that fail to
          diverge. That null result is publishable in its own right.
        </>
      }
      metaRow={
        <>
          <span>FD001 + FD003 · FedRep + cosine clustering · 3-round warmup</span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">scripts/run_fedccfa.py</span>
        </>
      }
      prev={{
        id: "FedRep",
        title: "Per-client heads",
        to: "/rq2-followups/fedrep",
      }}
      next={{
        id: "RQ7",
        title: "Security",
        to: "/rq7-story",
      }}
    >
      <ExperimentSection
        eyebrow="At a glance"
        title="The clustering works correctly — and discovers there is nothing to cluster."
        intro={
          <p>
            FedCCFA&apos;s final RMSE is essentially tied with FedRep
            (15.00 vs 14.91 macro). The clustering algorithm runs cleanly,
            but every round after the 3-round warmup it merges all 4
            clients into one cluster. We re-ran with similarity threshold
            0.99 — heads were still indistinguishable. This is a real
            architectural finding about the per-head capacity, not an
            implementation bug.
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "15.00", label: "FedCCFA macro RMSE" },
            { value: "14.91", label: "FedRep macro RMSE (essentially tied)" },
            { value: "1 / 4", label: "Final cluster count (expected 2)", tone: "bad" },
            { value: ">0.99", label: "Pairwise head similarity (rounds 4-50)" },
            { value: "130", label: "Per-client head params" },
            { value: "14 / 14", label: "Unit tests passing on FedCCFA module" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="The headline figure"
        title="FedCCFA RMSE is statistically tied with FedRep."
      >
        <ExplainedFigure
          artifactPath="results/rq2_fedccfa/headline_comparison_fd001+fd003.png"
          caption="FedCCFA vs FedRep + references"
          eyebrow="Figure 01"
          takeaway="No daylight between FedCCFA and FedRep on combined RMSE. The clustering refinement adds nothing because there is nothing to refine."
          explanation={
            <>
              <p>
                FedCCFA at 15.00, FedRep at 14.91 — a 0.09 RMSE
                difference, well within run-to-run noise on this test
                set. The clustering machinery activated, ran, made a
                decision every round, and did not improve the outcome
                because it always merged everything.
              </p>
              <p>
                On a different dataset — one where per-client heads{" "}
                <em>did</em> diverge — this same algorithm would
                presumably help. The null result here is conditional on
                this dataset and this architecture, not a refutation of
                clustered federated learning in general.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="The smoking gun"
        title="Cluster evolution heatmap — solid blue from round 2 onward."
      >
        <ExplainedFigure
          artifactPath="results/rq2_fedccfa/cluster_evolution_fd001+fd003.png"
          caption="Per-round cluster assignment heatmap"
          eyebrow="Figure 02"
          takeaway="After round 2, all 4 clients share one cluster ID for the next 48 rounds. The heads have nothing to distinguish themselves with."
          explanation={
            <>
              <p>
                Four rows (one per client) × 50 columns (one per round),
                cells coloured by cluster ID. From round 2 onward the
                heatmap is solid blue — all four clients belong to
                cluster 0 every round. The 3-round warmup was meant to
                give heads time to differentiate; in practice they
                converge to near-identical parameter vectors immediately.
              </p>
              <p>
                We re-ran with similarity threshold 0.99 (extremely
                strict) and got the same heatmap. The heads are not just
                close — they are nearly identical. This is the visual
                ground truth for the &quot;null result with diagnosis&quot;
                framing.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Why it didn't help"
        title="Three stacked causes — all architectural, none are bugs."
        intro={
          <>
            <p>
              The three reasons the heads fail to diverge are stacked
              and each is independently sufficient to explain the
              collapse:
            </p>
            <ol className="mt-4 list-decimal list-inside space-y-3 max-w-[78ch]">
              <li>
                <strong>Same initialisation.</strong> Vanilla FedAvg
                requires all clients to start from identical weights for
                cold-start convergence. The heads start at the same
                point.
              </li>
              <li>
                <strong>Tiny head capacity.</strong> Each head is a
                single linear layer: 64 inputs → 1 output = 65 parameters
                per head, 130 with both RUL and fault heads. There simply
                isn&apos;t enough parameter space for the heads to drift
                far apart, even if the data wanted them to.
              </li>
              <li>
                <strong>Shared averaged encoder.</strong> Every round
                resets the upstream features the heads see. Even if the
                heads briefly diverge during one round&apos;s local
                training, the next round&apos;s averaged encoder pulls
                their inputs back to a shared distribution, making
                divergent heads incoherent.
              </li>
            </ol>
          </>
        }
      >
        <></>
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Verdict + future work"
        title="Null result with diagnosis — points at a clean follow-up."
        intro={
          <>
            <p>
              On this dataset and this architecture, head clustering
              cannot help because the heads do not develop the diversity
              clustering needs to act on. The result is null but the
              diagnosis is precise.
            </p>
            <p>
              <strong>Two natural follow-ups</strong>: (a) larger
              per-client heads (multi-layer with non-linearities, more
              capacity for divergence), (b) cluster-aware initialisation
              (different head inits per cluster), perhaps combined. Both
              would test whether the architectural constraint is the
              binding one, or whether even with more head capacity the
              shared encoder pulls everything back together. That
              experiment is scoped but not run in this project.
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>
    </ExperimentLayout>
  );
}
