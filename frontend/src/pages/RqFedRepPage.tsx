import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * RQ2 follow-up: FedRep.
 *
 * Federate only the encoder + trunk. Each client keeps its own
 * RUL + fault heads, trained locally and never aggregated. This is
 * the architectural follow-up to RQ2's negative result, and it
 * produces the project's largest positive finding: +73% gap closed.
 */
export function RqFedRepPage() {
  return (
    <ExperimentLayout
      phaseId="RQ2 follow-up · FedRep"
      title="FedRep — per-client heads"
      lede={
        <>
          The architectural follow-up. Federate the shared{" "}
          <strong>encoder + trunk</strong> via FedAvg; each client keeps
          its own RUL and fault <strong>heads</strong> local. The shared
          representation learns fault-mode-agnostic degradation features;
          per-client heads avoid forcing one classifier to span both fault
          modes. <strong>+73% of the Non-IID gap closed</strong> — the
          project&apos;s largest positive finding.
        </>
      }
      metaRow={
        <>
          <span>FD001 + FD003 · 4 clients · 50 rounds · 2 local epochs · per-client heads</span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">scripts/run_fedrep.py</span>
        </>
      }
      prev={{
        id: "FedProx",
        title: "Proximal regularisation",
        to: "/rq2-followups/fedprox",
      }}
      next={{
        id: "FedCCFA",
        title: "Clustered heads",
        to: "/rq2-followups/fedccfa",
      }}
    >
      <ExperimentSection
        eyebrow="At a glance"
        title="Architecture beats both aggregation tricks and drift control."
        intro={
          <>
            <p>
              Macro RMSE (mean of per-client per-subset RMSEs) lands at{" "}
              <strong>14.91</strong> — closing 73% of the gap between
              vanilla FedAvg (17.95) and centralized (13.77). On FD001 the
              best client actually <em>beats</em> centralized (14.34 vs
              14.80). On FD003 it stays a few RMSE behind centralized but
              improves 3.5 RMSE over vanilla FedAvg.
            </p>
            <p>
              The trick: a 29,888-parameter shared encoder + a 130-
              parameter per-client head. Almost all the model is
              federated; the tiny head per client is what unlocks the
              gains.
            </p>
          </>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "14.91", label: "FedRep macro RMSE", tone: "good" },
            { value: "+73 %", label: "Non-IID gap closed", tone: "good" },
            { value: "14.34", label: "FD001 best (beats centralized's 14.80)", tone: "good" },
            { value: "29,888", label: "Shared encoder params" },
            { value: "130", label: "Per-client head params (×4)" },
            { value: "11 / 11", label: "Tests passing on FedRep module" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="The headline figure"
        title="FedRep is the only method that closes most of the structural Non-IID gap."
      >
        <ExplainedFigure
          artifactPath="results/rq2_fedrep/headline_comparison_fd001+fd003.png"
          caption="FedRep vs P6 references vs FedProx"
          eyebrow="Figure 01"
          takeaway="Centralized 13.77 → FedRep 14.91 → FedProx 17.70 → vanilla FedAvg 17.95 → local-only 17.92. FedRep is the visible jump."
          explanation={
            <>
              <p>
                Five bars on the same combined-test-set RMSE axis. The
                visual hierarchy is exactly the empirical layer hierarchy
                this trilogy proves: centralized is the upper bound,
                FedRep gets close to it, FedProx + RQ2 + vanilla FedAvg
                cluster ~4 RMSE above, local-only mean is the floor.
              </p>
              <p>
                Critically, FedRep is the <em>only</em> method tested in
                the entire project that gets close to centralized without
                actually pooling the data. That makes it the most
                operationally relevant result for a real airline
                consortium: per-client heads is the right architectural
                contract.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Per-client RMSE"
        title="Each client converges to its own near-optimal solution."
      >
        <ExplainedFigure
          artifactPath="results/rq2_fedrep/per_client_rmse_fd001+fd003.png"
          caption="Per-client test RMSE across rounds — each client evaluated on its own subset"
          eyebrow="Figure 02"
          takeaway="Each client's RMSE curve converges to a near-centralized number for ITS subset. The shared encoder doesn't hurt; the local head wins."
          explanation={
            <>
              <p>
                Four curves, one per client. FD001 clients converge to
                RMSE ~14.5; FD003 clients converge to RMSE ~15.2. Each
                client gets a model that is essentially as good as the
                centralized model would be on that client&apos;s data
                slice — because the encoder learned features useful to
                both subsets, and the head learned to map those features
                to that subset&apos;s specific output distribution.
              </p>
              <p>
                The macro RMSE (14.91) is the mean of these four
                per-client numbers. Reporting macro is the honest
                operational metric for FedRep: each airline deploys its
                own client&apos;s checkpoint, not a single combined one.
                Combined RMSE on the pooled test set is not the right
                comparison for this architecture.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Per-subset story"
        title="FD001 actually beats centralized."
      >
        <ExplainedFigure
          artifactPath="results/rq2_fedrep/per_subset_breakdown_fd001+fd003.png"
          caption="Per-subset RMSE — FedRep vs centralized vs vanilla FedAvg"
          eyebrow="Figure 03"
          takeaway="On FD001 FedRep (14.34) edges out centralized (14.80). On FD003 it stays ~3 RMSE behind centralized but improves +3.5 over vanilla FedAvg."
          explanation={
            <>
              <p>
                FedRep on FD001 lands at RMSE 14.34 — slightly{" "}
                <strong>better</strong> than centralized&apos;s 14.80.
                This isn&apos;t magic: FedRep&apos;s FD001 client trains
                a head specifically tuned to FD001&apos;s single-fault-mode
                distribution, while centralized trains one head that
                must straddle both fault modes. On the easier subset,
                specialisation wins.
              </p>
              <p>
                On FD003 (the harder, two-fault-mode subset) centralized
                still wins because it has access to FD001 examples too,
                which help the model generalise. But FedRep closes most
                of the gap there as well — well within deployable
                territory.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Verdict + caveat"
        title="The project's largest positive finding — with one honest caveat."
        intro={
          <>
            <p>
              FedRep is the strongest result in this project. The trick is
              architectural, not optimisational, and it points at a
              general principle: when client distributions are structurally
              heterogeneous, federate the parts that should be shared
              (representation) and localise the parts that should differ
              (classifier head).
            </p>
            <p>
              <strong>The caveat</strong>: there is no longer a single
              downloadable global model. Each client has its own head. A
              reviewer might object &quot;you didn&apos;t really federate;
              you just admitted defeat.&quot; The defence is FedCCFA-style:
              federated <em>representation</em> learning is a legitimate
              goal in its own right — the shared encoder IS the shared
              knowledge; the head is local interpretation. The next page
              (FedCCFA) tests whether a clustering refinement on top of
              FedRep can close even more of the gap.
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>
    </ExperimentLayout>
  );
}
