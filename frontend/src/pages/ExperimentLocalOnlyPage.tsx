import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 04 — Local-only baseline (4 isolated clients, FD001).
 *
 * Polished into the question-driven structure with the same
 * three-beat pattern. Adds a per-client table (transcribed from
 * results/04_local_only/per_client_best_fd001.csv) so the
 * "mean 15.02 ± 0.29" claim is auditable from a single block.
 */
export function ExperimentLocalOnlyPage() {
  return (
    <ExperimentLayout
      phaseId="Phase 04 · Local-only"
      title="Isolated per-client training"
      lede={
        <>
          What happens if every airline trains alone? Four independent
          50-epoch runs, same recipe as the centralized baseline, but
          each model sees only its 25-engine slice and shares nothing.
          Evaluated on the common test set so per-client numbers are
          directly comparable to Phase 03. The mean of the four sets
          the federation&apos;s <em>lower bound</em> — FedAvg must beat
          this for the federation to have a point.
        </>
      }
      metaRow={
        <>
          <span>
            FD001 · 4 clients × 50 epochs · stratified-by-lifetime
          </span>
          <span className="text-text-muted/50">·</span>
          <span className="font-mono-num">scripts/run_local_only.py</span>
        </>
      }
      prev={{
        id: "03",
        title: "Centralized baseline",
        to: "/experiments/03-centralized",
      }}
      next={{
        id: "05",
        title: "FedAvg IID baseline",
        to: "/experiments/05-fedavg",
      }}
    >
      <ExperimentSection
        eyebrow="Why this phase exists"
        title="Why measure isolation when nobody is proposing it as a solution?"
        intro={
          <>
            <p>
              Nobody recommends every airline training alone. The
              reason this baseline exists is the same reason every
              experiment needs both an upper bound (Phase 03) and a
              lower bound — without both, we cannot say what fraction
              of the available signal a federated method actually
              recovers. &quot;Closes 86 % of the gap&quot; needs a gap
              to measure.
            </p>
            <p>
              The setup is deliberately symmetric to Phase 03: same
              architecture, same optimizer, same cosine schedule, same
              test set. The only thing that changes is who sees what
              data. Any performance penalty is a clean attribution to
              the isolation itself.
            </p>
          </>
        }
      >
        <></>
      </ExperimentSection>

      <ExperimentSection
        eyebrow="At a glance"
        title="Penalty for isolation is real but small — about +1 RMSE on IID FD001."
        intro={
          <p>
            With balanced data and the same training recipe, isolated
            clients land at <strong>RMSE 15.02 ± 0.29</strong> versus
            centralized&apos;s 14.02. The 1-RMSE penalty is the
            federation&apos;s target — FedAvg should close most of it
            under IID. The interesting case (where this gap becomes
            huge) is the Non-IID partition at Phase 06.
          </p>
        }
      >
        <HeadlineNumbers
          items={[
            { value: "15.02 ± 0.29", label: "Mean per-client test RMSE" },
            { value: "+1.00", label: "Penalty vs centralized (14.02)" },
            { value: "0.973", label: "Mean AUPRC (down 0.014)" },
            { value: "0.923", label: "Mean F1 (down 0.039)" },
            { value: "client_3", label: "Weakest client (RMSE 15.50)" },
            { value: "82.1 s", label: "Wall-clock total" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 1"
        title="How much does isolation actually cost, in raw numbers?"
        intro={
          <p>
            <strong>What we need to know:</strong> the upper bound is
            14.02. The lower bound is whatever isolated training
            delivers. The gap between them is the budget the
            federation has to play with — every metric where the gap
            is large is one a federated method can prove its worth on.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/04_local_only/centralized_vs_local_fd001.png"
          caption="P3 centralized vs P4 local-only mean across RMSE / NASA / AUPRC / F1"
          eyebrow="Figure 01 · Findings"
          takeaway="Centralized wins every metric, but the margins are modest — 1.0 RMSE, ~50 NASA, 0.014 AUPRC, 0.039 F1. The federation has 1 RMSE to recover."
          explanation={
            <>
              <p>
                Side-by-side bars for each metric (lower is better for
                RMSE / NASA; higher is better for AUPRC / F1).
                Centralized wins on every metric — that&apos;s the
                upper bound — but the margins are modest. RMSE differs
                by 1.0 cycle, NASA by ~50 points, AUPRC by 0.014, F1
                by 0.039.
              </p>
              <p>
                Why so small? Because the Phase 01 partition is
                deliberately near-IID: every client&apos;s 25 engines
                were stratified by lifetime so the fault-positive rates
                and engine-life distributions match. With 4 433 windows
                per client (a quarter of the centralized total), each
                local model is statistically saturated for this
                architecture; pooling the clients into one helps only
                marginally.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 2"
        title="Are all four clients equally penalized, or does someone get unlucky?"
        intro={
          <p>
            <strong>What we need to know:</strong> the mean ± std
            hides per-client variance. We need to look at each
            client&apos;s best-epoch numbers individually — both to
            see whether the partition was honest and to understand
            what an unlucky split looks like in metric space.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/04_local_only/per_client_metrics_fd001.png"
          caption="Bar chart of best-epoch RMSE / NASA / AUPRC / F1 for each of the 4 clients"
          eyebrow="Figure 02 · Findings"
          takeaway="client_3 lags slightly (RMSE 15.50) because it received the shortest-lifetime engines — less degradation signal per engine."
          explanation={
            <>
              <p>
                Four groups of four bars, one group per client.
                Clients 1, 2, and 4 land within RMSE 14.76 – 14.96 —
                essentially equivalent. Client 3 is the visible
                outlier at RMSE 15.50. The reason is in the partition:
                the stratified-by-lifetime split gave client_3 the
                engines with the shortest mean lifetime (205.5 cycles
                vs 206 – 207 elsewhere), so fewer cycles of degradation
                signal per engine and slightly worse generalisation.
              </p>
              <p>
                In a real-world deployment this asymmetry would be
                much worse — small airlines, regional carriers, and
                operators with newer fleets have orders of magnitude
                less failure data than larger players. The natural
                unfairness of FL is why server-side reweighting is so
                often proposed (and why RQ2 tests it).
              </p>
            </>
          }
        />
        <PerClientLocalOnlyTable />
        <p className="text-[15px] text-text-dim mt-6 max-w-[82ch]">
          Each client converges at a different epoch (15 to 28), all
          well after Phase 03&apos;s epoch-5 best — predictable, since
          each model sees only a quarter of the data and consequently
          needs roughly four times more passes to land at its own
          optimum.
        </p>
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Question 3"
        title="Do the four local training trajectories look similar?"
        intro={
          <p>
            <strong>What we need to know:</strong> if the four
            clients&apos; loss curves overlap visibly, the partition
            is genuinely near-IID and any later FedAvg under this
            partition should converge cleanly. If they diverge, even
            the IID setting is going to be harder than expected. This
            is the prediction we&apos;ll get to test in Phase 05.
          </p>
        }
      >
        <ExplainedFigure
          artifactPath="results/04_local_only/loss_curves_fd001.png"
          caption="Training loss across 50 epochs, one curve per client (log scale)"
          eyebrow="Figure 03 · Findings"
          takeaway="All four curves overlap visibly. The partition is near-IID; FedAvg should converge cleanly. (Phase 06 will show the opposite picture under Non-IID.)"
          explanation={
            <>
              <p>
                Four loss curves are virtually overlapping. Each
                client hits its own best epoch between 15 and 28 —
                later than centralized&apos;s epoch 5 because each
                model sees only a quarter of the data. The
                4×-data argument for centralized shows up here as
                4× earlier convergence.
              </p>
              <p>
                The visual overlap is itself a finding: under
                near-IID partitioning, every client&apos;s local
                trajectory is essentially the same. This is exactly
                the setting where FedAvg should excel — averaging
                weights from similar gradient updates is close to
                averaging the gradients centrally. The contrasting
                Phase 06 picture (where local curves diverge sharply)
                is what makes the Non-IID challenge concrete.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="What this phase decided"
        title="The lower bound is 15.02 RMSE. The federation has 1.00 RMSE to recover."
        intro={
          <>
            <p>
              One number locked in (mean RMSE 15.02 ± 0.29), one
              partition validated (the stratified split is honest
              enough that all four clients land within ~0.7 RMSE of
              each other), one expected weak client identified
              (client_3, the shortest-lifetime bucket). The
              federation budget is now well-defined: <strong>+1.00
              RMSE / +52 NASA / +0.014 AUPRC / +0.039 F1</strong>{" "}
              relative to centralized.
            </p>
            <p>
              Phase 05 next runs the canonical FedAvg protocol over
              the same 4-client partition and checks how much of that
              budget weight-sharing alone can recover.
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
 * Per-client best-epoch numbers, transcribed from
 * results/04_local_only/per_client_best_fd001.csv.
 */
function PerClientLocalOnlyTable() {
  type Row = {
    client: string;
    bestEpoch: number;
    rmse: number;
    nasa: number;
    auprc: number;
    f1: number;
    seconds: number;
  };
  const rows: Row[] = [
    { client: "client_1", bestEpoch: 20, rmse: 14.76, nasa: 402.0, auprc: 0.9725, f1: 0.9412, seconds: 19.9 },
    { client: "client_2", bestEpoch: 28, rmse: 14.96, nasa: 355.5, auprc: 0.9834, f1: 0.9600, seconds: 19.6 },
    { client: "client_3", bestEpoch: 18, rmse: 15.50, nasa: 528.9, auprc: 0.9643, f1: 0.8936, seconds: 19.8 },
    { client: "client_4", bestEpoch: 15, rmse: 14.84, nasa: 348.5, auprc: 0.9720, f1: 0.8980, seconds: 20.1 },
  ];

  return (
    <div className="my-8 overflow-x-auto border border-border rounded-lg">
      <table className="w-full text-[14px] font-mono-num">
        <thead>
          <tr className="bg-bg-subtle text-text-muted text-[11.5px] uppercase tracking-[0.12em]">
            <th className="text-left px-5 py-3 font-medium font-sans">Client</th>
            <th className="text-right px-5 py-3 font-medium font-sans">Best epoch</th>
            <th className="text-right px-5 py-3 font-medium font-sans">RMSE</th>
            <th className="text-right px-5 py-3 font-medium font-sans">NASA</th>
            <th className="text-right px-5 py-3 font-medium font-sans">AUPRC</th>
            <th className="text-right px-5 py-3 font-medium font-sans">F1</th>
            <th className="text-right px-5 py-3 font-medium font-sans">Train (s)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((r) => (
            <tr key={r.client} className="text-text">
              <td className="px-5 py-3.5 font-sans font-medium">{r.client}</td>
              <td className="px-5 py-3.5 text-right">{r.bestEpoch}</td>
              <td className="px-5 py-3.5 text-right">{r.rmse.toFixed(2)}</td>
              <td className="px-5 py-3.5 text-right">{r.nasa.toFixed(1)}</td>
              <td className="px-5 py-3.5 text-right">{r.auprc.toFixed(3)}</td>
              <td className="px-5 py-3.5 text-right">{r.f1.toFixed(3)}</td>
              <td className="px-5 py-3.5 text-right text-text-dim">{r.seconds.toFixed(1)}</td>
            </tr>
          ))}
          <tr className="bg-bg-subtle text-text font-semibold">
            <td className="px-5 py-3.5 font-sans">Mean ± std</td>
            <td className="px-5 py-3.5 text-right text-text-dim">—</td>
            <td className="px-5 py-3.5 text-right">15.02 ± 0.29</td>
            <td className="px-5 py-3.5 text-right">408.8</td>
            <td className="px-5 py-3.5 text-right">0.973</td>
            <td className="px-5 py-3.5 text-right">0.923</td>
            <td className="px-5 py-3.5 text-right text-text-dim">82.1 total</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
