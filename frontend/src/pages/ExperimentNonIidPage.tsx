import { ExplainedFigure } from "../components/ExplainedFigure";
import {
  ExperimentLayout,
  ExperimentSection,
  HeadlineNumbers,
} from "../components/ExperimentLayout";

/**
 * Phase 06 — Non-IID baseline (FD001 + FD003 mixed across 4 clients).
 *
 * The flagship "where vanilla FedAvg breaks" phase. All later RQ
 * work uses this exact partition.
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
          FedAvg closes essentially <strong>0%</strong> of the RMSE
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
        eyebrow="At a glance"
        title="On structural Non-IID, vanilla FedAvg is statistically tied with the local-only mean."
        intro={
          <>
            <p>
              The combined-test-set RMSE shows centralized at{" "}
              <strong>13.77</strong>, FedAvg at <strong>17.95</strong>, and
              local-only mean at <strong>17.92 ± 1.52</strong>. The
              federation does not improve over isolation — the structural
              Non-IID gap is too wide for sample-count-weighted averaging
              to close. This is the canonical FedAvg failure mode and the
              starting point of the project&apos;s research arc.
            </p>
            <p>
              Worth noting:{" "}
              <em>
                FedAvg is still operationally valuable here even with no
                RMSE improvement
              </em>
              — its NASA score is 43% better than local-only, and it is
              the only model robust across both fault modes (the per-
              subset breakdown shows local models excel on their own
              subset and fail on the other).
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
            { value: "−43 %", label: "NASA score reduction vs local-only", tone: "good" },
            { value: "651 s", label: "Wall-clock total (3 methods)" },
          ]}
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 01"
        title="The three-way comparison — the headline image"
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/three_way_non_iid_fd001+fd003.png"
          caption="Centralized vs FedAvg vs local-only mean on the combined FD001+FD003 test set"
          takeaway="FedAvg and local-only are tied on RMSE; both are 4 RMSE worse than centralized. The gap is structural, not protocol-fixable."
          explanation={
            <>
              <p>
                Same three-bar format as Phase 05, very different result.
                Centralized at RMSE 13.77 (better than its FD001-only
                cousin because more data); FedAvg at 17.95; local-only
                mean at 17.92. The FedAvg and local-only bars are
                indistinguishable. The federation is not adding value
                <em>in the RMSE-on-combined-test-set sense</em>.
              </p>
              <p>
                But two other metrics tell a different story: NASA score is
                43% lower under FedAvg than local-only (1,647 vs 2,885)
                because local models pay huge late-prediction penalties on
                engines from the fault mode they never saw. AUPRC also
                wins (0.951 vs 0.924). The federation is operationally
                useful even when its headline RMSE is unchanged — it
                just doesn&apos;t look that way on a combined-test-set
                summary.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 02"
        title="Per-subset cross-evaluation — the asymmetry"
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/per_subset_breakdown_fd001+fd003.png"
          caption="Per-subset RMSE for each method (centralized, FedAvg, each local client)"
          takeaway="FD001-trained clients excel on FD001 and fail on FD003. FD003-trained clients do the mirror. FedAvg is the only model that doesn't catastrophically fail on either subset."
          explanation={
            <>
              <p>
                Two columns of bars (one per test subset, FD001 and
                FD003). Centralized wins both. Local-only models split
                cleanly: client_2 (best FD001-trained) lands at 15.0 on
                FD001 and 20.3 on FD003; client_4 (best FD003-trained)
                lands at 14.5 on FD003 and 16.4 on FD001. Each local
                model is good on the subset it saw and visibly worse on
                the one it didn&apos;t.
              </p>
              <p>
                FedAvg lands at 17.0 / 18.9 — never the best on either,
                but never catastrophic either. An operational consumer of
                FedAvg gets one model that&apos;s mediocre everywhere; an
                operational consumer of a local-only model has to choose
                which biased local model to deploy and accept that it
                will be wrong on competitors&apos; engines. The robustness
                story is real even without an RMSE win.
              </p>
              <p>
                This figure is the canonical &quot;what does structural
                Non-IID look like?&quot; visual. Every RQ2 / FedProx /
                FedRep figure later in the project compares against it.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 03"
        title="Centralized training curves on the combined set"
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/centralized_metrics_fd001+fd003.png"
          caption="Centralized 50-epoch curves on FD001+FD003 combined"
          takeaway="Centralized reaches RMSE 13.77 — better than P3's FD001-only 14.02 because the combined data is roughly 2× larger."
          explanation={
            <>
              <p>
                The centralized run on the combined dataset converges in
                a similar pattern to P3 (FD001-only) but lands at a
                slightly better best-epoch RMSE because the training set
                is roughly twice as large. This is the &quot;more data
                helps a converged architecture&quot; expectation. It is
                also the reference point that defines how much
                signal-in-the-aggregate the combined population contains —
                FedAvg fails to extract that signal because it can&apos;t
                see the combined distribution.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 04"
        title="FedAvg global model metrics across rounds"
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/fedavg_metrics_fd001+fd003.png"
          caption="FedAvg test metrics over 50 communication rounds on the combined test set"
          takeaway="The global model plateaus at RMSE ~18 within 10–15 rounds. Additional rounds don't help — the protocol can't recover what averaging biased updates loses."
          explanation={
            <>
              <p>
                The curves flatten almost immediately. Best round arrives
                early; subsequent rounds oscillate without recovering the
                centralized number. This is the visual signature of
                FedAvg&apos;s structural-Non-IID failure: the optimization
                <em>has</em> converged — to a saddle that&apos;s a convex
                combination of the FD001-bias optimum and the FD003-bias
                optimum, never to the joint optimum that the centralized
                run actually finds.
              </p>
              <p>
                Adding more rounds, more local epochs, or a longer cosine
                schedule doesn&apos;t change the answer. The fix is at a
                different layer of the protocol — at the local-step
                regularisation (FedProx) or the architectural sharing
                contract (FedRep), not at the aggregation rule.
              </p>
            </>
          }
        />
      </ExperimentSection>

      <ExperimentSection
        eyebrow="Figure 05"
        title="Local-only per-client metric bars"
      >
        <ExplainedFigure
          artifactPath="results/06_non_iid/local_only_metrics_fd001+fd003.png"
          caption="Best-epoch metrics per local-only client (combined test set)"
          takeaway="Each local model is good on its own half of the world. The averaged number hides individual best-case performance."
          explanation={
            <>
              <p>
                Four groups of bars, one per client. Combined-test-set
                RMSEs range from 15.5 (client_4, best) to 19.3 (client_3,
                worst). The mean ± std is 17.92 ± 1.52 — much wider
                spread than the IID Phase 04 result (15.02 ± 0.29).
                Heterogeneity in the data shows up as heterogeneity in
                local-only model quality.
              </p>
              <p>
                The per-subset breakdown (Figure 02) shows why: each
                client is much better on the subset it saw than the
                averaged figure suggests. A federation is the operational
                fix for the bad-on-competitor-engines half — even when
                vanilla FedAvg doesn&apos;t close the RMSE gap, it
                produces a single model usable on both subsets.
              </p>
            </>
          }
        />
      </ExperimentSection>
    </ExperimentLayout>
  );
}
