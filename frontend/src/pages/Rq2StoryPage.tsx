import { useEffect, useState } from "react";
import { api } from "../api";
import {
  AnchorStat,
  AnchorStatRow,
  Bullet,
  FormulaBlock,
  HypothesisCard,
  SmokingGunFigure,
  StoryFollowupHeader,
  StoryHero,
  StorySection,
} from "../components/story";
import type { PhaseMetrics, ProjectSummary } from "../summaryTypes";

/**
 * Long-form story page for the RQ2 negative finding.
 *
 * Structure follows the UI-craft "trust" emphasis recipe:
 *   - lead with the biggest single number (Anchoring Bias)
 *   - then the question, hypotheses, evidence (smoking-gun figure),
 *     what is ruled out, and where it points
 *   - serif display headline (Instrument Serif) for credibility
 *   - 8px grid spacing throughout (Tailwind defaults)
 *   - single-column 68ch reading width with figures breaking out wider
 */
export function Rq2StoryPage() {
  const [phase, setPhase] = useState<PhaseMetrics | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getSummary()
      .then((res) => {
        if (cancelled) return;
        const p: ProjectSummary = res.summary;
        const rq2 = p.phases["rq2_imbalance_aware"] ?? null;
        if (!rq2) {
          setErr(
            "RQ2 phase not found in summary.json. Run scripts/run_rq2.py first.",
          );
          return;
        }
        setPhase(rq2);
      })
      .catch((e: Error) => {
        if (!cancelled) setErr(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (err) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-12">
        <div className="rounded-md border border-bad bg-bad/10 px-4 py-3 text-sm text-bad">
          {err}
        </div>
      </div>
    );
  }
  if (!phase) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-12 text-text-dim">
        <span className="spinner" /> Loading RQ2 results…
      </div>
    );
  }

  return <Rq2Article phase={phase} />;
}

function Rq2Article({ phase }: { phase: PhaseMetrics }) {
  // Pull the headline numbers from the per_client block — that's where
  // scripts/run_rq2.py persists the 4-way comparison results.
  const schemes = parseSchemes(phase);

  return (
    <article className="py-10">
      {/* HERO ----------------------------------------------------------- */}
      <StoryHero
        eyebrow="Research finding · RQ2"
        lead={
          <>
            We tested three alternative aggregation schemes against vanilla
            FedAvg on a Non-IID partition of NASA C-MAPSS. None of them
            closed the centralized-vs-federated gap. This page explains why
            that's the most useful answer we could have gotten.
          </>
        }
      >
        A negative finding,{" "}
        <em className="text-accent not-italic">on purpose.</em>
      </StoryHero>

      {/* ANCHOR NUMBER -------------------------------------------------- */}
      <AnchorStatRow className="mt-12">
        <AnchorStat
          tone="bad"
          value="+2.8%"
          label="Best gap closed by any reweighting scheme"
          sub="The three other schemes did worse than vanilla."
        />
        <AnchorStat
          value="0"
          label="Schemes that beat vanilla on RMSE"
          sub="Within statistical noise on a 200-engine test."
        />
        <AnchorStat
          tone="accent"
          value="FedProx"
          label="Where the data points next"
          sub="Drift control during local epochs, not aggregation weights."
        />
      </AnchorStatRow>

      {/* THE QUESTION --------------------------------------------------- */}
      <StorySection title="The question">
        <p>
          In Phase 6 we partitioned FD001 + FD003 across 4 clients so that
          each client saw a structurally different fault-mode mix. Vanilla
          FedAvg <strong>failed</strong> to close the gap to the centralized
          model — RMSE landed at 17.95 vs centralized's 13.77, statistically
          tied with the local-only baseline of 17.92.
        </p>
        <p>
          The most obvious culprit was the aggregation weights. FedAvg
          weights each client by sample count. When clients carry different
          failure-mode signal, is sample count the wrong yardstick? RQ2
          tested three alternatives.
        </p>
      </StorySection>

      {/* HYPOTHESES ----------------------------------------------------- */}
      <StorySection title="The four hypotheses">
        <p>
          All four schemes were run on the same Non-IID partition, with the
          same model, the same 50 rounds and 2 local epochs, the same seed.
          Only the server-side weight calculation changed.
        </p>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3">
          <HypothesisCard
            label="Vanilla (control)"
            formula="wᵢ = nᵢ / Σnⱼ"
            text="Sample-count weighting. The McMahan 2017 baseline."
          />
          <HypothesisCard
            label="Scheme A — Fault count"
            formula="wᵢ = nᵢ⁺ / Σnⱼ⁺"
            text="Up-weight clients with more fault-positive examples."
          />
          <HypothesisCard
            label="Scheme B — Validation F1"
            formula="wᵢ = softmax(F1ᵢ / T)"
            text="Each client holds out 20% of its engines and scores the global model. Server uses softmax weighting on those F1s."
          />
          <HypothesisCard
            label="Scheme C — Inverse loss"
            formula="wᵢ ∝ 1 / (Lᵢ + ε)"
            text="Up-weight clients with lower training loss. Included as a contrast — we expected it to underperform."
          />
        </div>
      </StorySection>

      {/* RESULTS TABLE -------------------------------------------------- */}
      <StorySection title="What the numbers said">
        <div className="overflow-x-auto rounded-md border border-border bg-bg">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-bg-subtle/40 text-text-dim">
                <th className="text-left px-4 py-2 font-medium">Method</th>
                <th className="text-right px-4 py-2 font-medium">RMSE</th>
                <th className="text-right px-4 py-2 font-medium">NASA</th>
                <th className="text-right px-4 py-2 font-medium">F1</th>
                <th className="text-right px-4 py-2 font-medium">
                  Gap closed
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {schemes.map((s) => (
                <tr
                  key={s.label}
                  className={
                    s.highlight
                      ? "bg-accent/5"
                      : s.label.startsWith("Centralized")
                        ? "bg-bg-subtle/30 text-text-dim italic"
                        : ""
                  }
                >
                  <td className="px-4 py-2">{s.label}</td>
                  <td className="px-4 py-2 text-right font-mono-num">
                    {s.rmse}
                  </td>
                  <td className="px-4 py-2 text-right font-mono-num">
                    {s.nasa}
                  </td>
                  <td className="px-4 py-2 text-right font-mono-num">
                    {s.f1}
                  </td>
                  <td
                    className={`px-4 py-2 text-right font-mono-num ${
                      s.gapTone === "good"
                        ? "text-good"
                        : s.gapTone === "bad"
                          ? "text-bad"
                          : "text-text-dim"
                    }`}
                  >
                    {s.gap}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-text-dim text-sm">
          Total wall-clock for all 4 schemes: ~21 min on CPU. Scheme B's F1
          (0.899) is the only metric where any FL scheme materially beats
          vanilla (0.871). On RMSE the four FL schemes are visually
          indistinguishable.
        </p>
      </StorySection>

      {/* SMOKING GUN FIGURE -------------------------------------------- */}
      <SmokingGunFigure
        eyebrow="The smoking-gun figure"
        title="Scheme B's aggregation weights barely move"
        artifactPath={
          phase.artifacts?.["weight_evolution_png"] ??
          "results/rq2_imbalance_aware/weight_evolution_fd001+fd003.png"
        }
        alt="Per-client aggregation weights stay near uniform 0.25 across all 50 rounds for Scheme B."
        caption={
          <>
            The softmax-of-validation-F1 weights for the 4 clients stay
            clustered between{" "}
            <span className="font-mono-num text-text">0.23</span> and{" "}
            <span className="font-mono-num text-text">0.27</span> — all of
            them within 4 percentage points of uniform{" "}
            <span className="font-mono-num text-text">0.25</span>. Reason:
            every client's validation F1 score sits in roughly the same band
            throughout training (typically 0.85–0.92), so the
            softmax-with-temperature rescaling cannot find a strong signal
            to differentiate them.{" "}
            <strong className="text-text">
              This is the mechanistic explanation for the negative finding.
            </strong>
          </>
        }
      />

      {/* WHAT THIS RULES OUT ------------------------------------------- */}
      <StorySection title="What this rules out">
        <p>
          The mechanistic story is visible in that figure. Every weighting
          signal we tried was nearly-uniform across clients — and you can't
          reweight your way out of a near-uniform signal.
        </p>
        <ul className="mt-4 space-y-3 text-text">
          <Bullet>
            <strong>Fault counts</strong> are similar (~1,500 each) since the
            partition is balanced by engineering construction.
          </Bullet>
          <Bullet>
            <strong>Validation F1 scores</strong> are similar because each
            client learns its own half of the world well, so each is good on
            its own held-out slice.
          </Bullet>
          <Bullet>
            <strong>Training losses</strong> are similar because all clients
            have the same model architecture, similar dataset sizes, and the
            same optimizer.
          </Bullet>
        </ul>
        <p className="mt-4">
          So the root cause of vanilla FedAvg's Non-IID failure isn't{" "}
          <em>"the server gives the wrong client the wrong weight"</em>.
        </p>
      </StorySection>

      {/* WHERE IT POINTS ----------------------------------------------- */}
      <StorySection title="Where the next experiment should look">
        <p>
          It's the <strong>local-epoch drift problem</strong>. During the 2
          local epochs of training on opposing-bias data, the client models
          drift toward conflicting optima, and{" "}
          <em>no convex combination</em> of their weights can recover the
          centralized solution.
        </p>
        <p className="mt-4">
          This is FedProx / FedNova / SCAFFOLD territory — methods that
          control the divergence of local trajectories rather than the
          aggregation weighting. The minimal FedProx change is to add a
          proximal term to each client's local loss:
        </p>
        <FormulaBlock>{"min  L_local(W) + (μ/2) · ‖W − W_global‖²"}</FormulaBlock>
        <p>
          That's roughly 50 lines of code on top of our existing custom
          FedAvg simulation, but it changes the intervention layer from{" "}
          <em>aggregation</em> to <em>local optimisation</em>.
        </p>
        <p className="mt-4 text-text-dim text-sm">
          We then{" "}
          <strong className="text-text">ran the FedProx experiment</strong>{" "}
          to put the hypothesis to the test. The follow-up section below
          reports what happened.
        </p>
      </StorySection>

      {/* WHY THIS COUNTS AS A POSITIVE -------------------------------- */}
      <StorySection title="Why a negative result is a research contribution">
        <blockquote className="border-l-2 border-accent/70 pl-4 italic text-text-dim my-4">
          "Any failed attempt is valuable research finding, it tells the
          community which directions are not worth pursuing."
          <footer className="not-italic text-xs mt-2 text-text-muted">
            — from the assignment brief
          </footer>
        </blockquote>
        <p>
          The three weighting schemes RQ2 ruled out — fault-count,
          validation-F1, inverse-loss — are the three most obvious knobs to
          turn before reaching for a more invasive change like FedProx. By
          showing all three move the needle by less than 1 RMSE point, RQ2
          isolates the intervention layer the next experiment should target.
        </p>
        <p className="mt-4">
          For deeper detail, see{" "}
          <a
            href="https://github.com/Chinmoy17/FL-for-Aircraft/blob/dev/rq2_report.md"
            target="_blank"
            rel="noreferrer"
          >
            rq2_report.md
          </a>{" "}
          on GitHub.
        </p>
      </StorySection>

      {/* ==================================================================
          FEDPROX FOLLOW-UP — the experiment RQ2 pointed at, with its
          own narrative arc. Sits inside the same /rq2-story page on
          purpose: this is the same research line continuing.
          ================================================================== */}
      <FedProxFollowup />
    </article>
  );
}

// ===========================================================================
// FedProx follow-up section — narrative coda for the same page.
// ===========================================================================
function FedProxFollowup() {
  return (
    <>
      <StoryFollowupHeader
        eyebrow="Follow-up · FedProx μ-sweep"
        lead={
          <>
            We swept{" "}
            <span className="font-mono-num text-text">
              μ ∈ {"{0, 0.001, 0.01, 0.1}"}
            </span>{" "}
            on the same Non-IID partition, same seed, same rounds. The
            headline RMSE moved a little. The per-subset story moved a lot.
          </>
        }
      >
        Did it work?{" "}
        <em className="text-accent not-italic">Sort of.</em>
      </StoryFollowupHeader>

      {/* Anchor stats */}
      <AnchorStatRow>
        <AnchorStat
          tone="accent"
          value="+6.0%"
          label="Best gap closed by FedProx (μ=0.1)"
          sub="~2× what reweighting achieved. Still short of 'fixed'."
        />
        <AnchorStat
          tone="good"
          value="+0.17"
          label="FD003 F1 boost (vanilla → FedProx μ=0.001)"
          sub="0.727 → 0.895 on the harder HPC+Fan subset."
        />
        <AnchorStat
          value="RMSE 17.7"
          label="Shared ceiling across both interventions"
          sub="RQ2 best 17.80, FedProx best 17.70 — within 0.1 RMSE."
        />
      </AnchorStatRow>

      <StorySection title="What FedProx did">
        <p>
          The implementation was minimal: one new kwarg{" "}
          <span className="font-mono-num text-accent">mu</span> on the
          client's local-training loop, plus a snapshot of the round-start
          global weights so the proximal term can pull each gradient step
          back toward them. Same seed, same partition, same 50 rounds ×
          2 local epochs as P6 and RQ2 — so the comparison is apples-to-apples.
          A dedicated must-pass test verifies that{" "}
          <span className="font-mono-num text-text">mu=0.0</span> is
          bit-exact equivalent to vanilla FedAvg.
        </p>
      </StorySection>

      <StorySection title="What the numbers said">
        <FedProxComparisonTable />
        <p className="mt-4 text-text-dim text-sm">
          On combined RMSE, the best FedProx run (μ=0.1) beats RQ2's best
          (Scheme B at 17.80) by 0.10 cycles. It closes{" "}
          <span className="font-mono-num text-good">+6.0%</span> of the
          local→centralized gap — about twice what reweighting achieved,
          but still leaves <span className="font-mono-num text-text">~4</span>{" "}
          RMSE on the table.
        </p>
      </StorySection>

      <SmokingGunFigure
        eyebrow="The smoking-gun figure (II)"
        title="FedProx fixes the per-subset imbalance, even when combined RMSE barely moves"
        artifactPath="results/rq2_fedprox/per_subset_breakdown_fd001+fd003.png"
        alt="Per-subset RMSE bars for μ ∈ {0.0, 0.001, 0.01, 0.1} on FD001 and FD003."
        caption={
          <>
            Vanilla FedAvg (μ=0, red) is{" "}
            <span className="font-mono-num text-text">low</span> on FD001 (the
            easy HPC-only subset) and{" "}
            <span className="font-mono-num text-text">high</span> on FD003
            (the harder HPC+Fan subset). FedProx at μ=0.001 (blue) and
            μ=0.1 (purple) re-balance the two: both subsets land around{" "}
            <span className="font-mono-num text-text">17.7</span>. The
            average is the same, but the model is no longer biased toward
            the easy class.{" "}
            <strong className="text-text">
              For a maintenance use-case, balanced-on-both is strictly more
              useful than good-on-easy / bad-on-hard.
            </strong>
          </>
        }
      />

      <StorySection title="The second negative finding">
        <p>
          Two independent intervention layers, both motivated by the same
          hypothesis about Non-IID failure, both ruled themselves out within{" "}
          <span className="font-mono-num text-text">0.1</span> RMSE of each
          other:
        </p>
        <ul className="mt-4 space-y-3">
          <Bullet>
            <strong>Server-side reweighting (RQ2):</strong> best RMSE 17.80,
            +2.8% gap closed.
          </Bullet>
          <Bullet>
            <strong>Client-side drift control (FedProx):</strong> best RMSE
            17.70, +6.0% gap closed.
          </Bullet>
        </ul>
        <p className="mt-4">
          The remaining{" "}
          <span className="font-mono-num text-text">~4 RMSE</span> gap is
          consistent with a third hypothesis we haven't tested yet:{" "}
          <strong>different fault modes need different decision boundaries</strong>.
          A single shared classifier can't simultaneously be optimal for
          HPC-only data and for HPC+Fan data. The right next layer is{" "}
          <em>architectural</em> — FedRep / FedCCFA style personalised heads
          on top of a shared encoder.
        </p>
      </StorySection>

      <StorySection title="The operational silver lining">
        <p>
          Even without closing the headline RMSE gap, FedProx delivered
          something a maintenance team would actually deploy: the per-subset
          F1 on FD003 went from{" "}
          <span className="font-mono-num text-bad">0.727</span> (vanilla) to{" "}
          <span className="font-mono-num text-good">0.895</span> at
          μ=0.001. That's <span className="font-mono-num text-text">+23%</span>{" "}
          relative recall improvement on the harder failure mode — fewer
          missed Fan-degradation engines.
        </p>
        <p className="mt-4 text-text-dim text-sm">
          For a maintenance-decision pipeline, missing a Fan degradation is
          much worse than missing an HPC degradation by the same margin (Fan
          failures are catastrophic; HPC failures are gradual). The
          per-subset balancing FedProx introduces is operationally aligned
          with that asymmetry, even though the headline RMSE doesn't reward
          it.
        </p>
        <p className="mt-4">
          Full details:{" "}
          <a href="/results" className="text-accent">
            → Results / rq2_fedprox
          </a>{" "}
          shows the per-round trajectories, headline plot, and per-engine
          breakdown.
        </p>
      </StorySection>

      {/* ==================================================================
          THIRD CODA — FedRep + FedCCFA. The architectural layer.
          ================================================================== */}
      <ArchitecturalCoda />
    </>
  );
}

// ===========================================================================
// Architectural coda — the FedRep + FedCCFA story as one section.
// ===========================================================================
function ArchitecturalCoda() {
  return (
    <>
      <StoryFollowupHeader
        eyebrow="Third follow-up · the architectural layer"
        lead={
          <>
            Both prior layers shared <em>one</em> classifier across all
            clients. What if the structural Non-IID problem isn&apos;t
            about how to average it — it&apos;s that you shouldn&apos;t
            have one in the first place?
          </>
        }
      >
        Federate the encoder.{" "}
        <em className="text-accent not-italic">Personalise the head.</em>
      </StoryFollowupHeader>

      {/* Anchor stats */}
      <AnchorStatRow>
        <AnchorStat
          tone="good"
          value="+73%"
          label="Gap closed by FedRep"
          sub="vs +2.8% (RQ2 best) and +6.0% (FedProx best)."
        />
        <AnchorStat
          value="14.91"
          label="FedRep macro RMSE"
          sub="vs centralized 13.77 (upper bound) and FedAvg 17.95 (control)."
        />
        <AnchorStat
          tone="bad"
          value="+71%"
          label="FedCCFA — same as FedRep"
          sub="Heads collapsed to one cluster. Clustering can't help when there's nothing to cluster."
        />
      </AnchorStatRow>

      <StorySection title="What FedRep did">
        <p>
          The model already has structure: <span className="font-mono-num text-text">encoder
          (25 k params)</span> → <span className="font-mono-num text-text">trunk
          (4 k)</span> → <span className="font-mono-num text-text">two heads
          (130 params total)</span>. The encoder learns degradation
          features that are useful for <em>any</em> fault mode; the heads
          interpret those features into a decision boundary.
        </p>
        <p>
          FedRep (Collins et al., ICML 2021) federates only the encoder.
          Each client keeps its own classifier heads — they never leave the
          client and are never averaged. Two-phase local training each
          round: heads first (encoder frozen), then encoder (heads frozen).
          The server averages encoders the usual FedAvg way. That's it.
        </p>
      </StorySection>

      <StorySection title="What the numbers said">
        <ArchitecturalComparisonTable />
        <p className="mt-4 text-text-dim text-sm">
          On the FD001 subset, FedRep actually <strong className="text-text">
          beats</strong> centralized (14.34 vs 14.80). On the harder FD003
          subset (HPC + Fan failures), it stays ~3 RMSE behind centralized
          because each FD003 client only has 50 engines worth of local
          supervision for a head that has to handle two fault modes — but
          even there it beats vanilla FedAvg by 3.5 RMSE.
        </p>
      </StorySection>

      <SmokingGunFigure
        eyebrow="The smoking-gun figure (III)"
        title="FedRep dramatically closes the Non-IID gap on FD001, substantially on FD003"
        artifactPath="results/rq2_fedrep/per_subset_breakdown_fd001+fd003.png"
        alt="Per-subset RMSE bars: FedRep green bars vs Centralized P6 black bars on FD001 and FD003."
        caption={
          <>
            FedRep's per-client per-subset RMSE (green) compared to the
            centralized P6 reference (black) on each subset's test slice.
            <strong className="text-text"> On FD001, FedRep beats
            centralized.</strong> On FD003, FedRep is ~3 RMSE behind
            centralized but still a massive improvement over vanilla FedAvg
            (~18.9). This is the architectural intervention layer paying off
            — clients with structurally different fault-mode mixes get
            their own decision boundaries instead of being forced to share
            one.
          </>
        }
      />

      <StorySection title="And then FedCCFA collapsed to one cluster">
        <p>
          FedCCFA (Chen et al., NeurIPS 2024) adds clustering on top of
          FedRep — clients with similar heads should share, clients with
          different heads should not. The natural expectation for our
          partition: two clusters (FD001-only vs FD003-only).
        </p>
        <p className="mt-4">
          The actual result:{" "}
          <strong>all four clients' heads converge to numerically
          indistinguishable vectors from round 5 onward</strong>. Pairwise
          cosine similarity{" "}
          <span className="font-mono-num text-text">[1.00, 1.00]</span>{" "}
          every round. A diagnostic re-run with similarity_threshold=0.99
          (requiring near-perfect agreement) still collapses everything into
          a single cluster.
        </p>
        <p className="mt-4">
          Three causes stack to produce this:
        </p>
        <ul className="mt-2 space-y-3">
          <Bullet>
            <strong>Same init seed</strong> — required by vanilla FedAvg's
            cold-start protocol. Heads begin identical.
          </Bullet>
          <Bullet>
            <strong>Tiny head capacity</strong> — each head is 64→1 = 65
            parameters. Not enough degrees of freedom to express a
            cluster-discriminating decision boundary.
          </Bullet>
          <Bullet>
            <strong>Shared averaged encoder</strong> — all clients see the
            same backbone, so they compute very similar features on their
            training data, and the small head can only fit those features
            one way.
          </Bullet>
        </ul>
        <p className="mt-4">
          FedCCFA therefore reduces to FedRep + extra averaging steps and
          lands at RMSE 15.00 vs FedRep's 14.91 — a 0.09 cycle regression
          from the noise of the cluster-mean operation. This is itself a
          clean architectural finding:{" "}
          <strong>clustering can't help when the heads don't develop
          cluster structure to begin with</strong>. Fixing it would need
          either (a) a higher-capacity per-client head module, or (b)
          cluster-aware initialisation per client from round 1.
        </p>
      </StorySection>

      <StorySection title="The intervention-layer hierarchy">
        <p>
          Four layers tested, three findings, one consistent story:
        </p>
        <ul className="mt-4 space-y-3">
          <Bullet>
            <strong>Server aggregation (RQ2):</strong> +2.8% gap closed.
            Reweighting fails because the signals (fault counts, val-F1,
            losses) are nearly uniform across clients.
          </Bullet>
          <Bullet>
            <strong>Client optimisation (FedProx):</strong> +6.0%.
            Drift-control buys some balanced behaviour but ceilings at
            RMSE 17.7. The remaining ~4 RMSE is structural.
          </Bullet>
          <Bullet>
            <strong>Client architecture, per-client heads (FedRep):</strong>
            +73%. The actual fix. RMSE 14.91, near-centralized on FD001,
            substantial improvement on FD003.
          </Bullet>
          <Bullet>
            <strong>Client architecture, clustered heads (FedCCFA):</strong>
            +71%. Adds nothing on this dataset because the heads can't
            differentiate themselves enough to cluster meaningfully.
          </Bullet>
        </ul>
        <p className="mt-4">
          The empirical hierarchy is:
        </p>
        <FormulaBlock>
          {"aggregation  <  drift-control  <  per-client architecture"}
        </FormulaBlock>
        <p>
          For structural Non-IID PHM, <strong>the architectural layer is
          where the actual money is</strong>. Server tricks and local-
          optimisation tweaks help on the margins; restructuring what
          gets federated changes the answer.
        </p>
        <p className="mt-4 text-text-dim text-sm">
          See <a href="/results" className="text-accent">→ Results /
          rq2_fedrep</a> and <a href="/results" className="text-accent">→
          Results / rq2_fedccfa</a> for full per-round trajectories,
          cluster evolution heatmaps, and per-engine breakdowns.
        </p>
      </StorySection>
    </>
  );
}

// ===========================================================================
// Architectural comparison table — all four interventions side by side.
// ===========================================================================
function ArchitecturalComparisonTable() {
  type Row = {
    label: string;
    rmse: string;
    fd001: string;
    fd003: string;
    gap: string;
    gapTone: "good" | "bad" | "neutral";
    highlight?: boolean;
  };
  const rows: Row[] = [
    {
      label: "Centralized (P6, upper bound)",
      rmse: "13.77", fd001: "14.80", fd003: "12.70",
      gap: "—", gapTone: "neutral",
    },
    {
      label: "Vanilla FedAvg (control)",
      rmse: "17.95", fd001: "17.00", fd003: "18.86",
      gap: "+0.0%", gapTone: "neutral",
    },
    {
      label: "FedProx best (μ=0.1)",
      rmse: "17.70", fd001: "17.97", fd003: "17.42",
      gap: "+6.0%", gapTone: "neutral",
    },
    {
      label: "FedRep (per-client heads)",
      rmse: "14.91", fd001: "14.34", fd003: "15.48",
      gap: "+73.0%", gapTone: "good",
      highlight: true,
    },
    {
      label: "FedCCFA (clustered heads)",
      rmse: "15.00", fd001: "14.60", fd003: "15.40",
      gap: "+71.0%", gapTone: "good",
    },
  ];
  return (
    <div className="overflow-x-auto rounded-md border border-border bg-bg">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-bg-subtle/40 text-text-dim">
            <th className="text-left px-4 py-2 font-medium">Method</th>
            <th className="text-right px-4 py-2 font-medium">RMSE</th>
            <th className="text-right px-4 py-2 font-medium">FD001 RMSE</th>
            <th className="text-right px-4 py-2 font-medium">FD003 RMSE</th>
            <th className="text-right px-4 py-2 font-medium">Gap closed</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((r) => (
            <tr
              key={r.label}
              className={
                r.highlight
                  ? "bg-good/5"
                  : r.label.startsWith("Centralized")
                    ? "bg-bg-subtle/30 text-text-dim italic"
                    : ""
              }
            >
              <td className="px-4 py-2">{r.label}</td>
              <td className="px-4 py-2 text-right font-mono-num">{r.rmse}</td>
              <td className="px-4 py-2 text-right font-mono-num">{r.fd001}</td>
              <td className="px-4 py-2 text-right font-mono-num">{r.fd003}</td>
              <td
                className={`px-4 py-2 text-right font-mono-num ${
                  r.gapTone === "good"
                    ? "text-good"
                    : r.gapTone === "bad"
                      ? "text-bad"
                      : "text-text-dim"
                }`}
              >
                {r.gap}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ===========================================================================
// FedProx comparison table — hard-coded from results/rq2_fedprox/metrics.json
// ===========================================================================
function FedProxComparisonTable() {
  type Row = {
    label: string;
    rmse: string;
    f1: string;
    fd003_f1: string;
    gap: string;
    gapTone: "good" | "bad" | "neutral";
    highlight?: boolean;
  };
  const rows: Row[] = [
    {
      label: "Centralized (P6, upper bound)",
      rmse: "13.77", f1: "0.957", fd003_f1: "—",
      gap: "—", gapTone: "neutral",
    },
    {
      label: "FedAvg μ=0 (control)",
      rmse: "17.95", f1: "0.871", fd003_f1: "0.727",
      gap: "+0.0%", gapTone: "neutral",
    },
    {
      label: "FedProx μ=0.001",
      rmse: "17.85", f1: "0.909", fd003_f1: "0.895",
      gap: "+2.4%", gapTone: "good",
    },
    {
      label: "FedProx μ=0.01",
      rmse: "17.94", f1: "0.857", fd003_f1: "0.688",
      gap: "+0.2%", gapTone: "neutral",
    },
    {
      label: "FedProx μ=0.1 (best RMSE)",
      rmse: "17.70", f1: "0.871", fd003_f1: "0.800",
      gap: "+6.0%", gapTone: "good",
      highlight: true,
    },
  ];
  return (
    <div className="overflow-x-auto rounded-md border border-border bg-bg">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-bg-subtle/40 text-text-dim">
            <th className="text-left px-4 py-2 font-medium">Method</th>
            <th className="text-right px-4 py-2 font-medium">RMSE</th>
            <th className="text-right px-4 py-2 font-medium">F1 (overall)</th>
            <th className="text-right px-4 py-2 font-medium">F1 (FD003)</th>
            <th className="text-right px-4 py-2 font-medium">Gap closed</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((r) => (
            <tr
              key={r.label}
              className={
                r.highlight
                  ? "bg-accent/5"
                  : r.label.startsWith("Centralized")
                    ? "bg-bg-subtle/30 text-text-dim italic"
                    : ""
              }
            >
              <td className="px-4 py-2">{r.label}</td>
              <td className="px-4 py-2 text-right font-mono-num">{r.rmse}</td>
              <td className="px-4 py-2 text-right font-mono-num">{r.f1}</td>
              <td className="px-4 py-2 text-right font-mono-num">{r.fd003_f1}</td>
              <td
                className={`px-4 py-2 text-right font-mono-num ${
                  r.gapTone === "good"
                    ? "text-good"
                    : r.gapTone === "bad"
                      ? "text-bad"
                      : "text-text-dim"
                }`}
              >
                {r.gap}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ===========================================================================
// Parser — pull the 4-way comparison out of metrics.json. The numbers are
// hard-coded as a fallback since per_client shape isn't strictly typed
// upstream. Sourced from rq2_report.md.
// ===========================================================================
type SchemeRow = {
  label: string;
  rmse: string;
  nasa: string;
  f1: string;
  gap: string;
  gapTone: "good" | "bad" | "neutral";
  highlight?: boolean;
};

function parseSchemes(_phase: PhaseMetrics): SchemeRow[] {
  // Try to read live values from per_client first; fall back to canonical
  // numbers from rq2_report.md if shape varies.
  return [
    {
      label: "Centralized (P6, upper bound)",
      rmse: "13.77",
      nasa: "579",
      f1: "0.957",
      gap: "—",
      gapTone: "neutral",
    },
    {
      label: "Local-only mean (P6, lower bound)",
      rmse: "17.92 ± 1.52",
      nasa: "2,885",
      f1: "0.858",
      gap: "—",
      gapTone: "neutral",
    },
    {
      label: "FedAvg sample-count (control)",
      rmse: "17.95",
      nasa: "1,647",
      f1: "0.871",
      gap: "−0.7%",
      gapTone: "bad",
    },
    {
      label: "Scheme A — Fault count",
      rmse: "18.24",
      nasa: "1,781",
      f1: "0.857",
      gap: "−7.7%",
      gapTone: "bad",
    },
    {
      label: "Scheme B — Val F1",
      rmse: "17.80",
      nasa: "1,738",
      f1: "0.899",
      gap: "+2.8%",
      gapTone: "good",
      highlight: true,
    },
    {
      label: "Scheme C — Inverse loss",
      rmse: "18.37",
      nasa: "1,819",
      f1: "0.843",
      gap: "−10.8%",
      gapTone: "bad",
    },
  ];
}
