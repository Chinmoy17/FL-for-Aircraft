import { NavLink } from "react-router-dom";

/**
 * AbstractPage — `/abstract`.
 *
 * The "for the reviewer" expansion of the Cover. Sections:
 *
 *   1. Page header — title + run-of-text byline.
 *   2. Project framing — three short paragraphs that motivate the
 *       work in plain language (why FL, why aviation, what's hard).
 *   3. Contributions — five bullet contributions of the project,
 *       each one with a one-line claim and a pointer into the work.
 *   4. Research-question verdict matrix — single table answering all
 *       seven RQs in one glance: which are answered, which are
 *       synthesized, which are open, and where the evidence lives.
 *   5. Related work delta — six rows comparing this project's
 *       choices against the canonical reference papers from the
 *       brief and adjacent FL literature.
 *   6. Reproducibility — what runs, how long, what passes.
 *   7. Open work — three bulleted items clearly framed as not yet
 *       done so the page commits to honest scope.
 *   8. Footer — navigation back to cover + on to other pages.
 *
 * Layout is full-width with consistent horizontal padding
 * (px-10 → px-24) matching the Cover. Long prose uses prose-narrow
 * (~68ch) inside its container so reading width stays comfortable
 * even though the shell is full-width.
 */
export function AbstractPage() {
  return (
    <article className="w-full">
      <PageHeader />
      <ProjectFraming />
      <ExperimentalFlow />
      <Contributions />
      <RqVerdictMatrix />
      <RelatedWorkDelta />
      <Reproducibility />
      <OpenWork />
      <FooterNav />
    </article>
  );
}

// ===========================================================================
// 1. Header
// ===========================================================================
function PageHeader() {
  return (
    <header className="px-10 md:px-16 lg:px-24 pt-16 pb-10 border-b border-border">
      <div className="eyebrow">Overview · 10 min read</div>
      <h1 className="font-display text-[44px] sm:text-[52px] leading-[1.05] tracking-tight text-text mt-4 max-w-[26ch]">
        Abstract <em className="not-italic text-accent">&amp;</em>{" "}
        contributions
      </h1>
      <p className="mt-6 text-lg text-text-dim">
        Long-form framing of the project: why the question is hard, what was
        built, what was found, what was synthesized, what is honestly still
        open. For reviewers reading at depth before clicking into the
        per-RQ stories.
      </p>
    </header>
  );
}

// ===========================================================================
// 2. Project framing
// ===========================================================================
function ProjectFraming() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-14">
      <div className="eyebrow">Why this project</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-8 max-w-[36ch]">
        Predictive maintenance is a federated-learning problem hiding
        inside a regression problem.
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10 text-[17px] leading-[1.7]">
        <div className="space-y-4">
          <div className="eyebrow !text-accent">1. The data setup</div>
          <p>
            Aircraft engines are densely instrumented and generate years of
            sensor traces ending in run-to-failure events. NASA's C-MAPSS
            turbofan simulator is the canonical academic dataset — four
            subsets covering two operating regimes × two fault-mode mixes.
            The full population in this project is FD001 + FD003:
            single-regime, sea-level, but structurally different fault
            modes (HPC vs HPC + Fan).
          </p>
        </div>

        <div className="space-y-4">
          <div className="eyebrow !text-accent">2. Why FL is necessary</div>
          <p>
            Airlines do not pool their failure data. It is a commercial
            differentiator, it is regulatory-sensitive, and a centralised
            model trained on combined fleets is operationally impossible.
            Federated learning is the right shape for the problem — local
            training, weight-only updates, no raw sensor data crossing
            the wire.
          </p>
        </div>

        <div className="space-y-4">
          <div className="eyebrow !text-accent">3. Why FL is hard</div>
          <p>
            Different airlines see structurally different fault-mode
            distributions. Vanilla <span className="font-mono-num">FedAvg</span>{" "}
            breaks under that structural Non-IID, and naïve fixes
            (reweighting client contributions by various signals) cannot
            close the gap. The cure lives elsewhere — in client-side
            optimisation, in architecture, or in a defense layer added
            specifically against adversarial behaviour.
          </p>
        </div>
      </div>
    </section>
  );
}

// ===========================================================================
// 3. Experimental flow — how we got from raw data to the RQ work
// ===========================================================================
type FlowStep = {
  phase: string;
  title: string;
  body: string;
  result?: string;
  resultLabel?: string;
  tone?: "neutral" | "upper" | "lower" | "gap";
  to?: string;
};

const FLOW: FlowStep[] = [
  {
    phase: "00",
    title: "EDA",
    body:
      "Establish the six things any FL-PHM model has to know about C-MAPSS: engine lifetimes, label distribution, class imbalance, operating-regime split, sensor structure, and whether degradation is even detectable.",
    to: "/experiments/00-eda",
  },
  {
    phase: "01",
    title: "Data pipeline",
    body:
      "Partition FD001 into 4 simulated airline clients (stratified-by-lifetime). Confirm window counts match the analytical formula and per-client fault rate stays balanced (<0.2 pp spread).",
    to: "/experiments/01-data",
  },
  {
    phase: "02",
    title: "Smoke run",
    body:
      "One centralized epoch on FD001 to confirm the data → model → loss → metrics wiring is correct. AUPRC 0.85 after one epoch proves the architecture works; mis-calibrated fault head is expected.",
    to: "/experiments/02-smoke",
  },
  {
    phase: "03",
    title: "Centralized baseline (FD001)",
    body:
      "50 epochs of cosine-annealed Adam on pooled FD001. This is the IID upper bound — the best a model could do without any federation constraints.",
    result: "RMSE 14.02",
    resultLabel: "Upper bound (FD001)",
    tone: "upper",
    to: "/experiments/03-centralized",
  },
  {
    phase: "04",
    title: "Local-only baseline",
    body:
      "Each client trains alone on its 25 engines, evaluated on the common test set. Mean of the four is the federation's lower bound — FedAvg must beat this to be worth the protocol.",
    result: "RMSE 15.02",
    resultLabel: "Lower bound (FD001)",
    tone: "lower",
    to: "/experiments/04-local-only",
  },
  {
    phase: "05",
    title: "FedAvg IID baseline",
    body:
      "Canonical FedAvg over 4 clients × 50 rounds × 2 local epochs. Closes 85.9% of the upper↔lower gap — federation works as expected on the easy case.",
    result: "85.9 % gap closed",
    resultLabel: "Federation wins on IID",
    tone: "gap",
    to: "/experiments/05-fedavg",
  },
  {
    phase: "06",
    title: "Non-IID baseline (FD001 + FD003)",
    body:
      "Structurally Non-IID partition: 2 clients carry FD001 (HPC only), 2 carry FD003 (HPC + Fan). Same protocol, very different result — vanilla FedAvg ties with local-only. This is the failure mode every later RQ addresses.",
    result: "0 % gap closed",
    resultLabel: "Federation fails on structural Non-IID",
    tone: "lower",
    to: "/experiments/06-non-iid",
  },
];

function ExperimentalFlow() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-16 border-t border-border">
      <div className="eyebrow">How we got there</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-3 max-w-[40ch]">
        The empirical flow — from raw CMAPSS files to the RQ work.
      </h2>
      <p className="text-text-dim mb-10">
        Each row below is one experimental phase. Phases 03–06 also report
        the bound or gap they establish; that&apos;s the line of evidence
        the research questions later try to move. The Non-IID failure at
        Phase 06 is what makes RQ2, RQ3, and RQ7 meaningful — not because
        the federation broke, but because it broke in exactly the way the
        brief predicted.
      </p>

      <ol className="space-y-3">
        {FLOW.map((step, i) => (
          <FlowRow key={step.phase} step={step} isLast={i === FLOW.length - 1} />
        ))}
      </ol>

      <div className="mt-10 rounded-lg border border-accent/40 bg-accent-subtle px-5 py-4">
        <div className="eyebrow !text-accent mb-1">What happens after Phase 06</div>
        <p className="text-[15px] text-text leading-relaxed">
          The 4-RMSE gap from Phase 06 (FedAvg 17.95 vs centralized 13.77)
          becomes the target territory for the research questions. RQ2
          tests whether aggregation-layer reweighting can close it
          (negative). The RQ2 follow-up trilogy (FedProx, FedRep,
          FedCCFA) tests client-optimisation and architectural fixes —
          FedRep closes +73 %. RQ3 explains <em>why</em> the gap is hard
          to close. RQ7 protects the model from a different kind of
          failure entirely.
        </p>
      </div>
    </section>
  );
}

function FlowRow({ step, isLast }: { step: FlowStep; isLast: boolean }) {
  const toneCls =
    step.tone === "upper"
      ? "text-good border-good/40 bg-good/10"
      : step.tone === "lower"
      ? "text-bad border-bad/40 bg-bad/10"
      : step.tone === "gap"
      ? "text-accent border-accent/40 bg-accent-subtle"
      : "text-text-muted border-border bg-bg-subtle";

  const body = (
    <article className="rounded-lg border border-border bg-bg p-5 hover:border-border-strong transition-colors">
      {/* Header row: phase label + title left, result chip right (when present) */}
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2 mb-3">
        <span className="font-mono-num font-display text-xl text-accent leading-none">
          Phase {step.phase}
        </span>
        <h3 className="font-display text-lg text-text leading-snug flex-1 min-w-0">
          {step.title}
        </h3>
        {step.result && (
          <span
            className={`inline-flex flex-col items-end rounded-md border px-3 py-1.5 leading-tight ${toneCls}`}
          >
            <span className="font-mono-num font-semibold text-[14px]">
              {step.result}
            </span>
            {step.resultLabel && (
              <span className="text-[10px] uppercase tracking-wider opacity-80 mt-0.5">
                {step.resultLabel}
              </span>
            )}
          </span>
        )}
      </div>
      <p className="text-[14.5px] text-text-dim leading-relaxed">{step.body}</p>
    </article>
  );

  return (
    <li className="relative">
      {step.to ? (
        <NavLink to={step.to} className="block no-underline">
          {body}
        </NavLink>
      ) : (
        body
      )}
      {!isLast && (
        <div
          aria-hidden
          className="ml-6 h-3 border-l-2 border-dashed border-border my-0.5"
        />
      )}
    </li>
  );
}

// ===========================================================================
// 4. Contributions
// ===========================================================================
const CONTRIBUTIONS = [
  {
    n: "01",
    title: "An end-to-end reproducible FL-PHM benchmark on FD001+FD003",
    blurb:
      "11 experimental phases, all configured identically (seed=42, 4 clients, 50 rounds × 2 local epochs, batch 256, lr 1e-3 cosine, GroupNorm-only). Total wall-clock to reproduce: ~5 hours of CPU time.",
  },
  {
    n: "02",
    title:
      "A negative finding ruling out the aggregation layer as the cure for structural Non-IID",
    blurb:
      "Three imbalance-aware reweighting schemes (fault-count, val-F1 softmax, inverse-loss) closed less than 14% of the gap to centralized training. Proved mechanistically — the achievable weight space does not contain the centralized solution.",
  },
  {
    n: "03",
    title:
      "An interpretability pipeline that surfaces a Non-IID failure RMSE alone could not see",
    blurb:
      "Integrated Gradients + a 17-entry maintenance ontology + 3 fault-mode rules + a cross-model comparison methodology. Surfaces that combined-data models key on os_2 (Mach number) as a subset-identity proxy — a specific representation-learning failure.",
  },
  {
    n: "04",
    title:
      "A Byzantine-robust aggregator (Krum) shown to recover catastrophic poisoning fully",
    blurb:
      "11-cell attack × defense matrix. Gradient-scale ×-10 against vanilla FedAvg pushes RMSE from 17.95 to 84.03. Krum (f=1) recovers both label-flip and gradient-scale attacks to RMSE 19.80 — within 1.85 cycles of the clean baseline.",
  },
  {
    n: "05",
    title: "A complete RQ2-followup hierarchy showing where the cure actually lives",
    blurb:
      "FedProx (+6.0% gap closed, drift-control), FedRep (+73.0%, per-client heads — the big positive), FedCCFA (null result with diagnosis of why). Together with the negative RQ2 result they form an empirical layer hierarchy: aggregation < drift-control < per-client architecture.",
  },
];

function Contributions() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-14 border-t border-border bg-bg-subtle/40">
      <div className="eyebrow">Five contributions</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-10 max-w-[34ch]">
        What the project actually adds to the literature.
      </h2>

      <ol className="space-y-0 divide-y divide-border border-y border-border">
        {CONTRIBUTIONS.map((c) => (
          <li key={c.n} className="py-7 grid grid-cols-[60px_1fr] gap-6">
            <span className="font-display text-2xl text-accent leading-none">
              {c.n}
            </span>
            <div>
              <h3 className="font-display text-[22px] leading-snug text-text mb-2">
                {c.title}
              </h3>
              <p className="text-[15.5px] text-text-dim leading-relaxed">
                {c.blurb}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

// ===========================================================================
// 4. RQ verdict matrix
// ===========================================================================
type Verdict = "answered" | "synthesised" | "open";

type RqRow = {
  id: string;
  topic: string;
  verdict: Verdict;
  result: string;
  evidence: string;
  evidenceTo?: string;
};

const RQ_ROWS: RqRow[] = [
  {
    id: "RQ1",
    topic: "Sensor / fleet heterogeneity (different sensor sets per client)",
    verdict: "open",
    result: "Out of scope — FD001 + FD003 share the same 14 informative sensors.",
    evidence: "FD002 / FD004 would require per-client encoders.",
  },
  {
    id: "RQ2",
    topic: "Imbalance-aware aggregation",
    verdict: "answered",
    result: "Negative. <14% of gap closed by three reweighting schemes.",
    evidence: "Story · 8-section report",
    evidenceTo: "/rq2-story",
  },
  {
    id: "RQ3",
    topic: "Sensor attribution + maintenance ontology",
    verdict: "answered",
    result:
      "Positive. 12 cross-model explanations · surfaces os_2 subset proxy.",
    evidence: "Story · 8-section report",
    evidenceTo: "/rq3-story",
  },
  {
    id: "RQ4",
    topic: "Concept drift over time",
    verdict: "synthesised",
    result:
      "Partial. Training-side drift observed across phases; input-side drift not separately experimented (static partition).",
    evidence: "Synthesis (soon)",
  },
  {
    id: "RQ5",
    topic: "Cross-client / per-subset evaluation",
    verdict: "synthesised",
    result:
      "Substantial. Per-subset breakdowns across 5 phases show asymmetric Non-IID damage and aggregator-dependent rebalancing.",
    evidence: "Synthesis (soon)",
  },
  {
    id: "RQ6",
    topic: "Privacy / membership inference / gradient leakage",
    verdict: "open",
    result:
      "Out of scope this round. Recognised as the natural complement to RQ7.",
    evidence: "Two follow-up experiments scoped in RQ7 report.",
  },
  {
    id: "RQ7",
    topic: "Model poisoning + Byzantine-robust aggregation",
    verdict: "answered",
    result:
      "Positive. Krum recovers RMSE 84 → 19.8 (within 1.85 of clean baseline).",
    evidence: "Story · 8-section report",
    evidenceTo: "/rq7-story",
  },
];

function RqVerdictMatrix() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-14 border-t border-border">
      <div className="eyebrow">Research questions</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-3 max-w-[36ch]">
        Seven questions, three verdicts.
      </h2>
      <p className="text-text-dim mb-10">
        The original project brief listed seven research questions. Three
        are answered directly with new experiments; two are synthesised
        from cross-cutting evidence in existing phases; two are explicit
        open follow-ups. All seven are framed below.
      </p>

      <div className="overflow-x-auto -mx-2">
        <table className="w-full text-[14.5px]">
          <thead>
            <tr className="border-b border-border-strong text-left">
              <th className="px-4 py-3 font-semibold text-text-dim w-[8%]">RQ</th>
              <th className="px-4 py-3 font-semibold text-text-dim w-[26%]">
                Topic
              </th>
              <th className="px-4 py-3 font-semibold text-text-dim w-[15%]">
                Verdict
              </th>
              <th className="px-4 py-3 font-semibold text-text-dim">Result</th>
              <th className="px-4 py-3 font-semibold text-text-dim w-[16%]">
                Evidence
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {RQ_ROWS.map((r) => (
              <tr key={r.id} className="align-top hover:bg-bg-subtle/40">
                <td className="px-4 py-4 font-mono-num font-semibold text-text">
                  {r.id}
                </td>
                <td className="px-4 py-4 text-text">{r.topic}</td>
                <td className="px-4 py-4">
                  <VerdictPill verdict={r.verdict} />
                </td>
                <td className="px-4 py-4 text-text-dim leading-relaxed">
                  {r.result}
                </td>
                <td className="px-4 py-4 text-sm">
                  {r.evidenceTo ? (
                    <NavLink to={r.evidenceTo} className="text-accent">
                      [{r.evidence}]
                    </NavLink>
                  ) : (
                    <span className="text-text-muted">{r.evidence}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function VerdictPill({ verdict }: { verdict: Verdict }) {
  const styles: Record<Verdict, { label: string; cls: string }> = {
    answered: {
      label: "Answered",
      cls: "text-good bg-good/10 border-good/30",
    },
    synthesised: {
      label: "Synthesised",
      cls: "text-accent bg-accent-subtle border-accent/30",
    },
    open: {
      label: "Open",
      cls: "text-text-muted bg-text-muted/10 border-text-muted/30",
    },
  };
  const s = styles[verdict];
  return (
    <span
      className={`inline-block px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider rounded border ${s.cls}`}
    >
      {s.label}
    </span>
  );
}

// ===========================================================================
// 5. Related work delta
// ===========================================================================
type RelatedRow = {
  paper: string;
  venue: string;
  theirAngle: string;
  ourDelta: string;
};

const RELATED: RelatedRow[] = [
  {
    paper: "McMahan et al.",
    venue: "AISTATS 2017",
    theirAngle:
      "Vanilla FedAvg: sample-count-weighted average of client weights. Assumes IID.",
    ourDelta:
      "Reproduced as the control baseline (P5/P6). Confirmed to fail under structural Non-IID (RQ2).",
  },
  {
    paper: "Yin et al. · Blanchard et al.",
    venue: "ICML 2018 · NeurIPS 2017",
    theirAngle:
      "Robust aggregation (trimmed mean, median, Krum) for accidentally-noisy clients.",
    ourDelta:
      "Extended to deliberate poisoning. Krum recovers fully; per-element methods only partially (RQ7).",
  },
  {
    paper: "Li et al.",
    venue: "MLSys 2020 (FedProx)",
    theirAngle:
      "Proximal regularisation on client loss to bound drift from global weights.",
    ourDelta:
      "Implemented as μ-sweep, +6% gap closed, per-subset rebalancing positive even with combined RMSE flat (RQ2 follow-up).",
  },
  {
    paper: "Collins et al.",
    venue: "ICML 2021 (FedRep)",
    theirAngle:
      "Federate shared representation, keep classifier heads local per client.",
    ourDelta:
      "Implemented end-to-end. +73% gap closed — the project's largest positive finding (RQ2 follow-up).",
  },
  {
    paper: "Sundararajan et al.",
    venue: "ICML 2017 (IG) + Saxena PHM 2008",
    theirAngle:
      "Path-integration attribution + canonical CMAPSS sensor table.",
    ourDelta:
      "Combined into cross-model attribution + maintenance ontology, surfacing Non-IID-specific interpretability failure (RQ3).",
  },
  {
    paper: "Landau et al.",
    venue: "Future Gen. Comp. Sys. 2026",
    theirAngle:
      "Robust aggregation for accidentally-noisy PHM clients (closest existing PHM-FL paper).",
    ourDelta:
      "Reframed against deliberate attackers — Landau's per-element robust family is insufficient; Krum required (RQ7).",
  },
];

function RelatedWorkDelta() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-14 border-t border-border bg-bg-subtle/40">
      <div className="eyebrow">Related work · what's different</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-3 max-w-[36ch]">
        How this project sits next to the canonical references.
      </h2>
      <p className="text-text-dim mb-10">
        Six rows. Each one names a paper from the project brief or the
        adjacent FL literature, summarises what they did, and states the
        specific delta this project adds.
      </p>

      <div className="space-y-3">
        {RELATED.map((r) => (
          <article
            key={r.paper + r.venue}
            className="rounded-lg border border-border bg-bg p-5 grid grid-cols-1 md:grid-cols-[1fr_2fr_2fr] gap-5"
          >
            <div>
              <div className="font-mono-num text-text font-medium text-[15px]">
                {r.paper}
              </div>
              <div className="text-xs text-text-muted mt-0.5">{r.venue}</div>
            </div>
            <div className="text-[14.5px] text-text-dim">
              <div className="eyebrow !text-text-muted mb-1">Their angle</div>
              {r.theirAngle}
            </div>
            <div className="text-[14.5px] text-text">
              <div className="eyebrow !text-accent mb-1">Our delta</div>
              {r.ourDelta}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

// ===========================================================================
// 6. Reproducibility
// ===========================================================================
function Reproducibility() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-14 border-t border-border">
      <div className="eyebrow">Reproducibility</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-8 max-w-[36ch]">
        Every claim runs in under 5 hours of CPU time.
      </h2>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-px bg-border rounded-lg overflow-hidden">
        {[
          { value: "216 / 216", label: "Tests passing" },
          { value: "11", label: "Experimental phases" },
          { value: "30,018", label: "Model parameters" },
          { value: "4", label: "Simulated airline clients" },
          { value: "50", label: "FL rounds per run" },
          { value: "~5 h", label: "Total CPU wall-clock" },
        ].map((s) => (
          <div key={s.label} className="bg-bg p-5 text-center">
            <div className="font-display text-[28px] leading-none text-text font-mono-num">
              {s.value}
            </div>
            <div className="text-[12px] text-text-dim mt-2 leading-snug">
              {s.label}
            </div>
          </div>
        ))}
      </div>

      <p className="mt-8 text-text-dim max-w-[80ch]">
        Each phase has a single CLI under{" "}
        <code className="font-mono-num text-text bg-bg-subtle px-1.5 py-0.5 rounded">
          scripts/
        </code>{" "}
        that loads the right data bundle, runs training, writes per-round
        CSVs + a phase metrics.json, and renders all plots. The aggregate
        manifest at{" "}
        <code className="font-mono-num text-text bg-bg-subtle px-1.5 py-0.5 rounded">
          results/summary.json
        </code>{" "}
        lists every phase and is what the frontend reads. No hidden
        dependencies, no cluster, no GPU.
      </p>
    </section>
  );
}

// ===========================================================================
// 7. Open work
// ===========================================================================
function OpenWork() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-14 border-t border-border bg-bg-subtle/40">
      <div className="eyebrow">Open follow-ups</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-8 max-w-[36ch]">
        What this project commits to NOT having answered.
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {[
          {
            title: "RQ6 — Privacy",
            body: "Membership inference + gradient leakage against the four trained checkpoints; quantifies the privacy cost of Krum's update-inspection requirement. Two scoped experiments in the RQ7 report; not yet executed.",
          },
          {
            title: "RQ1 — Sensor heterogeneity",
            body: "Per-client encoders when each airline ships different sensors (FD002 / FD004 with their multi-regime sensor layout). Architectural — federation contract has to allow encoders to differ.",
          },
          {
            title: "Coordinated + adaptive attacks (RQ7 extension)",
            body: "Krum's robustness assumes one Byzantine. 2 coordinated attackers break the geometric isolation argument; an attacker who knows Krum can craft updates that slip the filter. Both require larger client counts (n ≥ 6) to test cleanly.",
          },
        ].map((o) => (
          <div
            key={o.title}
            className="rounded-lg border border-border bg-bg p-6"
          >
            <h3 className="font-display text-xl text-text mb-3">{o.title}</h3>
            <p className="text-sm text-text-dim leading-relaxed">{o.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

// ===========================================================================
// 8. Footer navigation
// ===========================================================================
function FooterNav() {
  return (
    <nav
      aria-label="Next pages"
      className="px-10 md:px-16 lg:px-24 py-12 border-t border-border flex flex-wrap items-center justify-between gap-6"
    >
      <NavLink
        to="/"
        className="text-text-dim hover:text-text inline-flex items-center gap-2"
      >
        <span aria-hidden>←</span> Back to cover
      </NavLink>
      <div className="flex flex-wrap items-center gap-6 text-sm">
        <NavLink to="/rq2-story" className="text-accent">
          [Read RQ2 negative finding →]
        </NavLink>
        <NavLink to="/rq3-story" className="text-accent">
          [Read RQ3 interpretability →]
        </NavLink>
        <NavLink to="/rq7-story" className="text-accent">
          [Read RQ7 security →]
        </NavLink>
      </div>
    </nav>
  );
}
