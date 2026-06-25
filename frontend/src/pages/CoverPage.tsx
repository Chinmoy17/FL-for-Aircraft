import { NavLink } from "react-router-dom";

/**
 * CoverPage — landing at "/".
 *
 * Rebuilt with the problem-first flow a reader-from-nowhere needs:
 *
 *   1. Hero               — title + byline + primary CTAs.
 *   2. The problem        — three short panels: data setup, why FL,
 *                            why FL is hard. A newcomer leaves this
 *                            section knowing exactly what we're
 *                            trying to do and why it's non-trivial.
 *   3. Tasks executed     — Task 1 / Task 2 / Task 3 from the
 *                            project brief, mapped to what shipped.
 *   4. Seven research     — the full 7-RQ list as a table with
 *      questions            verdict pills (answered / synthesised /
 *                            open). At-a-glance project scope.
 *   5. Three findings     — deeper card on each answered RQ.
 *   6. By the numbers     — six anchor stats for proof.
 *   7. Reading paths      — four entry points (abstract, baseline,
 *                            demo, reports).
 *   8. Quick links footer — source / dataset / author / tech.
 *
 * The accent slate-blue appears in exactly four places per the
 * 60/30/10 rule: primary CTA, verdict pills, link arrows, and the
 * single tinted Status panel under the RQ table.
 */
export function CoverPage() {
  return (
    <article className="w-full">
      <Hero />
      <TheProblem />
      <TasksExecuted />
      <ResearchQuestions />
      <ThreeFindings />
      <ByTheNumbers />
      <ReadingPaths />
      <QuickLinks />
    </article>
  );
}

// ===========================================================================
// 1. Hero
// ===========================================================================
function Hero() {
  return (
    <section className="px-10 md:px-16 lg:px-24 pt-16 pb-12 border-b border-border">
      <div className="eyebrow">PhD research · 2026</div>

      <h1
        className="
          font-display text-text
          text-[44px] sm:text-[56px] lg:text-[68px]
          leading-[1.05] tracking-tight
          mt-4 max-w-[16ch]
        "
      >
        Federated Learning for{" "}
        <em className="not-italic text-accent">Aircraft Engine</em> PHM
      </h1>

      <p className="mt-6 text-[19px] leading-relaxed text-text-dim max-w-[64ch]">
        Predicting when a jet engine will fail — without the airlines
        that own the data ever sharing it. A PhD research project
        spanning three answered questions, two synthesised ones, and
        two scoped follow-ups, on the NASA C-MAPSS turbofan dataset.
      </p>

      <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-text-muted">
        <span className="text-text">A PhD research project by Chinmoy Mitra</span>
        <Dot />
        <span>NASA C-MAPSS turbofan dataset</span>
        <Dot />
        <span>11 experimental phases · 216 tests passing</span>
      </div>

      <div className="mt-10 flex flex-wrap items-center gap-3">
        <NavLink
          to="/abstract"
          className="
            inline-flex items-center gap-2 px-4 py-2.5 rounded-md
            bg-accent text-[#fafaf7] font-semibold text-sm
            hover:brightness-110 transition
            focus-visible:outline-none focus-visible:ring-2
            focus-visible:ring-accent/40 focus-visible:ring-offset-2
            focus-visible:ring-offset-bg
          "
        >
          Read the abstract <span aria-hidden>→</span>
        </NavLink>

        <NavLink
          to="/demo"
          className="
            inline-flex items-center gap-2 px-4 py-2.5 rounded-md
            border border-border text-text font-medium text-sm
            hover:border-border-strong hover:bg-bg-subtle/40 transition
            focus-visible:outline-none focus-visible:ring-2
            focus-visible:ring-accent/30
          "
        >
          Try the live demo <span aria-hidden>→</span>
        </NavLink>

        <a
          href="https://github.com/Chinmoy17/FL-for-Aircraft"
          target="_blank"
          rel="noreferrer"
          className="
            inline-flex items-center gap-2 px-4 py-2.5 rounded-md
            border border-border text-text font-medium text-sm
            hover:border-border-strong hover:bg-bg-subtle/40 transition
          "
        >
          <GitHubMark />
          <span>GitHub</span>
          <span aria-hidden>↗</span>
        </a>
      </div>
    </section>
  );
}

function GitHubMark() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="currentColor"
      aria-hidden
    >
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function Dot() {
  return <span className="text-text-muted/50">·</span>;
}

// ===========================================================================
// 2. The problem
// ===========================================================================
function TheProblem() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-16">
      <div className="eyebrow">The problem</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-3 max-w-[36ch]">
        Predict when a jet engine will fail. Across multiple airlines.
        Without sharing their data.
      </h2>
      <p className="text-text-dim mb-10">
        Imagine you are the chief engineer at an airline. Every engine
        produces millions of sensor readings per flight. Your job is to
        predict, from those readings, when each engine will need
        maintenance — too early wastes tens of thousands of dollars,
        too late grounds an aircraft or worse. This problem is called{" "}
        <strong className="text-text">Remaining Useful Life (RUL)</strong>{" "}
        estimation, and it sits at the heart of Prognostics and Health
        Management (PHM).
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <ProblemPanel
          step="01"
          title="A single airline is data-starved"
          body={
            <>
              Modern aircraft engines rarely fail — exactly because
              maintenance teams are good at their job. One airline simply
              doesn&apos;t see enough run-to-failure events to train a
              reliable deep model. The obvious fix is to pool failure data
              across many airlines, multiplying the learning signal.
            </>
          }
        />
        <ProblemPanel
          step="02"
          title="But airlines won&apos;t share"
          body={
            <>
              Engine degradation records reveal which aircraft are
              approaching retirement, which routes face capacity risk, and
              what maintenance costs are coming. That information is
              commercially sensitive, regulatorily protected, and (per a
              2025 European data breach) actively targeted by attackers.
              Centralising raw sensor data is not an option.
            </>
          }
        />
        <ProblemPanel
          step="03"
          title="Federated Learning is the answer"
          body={
            <>
              Instead of moving data to a server, FL moves the model to
              each airline. Each operator trains a copy on its own data,
              then sends only the updated weights — never raw sensor
              readings — to a coordinator that averages them into a new
              global model. The cycle repeats until every airline has a
              model trained on every fleet&apos;s knowledge.
            </>
          }
        />
      </div>
    </section>
  );
}

function ProblemPanel({
  step,
  title,
  body,
}: {
  step: string;
  title: string;
  body: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-bg p-6">
      <div className="font-display text-accent text-2xl leading-none mb-3">
        {step}
      </div>
      <h3 className="font-display text-[22px] leading-snug text-text mb-3">
        {title}
      </h3>
      <p className="text-[15px] text-text-dim leading-relaxed">{body}</p>
    </div>
  );
}

// ===========================================================================
// 3. Tasks executed
// ===========================================================================
function TasksExecuted() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-16 border-t border-border bg-bg-subtle/40">
      <div className="eyebrow">What this project actually did</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-3 max-w-[36ch]">
        Three tasks. Eleven experimental phases. Three RQ reports.
      </h2>
      <p className="text-text-dim mb-10">
        The project brief asks for three deliverables: a federated baseline
        end-to-end, depth on at least one research question, and an honest
        look at where the work goes next. Each task below maps to its
        artifacts in the repo.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <TaskCard
          n="Task 1"
          title="Baseline implementation"
          body={
            <>
              A federated PHM pipeline with 4 simulated airline clients
              and a central FedAvg aggregator, jointly predicting RUL and
              early fault risk. Six experimental phases land here (00 EDA
              through 06 Non-IID baseline) plus a centralized upper bound
              and a local-only lower bound for comparison.
            </>
          }
          linkLabel="Walk the baseline phases →"
          linkTo="/experiments/00-eda"
        />
        <TaskCard
          n="Task 2"
          title="Research questions addressed"
          body={
            <>
              Three RQs answered directly with dedicated experiments
              (RQ2 negative, RQ3 positive, RQ7 positive). Two further RQs
              synthesised from cross-cutting evidence already in the
              project (RQ4 partial, RQ5 substantial). Plus an RQ2
              follow-up trilogy (FedProx, FedRep, FedCCFA) ruling in the
              right intervention layer.
            </>
          }
          linkLabel="See verdict on all 7 RQs ↓"
          linkTo="#research-questions"
        />
        <TaskCard
          n="Task 3"
          title="Future directions"
          body={
            <>
              RQ1 (sensor heterogeneity) and RQ6 (privacy) scoped
              transparently as open follow-ups, not claimed answered.
              Each RQ report ends with ranked next-experiments and a
              proposed novel synthesis (e.g. Krum + DP + per-client
              heads, RQ3-as-forensics for RQ7 attacks).
            </>
          }
          linkLabel="See open work on the abstract page →"
          linkTo="/abstract"
        />
      </div>
    </section>
  );
}

function TaskCard({
  n,
  title,
  body,
  linkLabel,
  linkTo,
}: {
  n: string;
  title: string;
  body: React.ReactNode;
  linkLabel: string;
  linkTo: string;
}) {
  return (
    <div className="rounded-lg border border-border bg-bg p-6 flex flex-col">
      <div className="eyebrow text-accent mb-2">{n}</div>
      <h3 className="font-display text-[22px] leading-snug text-text mb-3">
        {title}
      </h3>
      <p className="text-[15px] text-text-dim leading-relaxed mb-5">{body}</p>
      <NavLink
        to={linkTo}
        className="mt-auto text-accent text-sm font-medium hover:underline"
      >
        [{linkLabel}]
      </NavLink>
    </div>
  );
}

// ===========================================================================
// 4. Seven research questions
// ===========================================================================
type Verdict = "answered" | "synthesised" | "open";

type RqRow = {
  id: string;
  side: "Machine learning" | "Security";
  topic: string;
  oneLine: string;
  verdict: Verdict;
  to?: string;
};

const RQS: RqRow[] = [
  {
    id: "RQ1",
    side: "Machine learning",
    topic: "Sensor / fleet heterogeneity",
    oneLine:
      "Different airlines log different sensors. Map them into a shared representation.",
    verdict: "open",
  },
  {
    id: "RQ2",
    side: "Machine learning",
    topic: "Imbalance-aware aggregation",
    oneLine:
      "Rare failures get averaged away under FedAvg. Can smarter aggregation help?",
    verdict: "answered",
    to: "/rq2-story",
  },
  {
    id: "RQ3",
    side: "Machine learning",
    topic: "Sensor attribution + ontology",
    oneLine:
      "Turn one RUL prediction into a maintenance brief with attribution + fault rules.",
    verdict: "answered",
    to: "/rq3-story",
  },
  {
    id: "RQ4",
    side: "Machine learning",
    topic: "Concept drift over time",
    oneLine:
      "Sensor-to-RUL mapping shifts mid-training. Can we detect and recover within 10 rounds?",
    verdict: "synthesised",
    to: "/rq4-rq5-synthesis",
  },
  {
    id: "RQ5",
    side: "Machine learning",
    topic: "Validation under Non-IID",
    oneLine:
      "Cross-client scoring punishes difference, not poor quality. How much does this distort weights?",
    verdict: "synthesised",
    to: "/rq4-rq5-synthesis",
  },
  {
    id: "RQ6",
    side: "Security",
    topic: "Membership inference",
    oneLine:
      "An attacker capturing weight updates may recover information about private engine data.",
    verdict: "open",
  },
  {
    id: "RQ7",
    side: "Security",
    topic: "Model poisoning",
    oneLine:
      "A malicious airline biases the global model against a competitor. What defenses work?",
    verdict: "answered",
    to: "/rq7-story",
  },
];

function ResearchQuestions() {
  return (
    <section
      id="research-questions"
      className="px-10 md:px-16 lg:px-24 py-16 border-t border-border scroll-mt-4"
    >
      <div className="eyebrow">Seven research questions</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-3 max-w-[40ch]">
        The full project brief — answered, synthesised, or scoped as open.
      </h2>
      <p className="text-text-dim mb-8">
        The brief listed seven open research questions split between the
        machine-learning side (RQ1–5) and the security side (RQ6–7). The
        table below covers all seven, with verdict pills that match what
        the project actually delivered. Click any answered or synthesised
        row to jump into its story.
      </p>

      <div className="overflow-x-auto rounded-lg border border-border bg-bg">
        <table className="w-full text-[14.5px]">
          <thead>
            <tr className="border-b border-border-strong text-left bg-bg-subtle/40">
              <th className="px-4 py-3 font-semibold text-text-dim w-[8%]">RQ</th>
              <th className="px-4 py-3 font-semibold text-text-dim w-[14%]">
                Side
              </th>
              <th className="px-4 py-3 font-semibold text-text-dim w-[22%]">
                Topic
              </th>
              <th className="px-4 py-3 font-semibold text-text-dim">
                What the question asks
              </th>
              <th className="px-4 py-3 font-semibold text-text-dim w-[14%]">
                Verdict
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {RQS.map((r) => {
              // Clickable rows get group/cursor/hover treatment so the
              // affordance is obvious. The eye picks up the accent-coloured
              // topic + the trailing chevron together and reads them as
              // "this row is a link to the story".
              const isClickable = !!r.to;
              const baseCls =
                "group align-top transition-colors " +
                (isClickable
                  ? "hover:bg-accent/[0.04] cursor-pointer"
                  : "");
              const topicCls =
                "px-4 py-4 text-text transition-colors " +
                (isClickable ? "group-hover:text-accent" : "");
              return (
                <tr
                  key={r.id}
                  className={baseCls}
                  onClick={() => {
                    if (r.to) {
                      // SPA navigation via NavLink in last cell — row click
                      // just calls .click() on the embedded link.
                      const el = document.querySelector<HTMLAnchorElement>(
                        `a[data-rqrow="${r.id}"]`,
                      );
                      el?.click();
                    }
                  }}
                >
                  <td className="px-4 py-4 font-mono-num font-semibold text-text">
                    {r.id}
                  </td>
                  <td className="px-4 py-4 text-text-dim text-sm">
                    {r.side}
                  </td>
                  <td className={topicCls}>
                    <span className={isClickable ? "group-hover:underline" : ""}>
                      {r.topic}
                    </span>
                  </td>
                  <td className="px-4 py-4 text-text-dim leading-relaxed">
                    {r.oneLine}
                  </td>
                  <td className="px-4 py-4">
                    {r.to ? (
                      <NavLink
                        to={r.to}
                        data-rqrow={r.id}
                        className="inline-flex items-center gap-2 group/pill"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <VerdictPill verdict={r.verdict} />
                        <span
                          aria-hidden
                          className="text-text-muted text-base leading-none transition-transform group-hover:translate-x-1 group-hover:text-accent"
                        >
                          →
                        </span>
                      </NavLink>
                    ) : (
                      <VerdictPill verdict={r.verdict} />
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-6 text-sm text-text-muted">
        Three RQs are answered directly with dedicated experiments. Two
        more (RQ4, RQ5) are synthesised from cross-cutting evidence
        already produced by the answered ones. Two (RQ1, RQ6) are framed
        honestly as open follow-ups, not claimed — see the{" "}
        <NavLink to="/abstract" className="text-accent">
          abstract page
        </NavLink>{" "}
        for the full verdict matrix.
      </p>
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
// 5. Three findings (outcomes)
// ===========================================================================
type Finding = {
  rq: string;
  verdict: "positive" | "negative" | "null";
  title: string;
  blurb: string;
  headline: string;
  to: string;
};

const FINDINGS: Finding[] = [
  {
    rq: "RQ2",
    verdict: "negative",
    title: "Aggregation alone cannot fix structural Non-IID",
    blurb:
      "Three imbalance-aware reweighting schemes (fault count, val-F1 softmax, inverse loss) closed less than 14% of the gap to centralized training. The aggregator's weight space provably does not contain the centralized solution.",
    headline: "<14% of gap closed",
    to: "/rq2-story",
  },
  {
    rq: "RQ3",
    verdict: "positive",
    title: "Cross-model attribution exposes a Non-IID interpretability failure",
    blurb:
      "Integrated Gradients + a 17-entry maintenance ontology + 3 fault-mode rules. The same engine fed through 4 checkpoints reveals that combined-data models key on operational settings (Mach number) as a subset proxy — invisible from RMSE alone.",
    headline: "12 case studies, 14.8 s on CPU",
    to: "/rq3-story",
  },
  {
    rq: "RQ7",
    verdict: "positive",
    title: "Krum recovers from catastrophic Byzantine attack",
    blurb:
      "An 11-cell matrix (2 attacks × 3 defenses + baselines). Gradient scaling ×−10 against vanilla FedAvg pushes RMSE from 17.95 to 84.03. The geometric whole-update defense (Krum) restores it to 19.80 — within 1.85 cycles.",
    headline: "RMSE 84 → 19.8",
    to: "/rq7-story",
  },
];

function ThreeFindings() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-16 border-t border-border bg-bg-subtle/40">
      <div className="eyebrow">Outcomes — three findings</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-3 max-w-[40ch]">
        What the three answered research questions actually say.
      </h2>
      <p className="text-text-dim mb-10">
        Each card below is the headline result from one of the three
        answered RQs. The accompanying long-form story page expands the
        argument with mechanism, mathematical bounds, and figures; the
        technical report on the same RQ goes deeper still.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {FINDINGS.map((f) => (
          <FindingCard key={f.rq} finding={f} />
        ))}
      </div>
    </section>
  );
}

function FindingCard({ finding }: { finding: Finding }) {
  const verdictSymbol =
    finding.verdict === "positive"
      ? "+"
      : finding.verdict === "negative"
      ? "−"
      : "○";
  const verdictCls =
    finding.verdict === "positive"
      ? "text-good bg-good/10 border-good/30"
      : finding.verdict === "negative"
      ? "text-bad bg-bad/10 border-bad/30"
      : "text-text-muted bg-text-muted/10 border-text-muted/30";

  return (
    <NavLink
      to={finding.to}
      className="
        group block rounded-lg border border-border bg-bg p-6
        hover:border-border-strong hover:shadow-sm transition-all
        no-underline
      "
    >
      <div className="flex items-center gap-3 mb-3">
        <span
          className={`
            inline-flex items-center justify-center
            w-6 h-6 rounded-full border text-[12px] leading-none font-semibold
            ${verdictCls}
          `}
        >
          {verdictSymbol}
        </span>
        <span className="eyebrow !text-text-dim">{finding.rq}</span>
      </div>

      <h3 className="font-display text-[22px] leading-snug text-text mb-3">
        {finding.title}
      </h3>

      <p className="text-sm text-text-dim leading-relaxed mb-5">
        {finding.blurb}
      </p>

      <div className="flex items-baseline justify-between border-t border-border pt-4">
        <span className="font-mono-num text-text text-base">
          {finding.headline}
        </span>
        <span className="text-accent text-sm group-hover:underline">
          [Read story →]
        </span>
      </div>
    </NavLink>
  );
}

// ===========================================================================
// 6. By the numbers
// ===========================================================================
type Stat = {
  value: string;
  label: string;
  sub: string;
  tone?: "good" | "bad" | "neutral";
};

const STATS: Stat[] = [
  {
    value: "84.03",
    label: "Worst-case RMSE under undefended Byzantine attack",
    sub: "RQ7 · gradient scale ×−10 vs vanilla FedAvg — 4.7× the clean baseline.",
    tone: "bad",
  },
  {
    value: "19.80",
    label: "Krum-defended RMSE — both label-flip and gradient-scale",
    sub: "RQ7 · within 1.85 cycles of clean baseline. Recovery is essentially perfect.",
    tone: "good",
  },
  {
    value: "+73%",
    label: "Non-IID gap closed by per-client heads (FedRep)",
    sub: "RQ2 follow-up · architectural personalisation beats aggregation tweaks.",
    tone: "good",
  },
  {
    value: "12",
    label: "Cross-model attribution explanations",
    sub: "RQ3 · 3 engines × 4 trained checkpoints. Surfaces a Non-IID interpretability failure.",
  },
  {
    value: "30,018",
    label: "Model parameters · GroupNorm-only",
    sub: "Multi-task CNN — joint RUL regression + binary fault head. Federated-safe by design.",
  },
  {
    value: "216 / 216",
    label: "Unit and integration tests passing",
    sub: "Every aggregator, every attack, every defense, every loss function.",
  },
];

function ByTheNumbers() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-16 border-t border-border">
      <div className="eyebrow">By the numbers</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-10 max-w-[36ch]">
        What was measured, what was found.
      </h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-px bg-border rounded-lg overflow-hidden">
        {STATS.map((s) => (
          <StatCard key={s.label} stat={s} />
        ))}
      </div>
    </section>
  );
}

function StatCard({ stat }: { stat: Stat }) {
  const tone =
    stat.tone === "bad"
      ? "text-bad"
      : stat.tone === "good"
      ? "text-good"
      : "text-text";
  return (
    <div className="bg-bg p-6 flex flex-col gap-3">
      <div
        className={`font-display ${tone} text-[44px] leading-none tracking-tight font-mono-num`}
      >
        {stat.value}
      </div>
      <div className="text-sm font-medium text-text">{stat.label}</div>
      <div className="text-[13px] text-text-dim leading-snug">{stat.sub}</div>
    </div>
  );
}

// ===========================================================================
// 7. Reading paths
// ===========================================================================
type Path = {
  eyebrow: string;
  title: string;
  blurb: string;
  to: string;
};

const PATHS: Path[] = [
  {
    eyebrow: "Start here · 5 min",
    title: "Read the abstract",
    blurb:
      "Project framing, threat model, the seven research questions and their verdicts, related-work delta — all in one page.",
    to: "/abstract",
  },
  {
    eyebrow: "For the reviewer · 30 min",
    title: "Walk the empirical baseline",
    blurb:
      "Phases 00–06: EDA, data pipeline, centralized baseline, local-only floor, vanilla FedAvg, Non-IID partition. Every figure with a 2-paragraph explanation.",
    to: "/experiments/00-eda",
  },
  {
    eyebrow: "Interactive · 5 min",
    title: "Try the live demo",
    blurb:
      "Pick a trained checkpoint and a test engine. Backend runs Integrated Gradients on demand and returns a sensor-level explanation grounded in the maintenance ontology.",
    to: "/demo",
  },
  {
    eyebrow: "For the technical reader · 60 min",
    title: "Read the technical reports",
    blurb:
      "Three long-form .md reports (RQ2 / RQ3 / RQ7), each with 8 sections + TL;DR + artifact pointers. Closest to a chapter draft.",
    to: "/reports",
  },
];

function ReadingPaths() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-16 border-t border-border bg-bg-subtle/40">
      <div className="eyebrow">Where to start</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-10 max-w-[36ch]">
        Four entry points, depending on what you&apos;re here to find.
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {PATHS.map((p) => (
          <PathCard key={p.title} path={p} />
        ))}
      </div>
    </section>
  );
}

function PathCard({ path }: { path: Path }) {
  return (
    <NavLink
      to={path.to}
      className="
        group block rounded-lg border border-border bg-bg p-6
        hover:border-border-strong hover:shadow-sm transition-all
        no-underline
      "
    >
      <div className="eyebrow">{path.eyebrow}</div>
      <h3 className="font-display text-2xl text-text mt-3 mb-3 group-hover:text-accent transition-colors">
        {path.title}
      </h3>
      <p className="text-[14.5px] text-text-dim leading-relaxed">
        {path.blurb}
      </p>
    </NavLink>
  );
}

// ===========================================================================
// 8. Quick links
// ===========================================================================
function QuickLinks() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-12 border-t border-border">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 text-sm">
        <QuickLinkGroup
          eyebrow="Source"
          items={[
            { label: "GitHub repository ↗", href: "https://github.com/Chinmoy17/FL-for-Aircraft" },
            { label: "Branch: p7_demo ↗", href: "https://github.com/Chinmoy17/FL-for-Aircraft/tree/p7_demo" },
          ]}
        />
        <QuickLinkGroup
          eyebrow="Dataset"
          items={[
            { label: "NASA C-MAPSS ↗", href: "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/" },
            { label: "Saxena et al., PHM 2008 ↗", href: "https://ti.arc.nasa.gov/c/6/" },
          ]}
        />
        <QuickLinkGroup
          eyebrow="Author"
          items={[
            { label: "chinmoy17.github.io ↗", href: "https://chinmoy17.github.io/" },
            { label: "Research page ↗", href: "https://chinmoy17.github.io/research" },
          ]}
        />
        <QuickLinkGroup
          eyebrow="Tech"
          items={[
            { label: "PyTorch 2.12 + Captum 0.9", href: "" },
            { label: "FastAPI + Vite + React 19", href: "" },
          ]}
        />
      </div>
    </section>
  );
}

function QuickLinkGroup({
  eyebrow,
  items,
}: {
  eyebrow: string;
  items: { label: string; href: string }[];
}) {
  return (
    <div>
      <div className="eyebrow mb-3">{eyebrow}</div>
      <ul className="space-y-2">
        {items.map((it) =>
          it.href ? (
            <li key={it.label}>
              <a
                href={it.href}
                target="_blank"
                rel="noreferrer"
                className="text-text-dim hover:text-text"
              >
                {it.label}
              </a>
            </li>
          ) : (
            <li key={it.label} className="text-text-muted">
              {it.label}
            </li>
          ),
        )}
      </ul>
    </div>
  );
}
