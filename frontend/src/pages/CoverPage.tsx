import { NavLink } from "react-router-dom";

/**
 * CoverPage — landing page at "/".
 *
 * Six sections, full-width, designed under the ui-craft "trust"
 * intent:
 *
 *   1. Hero          — eyebrow + Instrument Serif title + subtitle +
 *                       byline + status badges + two CTAs.
 *   2. Abstract      — three short paragraphs of prose.
 *   3. By the numbers — six anchor stats. The first stat (the
 *                       worst-case attack damage RMSE 84.03) is
 *                       placed deliberately to anchor reviewer
 *                       expectations: "this project quantifies
 *                       Byzantine attack severity 4.7x baseline."
 *   4. Three findings — one card per answered research question.
 *                       Glance-readable verdict + headline number +
 *                       link.
 *   5. Reading paths  — four entry-point cards covering different
 *                       reviewer profiles (skim vs deep dive vs
 *                       try the demo vs read the reports).
 *   6. Quick links    — small footer with GitHub + reports + dataset.
 *
 * Visual hierarchy is enforced at three levels: 1) the Instrument
 * Serif title in the hero (biggest), 2) the section labels
 * (eyebrows + Instrument Serif), 3) body copy and metadata
 * (Inter, gray). The slate-blue accent appears in exactly four
 * places per the 60/30/10 rule: primary CTA, "+" / "−" verdict
 * badges, link arrows, and one tinted abstract panel.
 */
export function CoverPage() {
  return (
    <article className="w-full">
      <Hero />
      <Abstract />
      <ByTheNumbers />
      <ThreeFindings />
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
        Predicting jet-engine failures across multiple airlines without ever
        sharing their data. Three research questions answered — one
        negative, two positive — and one open security pipeline that
        recovers a 4.7× RMSE collapse to within 1.85 cycles of the clean
        baseline.
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
            bg-accent text-white font-medium text-sm
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
            text-text-dim font-medium text-sm
            hover:text-text hover:underline transition
          "
        >
          GitHub <span aria-hidden>↗</span>
        </a>
      </div>
    </section>
  );
}

function Dot() {
  return <span className="text-text-muted/50">·</span>;
}

// ===========================================================================
// 2. Abstract
// ===========================================================================
function Abstract() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-16">
      <div className="eyebrow">Abstract</div>

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-12">
        <div className="space-y-5 text-[17px] leading-[1.7] text-text">
          <p>
            Airlines hold the data that would make aircraft-engine{" "}
            <strong>predictive maintenance</strong> work: years of dense
            sensor traces ending in run-to-failure events. They do not
            share it. Each fleet's training data is a commercial
            differentiator and a regulatory headache, and a centralised
            model is operationally impossible.
          </p>
          <p>
            This project builds the{" "}
            <strong>federated alternative</strong> end-to-end on NASA's
            C-MAPSS turbofan dataset: a 30,018-parameter convolutional
            network trained across four simulated airline clients with
            structurally different fault-mode mixes (FD001 HPC-only vs
            FD003 HPC + Fan). Eleven experimental phases. Three of seven
            project research questions answered directly: one negative
            (server-side reweighting cannot close the Non-IID gap), two
            positive (Integrated-Gradients attribution surfaces an
            interpretability failure; Krum-based robust aggregation
            recovers a catastrophic Byzantine attack within 1.85 RMSE of
            the clean baseline).
          </p>
          <p>
            Two further questions (RQ4 concept drift, RQ5 cross-client
            evaluation) are{" "}
            <em className="text-text-dim">synthesised from existing
            evidence</em> rather than separately experimented; RQ1
            (sensor heterogeneity) and RQ6 (privacy) are explicitly
            scoped as open follow-up work. Every claim is reproducible
            from the repository in 55 minutes of CPU time.
          </p>
        </div>

        <aside
          className="
            rounded-lg border border-border bg-accent-subtle/60
            p-6 self-start
          "
        >
          <div className="eyebrow text-accent">Status</div>
          <ul className="mt-3 text-sm space-y-2.5 text-text">
            <li className="flex items-baseline gap-3">
              <span className="text-good text-base leading-none mt-0.5">+</span>
              <span>
                <strong>RQ3 — Interpretability:</strong> attribution +
                ontology + cross-model comparison shipped.
              </span>
            </li>
            <li className="flex items-baseline gap-3">
              <span className="text-good text-base leading-none mt-0.5">+</span>
              <span>
                <strong>RQ7 — Security:</strong> Krum recovers from
                catastrophic poisoning to within 1.85 RMSE.
              </span>
            </li>
            <li className="flex items-baseline gap-3">
              <span className="text-bad text-base leading-none mt-0.5">−</span>
              <span>
                <strong>RQ2 — Aggregation:</strong> imbalance-aware
                reweighting closes &lt;14% of Non-IID gap. Cure lives
                elsewhere.
              </span>
            </li>
            <li className="flex items-baseline gap-3">
              <span className="text-text-muted text-base leading-none mt-0.5">○</span>
              <span>
                <strong>RQ2 follow-ups:</strong> FedProx (+6%), FedRep
                (+73%), FedCCFA (null).
              </span>
            </li>
          </ul>
        </aside>
      </div>
    </section>
  );
}

// ===========================================================================
// 3. By the numbers
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
    sub: "RQ7 · gradient-scale ×−10 against vanilla FedAvg. 4.7× the clean baseline.",
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
    sub: "RQ2 follow-up · architectural personalisation, not aggregation, fixes structural heterogeneity.",
    tone: "good",
  },
  {
    value: "12",
    label: "Cross-model attribution explanations",
    sub: "RQ3 · 3 engines × 4 trained checkpoints, surfaces Non-IID interpretability failure.",
  },
  {
    value: "30,018",
    label: "Model parameters · GroupNorm, no BN",
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
    <section className="px-10 md:px-16 lg:px-24 py-16 border-t border-border bg-bg-subtle/40">
      <div className="eyebrow">By the numbers</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-10 max-w-[28ch]">
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
// 4. Three findings
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
      "Three imbalance-aware reweighting schemes (fault count, val-F1 softmax, inverse loss) closed less than 14% of the gap to centralized training. The weight space provably does not contain the centralized solution.",
    headline: "<14% of gap closed",
    to: "/rq2-story",
  },
  {
    rq: "RQ3",
    verdict: "positive",
    title: "Cross-model attribution surfaces a Non-IID interpretability failure",
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
    <section className="px-10 md:px-16 lg:px-24 py-16 border-t border-border">
      <div className="eyebrow">Three findings</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-10 max-w-[34ch]">
        What the three answered research questions actually say.
      </h2>

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
// 5. Reading paths
// ===========================================================================
type Path = {
  eyebrow: string;
  title: string;
  blurb: string;
  to: string;
  external?: boolean;
};

const PATHS: Path[] = [
  {
    eyebrow: "Start here · 5 min",
    title: "Read the abstract",
    blurb:
      "Project framing, threat model, the three answered research questions and their verdicts in one page.",
    to: "/abstract",
  },
  {
    eyebrow: "For the reviewer · 30 min",
    title: "Walk the empirical baseline",
    blurb:
      "Phases 00–06: EDA, data pipeline, centralized baseline, local-only floor, vanilla FedAvg, Non-IID partition. Every figure with a 2-paragraph explanation.",
    to: "/", // experiments index — will route once built
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
    to: "/", // reports page — will route once built
  },
];

function ReadingPaths() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-16 border-t border-border bg-bg-subtle/40">
      <div className="eyebrow">Where to start</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-10 max-w-[36ch]">
        Four entry points, depending on what you're here to find.
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
// 6. Quick links
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
