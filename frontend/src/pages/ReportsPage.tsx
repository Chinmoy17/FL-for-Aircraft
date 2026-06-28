import { NavLink } from "react-router-dom";

/**
 * ReportsPage — `/reports`.
 *
 * Browse the three long-form technical reports. Each report is a
 * markdown file living at the repo root (rq2_report.md / rq3_report.md /
 * rq7_report.md) following the same 8-section template. This page
 * gives reviewers:
 *
 *   - A one-page overview of all three reports' shapes and sizes.
 *   - A "what's inside" breakdown for each one.
 *   - Two clear actions per report: read on GitHub (rendered MD) and
 *     download raw .md.
 *   - A link back to the corresponding /rq*-story page that
 *     summarises the same content interactively.
 */
export function ReportsPage() {
  return (
    <article className="w-full">
      <Header />
      <ReportsGrid />
      <Methodology />
    </article>
  );
}

// ===========================================================================
// Header
// ===========================================================================
function Header() {
  return (
    <header className="px-10 md:px-16 lg:px-24 pt-16 pb-10 border-b border-border">
      <div className="eyebrow">Long-form · 60 min read</div>
      <h1 className="font-display text-[44px] sm:text-[52px] leading-[1.2] tracking-tight text-text mt-6 max-w-[28ch]">
        Technical{" "}
        <em className="not-italic text-accent">reports</em>
      </h1>
      <p className="mt-6 text-lg text-text-dim">
        Three long-form markdown reports — one per answered research
        question. Each follows the same 8-section template (problem,
        previous work, dataset, methods, experiment, mechanism, future
        directions, caveats) plus a TL;DR and artifact-pointer appendix.
        Closest to a thesis-chapter draft this project ships.
      </p>
      <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-text-muted">
        <span>3 reports · 132 KB · 2,134 lines · 8-section template</span>
        <span className="text-text-muted/50">·</span>
        <span className="font-mono-num">repo root: rq*_report.md</span>
      </div>
    </header>
  );
}

// ===========================================================================
// Reports grid
// ===========================================================================
type ReportInfo = {
  rq: string;
  verdict: "positive" | "negative";
  title: string;
  filename: string;
  sizeKb: number;
  lines: number;
  sections: string[];
  blurb: string;
  storyTo: string;
  storyLabel: string;
};

const REPORTS: ReportInfo[] = [
  {
    rq: "RQ2",
    verdict: "negative",
    title: "Imbalance-Aware Aggregation",
    filename: "rq2_report.md",
    sizeKb: 35,
    lines: 586,
    sections: [
      "Why aggregation alone can't fix structural Non-IID",
      "Four reweighting schemes (vanilla, fault-count, val-F1, inverse-loss)",
      "Per-round weight evolution — why all four collapse to ~uniform",
      "The mechanistic argument: weight space ≠ centralised solution",
      "Pointer to the FedProx / FedRep follow-up trilogy",
    ],
    blurb:
      "The negative finding. Three imbalance-aware reweighting schemes (fault-count, val-F1 softmax, inverse-loss) closed less than 14% of the gap to centralized training. Proved mechanistically that the aggregator's weight space does not contain the centralised solution. Points at the right intervention layer — client optimisation (FedProx) or per-client architecture (FedRep).",
    storyTo: "/rq2-story",
    storyLabel: "Read the interactive RQ2 story",
  },
  {
    rq: "RQ3",
    verdict: "positive",
    title: "Sensor Attribution & Maintenance Ontology",
    filename: "rq3_report.md",
    sizeKb: 49,
    lines: 779,
    sections: [
      "Why a black-box RUL prediction is operationally useless without a 'because'",
      "Integrated Gradients vs SHAP — why we pick IG for this domain",
      "The 17-entry sensor ontology + 3 fault-mode rules with reciprocal-rank matching",
      "12 cross-model attribution explanations (3 engines × 4 checkpoints)",
      "The os_2 subset-proxy finding RMSE alone could not see",
      "Future direction: interpretability-as-forensics for RQ7 attacks",
    ],
    blurb:
      "The interpretability pipeline. Integrated Gradients + maintenance ontology + 3 fault-mode rules + a deterministic narrative + cross-model comparison. Surfaces that combined-data models key on os_2 (Mach number) as a subset-identity proxy — a Non-IID interpretability failure that explains RQ2's negative finding qualitatively, not just numerically.",
    storyTo: "/rq3-story",
    storyLabel: "Read the interactive RQ3 story",
  },
  {
    rq: "RQ7",
    verdict: "positive",
    title: "Model Poisoning + Byzantine-Robust Aggregation",
    filename: "rq7_report.md",
    sizeKb: 47,
    lines: 769,
    sections: [
      "Threat model: one malicious airline targeting a competitor's fleet",
      "Two attacks (label-flip stealthy + gradient-scale ×−10 catastrophic)",
      "Three defenses (trimmed mean, median, Krum geometric)",
      "The 11-cell experiment matrix and headline numbers",
      "Why Krum works completely and per-element defenses only partially",
      "Future direction: triple-defense FL (Krum + DP + FedRep)",
    ],
    blurb:
      "The security finding. Gradient-scale ×−10 against vanilla FedAvg pushes RMSE from 17.95 to 84.03 (4.7× collapse). Krum's geometric whole-update isolation recovers both attacks to RMSE 19.80 — within 1.85 cycles of the clean baseline. Includes the per-element trimmed-mean = median degeneracy at n=4 as a mathematical finding, not a bug.",
    storyTo: "/rq7-story",
    storyLabel: "Read the interactive RQ7 story",
  },
];

const GH_RAW = "https://raw.githubusercontent.com/Chinmoy17/FL-for-Aircraft/p7_demo";
const GH_BLOB = "https://github.com/Chinmoy17/FL-for-Aircraft/blob/p7_demo";

function ReportsGrid() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-14">
      <div className="space-y-6">
        {REPORTS.map((r) => (
          <ReportCard key={r.rq} report={r} />
        ))}
      </div>
    </section>
  );
}

function ReportCard({ report }: { report: ReportInfo }) {
  const verdictCls =
    report.verdict === "positive"
      ? "text-good bg-good/10 border-good/30"
      : "text-bad bg-bad/10 border-bad/30";
  const verdictSym = report.verdict === "positive" ? "+" : "−";

  return (
    <article className="rounded-lg border border-border bg-bg p-8 lg:p-10 grid grid-cols-1 lg:grid-cols-[1fr_1.7fr] gap-8 lg:gap-12">
      {/* Left column: identity + actions */}
      <div className="flex flex-col">
        <div className="flex items-center gap-3 mb-4">
          <span
            className={`inline-flex items-center justify-center w-6 h-6 rounded-full border text-[12px] leading-none font-semibold ${verdictCls}`}
          >
            {verdictSym}
          </span>
          <span className="eyebrow !text-text-dim">{report.rq}</span>
        </div>

        <h2 className="font-display text-[28px] leading-snug text-text mb-4">
          {report.title}
        </h2>

        <dl className="text-[13px] text-text-muted space-y-1.5 font-mono-num mb-6">
          <div className="flex items-baseline gap-3">
            <dt className="w-16 text-text-muted/80">File</dt>
            <dd className="text-text">{report.filename}</dd>
          </div>
          <div className="flex items-baseline gap-3">
            <dt className="w-16 text-text-muted/80">Size</dt>
            <dd className="text-text">{report.sizeKb} KB</dd>
          </div>
          <div className="flex items-baseline gap-3">
            <dt className="w-16 text-text-muted/80">Lines</dt>
            <dd className="text-text">{report.lines}</dd>
          </div>
        </dl>

        <div className="mt-auto flex flex-col gap-2">
          <a
            href={`${GH_BLOB}/${report.filename}`}
            target="_blank"
            rel="noreferrer"
            className="
              inline-flex items-center justify-between gap-3
              px-4 py-2.5 rounded-md
              bg-accent text-[#fafaf7] font-semibold text-sm
              hover:brightness-110 transition no-underline
            "
          >
            <span>Read on GitHub (rendered)</span>
            <span aria-hidden>↗</span>
          </a>
          <a
            href={`${GH_RAW}/${report.filename}`}
            target="_blank"
            rel="noreferrer"
            className="
              inline-flex items-center justify-between gap-3
              px-4 py-2.5 rounded-md
              border border-border text-text font-medium text-sm
              hover:border-border-strong hover:bg-bg-subtle/40 transition
              no-underline
            "
          >
            <span>Download raw .md</span>
            <span aria-hidden>↓</span>
          </a>
          <NavLink
            to={report.storyTo}
            className="
              inline-flex items-center gap-2 px-4 py-2.5
              text-accent text-sm font-medium
              hover:underline
            "
          >
            [{report.storyLabel} →]
          </NavLink>
        </div>
      </div>

      {/* Right column: blurb + section breakdown */}
      <div>
        <p className="text-[15.5px] text-text leading-relaxed mb-6">
          {report.blurb}
        </p>

        <div className="eyebrow mb-3">What&apos;s inside</div>
        <ul className="space-y-2.5 text-[14.5px] text-text-dim">
          {report.sections.map((s) => (
            <li key={s} className="pl-5 relative leading-snug">
              <span
                aria-hidden
                className="absolute left-0 top-2 w-1.5 h-1.5 rounded-full bg-accent"
              />
              <span className="text-text">{s}</span>
            </li>
          ))}
        </ul>
      </div>
    </article>
  );
}

// ===========================================================================
// Methodology footer
// ===========================================================================
function Methodology() {
  return (
    <section className="px-10 md:px-16 lg:px-24 py-14 border-t border-border bg-bg-subtle/40">
      <div className="eyebrow">Report template</div>
      <h2 className="font-display text-3xl text-text mt-3 mb-4 max-w-[36ch]">
        Every report follows the same 8-section structure.
      </h2>
      <p className="text-text-dim mb-8">
        Standardising the report shape means a reviewer can move between
        them without re-learning the layout each time. The sections are
        ordered to mirror how the research itself happened — start with
        the problem, then say what already exists, then build, then
        measure, then mechanically explain, then admit limitations.
      </p>

      <ol className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 list-none p-0">
        {[
          { n: "1", label: "The problem", body: "Project framing, threat model, what this RQ is NOT about." },
          { n: "2", label: "Previous work", body: "Canonical references + what's new vs each." },
          { n: "3", label: "Our dataset", body: "Why FD001+FD003 specifically; what's controlled." },
          { n: "4", label: "Methods", body: "Implementation walkthrough with code pointers." },
          { n: "5", label: "Experiment", body: "Headline numbers, full tables, smoking-gun figure." },
          { n: "6", label: "Mechanism", body: "Why the result is what it is; mathematical argument." },
          { n: "7", label: "Future directions", body: "Ranked next experiments, novel synthesis ideas." },
          { n: "8", label: "Caveats", body: "Honest scope statement: what doesn't generalise." },
        ].map((s) => (
          <li key={s.n} className="rounded-md border border-border bg-bg p-5">
            <div className="font-display text-accent text-xl leading-none mb-2">
              {s.n}
            </div>
            <div className="font-medium text-text text-[14.5px] mb-1.5">
              {s.label}
            </div>
            <div className="text-[13px] text-text-dim leading-snug">
              {s.body}
            </div>
          </li>
        ))}
      </ol>

      <p className="mt-10 text-sm text-text-muted">
        Each report also includes a 10-bullet TL;DR section and an artifact-
        pointer appendix linking back to the specific source files, plots,
        and per-round CSVs the report describes. The combined web
        equivalent (the <NavLink to="/rq2-story" className="text-accent">interactive story pages</NavLink>) lifts roughly half the
        content into the academic-shell story format; the markdown reports
        are the canonical source of truth.
      </p>
    </section>
  );
}
