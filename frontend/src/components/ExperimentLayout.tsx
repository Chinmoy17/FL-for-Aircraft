import { type ReactNode } from "react";
import { NavLink } from "react-router-dom";

/**
 * Shared layout for each experiment-phase page (00–06).
 *
 * Sets up consistent chrome: header strip (phase id + title + lede),
 * generous full-width sections with horizontal padding matching the
 * rest of the academic shell, and a footer "previous / next" nav.
 *
 * Per-phase pages render their content inside <ExperimentSection>s
 * and embed figures via <ExplainedFigure>.
 */

export type PhaseNav = { id: string; title: string; to: string };

export function ExperimentLayout({
  phaseId,
  title,
  lede,
  metaRow,
  children,
  prev,
  next,
}: {
  phaseId: string;
  title: string;
  lede: ReactNode;
  metaRow?: ReactNode;
  children: ReactNode;
  prev?: PhaseNav;
  next?: PhaseNav;
}) {
  return (
    <article className="w-full">
      <header className="px-10 md:px-16 lg:px-24 pt-16 pb-10 border-b border-border">
        <div className="eyebrow font-mono-num">{phaseId}</div>
        <h1 className="font-display text-[44px] sm:text-[52px] leading-[1.05] tracking-tight text-text mt-4 max-w-[26ch]">
          {title}
        </h1>
        <p className="mt-6 text-lg text-text-dim max-w-[68ch]">{lede}</p>
        {metaRow && (
          <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-text-muted">
            {metaRow}
          </div>
        )}
      </header>

      <div className="px-10 md:px-16 lg:px-24 py-14">{children}</div>

      <ExperimentFooterNav prev={prev} next={next} />
    </article>
  );
}

/**
 * One labelled section inside an experiment page. Matches the
 * eyebrow + Instrument Serif heading rhythm used everywhere.
 */
export function ExperimentSection({
  eyebrow,
  title,
  intro,
  children,
}: {
  eyebrow?: string;
  title?: string;
  intro?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="mb-16 first:mt-0">
      {eyebrow && <div className="eyebrow">{eyebrow}</div>}
      {title && (
        <h2 className="font-display text-3xl text-text mt-3 mb-4 max-w-[36ch]">
          {title}
        </h2>
      )}
      {intro && (
        <div className="text-text-dim text-[16px] leading-[1.7] max-w-[78ch] mb-6 space-y-3 [&>p]:m-0">
          {intro}
        </div>
      )}
      {children}
    </section>
  );
}

/**
 * Headline-numbers strip — repeating pattern across phases. Each entry
 * is a short value + label pair. Renders as a divided horizontal grid.
 */
export function HeadlineNumbers({
  items,
}: {
  items: { value: string; label: string; tone?: "good" | "bad" }[];
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-px bg-border rounded-lg overflow-hidden my-8">
      {items.map((it) => {
        const tone =
          it.tone === "bad"
            ? "text-bad"
            : it.tone === "good"
            ? "text-good"
            : "text-text";
        return (
          <div key={it.label} className="bg-bg p-5">
            <div
              className={`font-display ${tone} text-[28px] leading-none tracking-tight font-mono-num`}
            >
              {it.value}
            </div>
            <div className="text-[12.5px] text-text-dim mt-2 leading-snug">
              {it.label}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ===========================================================================
// Footer prev/next
// ===========================================================================
function ExperimentFooterNav({
  prev,
  next,
}: {
  prev?: PhaseNav;
  next?: PhaseNav;
}) {
  return (
    <nav
      aria-label="Page navigation"
      className="px-10 md:px-16 lg:px-24 py-10 border-t border-border flex flex-wrap items-center justify-between gap-6"
    >
      {prev ? (
        <NavLink
          to={prev.to}
          className="block group max-w-[40%] no-underline"
        >
          <div className="eyebrow">← Previous</div>
          <div className="mt-1 text-text group-hover:text-accent transition-colors">
            <span className="font-mono-num text-text-muted text-sm">
              {prev.id}
            </span>{" "}
            · {prev.title}
          </div>
        </NavLink>
      ) : (
        <span />
      )}
      {next ? (
        <NavLink
          to={next.to}
          className="block group text-right max-w-[40%] no-underline"
        >
          <div className="eyebrow">Next →</div>
          <div className="mt-1 text-text group-hover:text-accent transition-colors">
            <span className="font-mono-num text-text-muted text-sm">
              {next.id}
            </span>{" "}
            · {next.title}
          </div>
        </NavLink>
      ) : (
        <span />
      )}
    </nav>
  );
}
