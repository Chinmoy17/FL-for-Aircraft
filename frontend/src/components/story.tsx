/**
 * Shared atoms for the long-form RQ story pages.
 *
 * Both /rq2-story and /rq3-story follow the same Distill-style template:
 *
 *   - editorial hero with eyebrow + serif display headline
 *   - 3 anchor statistics (Anchoring Bias)
 *   - Section blocks with 68ch reading width
 *   - hypothesis / methodology cards
 *   - bulleted lists with accent dots
 *   - inline formula blocks
 *   - one full-width smoking-gun figure
 *
 * Keeping these atoms in one place guarantees both pages share spacing,
 * type ramp, and color usage. Per UI-craft trust-emphasis guidance:
 * 8px grid, single accent color, AA contrast throughout.
 */
import { figureUrl } from "../api";

// ---------------------------------------------------------------------------
// Section — academic-shell-aligned section block. The content fills the
// section's available width (no inner max-w cap on body text) so the
// right edge of paragraphs lines up with the right edge of figures /
// anchor-stat rows above and below. Readability stays acceptable because
// the parent article caps at max-w-5xl (1024 px) and the section's
// horizontal padding (96 px each side) trims the inner content to
// roughly 832 px on a wide display — about 100 chars at the body type
// size, which is the wide end of typeset academic columns.
// ---------------------------------------------------------------------------
export function StorySection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="px-10 md:px-16 lg:px-24 mt-16">
      <h2 className="font-display text-2xl md:text-3xl tracking-tight text-text mb-4">
        {title}
      </h2>
      <div className="space-y-4 text-[16px] leading-[1.7] text-text">
        {children}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// StoryFollowupHeader — the in-page transition used to open a coda
// (e.g. the FedProx and FedRep + FedCCFA follow-ups inside /rq2-story).
// Replaces the previous max-w-3xl mx-auto + text-center pattern with
// the same shell-aligned, left-justified rhythm the rest of the page
// uses — no dead space on the right, no centered narrow column.
// ---------------------------------------------------------------------------
export function StoryFollowupHeader({
  eyebrow,
  children,
  lead,
}: {
  eyebrow: string;
  /** Headline content — fragment so callers control the accent placement. */
  children: React.ReactNode;
  lead: React.ReactNode;
}) {
  return (
    <header className="px-10 md:px-16 lg:px-24 mt-24 pt-12 border-t border-border">
      <div className="eyebrow">{eyebrow}</div>
      <h2 className="mt-3 font-display text-[34px] sm:text-[42px] lg:text-[48px] leading-[1.08] tracking-tight text-text max-w-[26ch]">
        {children}
      </h2>
      <p className="mt-5 text-lg text-text-dim leading-relaxed">
        {lead}
      </p>
    </header>
  );
}

// ---------------------------------------------------------------------------
// AnchorStat — one big number used in the 3-up anchor row that follows the
// hero / follow-up headers. Left-aligned text (not centered) so it sits
// in the same visual rhythm as the rest of the page — no dead space on
// the right of wide displays, consistent edge with section headings.
// ---------------------------------------------------------------------------
type StatTone = "neutral" | "good" | "bad" | "accent";

const statToneClass: Record<StatTone, string> = {
  neutral: "text-text",
  good: "text-good",
  bad: "text-bad",
  accent: "text-accent",
};

export function AnchorStat({
  value,
  label,
  sub,
  tone = "neutral",
}: {
  value: string;
  label: string;
  sub: string;
  tone?: StatTone;
}) {
  return (
    <div className="rounded-md border border-border bg-bg-subtle p-5">
      <div
        className={`font-display text-[32px] leading-none font-mono-num tracking-tight ${statToneClass[tone]}`}
      >
        {value}
      </div>
      <div className="mt-3 text-[11.5px] uppercase tracking-[0.12em] text-text-dim font-semibold">
        {label}
      </div>
      <div className="mt-2 text-[13px] text-text-muted leading-snug">{sub}</div>
    </div>
  );
}

/**
 * Standard wrapper for a 3-up anchor-stat row. Sits at the section's
 * left padding edge (same px as StorySection / StoryHero / SmokingGunFigure)
 * so the three cards line up with the page's content edge instead of
 * floating in a narrow centered column. No internal max-w cap — the
 * row fills whatever its parent container allows, so the right edge of
 * the anchor row matches the right edge of headings / figures / captions
 * on the same article (the article-level cap is the one place that
 * decides where the article visually ends).
 */
export function AnchorStatRow({
  children,
  className = "mt-10",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`px-10 md:px-16 lg:px-24 ${className} grid grid-cols-1 sm:grid-cols-3 gap-4`}
    >
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// HypothesisCard — short card used to describe one experimental scheme,
// model, or method. Title, formula/code line, one-sentence explanation.
// ---------------------------------------------------------------------------
export function HypothesisCard({
  label,
  formula,
  text,
}: {
  label: string;
  formula?: string;
  text: string;
}) {
  return (
    <div className="rounded-md border border-border bg-bg-subtle p-4">
      <div className="text-xs uppercase tracking-wider text-text-dim font-medium">
        {label}
      </div>
      {formula && (
        <div className="mt-2 font-mono-num text-sm text-accent">{formula}</div>
      )}
      <p className="mt-2 text-sm text-text-dim leading-snug">{text}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Bullet — accent-dot list item, more editorial than default <li>.
// ---------------------------------------------------------------------------
export function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="pl-5 relative">
      <span
        aria-hidden
        className="absolute left-0 top-2 w-1.5 h-1.5 rounded-full bg-accent"
      />
      <span className="text-text">{children}</span>
    </li>
  );
}

// ---------------------------------------------------------------------------
// FormulaBlock — inline math / pseudo-code, accent-colored, centered.
// ---------------------------------------------------------------------------
export function FormulaBlock({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-4 rounded-md border border-border bg-bg px-4 py-3 font-mono-num text-sm text-accent text-center">
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SmokingGunFigure — left-aligned eyebrow + title (matches academic-shell
// rhythm) followed by the figure. Caption sits below the image, constrained
// to a comfortable reading width. Click → opens in new tab.
// ---------------------------------------------------------------------------
export function SmokingGunFigure({
  eyebrow,
  title,
  artifactPath,
  caption,
  alt,
}: {
  eyebrow: string;
  title: string;
  artifactPath: string;
  caption: React.ReactNode;
  alt: string;
}) {
  const url = figureUrl(artifactPath);
  return (
    <section className="px-10 md:px-16 lg:px-24 mt-16">
      <p className="eyebrow">{eyebrow}</p>
      <h2 className="mt-3 font-display text-2xl md:text-3xl tracking-tight text-text mb-6">
        {title}
      </h2>
      <figure>
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="block rounded-md overflow-hidden border border-border bg-white hover:border-border-strong transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
        >
          <img src={url} alt={alt} className="w-full h-auto" />
        </a>
        <figcaption className="mt-4 text-sm text-text-dim leading-relaxed">
          {caption}
        </figcaption>
      </figure>
    </section>
  );
}

// ---------------------------------------------------------------------------
// StoryHero — academic-shell-aligned hero. Left-aligned (not centered)
// to match the experiment-page rhythm. Eyebrow + Instrument Serif display
// headline + lead paragraph.
// ---------------------------------------------------------------------------
export function StoryHero({
  eyebrow,
  children,
  lead,
}: {
  eyebrow: string;
  /** Headline content; pass a fragment so callers control where the accent goes. */
  children: React.ReactNode;
  lead: React.ReactNode;
}) {
  return (
    <header className="px-10 md:px-16 lg:px-24 pt-16 pb-10 border-b border-border">
      <div className="eyebrow">{eyebrow}</div>
      <h1 className="font-display text-[44px] sm:text-[52px] lg:text-[60px] leading-[1.05] tracking-tight text-text mt-4 max-w-[22ch]">
        {children}
      </h1>
      <p className="mt-6 text-lg text-text-dim leading-relaxed">
        {lead}
      </p>
    </header>
  );
}
