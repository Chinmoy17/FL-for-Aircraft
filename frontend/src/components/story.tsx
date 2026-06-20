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
// Section — 68ch reading width, top spacing scales with the article's rhythm.
// ---------------------------------------------------------------------------
export function StorySection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="max-w-3xl mx-auto mt-16">
      <h2 className="text-xl font-semibold tracking-tight text-text mb-4">
        {title}
      </h2>
      <div className="space-y-4 text-[16px] leading-[1.7] text-text">
        {children}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// AnchorStat — the three big numbers under each hero. Anchoring Bias says
// the first number a reader sees calibrates their judgement of the rest.
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
    <div className="rounded-md border border-border bg-bg-subtle p-4 text-center">
      <div
        className={`text-3xl font-semibold font-mono-num ${statToneClass[tone]}`}
      >
        {value}
      </div>
      <div className="mt-2 text-xs uppercase tracking-wider text-text-dim font-medium">
        {label}
      </div>
      <div className="mt-2 text-xs text-text-muted leading-snug">{sub}</div>
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
// SmokingGunFigure — one full-width figure (max-w-5xl), serif eyebrow,
// long caption explaining the mechanism. Click → opens in new tab.
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
    <section className="max-w-5xl mx-auto mt-16 px-6">
      <div className="text-center">
        <p className="text-xs uppercase tracking-[0.18em] text-text-dim font-medium">
          {eyebrow}
        </p>
        <h2 className="mt-3 text-xl font-semibold tracking-tight text-text">
          {title}
        </h2>
      </div>
      <figure className="mt-6">
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="block rounded-md overflow-hidden border border-border hover:border-border-strong transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
        >
          <img
            src={url}
            alt={alt}
            className="w-full h-auto bg-white"
          />
        </a>
        <figcaption className="mt-4 max-w-2xl mx-auto text-sm text-text-dim leading-relaxed">
          {caption}
        </figcaption>
      </figure>
    </section>
  );
}

// ---------------------------------------------------------------------------
// StoryHero — eyebrow + Instrument Serif display headline + lead paragraph.
// The accent-colored italic span lets each page have its own micro-flourish.
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
    <header className="max-w-3xl mx-auto text-center">
      <p className="text-xs uppercase tracking-[0.18em] text-text-dim font-medium">
        {eyebrow}
      </p>
      <h1
        style={{ fontFamily: "var(--font-display)" }}
        className="mt-4 text-5xl md:text-6xl leading-[1.05] tracking-tight text-text"
      >
        {children}
      </h1>
      <p className="mt-6 text-lg text-text-dim leading-relaxed">{lead}</p>
    </header>
  );
}
