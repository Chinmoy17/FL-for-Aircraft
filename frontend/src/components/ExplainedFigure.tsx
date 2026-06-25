import { useState, type ReactNode } from "react";
import { figureUrl } from "../api";

/**
 * ExplainedFigure — the only figure component the new experiment pages
 * should use. The contract is enforced at the type level: caller MUST
 * supply both `caption` (one-line technical label) AND `explanation`
 * (2+ paragraphs of "what the figure shows + why it matters").
 *
 * This exists because the previous frontend rendered bare images with
 * nothing more than a filename caption, which left the EDA / results
 * pages effectively as image dumps. The new contract says: if you want
 * to show a figure, you commit to explaining it.
 *
 * Layout: image on left (~60% width on lg), explanation block on right
 * (~40%), stacked on smaller viewports. Click image to open full-size
 * in new tab. If the image fails to load (e.g. backend not running),
 * we render a clear inline placeholder instead of the browser's broken-
 * image icon — so the explanation is still readable.
 */
export type ExplainedFigureProps = {
  /** Path relative to the repo's results/ directory, OR a full
   *  results/ path. Identical to the contract of <Figure>. */
  artifactPath: string;
  /** Short technical label rendered as the figure title. */
  caption: string;
  /** Long-form explanation. Use plain text or JSX (one or more <p> elements). */
  explanation: ReactNode;
  /** Optional eyebrow above the figure title (e.g. "Figure 03"). */
  eyebrow?: string;
  /** Optional pull-out takeaways — one or two phrases shown prominently. */
  takeaway?: string;
  /**
   * When true, render image and explanation side-by-side in a 3:2 grid.
   * Default false (image on top at max-w-4xl, text below at 78ch). Only
   * use compact for small chart figures where the heights happen to be
   * close.
   */
  compact?: boolean;
};

export function ExplainedFigure({
  artifactPath,
  caption,
  explanation,
  eyebrow,
  takeaway,
  compact = false,
}: ExplainedFigureProps) {
  const url = figureUrl(artifactPath);
  const [hasError, setHasError] = useState(false);

  const figure = (
    <figure className={compact ? "" : "max-w-4xl"}>
      {hasError ? (
        <FigurePlaceholder artifactPath={artifactPath} />
      ) : (
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="
            block rounded-md overflow-hidden border border-border bg-white
            hover:border-border-strong transition-colors
            focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40
          "
          title="Open full image in new tab"
        >
          <img
            src={url}
            alt={caption}
            loading="lazy"
            className="w-full h-auto"
            onError={() => setHasError(true)}
          />
        </a>
      )}
      <figcaption className="mt-3 text-[13px] text-text-muted font-mono-num">
        {artifactPath}
      </figcaption>
    </figure>
  );

  const explanationBlock = (
    <div>
      {eyebrow && <div className="eyebrow mb-2">{eyebrow}</div>}
      <h3 className="font-display text-2xl leading-snug text-text mb-4">
        {caption}
      </h3>
      {takeaway && (
        <div
          className="
            mb-5 border-l-2 border-accent
            pl-4 py-1
            text-[15.5px] text-text italic leading-relaxed
          "
        >
          {takeaway}
        </div>
      )}
      <div
        className="
          text-[15px] leading-[1.7] text-text-dim
          space-y-3 [&>p]:m-0
        "
      >
        {explanation}
      </div>
    </div>
  );

  // Default: stacked vertical (paper style) — image on top with caption
  // immediately under, then the explanation block below at a comfortable
  // reading width. No height-mismatch dead space.
  if (!compact) {
    return (
      <section className="my-12">
        {figure}
        <div className="mt-8 max-w-[78ch]">{explanationBlock}</div>
      </section>
    );
  }

  // Opt-in compact layout: side-by-side for small marginalia figures.
  return (
    <section className="my-14 grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-8 lg:gap-10 items-start">
      {figure}
      {explanationBlock}
    </section>
  );
}

/**
 * Inline placeholder shown when a figure fails to load — typically because
 * the backend is not running. Keeps the page readable instead of showing
 * a broken-image icon.
 */
function FigurePlaceholder({ artifactPath }: { artifactPath: string }) {
  return (
    <div
      role="img"
      aria-label={`Figure unavailable: ${artifactPath}`}
      className="
        flex flex-col items-center justify-center gap-3
        rounded-md border border-dashed border-border-strong bg-bg-subtle
        px-6 py-12 text-center
      "
    >
      <svg
        width="28"
        height="28"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-text-muted"
        aria-hidden
      >
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <circle cx="9" cy="9" r="2" />
        <path d="M21 15l-5-5L5 21" />
      </svg>
      <div className="text-sm text-text-dim font-medium">
        Figure not available
      </div>
      <div className="text-[13px] text-text-muted max-w-[42ch] leading-snug">
        The backend at <span className="font-mono-num">/api/figures</span> did
        not respond. If you&apos;re running locally, start the API:
      </div>
      <pre className="font-mono-num text-[12.5px] text-text bg-bg border border-border rounded px-3 py-2 max-w-full overflow-x-auto">
        python -m uvicorn backend.main:app --port 8000
      </pre>
    </div>
  );
}
