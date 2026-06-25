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
 * Layout: figure renders FULL-WIDTH on top (big, sharp — fills the
 * section's available width so multi-panel matplotlib exports stay
 * legible on extended displays). The findings block sits BELOW the
 * image, left-aligned, capped at a generous reading width (max-w-4xl
 * ≈ 896 px) so prose lines stay readable without floating in a
 * narrow centered column. The structure inside a page section is:
 *
 *     Section heading  →  WHAT WE NEED TO KNOW (question + why)
 *     Image (full-width, sharp, click for full-size)
 *     Findings block   →  WHAT WE FOUND (eyebrow + title + takeaway + body)
 *
 * Click image to open full-size in new tab. If the image fails to
 * load (e.g. backend not running), we render a clear inline
 * placeholder instead of the browser's broken-image icon — so the
 * explanation is still readable.
 */
export type ExplainedFigureProps = {
  /** Path relative to the repo's results/ directory, OR a full
   *  results/ path. Identical to the contract of <Figure>. */
  artifactPath: string;
  /** Short technical label rendered as the findings title. */
  caption: string;
  /** Long-form findings. Use plain text or JSX (one or more <p> elements). */
  explanation: ReactNode;
  /** Optional eyebrow above the findings title (e.g. "Figure 03"). */
  eyebrow?: string;
  /** Optional pull-out takeaway — one sentence shown prominently. */
  takeaway?: string;
};

export function ExplainedFigure({
  artifactPath,
  caption,
  explanation,
  eyebrow,
  takeaway,
}: ExplainedFigureProps) {
  const url = figureUrl(artifactPath);
  const [hasError, setHasError] = useState(false);

  const figure = (
    <figure className="w-full">
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
      <figcaption className="mt-2.5 text-[12px] text-text-muted font-mono-num">
        {artifactPath}
      </figcaption>
    </figure>
  );

  // Findings block — sits BELOW the image. Eyebrow + findings title
  // + accent pull-quote + prose. Left-aligned, capped at max-w-4xl so
  // line lengths stay readable on wide displays. Never centered with
  // mx-auto — the block hugs the section's left edge so the eye
  // continues naturally from the image above to the findings below.
  //
  // TYPOGRAPHY CONTRAST WITH THE QUESTION HEADING:
  // The page's question heading uses Instrument Serif (font-display)
  // — formal, paper-like. The findings title uses Inter (font-ui)
  // semibold — a precise sans-serif "empirical statement" voice. The
  // serif/sans split makes the question-vs-answer roles visually
  // unambiguous without relying on extra borders or pills.
  const findingsBlock = (
    <div className="max-w-4xl">
      {eyebrow && (
        <div className="text-[10.5px] font-semibold tracking-[0.16em] uppercase text-accent mb-2.5">
          {eyebrow}
        </div>
      )}
      <h3
        className="
          font-ui font-semibold text-text
          text-[22px] sm:text-[24px] lg:text-[26px]
          leading-[1.25] tracking-[-0.01em]
          mb-5
        "
      >
        {caption}
      </h3>
      {takeaway && (
        <div
          className="
            my-5 border-l-[3px] border-accent
            pl-5 py-1
            text-[18px] text-text italic leading-[1.55] font-display
          "
        >
          {takeaway}
        </div>
      )}
      <div
        className="
          text-[17px] leading-[1.7] text-text-dim
          space-y-4 [&>p]:m-0
        "
      >
        {explanation}
      </div>
    </div>
  );

  return (
    <section className="my-12">
      {figure}
      <div className="mt-8">{findingsBlock}</div>
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
