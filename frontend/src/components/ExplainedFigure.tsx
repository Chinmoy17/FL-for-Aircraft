import { type ReactNode } from "react";
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
 * in new tab.
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
  /** When true, the image takes full width; explanation appears underneath. */
  fullWidth?: boolean;
};

export function ExplainedFigure({
  artifactPath,
  caption,
  explanation,
  eyebrow,
  takeaway,
  fullWidth = false,
}: ExplainedFigureProps) {
  const url = figureUrl(artifactPath);

  const figure = (
    <figure>
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
        />
      </a>
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

  if (fullWidth) {
    return (
      <section className="my-12">
        {figure}
        <div className="mt-6 max-w-[78ch]">{explanationBlock}</div>
      </section>
    );
  }

  return (
    <section className="my-14 grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-8 lg:gap-10 items-start">
      {figure}
      {explanationBlock}
    </section>
  );
}
