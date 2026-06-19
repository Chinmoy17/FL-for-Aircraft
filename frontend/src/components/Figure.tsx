import { figureUrl } from "../api";

/**
 * Renders a figure pulled from /api/figures/{path}.
 *
 * Per the user's instruction the click opens the full image in a NEW TAB
 * (no lightbox modal). Each figure is wrapped in an <a target="_blank"> so
 * keyboard users get a proper focusable affordance.
 */
export function Figure({
  artifactPath,
  caption,
  className = "",
}: {
  artifactPath: string;
  caption?: string;
  className?: string;
}) {
  const url = figureUrl(artifactPath);
  return (
    <figure className={`group ${className}`}>
      <a
        href={url}
        target="_blank"
        rel="noreferrer"
        className="block rounded-md overflow-hidden border border-border bg-bg hover:border-border-strong transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
      >
        <img
          src={url}
          alt={caption ?? artifactPath}
          loading="lazy"
          className="w-full h-auto bg-white"
        />
      </a>
      {caption && (
        <figcaption className="mt-2 text-xs text-text-dim leading-relaxed">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

/**
 * Grid of figures. Used by the generic per-phase page renderer to show
 * everything in the metrics.json `artifacts` map.
 */
export function FigureGrid({
  artifacts,
}: {
  artifacts: Record<string, string>;
}) {
  const entries = Object.entries(artifacts).filter(([, p]) =>
    /\.(png|jpe?g|svg)$/i.test(p),
  );
  if (entries.length === 0) {
    return <p className="text-sm text-text-dim">No figures recorded.</p>;
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
      {entries.map(([key, path]) => (
        <Figure
          key={key}
          artifactPath={path}
          caption={artifactLabel(key)}
        />
      ))}
    </div>
  );
}

function artifactLabel(key: string): string {
  // Turn `loss_curve_png` → `Loss curve`
  return key
    .replace(/_png$/i, "")
    .replace(/_/g, " ")
    .replace(/^./, (c) => c.toUpperCase());
}
