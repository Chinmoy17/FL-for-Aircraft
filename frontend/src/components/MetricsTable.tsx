/**
 * Generic key/value table for metric blocks (e.g. test.rul, test.fault,
 * timing, config). Caller passes the source object and an optional
 * formatter per key.
 */
export function MetricsTable({
  rows,
  caption,
}: {
  rows: Array<[string, unknown]>;
  caption?: string;
}) {
  if (rows.length === 0) {
    return <p className="text-sm text-text-dim">—</p>;
  }
  return (
    <div className="rounded-md border border-border overflow-hidden bg-bg">
      {caption && (
        <div className="px-4 py-2 text-xs uppercase tracking-wider text-text-dim font-medium border-b border-border bg-bg-subtle/50">
          {caption}
        </div>
      )}
      <table className="w-full text-sm">
        <tbody className="divide-y divide-border">
          {rows.map(([k, v]) => (
            <tr key={k}>
              <th
                scope="row"
                className="px-4 py-1.5 text-left font-normal text-text-dim w-1/2 align-top"
              >
                {k}
              </th>
              <td className="px-4 py-1.5 text-right font-mono-num text-text">
                {formatValue(v)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    // Heuristic: small floats get 4 dp; ints stay ints; large ints get commas.
    if (Number.isInteger(v)) return v.toLocaleString();
    if (Math.abs(v) >= 100) return v.toFixed(1);
    if (Math.abs(v) >= 1) return v.toFixed(3);
    return v.toFixed(4);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return v.join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/**
 * Convenience: build rows from a Record, sorted by insertion order, skipping
 * undefined / null entries.
 */
export function rowsFromRecord(
  obj: Record<string, unknown> | undefined | null,
): Array<[string, unknown]> {
  if (!obj) return [];
  return Object.entries(obj).filter(([, v]) => v !== null && v !== undefined);
}
