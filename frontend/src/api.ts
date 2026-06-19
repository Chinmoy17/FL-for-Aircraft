/**
 * Thin typed wrapper around `fetch()` for the FastAPI backend.
 *
 * In dev, Vite proxies `/api/*` to `http://localhost:8000` (see vite.config.ts),
 * so the same code works both locally and when both halves are served from the
 * same Docker container in production.
 */
import type {
  CheckpointsResponse,
  EnginesResponse,
  PredictRequest,
  PredictResponse,
} from "./types";
import type { SummaryEnvelope } from "./summaryTypes";

async function jsonGet<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`GET ${path} failed (${res.status}): ${body}`);
  }
  return (await res.json()) as T;
}

async function jsonPost<TReq, TRes>(path: string, body: TReq): Promise<TRes> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errBody = await res.text().catch(() => "");
    throw new Error(`POST ${path} failed (${res.status}): ${errBody}`);
  }
  return (await res.json()) as TRes;
}

export const api = {
  listCheckpoints: () => jsonGet<CheckpointsResponse>("/api/checkpoints"),
  listEngines: (key: string) =>
    jsonGet<EnginesResponse>(
      `/api/checkpoints/${encodeURIComponent(key)}/engines`,
    ),
  predict: (req: PredictRequest) =>
    jsonPost<PredictRequest, PredictResponse>("/api/predict", req),
  getSummary: () => jsonGet<SummaryEnvelope>("/api/summary"),
};

/**
 * Build a URL the browser can fetch directly (img src, anchor href, etc.)
 * for a figure path stored in metrics.json under the `artifacts` map.
 *
 * Backend serves `/api/figures/{rel_path}` where rel_path is relative
 * to `results/`. metrics.json stores paths as `results/<phase>/<file>.png`,
 * so we strip the leading "results/" prefix.
 */
export function figureUrl(artifactPath: string): string {
  const rel = artifactPath.replace(/^results\//, "");
  return `/api/figures/${rel}`;
}
