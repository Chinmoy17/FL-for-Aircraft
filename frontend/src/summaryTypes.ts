/**
 * Type extension for /api/summary — `summary` field is the contents of
 * results/summary.json. Mirrors src/fl_aircraft/utils/results.py.
 */
import type { Explanation } from "./types";

export interface SummaryEnvelope {
  summary: ProjectSummary;
}

export interface ProjectSummary {
  project: string;
  generated_at: string;
  git_commit: string;
  phases: Record<string, PhaseMetrics>;
}

export type PhaseId =
  | "00_eda"
  | "01_data"
  | "02_smoke"
  | "03_centralized"
  | "04_local_only"
  | "05_fedavg"
  | "06_non_iid"
  | "rq2_imbalance_aware"
  | "rq3_explanations";

export interface PhaseMetrics {
  phase_id: string;
  phase_name: string;
  generated_at?: string;
  subset?: string;
  interpretation?: string;
  config?: Record<string, unknown>;
  timing?: Record<string, number>;
  summary?: Record<string, unknown>;
  train?: TrainBlock | null;
  test?: TestBlock | null;
  per_subset?: Record<string, TestBlock>;
  per_client?: Record<string, unknown>;
  artifacts?: Record<string, string>;
}

export interface TrainBlock {
  epochs?: TrainEpoch[];
  per_round?: unknown[];
  [k: string]: unknown;
}

export interface TrainEpoch {
  epoch?: number;
  loss_total?: number;
  loss_rul?: number;
  loss_fault?: number;
  rmse?: number;
  [k: string]: unknown;
}

export interface TestBlock {
  rul?: Record<string, number>;
  fault?: Record<string, number>;
  [k: string]: unknown;
}

export type { Explanation };
