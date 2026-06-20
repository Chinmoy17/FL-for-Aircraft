/**
 * TypeScript mirrors of backend/schemas.py.
 *
 * Keep these in sync by hand for now. If the API grows, we can switch to
 * generating them with `openapi-typescript` against /openapi.json.
 */

export interface CheckpointSummary {
  key: string;
  display_name: string;
  training_subsets: string[];
  test_engine_count: number;
  checkpoint_file: string;
}

export interface CheckpointsResponse {
  checkpoints: CheckpointSummary[];
}

export interface EngineSummary {
  engine_id: number;
  subset: string;
  true_rul: number;
  true_fault: number;
}

export interface EnginesResponse {
  checkpoint_key: string;
  engines: EngineSummary[];
}

export interface TopSensor {
  column: string;
  name: string;
  description: string;
  subsystem: string;
  contribution: number;
}

export interface FaultModeInfo {
  fault_mode: string;
  affected_components: string[];
  recommended_action: string;
  confidence_label: string;
}

export interface Explanation {
  predicted_rul: number;
  fault_probability: number;
  target_head: string;
  top_sensors: TopSensor[];
  inferred_fault_mode: FaultModeInfo | null;
  narrative: string;
  narrative_llm: string | null;
  convergence_delta: number;
  notes: string[];
}

export interface PredictResponse {
  checkpoint_key: string;
  checkpoint_display_name: string;
  engine_id: number;
  subset: string;
  true_rul: number;
  true_fault: number;
  explanation: Explanation;
}

export interface PredictRequest {
  checkpoint_key: string;
  engine_id: number;
  top_k?: number;
  use_llm?: boolean;
}
