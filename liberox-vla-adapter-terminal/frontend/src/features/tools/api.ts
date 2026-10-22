import { api } from "../../api/client";
import type { OfflineJob } from "../run-control/types";

export type CriticDefaults = {
  train_steps: number;
  batch_size: number;
  reward_sources: string[];
  default_reward_source: string | null;
};

export type CriticRequest = {
  dataset_id: string;
  reward_source: string;
  task_id: string;
  policy_id: string;
  max_steps: number;
  open_loop_steps: number;
  seed: number;
  init_state_index: number;
  train_steps: number;
  batch_size: number;
};

export type CriticDiagnostic = OfflineJob & {
  diagnostic_summary?: Record<string, unknown>;
  critic_progress?: {
    phase?: string;
    rollout_step?: number;
    rollout_steps?: number;
    train_step?: number;
    train_steps?: number;
    q_loss?: number | null;
    value_loss?: number | null;
  };
  artifacts?: Record<string, string>;
};

const root = "/api/tools/critic";
export const getCriticDefaults = (datasetId: string) => api<CriticDefaults>(
  `${root}/defaults?${new URLSearchParams({ dataset_id: datasetId })}`,
);
export const startCriticDiagnostic = (body: CriticRequest) => api<CriticDiagnostic>(root, {
  method: "POST", body: JSON.stringify(body),
});
export const listCriticDiagnostics = () => api<{ jobs: CriticDiagnostic[] }>(root);
export const getCriticDiagnostic = (id: string) => api<CriticDiagnostic>(`${root}/${encodeURIComponent(id)}`);
export const stopCriticDiagnostic = (id: string) => api<CriticDiagnostic>(`${root}/${encodeURIComponent(id)}/stop`, {
  method: "POST",
});
export const deleteCriticDiagnostic = (id: string) => api<{ deleted: string }>(`${root}/${encodeURIComponent(id)}`, {
  method: "DELETE", body: JSON.stringify({ confirm_job_id: id }),
});
