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

export type ReturnDefaults = { return_samples: number; policy_ids: string[] };

export type ReturnRequest = {
  task_id: string;
  policy_id: string;
  max_steps: number;
  open_loop_steps: number;
  seed: number;
  init_state_index: number;
  return_samples: number;
};

export type ReturnDistribution = OfflineJob & {
  return_summary?: Record<string, unknown>;
  return_progress?: { phase?: string; rollout_step?: number; rollout_steps?: number; success?: boolean };
  artifacts?: Record<string, string>;
};

/** One policy query: every candidate's critic score and return distribution (per-candidate arrays). */
export type ReturnQuery = {
  step: number;
  chosen: number;
  scores: number[];
  mean: number[];
  q10: number[];
  median: number[];
  q90: number[];
  /** Sample counts per candidate over the shared ``edges``. */
  histograms: number[][];
};

export type ReturnData = {
  schema_version: 1;
  fps: number;
  action_count: number;
  edges: number[];
  samples: number;
  reference: { available: boolean; reason: string | null; values: number[] | null };
  queries: ReturnQuery[];
};

/** Queue jobs of one analysis tool, all under ``root``. */
export type ToolEndpoints<Job extends OfflineJob, Request> = {
  start: (body: Request) => Promise<Job>;
  list: () => Promise<{ jobs: Job[] }>;
  get: (id: string) => Promise<Job>;
  stop: (id: string) => Promise<Job>;
  remove: (id: string) => Promise<{ deleted: string }>;
};

const toolEndpoints = <Job extends OfflineJob, Request>(root: string): ToolEndpoints<Job, Request> => ({
  start: (body) => api<Job>(root, { method: "POST", body: JSON.stringify(body) }),
  list: () => api<{ jobs: Job[] }>(root),
  get: (id) => api<Job>(`${root}/${encodeURIComponent(id)}`),
  stop: (id) => api<Job>(`${root}/${encodeURIComponent(id)}/stop`, { method: "POST" }),
  remove: (id) => api<{ deleted: string }>(`${root}/${encodeURIComponent(id)}`, {
    method: "DELETE", body: JSON.stringify({ confirm_job_id: id }),
  }),
});

const critic = toolEndpoints<CriticDiagnostic, CriticRequest>("/api/tools/critic");
export const getCriticDefaults = (datasetId: string) => api<CriticDefaults>(
  `/api/tools/critic/defaults?${new URLSearchParams({ dataset_id: datasetId })}`,
);
export const startCriticDiagnostic = (body: CriticRequest) => critic.start(body);
export const listCriticDiagnostics = () => critic.list();
export const getCriticDiagnostic = (id: string) => critic.get(id);
export const stopCriticDiagnostic = (id: string) => critic.stop(id);
export const deleteCriticDiagnostic = (id: string) => critic.remove(id);

const returns = toolEndpoints<ReturnDistribution, ReturnRequest>("/api/tools/returns");
export const getReturnDefaults = () => api<ReturnDefaults>("/api/tools/returns/defaults");
export const startReturnDistribution = (body: ReturnRequest) => returns.start(body);
export const listReturnDistributions = () => returns.list();
export const getReturnDistribution = (id: string) => returns.get(id);
export const stopReturnDistribution = (id: string) => returns.stop(id);
export const deleteReturnDistribution = (id: string) => returns.remove(id);
/** The replay data written by the analysis phase (``returns.json`` artifact URL). */
export const getReturnData = (url: string) => api<ReturnData>(url);
