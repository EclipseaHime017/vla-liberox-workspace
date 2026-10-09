import { api } from "../../api/client";

export type LabMode = "plan_only" | "localize";
export type LabConfig = {
  model_id: string; revision: string; environment: string; cameras: string[];
  mode?: LabMode;
  coarse_fps?: number; window_seconds?: number;
};
export type LabDefaults = { config: LabConfig; model_available: boolean; environment_available: boolean; message: string | null };
export type Experiment = {
  id: string; status: string; error: string | null; created_at: string; run_ids: string[]; config: LabConfig;
  progress: null | { stage: string; completed_runs: number; failed_runs: number; total_runs: number;
    current_run: string | null; calls: number; window_index: number; window_total: number;
    elapsed_seconds: number; estimated_remaining_seconds: number | null; error: string | null;
    runs?: Array<{ run_id: string; status: string; error: string | null }> };
};
export type Milestone = { id: string; label: string; achieved_when: string; lost_when: string;
  initial_state?: string; initial_reason?: string; depends_on?: string[]; requirement_ids?: string[] };
export type TaskRequirement = { id: string; instruction_span: string; condition: string; depends_on: string[] };
type Interval = { start_step: number; end_step: number };
export type RegionStatus = "confirmed" | "uncertain" | "outside" | "pending";
export type GoalRange = {
  stage_id: string;
  regions: Array<Interval & { status: RegionStatus; window_indices: number[] }>;
  passes: Array<{ steps: number[]; labels: RegionStatus[]; reason?: string }>;
};
export type LabResult = {
  schema_version: number; goal_ranges: GoalRange[];
  mode?: LabMode; task_contract?: { requirements: TaskRequirement[] } | null;
  localization?: { assigned_goals: number; total_goals: number; windows_completed?: number; windows_total?: number; mode: string };
  sampling?: { fps: number; steps: number[]; window_seconds?: number; stride_seconds?: number };
  grounding?: { description: string; uncertain: boolean } | null;
  status: string; error: string | null; action_count: number; control_hz: number; success_step: number | null;
  source: { prompt: string }; plan: null | { stages: Milestone[]; notes: string; initial_scene?: string };
  calls: Array<{ name: string; attempt: number; steps: number[]; prompt: string;
    raw_response: string; seconds: number; valid: boolean; error?: string }>; cuda_peak_memory_gib?: number;
};
const root = "/api/annotation-lab";
const path = (id: string) => `${root}/experiments/${encodeURIComponent(id)}`;
export const labDefaults = () => api<LabDefaults>(`${root}/config`);
export const listExperiments = () => api<Experiment[]>(`${root}/experiments`);
export const startExperiment = (run_ids: string[], options: Partial<LabConfig>) => api<Experiment>(`${root}/experiments`, {
  method: "POST", body: JSON.stringify({ run_ids, options }),
});
export const getExperiment = (id: string) => api<Experiment>(path(id));
export const stopExperiment = (id: string) => api<Experiment>(`${path(id)}/stop`, { method: "POST" });
export const getLabResult = (id: string, runId: string) => api<LabResult | null>(`${path(id)}/runs/${encodeURIComponent(runId)}`);
export const evidenceUrl = (id: string, runId: string, camera: string, step: number) =>
  `${path(id)}/runs/${encodeURIComponent(runId)}/evidence/${encodeURIComponent(camera)}/${step}`;
export const isActive = (status: string) => ["QUEUED", "STARTING", "RUNNING", "STOPPING"].includes(status);
export const memberStatus = (experiment: Experiment, runId: string) => {
  const member = experiment.progress?.runs?.find((item) => item.run_id === runId);
  if (member) return member.status;
  if (experiment.progress?.current_run === runId) return isActive(experiment.status) ? "RUNNING" : "执行中断";
  if (!experiment.progress?.runs) return `批次 ${experiment.status} · 单条状态未知`;
  return isActive(experiment.status) ? "等待处理" : "未处理";
};
