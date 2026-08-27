import { api } from "../../api/client";
import type {
  Bootstrap, DatasetPreview, DatasetSelection, DatasetSummary, EvaluationConfig,
  EvaluationFilters, EvaluationPreview, EvaluationRecord, OfflineJob, Session,
  PaginatedRuns, PolicyDetail, PolicyInfo, TensorBoardStatus, TrajectoryDetail,
  TrainingDataset, TrainingDefaults,
} from "./types";

export const getBootstrap = () => api<Bootstrap>("/api/bootstrap");
export const listRuns = () => api<Session[]>("/api/runs");
export const getDatasetSummary = () => api<DatasetSummary>("/api/datasets/summary");
export const listDatasetRuns = (taskId: string, page = 1, pageSize = 5) => api<PaginatedRuns>(
  "/api/datasets/runs?task_id=" + encodeURIComponent(taskId)
  + `&page=${page}&page_size=${pageSize}`,
);
export const getTrajectoryDetail = (runId: string) => api<TrajectoryDetail>(
  `/api/datasets/runs/${encodeURIComponent(runId)}`,
);
export const evaluateTrajectories = (body: {
  task_id: string; run_ids: string[] | null; overwrite: boolean;
}) => api<{
  kind: "trajectory_evaluation"; status: string; job: OfflineJob | null;
  selected_count: number; skipped_count: number; skipped_run_ids: string[]; message?: string;
}>("/api/datasets/evaluations", { method: "POST", body: JSON.stringify(body) });
export const datasetExportUrl = (taskId: string) => (
  "/api/datasets/export?task_id=" + encodeURIComponent(taskId)
);

export const previewTrainingDataset = (taskId: string, selection: DatasetSelection) =>
  api<DatasetPreview>("/api/training-datasets/preview", {
    method: "POST", body: JSON.stringify({ task_id: taskId, selection }),
  });

export const createTrainingDataset = (body: {
  name: string; task_id: string; selection: DatasetSelection;
  validation_fraction: number; split_seed: number; success_consecutive_steps: number;
}) => api<TrainingDataset>("/api/training-datasets", {
  method: "POST", body: JSON.stringify(body),
});

export const deriveTrainingDataset = (parentId: string, body: {
  name: string; selection: DatasetSelection; validation_fraction: number;
  split_seed: number; success_consecutive_steps: number;
}) => api<TrainingDataset>(`/api/training-datasets/${encodeURIComponent(parentId)}/derive`, {
  method: "POST", body: JSON.stringify(body),
});

export const deleteTrainingDataset = (id: string, force = false) => api<{
  deleted: string; annotation_status: string;
  source_runs_deleted: boolean; shared_cache_deleted: boolean;
  retained_training_jobs: string[];
}>(`/api/training-datasets/${encodeURIComponent(id)}`, {
  method: "DELETE", body: JSON.stringify({ confirm_dataset_id: id, force }),
});

export const listTrainingDatasets = (taskId?: string) => api<TrainingDataset[]>(
  "/api/training-datasets" + (taskId ? `?task_id=${encodeURIComponent(taskId)}` : ""),
);
export const verifyTrainingDataset = (id: string) => api<TrainingDataset>(
  `/api/training-datasets/${encodeURIComponent(id)}/verify`, { method: "POST" },
);
export const annotateTrainingDataset = (id: string) => api<OfflineJob>(
  `/api/training-datasets/${encodeURIComponent(id)}/annotations`, { method: "POST" },
);
export const listOfflineJobs = () => api<OfflineJob[]>("/api/jobs");
export const getOfflineJob = (id: string) => api<OfflineJob>(`/api/jobs/${encodeURIComponent(id)}`);
export const stopOfflineJob = (id: string) => api<OfflineJob>(
  `/api/jobs/${encodeURIComponent(id)}/stop`, { method: "POST" },
);
export const getJobLogs = (id: string, offset = 0) => api<{
  offset: number; next_offset: number; text: string;
}>(`/api/jobs/${encodeURIComponent(id)}/logs?offset=${offset}`);
export const getTrainingDefaults = (datasetId?: string) => api<TrainingDefaults>(
  "/api/training/defaults" + (datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : ""),
);
export const startTraining = (datasetId: string, parameters: Record<string, unknown>) =>
  api<OfflineJob>("/api/training-runs", {
    method: "POST", body: JSON.stringify({ dataset_id: datasetId, parameters }),
  });
export const getTensorBoard = () => api<TensorBoardStatus>("/api/tensorboard");
export const startTensorBoard = () => api<TensorBoardStatus>(
  "/api/tensorboard/start", { method: "POST" },
);

export const listModels = () => api<PolicyInfo[]>("/api/models");
export const getModel = (id: string) => api<PolicyDetail>(
  `/api/models/${encodeURIComponent(id)}`,
);
export const renameModel = (id: string, label: string) => api<PolicyDetail>(
  `/api/models/${encodeURIComponent(id)}`,
  { method: "PATCH", body: JSON.stringify({ label }) },
);
export const copyModel = (id: string, label: string) => api<PolicyDetail>(
  `/api/models/${encodeURIComponent(id)}/copy`,
  { method: "POST", body: JSON.stringify({ label }) },
);
export const deleteModel = (id: string) => api<{ deleted: string }>(
  `/api/models/${encodeURIComponent(id)}`,
  { method: "DELETE", body: JSON.stringify({ confirm_policy_id: id }) },
);

export const previewEvaluation = (config: EvaluationConfig) => api<EvaluationPreview>(
  "/api/evaluations/preview", { method: "POST", body: JSON.stringify(config) },
);
type EvaluationJobResponse = OfflineJob | { job: OfflineJob };
const unwrapEvaluationJob = (response: EvaluationJobResponse) => (
  "job" in response ? response.job : response
);
export const startEvaluation = (config: EvaluationConfig) => api<EvaluationJobResponse>(
  "/api/evaluations", { method: "POST", body: JSON.stringify(config) },
).then(unwrapEvaluationJob);
export const listEvaluations = (filters: EvaluationFilters = {}) => {
  const query = new URLSearchParams();
  Object.entries(filters).forEach(([name, value]) => {
    if (value) query.set(name, value);
  });
  const suffix = query.size ? `?${query.toString()}` : "";
  return api<EvaluationRecord[]>(`/api/evaluations${suffix}`);
};
export const getEvaluation = (id: string) => api<EvaluationRecord>(
  `/api/evaluations/${encodeURIComponent(id)}`,
);
export const stopEvaluation = (id: string) => api<EvaluationJobResponse>(
  `/api/evaluations/${encodeURIComponent(id)}/stop`, { method: "POST" },
).then(unwrapEvaluationJob);
export const deleteEvaluation = (id: string, confirmation: string) => api<{ deleted: string }>(
  `/api/evaluations/${encodeURIComponent(id)}`, {
    method: "DELETE", body: JSON.stringify({ confirm_evaluation_id: confirmation }),
  },
);
