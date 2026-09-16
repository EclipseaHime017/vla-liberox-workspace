import { api } from "../../api/client";
import type {
  Bootstrap, DatasetPreview, DatasetSelection, DatasetSummary, EvaluationConfig,
  EvaluationFilters, EvaluationPreview, EvaluationRecord, OfflineJob, Session,
  PaginatedRuns, PolicyDetail, PolicyInfo, TensorBoardStatus, TrajectoryDetail,
  TrainingDataset, TrainingDefaults, StageAnnotation, StageKeyframe, RewardParameters,
  RewardSource, TrainingRewardSource, EvaluationOperation,
} from "./types";

export const getBootstrap = () => api<Bootstrap>("/api/bootstrap");
export const listRuns = () => api<Session[]>("/api/runs");
export const getDatasetSummary = () => api<DatasetSummary>("/api/datasets/summary");
function taskQuery(taskId?: string, taskIds?: string[]) {
  const query = new URLSearchParams();
  if (taskId) query.set("task_id", taskId);
  if (taskIds !== undefined) (taskIds.length ? taskIds : [""]).forEach((id) => query.append("task_ids", id));
  return query;
}
export const listDatasetRuns = (taskId: string, page = 1, pageSize = 5, taskIds?: string[]) => {
  const query = taskQuery(taskId, taskIds);
  query.set("page", String(page)); query.set("page_size", String(pageSize));
  return api<PaginatedRuns>(`/api/datasets/runs?${query}`);
};
export const getTrajectoryDetail = (runId: string, datasetId?: string) => {
  const query = new URLSearchParams();
  if (datasetId) query.set("dataset_id", datasetId);
  return api<TrajectoryDetail>(`/api/datasets/runs/${encodeURIComponent(runId)}${query.size ? `?${query}` : ""}`);
};
export const getStageAnnotation = (runId: string) => api<StageAnnotation>(
  `/api/datasets/runs/${encodeURIComponent(runId)}/stage-annotation`,
);
export const saveStageAnnotation = (runId: string, body: {
  keyframes: StageKeyframe[]; exponent?: number; revision: string | null;
}) => api<StageAnnotation>(`/api/datasets/runs/${encodeURIComponent(runId)}/stage-annotation`, {
  method: "PUT", body: JSON.stringify(body),
});
export const setTrajectoryTestLabel = (runId: string, isTest: boolean) => api<{
  run_id: string; task_id: string | null; is_test: boolean;
  excluded_from_training_packages: boolean;
  excluded_from_default_batch_evaluation: boolean;
}>(`/api/datasets/runs/${encodeURIComponent(runId)}/labels`, {
  method: "PATCH", body: JSON.stringify({ is_test: isTest }),
});
export const evaluateTrajectories = (body: {
  task_id: string; run_ids: string[] | null; overwrite: boolean;
  evaluators: Array<"rynnvalue" | "robometer">;
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

export const listTrainingDatasets = (taskId?: string, taskIds?: string[]) => {
  const query = taskQuery(taskId, taskIds);
  return api<TrainingDataset[]>("/api/training-datasets" + (query.size ? `?${query}` : ""));
};
export const verifyTrainingDataset = (id: string) => api<TrainingDataset>(
  `/api/training-datasets/${encodeURIComponent(id)}/verify`, { method: "POST" },
);
export const annotateTrainingDataset = (
  id: string, parameters: RewardParameters & { source: RewardSource | EvaluationOperation },
) => api<OfflineJob>(
  `/api/training-datasets/${encodeURIComponent(id)}/annotations`, {
    method: "POST",
    body: JSON.stringify(parameters),
  },
);
export const getDatasetRewardConfig = (id: string) => api<Record<RewardSource | EvaluationOperation, RewardParameters>>(
  `/api/training-datasets/${encodeURIComponent(id)}/reward-config`,
);
export const listTrainingDatasetMembers = (id: string, page = 1, pageSize = 5) => api<PaginatedRuns>(
  `/api/training-datasets/${encodeURIComponent(id)}/members?page=${page}&page_size=${pageSize}`,
);
export const listOfflineJobs = () => api<OfflineJob[]>("/api/jobs");
export const getOfflineJob = (id: string) => api<OfflineJob>(`/api/jobs/${encodeURIComponent(id)}`);
export const stopOfflineJob = (id: string) => api<OfflineJob>(
  `/api/jobs/${encodeURIComponent(id)}/stop`, { method: "POST" },
);
export const getJobLogs = (id: string, offset = 0) => api<{
  offset: number; next_offset: number; text: string;
}>(`/api/jobs/${encodeURIComponent(id)}/logs?offset=${offset}`);
export const getTrainingDefaults = (datasetId?: string, rewardSource?: TrainingRewardSource) => {
  const query = new URLSearchParams();
  if (datasetId) query.set("dataset_id", datasetId);
  if (rewardSource) query.set("reward_source", rewardSource);
  return api<TrainingDefaults>(`/api/training/defaults${query.size ? `?${query}` : ""}`);
};
export const startTraining = (datasetId: string, parameters: Record<string, unknown>) =>
  api<OfflineJob>("/api/training-runs", {
    method: "POST", body: JSON.stringify({ dataset_id: datasetId, parameters }),
  });
export const enqueueTraining = (datasetId: string, parameters: Record<string, unknown>) =>
  api<OfflineJob>("/api/training-queue", {
    method: "POST", body: JSON.stringify({ dataset_id: datasetId, parameters }),
  });
export const getTrainingQueue = () => api<import("./types").TrainingQueueState>("/api/training-queue");
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
    if (Array.isArray(value)) (value.length ? value : [""]).forEach((id) => query.append(name, id));
    else if (value) query.set(name, value);
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
