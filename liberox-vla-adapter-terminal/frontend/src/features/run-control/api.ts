import { api } from "../../api/client";
import type {
  Bootstrap, DatasetPreview, DatasetSelection, DatasetSummary, OfflineJob,
  Session, TensorBoardStatus, TrainingDataset, TrainingDefaults,
} from "./types";

export const getBootstrap = () => api<Bootstrap>("/api/bootstrap");
export const listRuns = () => api<Session[]>("/api/runs");
export const getDatasetSummary = () => api<DatasetSummary>("/api/datasets/summary");
export const listDatasetRuns = (taskId: string) => api<Session[]>(
  "/api/datasets/runs?task_id=" + encodeURIComponent(taskId),
);
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

export const deleteTrainingDataset = (id: string) => api<{
  deleted: string; annotation_status: string;
  source_runs_deleted: boolean; shared_cache_deleted: boolean;
}>(`/api/training-datasets/${encodeURIComponent(id)}`, {
  method: "DELETE", body: JSON.stringify({ confirm_dataset_id: id }),
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
