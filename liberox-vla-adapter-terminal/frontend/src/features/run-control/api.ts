import { api } from "../../api/client";
import type { Bootstrap, DatasetSummary, Session } from "./types";

export const getBootstrap = () => api<Bootstrap>("/api/bootstrap");
export const listRuns = () => api<Session[]>("/api/runs");
export const getDatasetSummary = () => api<DatasetSummary>("/api/datasets/summary");
export const listDatasetRuns = (taskId: string) => api<Session[]>(
  "/api/datasets/runs?task_id=" + encodeURIComponent(taskId),
);
export const datasetExportUrl = (taskId: string) => (
  "/api/datasets/export?task_id=" + encodeURIComponent(taskId)
);
