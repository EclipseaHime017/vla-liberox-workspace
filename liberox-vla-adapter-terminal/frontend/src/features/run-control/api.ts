import { api } from "../../api/client";
import type { Bootstrap, DatasetSummary, Session } from "./types";

export const getBootstrap = () => api<Bootstrap>("/api/bootstrap");
export const listRuns = () => api<Session[]>("/api/runs");
export const getDatasetSummary = () => api<DatasetSummary>("/api/datasets/summary");
