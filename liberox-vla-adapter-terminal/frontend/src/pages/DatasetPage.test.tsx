import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetPage } from "./DatasetPage";
import * as api from "../features/run-control/api";
import type { Bootstrap, DatasetSummary, TrainingDataset } from "../features/run-control/types";

vi.mock("../features/run-control/api", () => ({
  annotateTrainingDataset: vi.fn(), createTrainingDataset: vi.fn(), datasetExportUrl: vi.fn(),
  deriveTrainingDataset: vi.fn(), deleteTrainingDataset: vi.fn(), evaluateTrajectories: vi.fn(),
  getBootstrap: vi.fn(), getDatasetSummary: vi.fn(), getTrajectoryDetail: vi.fn(),
  listDatasetRuns: vi.fn(), listOfflineJobs: vi.fn(), listTrainingDatasets: vi.fn(),
  previewTrainingDataset: vi.fn(), verifyTrainingDataset: vi.fn(), setTrajectoryTestLabel: vi.fn(),
}));
vi.mock("../features/training/JobMonitor", () => ({ JobMonitor: () => null }));
vi.mock("../features/dataset/TrajectoryDetail", () => ({ TrajectoryDetail: () => null }));

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.getBootstrap).mockResolvedValue({ task: { task_id: "task" },
    task_catalog: [{ task_id: "task", prompt: "pick bowl" }] } as unknown as Bootstrap);
  vi.mocked(api.getDatasetSummary).mockResolvedValue({
    tasks: [], dataset_root: "/tmp/datasets", catalog: "/tmp/catalog",
  } as unknown as DatasetSummary);
  vi.mocked(api.listDatasetRuns).mockResolvedValue({
    items: [], total: 0, eligible_count: 0, evaluated_count: 0, rynn_evaluated_count: 0,
    robometer_evaluated_count: 0, both_evaluated_count: 0, test_count: 0,
    page: 1, page_size: 5, pages: 1,
  });
  vi.mocked(api.listOfflineJobs).mockResolvedValue([]);
  vi.mocked(api.annotateTrainingDataset).mockResolvedValue({
    id: "annotation", kind: "annotation", status: "STARTING",
  } as Awaited<ReturnType<typeof api.annotateTrainingDataset>>);
});
afterEach(cleanup);

describe("dataset cumulative reward selection", () => {
  it.each([false, true])("submits cumulative reward as a boolean when changed from %s", async (initial) => {
    vi.mocked(api.listTrainingDatasets).mockResolvedValue([{
      id: "dataset", name: "Frozen dataset", member_count: 1, action_count: 16, chunk_count: 2,
      integrity_status: "HEALTHY", annotation_status: "READY",
      annotation_config: { max_frames: 8, accumulate_primitive_steps: initial },
    } as TrainingDataset]);
    render(<DatasetPage />);
    fireEvent.click(await screen.findByRole("button", { name: "打包训练数据集" }));
    const cumulative = await screen.findByRole("combobox", { name: "cumulative reward" }) as HTMLSelectElement;
    expect(cumulative.value).toBe(String(initial));
    expect(Array.from(cumulative.options, (option) => option.text)).toEqual(["Off", "On"]);
    expect(screen.queryByRole("checkbox", { name: /cumulative reward|accumulate primitive steps|累计 chunk 内 20 Hz 奖励/ })).toBeNull();
    expect(screen.queryByText("累计 chunk 内 20 Hz 奖励")).toBeNull();
    fireEvent.change(cumulative, { target: { value: String(!initial) } });
    fireEvent.click(screen.getByRole("button", { name: "生成新评价并切换" }));
    await waitFor(() => expect(api.annotateTrainingDataset).toHaveBeenCalledWith("dataset", 8, !initial));
  });
});
