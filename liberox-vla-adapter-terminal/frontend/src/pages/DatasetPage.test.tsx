import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
  getDatasetRewardConfig: vi.fn(), listTrainingDatasetMembers: vi.fn(),
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
  vi.mocked(api.getDatasetRewardConfig).mockResolvedValue({
    sparse: { gamma: .99, accumulate_primitive_steps: false },
    stage: { gamma: .99, stage_exponent: 2, accumulate_primitive_steps: false },
    rynnvalue: { gamma: .99, shaping_weight: .1, max_frames: 4, batch_size: 1, accumulate_primitive_steps: false, checkpoint: "Rynn-4B", revision: "rynn-revision" },
    robometer: { sampling_hz: 3, batch_size: 1, prefix_frames: 4, checkpoint: "Robo-LIBERO", revision: "robo-revision" },
  });
});
afterEach(cleanup);

describe("dataset evaluation configuration", () => {
  it("freezes membership without launching any evaluator", async () => {
    vi.mocked(api.listTrainingDatasets).mockResolvedValue([]);
    vi.mocked(api.previewTrainingDataset).mockResolvedValue({ task_id: "task", eligible_count: 1, selected_count: 1,
      action_count: 16, chunk_count: 2, categories: {}, run_ids: ["run"], runs: [] });
    vi.mocked(api.createTrainingDataset).mockResolvedValue({ id: "frozen", annotation_status: "NOT_STARTED" } as TrainingDataset);
    render(<DatasetPage />);
    fireEvent.click(await screen.findByRole("button", { name: "打包训练数据集" }));
    fireEvent.click(screen.getByRole("button", { name: "创建训练数据集" }));
    fireEvent.click(screen.getByRole("button", { name: "预览选择结果" }));
    await waitFor(() => expect((screen.getByRole("button", { name: "冻结数据集" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "冻结数据集" }));
    await waitFor(() => expect(api.createTrainingDataset).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.queryByRole("button", { name: "冻结数据集" })).toBeNull());
    expect(api.annotateTrainingDataset).not.toHaveBeenCalled();
    expect(api.evaluateTrajectories).not.toHaveBeenCalled();
  });

  it.each([false, true])("submits cumulative reward as a boolean when changed from %s", async (initial) => {
    vi.mocked(api.listTrainingDatasets).mockResolvedValue([{
      id: "dataset", name: "Frozen dataset", member_count: 1, action_count: 16, chunk_count: 2,
      integrity_status: "HEALTHY", annotation_status: "READY",
      annotation_config: { max_frames: 8, accumulate_primitive_steps: initial },
      reward_version_id: "old-version", evaluation_versions: [{ id: "old-version", evaluator: "rynnvalue", status: "COMPLETED",
        parameters: { max_frames: 8, accumulate_primitive_steps: initial }, created_at: "" }],
    } as TrainingDataset]);
    render(<DatasetPage />);
    fireEvent.click(await screen.findByRole("button", { name: "打包训练数据集" }));
    expect(screen.queryByRole("combobox", { name: "cumulative reward" })).toBeNull();
    expect(api.getDatasetRewardConfig).not.toHaveBeenCalled();
    fireEvent.click(await screen.findByRole("button", { name: "配置" }));
    const cumulative = await screen.findByRole("combobox", { name: "cumulative reward" }) as HTMLSelectElement;
    expect(cumulative.value).toBe(String(initial));
    expect(Array.from(cumulative.options, (option) => option.text)).toEqual(["Off", "On"]);
    expect(screen.queryByRole("checkbox", { name: /cumulative reward|accumulate primitive steps|累计 chunk 内 20 Hz 奖励/ })).toBeNull();
    expect(screen.queryByText("累计 chunk 内 20 Hz 奖励")).toBeNull();
    fireEvent.change(cumulative, { target: { value: String(!initial) } });
    fireEvent.click(screen.getByRole("button", { name: "重新评价" }));
    await waitFor(() => expect(api.annotateTrainingDataset).toHaveBeenCalledWith("dataset", expect.objectContaining({
      source: "rynnvalue", max_frames: 8, accumulate_primitive_steps: !initial, force_model: false,
    })));
  });

  it("offers independent Stage/global overwrite and Robometer model-refresh settings", async () => {
    vi.mocked(api.listTrainingDatasets).mockResolvedValue([{
      id: "dataset", name: "Dataset", integrity_status: "HEALTHY", annotation_status: "NOT_STARTED", evaluation_versions: [],
    } as unknown as TrainingDataset]);
    render(<DatasetPage />);
    fireEvent.click(await screen.findByRole("button", { name: "打包训练数据集" }));
    fireEvent.click(await screen.findByRole("button", { name: "配置" }));
    const source = await screen.findByLabelText("评价类型");
    fireEvent.change(source, { target: { value: "stage" } });
    fireEvent.change(screen.getByLabelText("Stage 插值指数 p"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("同步覆盖全局评价"), { target: { value: "true" } });
    expect(screen.queryByLabelText("max_frames")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "开始评价" }));
    await waitFor(() => expect(api.annotateTrainingDataset).toHaveBeenCalledWith("dataset", {
      source: "stage", gamma: .99, stage_exponent: 4, accumulate_primitive_steps: false, overwrite_global: true,
    }));
    await waitFor(() => expect((source as HTMLSelectElement).disabled).toBe(false));
    fireEvent.change(source, { target: { value: "robometer" } });
    expect(screen.queryByLabelText("Discount γ")).toBeNull();
    expect((screen.getByLabelText("前缀帧数") as HTMLInputElement).value).toBe("4");
    expect((screen.getByLabelText("前缀帧数") as HTMLInputElement).disabled).toBe(true);
    expect(screen.getByTitle("固定 revision：robo-revision").textContent).toBe("Robo-LIBERO");
    expect((screen.getByLabelText("同步覆盖全局评价") as HTMLSelectElement).value).toBe("false");
    fireEvent.change(screen.getByLabelText("评价 fps"), { target: { value: "2" } });
    fireEvent.change(screen.getByLabelText("重新运行模型"), { target: { value: "true" } });
    fireEvent.click(screen.getByRole("button", { name: "开始评价" }));
    await waitFor(() => expect(api.annotateTrainingDataset).toHaveBeenLastCalledWith("dataset", {
      source: "robometer", sampling_hz: 2, batch_size: 1, force_model: true, overwrite_global: false,
    }));
  });

  it("opens current dataset results through paginated members without version controls", async () => {
    vi.mocked(api.listTrainingDatasets).mockResolvedValue([{
      id: "dataset", name: "Dataset", integrity_status: "HEALTHY", annotation_status: "READY", reward_version_id: "v2",
      evaluation_versions: ["v1", "v2"].map((id) => ({ id, evaluator: "stage", status: "COMPLETED", parameters: { stage_exponent: id === "v1" ? 2 : 4 }, created_at: "" })),
    } as TrainingDataset]);
    vi.mocked(api.listTrainingDatasetMembers).mockImplementation(async (_id, page = 1, pageSize = 5) => ({
      items: [{ id: "member", status: "COMPLETED", source_type: "manual", action_count: 10, success: true }],
      page, page_size: pageSize, pages: 2, total: 6,
    } as Awaited<ReturnType<typeof api.listTrainingDatasetMembers>>));
    render(<DatasetPage />);
    fireEvent.click(await screen.findByRole("button", { name: "打包训练数据集" }));
    fireEvent.click(await screen.findByRole("button", { name: "配置" }));
    await screen.findByLabelText("Stage 插值指数 p");
    expect(screen.queryByText(/v1|v2|历史版本/)).toBeNull();
    expect(screen.queryByRole("button", { name: "启用" })).toBeNull();
    expect(screen.queryByLabelText("评价版本")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "成员" }));
    const members = await screen.findByRole("region", { name: "Dataset 成员" });
    await waitFor(() => expect(api.listTrainingDatasetMembers).toHaveBeenCalledWith("dataset", 1, 5));
    const size = within(members).getByLabelText("每页") as HTMLSelectElement;
    expect(Array.from(size.options, (option) => option.value)).toEqual(["5", "10", "20", "50"]);
    fireEvent.change(size, { target: { value: "50" } });
    await waitFor(() => expect(api.listTrainingDatasetMembers).toHaveBeenCalledWith("dataset", 1, 50));
    fireEvent.click(await within(members).findByRole("button", { name: "详情" }));
    await waitFor(() => expect(api.getTrajectoryDetail).toHaveBeenCalledWith("member", "dataset"));
  });

  it("offers reevaluation of legacy results without exposing internal versions", async () => {
    vi.mocked(api.listTrainingDatasets).mockResolvedValue([{
      id: "dataset", name: "Dataset", integrity_status: "HEALTHY", annotation_status: "READY",
      evaluation_versions: [{ id: "legacy", evaluator: "rynnvalue", status: "READY", legacy: true, parameters: {}, created_at: "" }],
    } as TrainingDataset]);
    render(<DatasetPage />);
    fireEvent.click(await screen.findByRole("button", { name: "打包训练数据集" }));
    fireEvent.click(await screen.findByRole("button", { name: "配置" }));
    fireEvent.click(await screen.findByRole("button", { name: "重新评价" }));
    expect(screen.queryByText(/legacy/)).toBeNull();
    expect(screen.queryByRole("button", { name: "启用" })).toBeNull();
    await waitFor(() => expect(api.annotateTrainingDataset).toHaveBeenCalledWith("dataset", expect.objectContaining({ source: "rynnvalue", force_model: false })));
  });

  it("uses the overwrite choice when evaluating selected trajectories", async () => {
    vi.mocked(api.listTrainingDatasets).mockResolvedValue([]);
    vi.mocked(api.listDatasetRuns).mockResolvedValue({
      items: [{ id: "run", status: "COMPLETED", training_eligible: true }], total: 1, eligible_count: 1,
      evaluated_count: 1, rynn_evaluated_count: 1, robometer_evaluated_count: 0, both_evaluated_count: 0,
      test_count: 0, page: 1, page_size: 5, pages: 1,
    } as Awaited<ReturnType<typeof api.listDatasetRuns>>);
    vi.mocked(api.evaluateTrajectories).mockResolvedValue({ job: { id: "evaluation", kind: "trajectory_evaluation", status: "STARTING" } } as Awaited<ReturnType<typeof api.evaluateTrajectories>>);
    render(<DatasetPage />);
    fireEvent.click(await screen.findByLabelText("选择 run"));
    fireEvent.click(screen.getByLabelText("覆盖已有评价"));
    fireEvent.click(screen.getByRole("button", { name: "评价所选（覆盖已有）" }));
    await waitFor(() => expect(api.evaluateTrajectories).toHaveBeenCalledWith({
      task_id: "task", run_ids: ["run"], evaluators: ["rynnvalue"], overwrite: true,
    }));
  });

  it("refreshes pending global evaluation without replacing the detail screen", async () => {
    vi.mocked(api.listTrainingDatasets).mockResolvedValue([]);
    vi.mocked(api.listDatasetRuns).mockResolvedValue({ items: [{ id: "run", status: "COMPLETED" }],
      total: 1, page: 1, page_size: 5, pages: 1 } as Awaited<ReturnType<typeof api.listDatasetRuns>>);
    vi.mocked(api.getTrajectoryDetail)
      .mockResolvedValueOnce({ run: { id: "run" }, global_evaluation_pending: true } as Awaited<ReturnType<typeof api.getTrajectoryDetail>>)
      .mockResolvedValue({ run: { id: "run" }, global_evaluation_pending: false } as Awaited<ReturnType<typeof api.getTrajectoryDetail>>);
    render(<DatasetPage />);
    fireEvent.click(await screen.findByRole("button", { name: "详情" }));
    await waitFor(() => expect(api.getTrajectoryDetail).toHaveBeenCalledTimes(2), { timeout: 2000 });
    expect(api.getTrajectoryDetail).toHaveBeenLastCalledWith("run");
    expect(screen.queryByRole("button", { name: "详情" })).toBeNull();
  });
});
