import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingPage } from "./TrainingPage";
import * as api from "../features/run-control/api";
import type { Bootstrap, TrainingDataset, TrainingDefaults } from "../features/run-control/types";

vi.mock("../features/run-control/api", () => ({
  getBootstrap: vi.fn(), getTensorBoard: vi.fn(), getTrainingDefaults: vi.fn(),
  listOfflineJobs: vi.fn(), listTrainingDatasets: vi.fn(), startTensorBoard: vi.fn(), startTraining: vi.fn(),
  getDatasetRewardConfig: vi.fn(), annotateTrainingDataset: vi.fn(), verifyTrainingDataset: vi.fn(),
}));
vi.mock("../features/training/JobMonitor", () => ({ JobMonitor: () => null }));
const defaults: TrainingDefaults = {
  basic: { train_steps: 10000, micro_batch_size: 1 },
  advanced: { reward_source: "stage", reward_stage_exponent: 4, reward_gamma: .92,
    reward_shaping_weight: .1, reward_accumulate_primitive_steps: false, beta: 3 },
  reward_version: { id: "version-p4", evaluator: "stage", status: "COMPLETED", parameters: { stage_exponent: 4, gamma: .92 }, created_at: "" },
  reward_parameters_locked: false,
  reward_locked_parameters: ["reward_source", "reward_stage_exponent", "reward_shaping_weight", "reward_rynnvalue"],
  reward_editable_parameters: ["reward_gamma", "reward_accumulate_primitive_steps"],
  monitoring: {}, fixed: {}, environments: { training: "vla-liberox" }, checkpoints: [],
};
const datasets = [
  { id: "ready", name: "Ready dataset", member_count: 5, integrity_status: "HEALTHY", annotation_status: "READY", reward_version_id: "version-p4" },
  { id: "unready", name: "Unready dataset", member_count: 3, integrity_status: "HEALTHY", annotation_status: "NOT_STARTED" },
  { id: "broken", name: "Broken dataset", member_count: 3, integrity_status: "BROKEN", annotation_status: "READY" },
] as TrainingDataset[];

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.getBootstrap).mockResolvedValue({ task: { task_id: "task" },
    task_catalog: [{ task_id: "task", prompt: "pick bowl" }] } as unknown as Bootstrap);
  vi.mocked(api.getTrainingDefaults).mockImplementation(async (datasetId) => datasetId === "unready"
    ? { ...defaults, reward_version: null } : defaults);
  vi.mocked(api.listOfflineJobs).mockResolvedValue([]);
  vi.mocked(api.listTrainingDatasets).mockResolvedValue(datasets);
  vi.mocked(api.getTensorBoard).mockResolvedValue({ url: "/board", running: false, managed: false, pid: null, logdir: "/tmp" });
  vi.mocked(api.startTraining).mockResolvedValue({ id: "training", kind: "training", status: "STARTING" } as Awaited<ReturnType<typeof api.startTraining>>);
  vi.mocked(api.startTensorBoard).mockResolvedValue({ url: "/board", running: true, managed: true, pid: 123, logdir: "/tmp" });
});
afterEach(cleanup);

describe("dataset-pinned training reward", () => {
  it("shows legacy evaluation guidance without disabling dataset configuration", async () => {
    const message = "旧评价缺少训练快照，请在配置数据集评价中重新生成该类型结果";
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (id) => id ? {
      ...defaults, reward_availability: { ready: false, origin: "dataset", message },
    } : defaults);
    render(<TrainingPage />);
    expect(await screen.findByText(message)).toBeTruthy();
    expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "配置数据集评价" }) as HTMLButtonElement).disabled).toBe(false);
    expect(api.startTraining).not.toHaveBeenCalled();
  });

  it("clears a previous defaults error when another reward source loads successfully", async () => {
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (id, source) => {
      if (id && source === "stage") throw new Error("Missing saved evaluation config");
      return { ...defaults, reward_availability: { ready: true, origin: "global" } };
    });
    render(<TrainingPage />);
    await screen.findByText(/Missing saved evaluation config/);
    expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Reward 来源"), { target: { value: "sparse" } });
    await waitFor(() => expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(false));
    expect(screen.queryByText(/Missing saved evaluation config/)).toBeNull();
  });

  it("selects global rewards without requiring a dataset-local result ID", async () => {
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (id, source) => id ? {
      ...defaults, reward_version: null,
      advanced: { ...defaults.advanced, reward_source: source ?? "stage" },
      reward_availability: { ready: true, origin: "global" },
    } : defaults);
    render(<TrainingPage />);
    const start = await screen.findByRole("button", { name: "开始训练" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    expect(screen.getByText(/使用逐轨迹全局结果/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Reward 来源"), { target: { value: "sparse" } });
    await waitFor(() => expect(api.getTrainingDefaults).toHaveBeenLastCalledWith("ready", "sparse"));
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(start);
    await waitFor(() => expect(api.startTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_source: "sparse", reward_version_id: null,
    })));
    expect(screen.queryByRole("option", { name: "Robometer" })).toBeNull();
  });

  it("ignores delayed defaults for a previously selected reward source", async () => {
    let resolveSparse: (value: TrainingDefaults) => void = () => {};
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (id, source) => {
      if (id && source === "sparse") return new Promise((resolve) => { resolveSparse = resolve; });
      return defaults;
    });
    render(<TrainingPage />);
    const start = await screen.findByRole("button", { name: "开始训练" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    fireEvent.change(screen.getByLabelText("Reward 来源"), { target: { value: "sparse" } });
    await waitFor(() => expect(api.getTrainingDefaults).toHaveBeenLastCalledWith("ready", "sparse"));
    expect((start as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Reward 来源"), { target: { value: "stage" } });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    resolveSparse({ ...defaults, reward_version: null, advanced: { ...defaults.advanced, reward_gamma: .1 } });
    await waitFor(() => expect((screen.getByLabelText("Discount ratio γ") as HTMLInputElement).value).toBe("0.92"));
    fireEvent.click(start);
    await waitFor(() => expect(api.startTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_source: "stage", reward_version_id: "version-p4",
    })));
  });

  it("opens the same inline dataset configuration from training", async () => {
    vi.mocked(api.getDatasetRewardConfig).mockResolvedValue({ sparse: { gamma: .99 }, stage: { gamma: .99, stage_exponent: 2 },
      rynnvalue: { gamma: .99, max_frames: 4 }, robometer: { prefix_frames: 4 } });
    render(<TrainingPage />);
    fireEvent.click(await screen.findByRole("button", { name: "配置数据集评价" }));
    expect(await screen.findByLabelText("评价类型")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "删除数据集" })).toBeNull();
  });
  it.each([false, true])("overrides cumulative reward=%s and gamma only for this training run", async (cumulative) => {
    vi.mocked(api.getTrainingDefaults).mockResolvedValue({ ...defaults,
      advanced: { ...defaults.advanced, reward_accumulate_primitive_steps: cumulative } });
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(false));
    expect((screen.getByLabelText("Reward 来源") as HTMLSelectElement).disabled).toBe(false);
    expect(screen.queryByLabelText("Shape reward 系数 κ")).toBeNull();
    expect(screen.queryByLabelText("Stage 插值指数 p")).toBeNull();
    expect(screen.queryByText(/version-p4/)).toBeNull();
    const selector = screen.getByLabelText("cumulative reward") as HTMLSelectElement;
    expect(selector.disabled).toBe(false);
    expect(selector.value).toBe(String(cumulative));
    expect(Array.from(selector.options, (option) => option.text)).toEqual(["Off", "On"]);
    const gamma = screen.getByLabelText("Discount ratio γ") as HTMLInputElement;
    expect(gamma.disabled).toBe(false);
    expect(gamma.value).toBe("0.92");
    fireEvent.change(gamma, { target: { value: "0.95" } });
    fireEvent.change(selector, { target: { value: String(!cumulative) } });
    expect((screen.getByLabelText("训练步数") as HTMLInputElement).disabled).toBe(false);
    fireEvent.change(screen.getByLabelText("训练步数"), { target: { value: "20000" } });
    fireEvent.click(screen.getByRole("button", { name: "开始训练" }));
    await waitFor(() => expect(api.startTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_version_id: "version-p4", reward_source: "stage", reward_stage_exponent: 4,
      reward_gamma: .95, reward_accumulate_primitive_steps: !cumulative, train_steps: 20000,
    })));
    expect(defaults.reward_version?.parameters).toEqual({ stage_exponent: 4, gamma: .92 });
  });

  it("allows inspecting an unevaluated frozen dataset but blocks training", async () => {
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByLabelText("冻结数据集") as HTMLSelectElement).value).toBe("ready"));
    expect(screen.queryByRole("option", { name: /Broken dataset/ })).toBeNull();
    fireEvent.change(screen.getByLabelText("冻结数据集"), { target: { value: "unready" } });
    await screen.findByText(/请先完成 Stage-based 评价/);
    expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(true);
    expect(api.startTraining).not.toHaveBeenCalled();
  });

  it("loads each dataset's reward defaults while retaining unrelated training edits", async () => {
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (datasetId) => datasetId === "unready"
      ? { ...defaults, advanced: { ...defaults.advanced, reward_source: "sparse", reward_gamma: .95 },
        reward_version: { ...defaults.reward_version!, id: "sparse-v1", evaluator: "sparse" } } : defaults);
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.change(screen.getByLabelText("训练步数"), { target: { value: "77" } });
    fireEvent.change(screen.getByLabelText("冻结数据集"), { target: { value: "unready" } });
    fireEvent.change(screen.getByLabelText("Reward 来源"), { target: { value: "sparse" } });
    await waitFor(() => expect((screen.getByLabelText("Discount ratio γ") as HTMLInputElement).value).toBe("0.95"));
    expect((screen.getByLabelText("Reward 来源") as HTMLSelectElement).value).toBe("sparse");
    expect((screen.getByLabelText("训练步数") as HTMLInputElement).value).toBe("77");
    fireEvent.click(screen.getByRole("button", { name: "开始训练" }));
    await waitFor(() => expect(api.startTraining).toHaveBeenCalledWith("unready", expect.objectContaining({ reward_version_id: "sparse-v1", reward_gamma: .95, train_steps: 77 })));
  });

  it("shows a rejected pinned-version error without starting TensorBoard", async () => {
    vi.mocked(api.startTraining).mockRejectedValue(new Error("Reward version artifacts missing"));
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "开始训练" }));
    await screen.findByText(/Reward version artifacts missing/);
    expect(api.startTensorBoard).not.toHaveBeenCalled();
  });
});
