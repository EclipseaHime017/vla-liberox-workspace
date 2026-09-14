import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingPage } from "./TrainingPage";
import * as api from "../features/run-control/api";
import type { Bootstrap, TrainingDataset, TrainingDefaults } from "../features/run-control/types";

vi.mock("../features/run-control/api", () => ({
  getBootstrap: vi.fn(), getTensorBoard: vi.fn(), getTrainingDefaults: vi.fn(),
  listOfflineJobs: vi.fn(), listTrainingDatasets: vi.fn(), startTensorBoard: vi.fn(), startTraining: vi.fn(),
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
  it.each([false, true])("overrides cumulative reward=%s and gamma only for this training run", async (cumulative) => {
    vi.mocked(api.getTrainingDefaults).mockResolvedValue({ ...defaults,
      advanced: { ...defaults.advanced, reward_accumulate_primitive_steps: cumulative } });
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(false));
    expect((screen.getByLabelText("Reward 来源") as HTMLSelectElement).disabled).toBe(true);
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
    await screen.findByText("请先在数据集配置中完成评价，再开始训练。");
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
