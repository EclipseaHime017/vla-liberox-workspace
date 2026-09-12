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
  advanced: { reward_source: "rynnvalue", reward_stage_exponent: 2, reward_gamma: .99,
    reward_shaping_weight: .1, reward_accumulate_primitive_steps: false, beta: 3 },
  monitoring: {}, fixed: {}, environments: { training: "vla-liberox" }, checkpoints: [],
};
const datasets = [
  { id: "ready", name: "Rynn ready", member_count: 5, integrity_status: "HEALTHY", annotation_status: "READY" },
  { id: "no-rynn", name: "No Rynn annotation", member_count: 3, integrity_status: "HEALTHY", annotation_status: "NOT_STARTED" },
  { id: "broken", name: "Broken dataset", member_count: 3, integrity_status: "BROKEN", annotation_status: "READY" },
] as TrainingDataset[];

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.getBootstrap).mockResolvedValue({ task: { task_id: "task" },
    task_catalog: [{ task_id: "task", prompt: "pick bowl" }] } as unknown as Bootstrap);
  vi.mocked(api.getTrainingDefaults).mockResolvedValue(defaults);
  vi.mocked(api.listOfflineJobs).mockResolvedValue([]);
  vi.mocked(api.listTrainingDatasets).mockResolvedValue(datasets);
  vi.mocked(api.getTensorBoard).mockResolvedValue({ url: "/board", running: false, managed: false, pid: null, logdir: "/tmp" });
});
afterEach(cleanup);

describe("training reward selection", () => {
  it.each([false, true])("submits cumulative reward as a boolean when changed from %s", async (initial) => {
    vi.mocked(api.getTrainingDefaults).mockResolvedValue({
      ...defaults, advanced: { ...defaults.advanced, reward_accumulate_primitive_steps: initial },
    });
    vi.mocked(api.startTraining).mockResolvedValue({ id: "training", kind: "training", status: "STARTING" } as Awaited<ReturnType<typeof api.startTraining>>);
    vi.mocked(api.startTensorBoard).mockResolvedValue({ url: "/board", running: true, managed: true, pid: 123, logdir: "/tmp" });
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByLabelText("冻结数据集") as HTMLSelectElement).value).toBe("ready"));
    const cumulative = screen.getByLabelText("cumulative reward") as HTMLSelectElement;
    expect(cumulative.tagName).toBe("SELECT");
    expect(cumulative.value).toBe(String(initial));
    expect(Array.from(cumulative.options, (option) => option.text)).toEqual(["Off", "On"]);
    expect(screen.queryByRole("checkbox", { name: /cumulative reward|累计 chunk 内 primitive-step reward/, hidden: true })).toBeNull();
    expect(screen.queryByText("累计 chunk 内 primitive-step reward")).toBeNull();
    fireEvent.change(cumulative, { target: { value: String(!initial) } });
    await waitFor(() => expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "开始训练" }));
    await waitFor(() => expect(api.startTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_accumulate_primitive_steps: !initial,
    })));
  });

  it("offers three reward sources and only gates RynnValue on model evaluation", async () => {
    render(<TrainingPage />);
    const source = await screen.findByLabelText("Reward 来源");
    await waitFor(() => expect((screen.getByLabelText("冻结数据集") as HTMLSelectElement).value).toBe("ready"));
    expect(screen.queryByRole("option", { name: /No Rynn/ })).toBeNull();
    fireEvent.change(source, { target: { value: "stage" } });
    expect(screen.getByRole("option", { name: /No Rynn/ })).toBeTruthy();
    expect(screen.queryByRole("option", { name: /Broken dataset/ })).toBeNull();
    expect((screen.getByLabelText("Stage 插值指数 p") as HTMLInputElement).disabled).toBe(false);
    expect((screen.getByLabelText("Shape reward 系数 κ") as HTMLInputElement).disabled).toBe(true);
    fireEvent.change(source, { target: { value: "sparse" } });
    expect(screen.getByRole("option", { name: /No Rynn/ })).toBeTruthy();
    expect((screen.getByLabelText("Stage 插值指数 p") as HTMLInputElement).disabled).toBe(true);
    expect(screen.queryByText(/以上奖励设置只重建/)).toBeNull();
    expect(screen.queryByLabelText("启用 RynnValue Shape Reward")).toBeNull();
  });

  it("sends stage source and exponent without the old checkbox and displays missing-label errors", async () => {
    vi.mocked(api.startTraining).mockRejectedValue(new Error("Stage annotation missing: episode-3"));
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByLabelText("冻结数据集") as HTMLSelectElement).value).toBe("ready"));
    fireEvent.change(screen.getByLabelText("Reward 来源"), { target: { value: "stage" } });
    fireEvent.change(screen.getByLabelText("冻结数据集"), { target: { value: "no-rynn" } });
    fireEvent.change(screen.getByLabelText("Stage 插值指数 p"), { target: { value: "3" } });
    await waitFor(() => expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "开始训练" }));
    await screen.findByText(/Stage annotation missing: episode-3/);
    expect(api.startTraining).toHaveBeenCalledWith("no-rynn", expect.objectContaining({
      reward_source: "stage", reward_stage_exponent: 3, reward_gamma: .99, beta: 3,
    }));
    expect(vi.mocked(api.startTraining).mock.calls[0][1]).not.toHaveProperty("reward_rynnvalue");
    expect(api.startTensorBoard).not.toHaveBeenCalled();
  });

  it.each(["stage", "sparse"])("starts %s on a healthy frozen dataset without RynnValue labels", async (source) => {
    vi.mocked(api.startTraining).mockResolvedValue({ id: "training", kind: "training", status: "STARTING" } as Awaited<ReturnType<typeof api.startTraining>>);
    vi.mocked(api.startTensorBoard).mockResolvedValue({ url: "/board", running: true, managed: true, pid: 123, logdir: "/tmp" });
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByLabelText("冻结数据集") as HTMLSelectElement).value).toBe("ready"));
    fireEvent.change(screen.getByLabelText("Reward 来源"), { target: { value: source } });
    fireEvent.change(screen.getByLabelText("冻结数据集"), { target: { value: "no-rynn" } });
    await waitFor(() => expect((screen.getByRole("button", { name: "开始训练" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "开始训练" }));
    await waitFor(() => expect(api.startTraining).toHaveBeenCalledWith("no-rynn", expect.objectContaining({ reward_source: source })));
    expect(api.startTraining).toHaveBeenCalledTimes(1);
  });

  it("reads legacy sparse defaults but submits only the new source field", async () => {
    vi.mocked(api.getTrainingDefaults).mockResolvedValue({ ...defaults, advanced: { reward_rynnvalue: false } });
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByLabelText("Reward 来源") as HTMLSelectElement).value).toBe("sparse"));
    expect(screen.getByRole("option", { name: /No Rynn/ })).toBeTruthy();
  });
});
