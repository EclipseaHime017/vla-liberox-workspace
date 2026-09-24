import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingPage } from "./TrainingPage";
import * as api from "../features/run-control/api";
import type { Bootstrap, OfflineJob, TrainingDataset, TrainingDefaults } from "../features/run-control/types";

vi.mock("../features/run-control/api", () => ({
  getBootstrap: vi.fn(), getTensorBoard: vi.fn(), getTrainingDefaults: vi.fn(),
  listOfflineJobs: vi.fn(), listTrainingDatasets: vi.fn(), startTensorBoard: vi.fn(), enqueueTraining: vi.fn(),
  getTrainingQueue: vi.fn(), getOfflineJob: vi.fn(), stopOfflineJob: vi.fn(),
  getDatasetRewardConfig: vi.fn(), annotateTrainingDataset: vi.fn(), verifyTrainingDataset: vi.fn(),
}));
vi.mock("../features/training/JobMonitor", () => ({ JobMonitor: ({ initial }: { initial: OfflineJob }) =>
  <output data-testid="monitor">{initial.id}:{initial.status}</output> }));
const runningJob: OfflineJob = {
  id: "running", kind: "training", status: "RUNNING", dataset_id: "ready",
  created_at: "2026-09-16T01:00:00Z", started_at: null, completed_at: null,
  stage: "train", stage_label: "训练中", error: null, output_path: "/tmp", log_size: 0,
  parameters: { dataset_name: "First dataset", micro_batch_size: 1, train_steps: 10000, seed: 7 },
};
const defaults: TrainingDefaults = {
  basic: { train_steps: 10000, micro_batch_size: 1 },
  advanced: { reward_source: "final", reward_stage_exponent: 4, reward_gamma: .92,
    reward_shaping_weight: .1, reward_accumulate_primitive_steps: false, beta: 3 },
  reward_version: { id: "version-p4", evaluator: "final", status: "COMPLETED", parameters: { stage_exponent: 4, gamma: .92 }, created_at: "" },
  reward_parameters_locked: false,
  reward_locked_parameters: ["reward_source", "reward_stage_exponent", "reward_shaping_weight", "reward_rynnvalue"],
  reward_editable_parameters: ["reward_gamma", "reward_accumulate_primitive_steps"],
  monitoring: {}, fixed: {}, environments: { training: "vla-liberox" }, checkpoints: [],
};
const rynnDefaults: TrainingDefaults = {
  ...defaults,
  advanced: { ...defaults.advanced, reward_source: "rynnvalue", reward_gamma: .87,
    reward_shaping_weight: .2, reward_accumulate_primitive_steps: true },
  reward_version: { id: "rynn-v1", evaluator: "rynnvalue", status: "COMPLETED",
    parameters: { gamma: .87, shaping_weight: .2 }, created_at: "" },
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
  vi.mocked(api.getTrainingQueue).mockResolvedValue({ jobs: [], waiting_reason: null });
  vi.mocked(api.listTrainingDatasets).mockResolvedValue(datasets);
  vi.mocked(api.getTensorBoard).mockResolvedValue({ url: "/board", running: false, managed: false, pid: null, logdir: "/tmp" });
  vi.mocked(api.enqueueTraining).mockResolvedValue({ id: "training", kind: "training", status: "QUEUED",
    dataset_id: "ready", created_at: "2026-09-16T01:00:00Z", parameters: { dataset_name: "Ready dataset" },
    started_at: null, completed_at: null, stage: "queued", stage_label: "等待中", error: null,
    output_path: "/tmp", log_size: 0 });
  vi.mocked(api.startTensorBoard).mockResolvedValue({ url: "/board", running: true, managed: true, pid: 123, logdir: "/tmp" });
});
afterEach(cleanup);

describe("dataset-pinned training reward", () => {
  it("switches between coexisting Final Reward and RynnValue without changing other training edits", async () => {
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (_id, source) =>
      source === "rynnvalue" ? rynnDefaults : defaults);
    render(<TrainingPage />);
    const start = await screen.findByRole("button", { name: "注册训练任务" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    const reward = screen.getByLabelText("训练奖励") as HTMLSelectElement;
    expect(Array.from(reward.options, (option) => option.text)).toEqual(["Final Reward", "RynnValue"]);
    fireEvent.change(screen.getByLabelText("训练步数"), { target: { value: "77" } });
    fireEvent.change(reward, { target: { value: "rynnvalue" } });
    await waitFor(() => expect(api.getTrainingDefaults).toHaveBeenLastCalledWith("ready", "rynnvalue"));
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    expect((screen.getByLabelText("Discount ratio γ") as HTMLInputElement).value).toBe("0.87");
    fireEvent.click(start);
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_source: "rynnvalue", reward_version_id: "rynn-v1", reward_gamma: .87,
      reward_shaping_weight: .2, reward_accumulate_primitive_steps: true, train_steps: 77,
    })));
    await waitFor(() => expect(reward.disabled).toBe(false));
    fireEvent.change(reward, { target: { value: "final" } });
    await waitFor(() => expect((screen.getByLabelText("Discount ratio γ") as HTMLInputElement).value).toBe("0.92"));
    fireEvent.click(start);
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenLastCalledWith("ready", expect.objectContaining({
      reward_source: "final", reward_version_id: "version-p4", reward_gamma: .92,
      reward_shaping_weight: .1, reward_accumulate_primitive_steps: false, train_steps: 77,
    })));
  });

  it("does not use available Final Reward when the selected RynnValue result is missing", async () => {
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (_id, source) => source === "rynnvalue"
      ? { ...rynnDefaults, reward_version: null, reward_availability: { ready: false, origin: "global" } } : defaults);
    render(<TrainingPage />);
    const start = await screen.findByRole("button", { name: "注册训练任务" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    fireEvent.change(screen.getByLabelText("训练奖励"), { target: { value: "rynnvalue" } });
    await screen.findByText(/请先完成 RynnValue 评价/);
    expect((start as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(start);
    expect(api.enqueueTraining).not.toHaveBeenCalled();
  });

  it("ignores late reward defaults and never submits the previously selected version", async () => {
    let resolveRynn: (value: TrainingDefaults) => void = () => {};
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (_id, source) => source === "rynnvalue"
      ? new Promise((resolve) => { resolveRynn = resolve; }) : defaults);
    render(<TrainingPage />);
    const start = await screen.findByRole("button", { name: "注册训练任务" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    fireEvent.change(screen.getByLabelText("训练奖励"), { target: { value: "rynnvalue" } });
    await waitFor(() => expect(api.getTrainingDefaults).toHaveBeenLastCalledWith("ready", "rynnvalue"));
    expect((start as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(start);
    expect(api.enqueueTraining).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("训练奖励"), { target: { value: "final" } });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    resolveRynn(rynnDefaults);
    await waitFor(() => expect((screen.getByLabelText("Discount ratio γ") as HTMLInputElement).value).toBe("0.92"));
    fireEvent.click(start);
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_source: "final", reward_version_id: "version-p4",
    })));
  });

  it("keeps RynnValue cumulative editable when unused fusion settings are multiplicative", async () => {
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (_id, source) => source === "rynnvalue"
      ? { ...rynnDefaults, advanced: { ...rynnDefaults.advanced, reward_fusion_mode: "multiplicative" } }
      : { ...defaults, advanced: { ...defaults.advanced, reward_fusion_mode: "multiplicative" },
          reward_editable_parameters: ["reward_gamma"] });
    render(<TrainingPage />);
    const start = await screen.findByRole("button", { name: "注册训练任务" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    expect(screen.queryByLabelText("cumulative reward")).toBeNull();
    fireEvent.change(screen.getByLabelText("训练奖励"), { target: { value: "rynnvalue" } });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    expect((screen.getByLabelText("cumulative reward") as HTMLSelectElement).value).toBe("true");
    fireEvent.click(start);
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_source: "rynnvalue", reward_accumulate_primitive_steps: true,
    })));
  });

  it("trains BC on an unannotated dataset without sending reward or critic settings", async () => {
    render(<TrainingPage />);
    await screen.findByRole("option", { name: /Unready dataset/ });
    fireEvent.change(screen.getByLabelText("冻结数据集"), { target: { value: "unready" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "注册训练任务" }).hasAttribute("disabled")).toBe(true));
    fireEvent.change(screen.getByLabelText("训练方法"), { target: { value: "bc" } });
    await waitFor(() => expect(api.getTrainingDefaults).toHaveBeenCalledWith("unready", undefined, "bc"));
    await waitFor(() => expect(screen.getByRole("button", { name: "注册训练任务" }).hasAttribute("disabled")).toBe(false));
    expect(screen.queryByLabelText("Discount ratio γ")).toBeNull();
    expect(screen.queryByLabelText("训练奖励")).toBeNull();
    expect(screen.queryByLabelText("Critic warmup")).toBeNull();
    expect(screen.getByLabelText("Policy LR warmup")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "注册训练任务" }));
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalled());
    const [datasetId, parameters] = vi.mocked(api.enqueueTraining).mock.calls[0];
    expect(datasetId).toBe("unready");
    expect(parameters.algorithm).toBe("bc");
    expect(Object.keys(parameters).some((key) => key.startsWith("reward_") || key.startsWith("critic_"))).toBe(false);
    fireEvent.change(screen.getByLabelText("训练方法"), { target: { value: "iql" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "注册训练任务" }).hasAttribute("disabled")).toBe(true));
  });

  it("registers independent tasks while training and preserves the active monitor", async () => {
    vi.mocked(api.listOfflineJobs).mockResolvedValue([runningJob]);
    vi.mocked(api.getTrainingQueue).mockResolvedValue({ jobs: [runningJob], waiting_reason: null });
    vi.mocked(api.getOfflineJob).mockResolvedValue(runningJob);
    vi.mocked(api.getTrainingDefaults).mockResolvedValue({ ...defaults,
      checkpoints: [{ path: "/previous.pt", label: "Previous checkpoint" }] });
    vi.mocked(api.enqueueTraining).mockImplementation(async (id, parameters) => ({ ...runningJob,
      id: `queued-${parameters.micro_batch_size}`, status: "QUEUED", dataset_id: id,
      parameters: { ...parameters, dataset_name: "New dataset" },
    }));
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByRole("button", { name: "注册训练任务" }) as HTMLButtonElement).disabled).toBe(false));
    expect(screen.queryByLabelText("断点恢复")).toBeNull();
    fireEvent.change(screen.getByLabelText("Micro batch size"), { target: { value: "4" } });
    fireEvent.click(screen.getByRole("button", { name: "注册训练任务" }));
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalledWith("ready",
      expect.objectContaining({ micro_batch_size: 4, resume_checkpoint: null })));
    await screen.findByText(/Batch 4/);
    expect(screen.getByTestId("monitor").textContent).toBe("running:RUNNING");
    await waitFor(() => expect((screen.getByRole("button", { name: "注册训练任务" }) as HTMLButtonElement).disabled).toBe(false));
    expect(screen.getByRole("button", { name: "注册训练任务" }).closest("details")).toBeNull();
    expect(screen.queryByRole("button", { name: "新增训练任务" })).toBeNull();
    expect((screen.getByLabelText("Micro batch size") as HTMLInputElement).value).toBe("4");
    fireEvent.change(screen.getByLabelText("Micro batch size"), { target: { value: "8" } });
    fireEvent.change(screen.getByLabelText("Discount ratio γ"), { target: { value: ".95" } });
    fireEvent.click(screen.getByRole("button", { name: "注册训练任务" }));
    await screen.findByText(/Batch 8/);
    expect(screen.getByText(/Batch 4/)).toBeTruthy();
    expect(vi.mocked(api.enqueueTraining).mock.calls[0][1].reward_gamma).toBe(.92);
    expect(vi.mocked(api.enqueueTraining).mock.calls[1][1].reward_gamma).toBe(.95);
  });

  it("restores pending tasks and cancels only the selected item", async () => {
    const pending = { ...runningJob, id: "pending", status: "QUEUED" as const };
    vi.mocked(api.listOfflineJobs).mockResolvedValue([runningJob]);
    vi.mocked(api.getTrainingQueue).mockResolvedValue({ jobs: [runningJob, pending], waiting_reason: null });
    vi.mocked(api.getOfflineJob).mockResolvedValue(runningJob);
    vi.mocked(api.stopOfflineJob).mockResolvedValue({ ...pending, status: "CANCELED" });
    render(<TrainingPage />);
    fireEvent.click(await screen.findByRole("button", { name: "取消排队" }));
    await waitFor(() => expect(api.stopOfflineJob).toHaveBeenCalledExactlyOnceWith("pending"));
    expect(screen.getByRole("button", { name: "停止本轮" })).toBeTruthy();
    expect(screen.getByTestId("monitor").textContent).toBe("running:RUNNING");
  });

  it("advances to the next running task even if the previous websocket was stale", async () => {
    const second = { ...runningJob, id: "second" };
    vi.mocked(api.listOfflineJobs).mockResolvedValue([runningJob]);
    vi.mocked(api.getTrainingQueue).mockResolvedValueOnce({ jobs: [runningJob], waiting_reason: null })
      .mockResolvedValue({ jobs: [{ ...runningJob, status: "COMPLETED" }, second], waiting_reason: null });
    vi.mocked(api.getOfflineJob).mockImplementation(async (id) => id === "second" ? second : runningJob);
    render(<TrainingPage />);
    await waitFor(() => expect(screen.getByTestId("monitor").textContent).toBe("running:RUNNING"));
    await waitFor(() => expect(screen.getByTestId("monitor").textContent).toBe("second:RUNNING"), { timeout: 4500 });
  });

  it("does not override deliberate inspection of a waiting task during polling", async () => {
    const pending = { ...runningJob, id: "pending", status: "QUEUED" as const };
    vi.mocked(api.listOfflineJobs).mockResolvedValue([runningJob]);
    vi.mocked(api.getTrainingQueue).mockResolvedValue({ jobs: [runningJob, pending], waiting_reason: null });
    vi.mocked(api.getOfflineJob).mockImplementation(async (id) => id === "pending" ? pending : runningJob);
    render(<TrainingPage />);
    await screen.findByRole("button", { name: "取消排队" });
    fireEvent.click(screen.getAllByRole("button", { name: "查看进度" })[1]);
    await waitFor(() => expect(screen.getByTestId("monitor").textContent).toBe("pending:QUEUED"));
    await waitFor(() => expect(api.getTrainingQueue).toHaveBeenCalledTimes(2), { timeout: 4500 });
    expect(screen.getByTestId("monitor").textContent).toBe("pending:QUEUED");
    expect(screen.getByRole("button", { name: "跟随当前训练" })).toBeTruthy();
  });

  it.each(["multiplicative", "mixed"])("%s results hide cumulative and submit macro even with a stale On default", async (mode) => {
    vi.mocked(api.getTrainingDefaults).mockResolvedValue({ ...defaults,
      advanced: { ...defaults.advanced, reward_fusion_mode: mode === "mixed" ? "additive" : mode,
        reward_accumulate_primitive_steps: true }, reward_editable_parameters: ["reward_gamma"] });
    render(<TrainingPage />);
    const start = await screen.findByRole("button", { name: "注册训练任务" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    expect(screen.queryByLabelText("cumulative reward")).toBeNull();
    fireEvent.change(screen.getByLabelText("Discount ratio γ"), { target: { value: ".95" } });
    fireEvent.click(start);
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_accumulate_primitive_steps: false, reward_gamma: .95,
    })));
  });

  it("starts a newly frozen dataset directly from global labels without an evaluation request", async () => {
    vi.mocked(api.listTrainingDatasets).mockResolvedValue([datasets[1]]);
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (id) => id ? {
      ...defaults, reward_version: null, reward_availability: { ready: true, origin: "global" },
    } : defaults);
    render(<TrainingPage />);
    const start = await screen.findByRole("button", { name: "注册训练任务" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    expect(screen.getByText(/使用逐轨迹全局结果/)).toBeTruthy();
    fireEvent.click(start);
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalledWith("unready",
      expect.objectContaining({ reward_source: "final", reward_version_id: null })));
    expect(api.annotateTrainingDataset).not.toHaveBeenCalled();
  });

  it("shows legacy evaluation guidance without disabling dataset configuration", async () => {
    const message = "旧评价缺少训练快照，请在配置数据集评价中重新生成该类型结果";
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (id) => id ? {
      ...defaults, reward_availability: { ready: false, origin: "dataset", message },
    } : defaults);
    render(<TrainingPage />);
    expect(await screen.findByText(message)).toBeTruthy();
    expect((screen.getByRole("button", { name: "注册训练任务" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "配置数据集评价" }) as HTMLButtonElement).disabled).toBe(false);
    expect(api.enqueueTraining).not.toHaveBeenCalled();
  });

  it("clears a previous defaults error when another dataset loads successfully", async () => {
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (id, source) => {
      if (id === "ready") throw new Error("Missing saved evaluation config");
      return { ...defaults, reward_availability: { ready: true, origin: "global" } };
    });
    render(<TrainingPage />);
    await screen.findByText(/Missing saved evaluation config/);
    expect((screen.getByRole("button", { name: "注册训练任务" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("冻结数据集"), { target: { value: "unready" } });
    await waitFor(() => expect((screen.getByRole("button", { name: "注册训练任务" }) as HTMLButtonElement).disabled).toBe(false));
    expect(screen.queryByText(/Missing saved evaluation config/)).toBeNull();
  });

  it("selects global rewards without requiring a dataset-local result ID", async () => {
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (id, source) => id ? {
      ...defaults, reward_version: null,
      advanced: { ...defaults.advanced, reward_source: source ?? "final" },
      reward_availability: { ready: true, origin: "global" },
    } : defaults);
    render(<TrainingPage />);
    const start = await screen.findByRole("button", { name: "注册训练任务" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    expect(screen.getByText(/使用逐轨迹全局结果/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("训练奖励"), { target: { value: "rynnvalue" } });
    await waitFor(() => expect(api.getTrainingDefaults).toHaveBeenLastCalledWith("ready", "rynnvalue"));
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(start);
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_source: "rynnvalue", reward_version_id: null,
    })));
    expect(screen.queryByRole("option", { name: "Robometer" })).toBeNull();
  });

  it("ignores delayed defaults for a previously selected dataset", async () => {
    let resolveSparse: (value: TrainingDefaults) => void = () => {};
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (id, source) => {
      if (id === "unready") return new Promise((resolve) => { resolveSparse = resolve; });
      return defaults;
    });
    render(<TrainingPage />);
    const start = await screen.findByRole("button", { name: "注册训练任务" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    fireEvent.change(screen.getByLabelText("冻结数据集"), { target: { value: "unready" } });
    await waitFor(() => expect(api.getTrainingDefaults).toHaveBeenLastCalledWith("unready", "final"));
    expect((start as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("冻结数据集"), { target: { value: "ready" } });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    resolveSparse({ ...defaults, reward_version: null, advanced: { ...defaults.advanced, reward_gamma: .1 } });
    await waitFor(() => expect((screen.getByLabelText("Discount ratio γ") as HTMLInputElement).value).toBe("0.92"));
    fireEvent.click(start);
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_source: "final", reward_version_id: "version-p4",
    })));
  });

  it("opens the same inline dataset configuration from training", async () => {
    vi.mocked(api.getDatasetRewardConfig).mockResolvedValue({ sparse: { gamma: .99 }, stage: { gamma: .99, stage_exponent: 2 },
      rynnvalue: { gamma: .99, max_frames: 4 }, robometer: { prefix_frames: 4 }, final: { gamma: .99, alpha: .5, stage_exponent: 2, shaping_weight: .1 }, all: {} });
    render(<TrainingPage />);
    fireEvent.click(await screen.findByRole("button", { name: "配置数据集评价" }));
    expect(await screen.findByLabelText("评价类型")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "删除数据集" })).toBeNull();
  });
  it.each([false, true])("overrides cumulative reward=%s and gamma only for this training run", async (cumulative) => {
    vi.mocked(api.getTrainingDefaults).mockResolvedValue({ ...defaults,
      advanced: { ...defaults.advanced, reward_accumulate_primitive_steps: cumulative } });
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByRole("button", { name: "注册训练任务" }) as HTMLButtonElement).disabled).toBe(false));
    expect((screen.getByLabelText("训练奖励") as HTMLSelectElement).value).toBe("final");
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
    fireEvent.click(screen.getByRole("button", { name: "注册训练任务" }));
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalledWith("ready", expect.objectContaining({
      reward_version_id: "version-p4", reward_source: "final", reward_stage_exponent: 4,
      reward_gamma: .95, reward_accumulate_primitive_steps: !cumulative, train_steps: 20000,
    })));
    expect(defaults.reward_version?.parameters).toEqual({ stage_exponent: 4, gamma: .92 });
  });

  it("allows inspecting an unevaluated frozen dataset but blocks training", async () => {
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByLabelText("冻结数据集") as HTMLSelectElement).value).toBe("ready"));
    expect(screen.queryByRole("option", { name: /Broken dataset/ })).toBeNull();
    fireEvent.change(screen.getByLabelText("冻结数据集"), { target: { value: "unready" } });
    await screen.findByText(/请先完成 Final Reward 评价/);
    expect((screen.getByRole("button", { name: "注册训练任务" }) as HTMLButtonElement).disabled).toBe(true);
    expect(api.enqueueTraining).not.toHaveBeenCalled();
  });

  it("loads each dataset's reward defaults while retaining unrelated training edits", async () => {
    vi.mocked(api.getTrainingDefaults).mockImplementation(async (datasetId) => datasetId === "unready"
      ? { ...defaults, advanced: { ...defaults.advanced, reward_source: "final", reward_gamma: .95 },
        reward_version: { ...defaults.reward_version!, id: "sparse-v1", evaluator: "final" } } : defaults);
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByRole("button", { name: "注册训练任务" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.change(screen.getByLabelText("训练步数"), { target: { value: "77" } });
    fireEvent.change(screen.getByLabelText("冻结数据集"), { target: { value: "unready" } });
    await waitFor(() => expect((screen.getByLabelText("Discount ratio γ") as HTMLInputElement).value).toBe("0.95"));
    expect((screen.getByLabelText("训练奖励") as HTMLSelectElement).value).toBe("final");
    expect((screen.getByLabelText("训练步数") as HTMLInputElement).value).toBe("77");
    fireEvent.click(screen.getByRole("button", { name: "注册训练任务" }));
    await waitFor(() => expect(api.enqueueTraining).toHaveBeenCalledWith("unready", expect.objectContaining({ reward_version_id: "sparse-v1", reward_gamma: .95, train_steps: 77 })));
  });

  it("shows a rejected pinned-version error without starting TensorBoard", async () => {
    vi.mocked(api.enqueueTraining).mockRejectedValue(new Error("Reward version artifacts missing"));
    render(<TrainingPage />);
    await waitFor(() => expect((screen.getByRole("button", { name: "注册训练任务" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "注册训练任务" }));
    await screen.findByText(/Reward version artifacts missing/);
    expect(api.startTensorBoard).not.toHaveBeenCalled();
  });
});
