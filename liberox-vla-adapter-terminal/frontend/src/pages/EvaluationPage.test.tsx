import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Bootstrap, EvaluationPreview, EvaluationRecord, OfflineJob } from "../features/run-control/types";

const mocks = vi.hoisted(() => ({
  getBootstrap: vi.fn(), listEvaluations: vi.fn(), listOfflineJobs: vi.fn(),
  previewEvaluation: vi.fn(), enqueueEvaluation: vi.fn(), getEvaluation: vi.fn(),
  getEvaluationQueue: vi.fn(), getOfflineJob: vi.fn(), stopEvaluation: vi.fn(), deleteEvaluation: vi.fn(),
}));
vi.mock("../features/run-control/api", () => ({
  ...mocks,
}));
vi.mock("../features/evaluation/EvaluationMonitor", () => ({
  EvaluationMonitor: ({ initial }: { initial: OfflineJob }) => <div data-testid="monitor">测试监视器 {initial.id}:{initial.status}</div>,
}));
import { EvaluationPage } from "./EvaluationPage";

const task = { task_id: "task-a", level: "LEVEL1", task_name: "task-a", prompt: "stack bowls", init_state_count: 3, init_state_index_min: 0, init_state_index_max: 2 };
const policy = { policy_id: "base", label: "Object-Pro", base_checkpoint: "base", stats_key: "stats", kind: "base" as const, training_step: null, compatibility_sha256: null };
const bootstrap = {
  config: { max_steps: 300, open_loop_steps: 8, seed: 7 },
  model: { policy_id: "base" }, policy_catalog: [policy], task, task_catalog: [task],
} as Bootstrap;
const config = { task_id: "task-a", policy_id: "base", trials: 100, max_steps: 300, open_loop_steps: 8, realtime: true, init_state_indices: null, base_seed: 7, seed_count: null, schedule_seed: 7 };
const preview: EvaluationPreview = {
  config, schedule: [{ trial_index: 0, init_state_index: 0, seed: 7 }],
  schedule_sha256: "a".repeat(64), init_state_counts: { "0": 34, "1": 33, "2": 33 },
  seed_counts: { "7": 3 }, combination_counts: {}, estimated_duration_seconds: 1500,
};
const aggregate = {
  total_trials: 10, attempted_trials: 10, completed_trials: 10, successes: 7,
  failures: 3, errors: 0, success_rate: .7, wilson_lower: .4, wilson_upper: .89,
  completion_rate: 1, by_init_state: {}, by_seed: {}, by_combination: {},
  first_success_step_mean: 30, policy_queries_mean: 38, inference_latency_ms_mean: 50,
  measured_control_hz_mean: 20, elapsed_seconds_mean: 15,
};
const record = {
  id: "eval-history", status: "COMPLETED", created_at: "2026-08-26T00:00:00Z",
  started_at: "2026-08-26T00:00:00Z", completed_at: "2026-08-26T00:10:00Z",
  task_id: "task-a", task_name: "task-a", task_prompt: "stack bowls", policy_id: "base",
  policy_label: "Object-Pro", base_checkpoint: "base", overlay_id: null, training_step: null,
  compatibility_sha256: null, config: { ...config, trials: 10 }, schedule: [], schedule_sha256: "hash",
  success_rule: { done_consecutive_steps: 5, success_latched: true, run_full_horizon: true, errors_in_denominator: true },
  trials: [], aggregate, model_load_seconds: 20, wall_time_seconds: 600,
  simulated_time_seconds: 150, error: null, output_path: "/tmp/eval",
} as EvaluationRecord;
const running: OfflineJob = {
  id: "eval-active", kind: "evaluation", status: "RUNNING", dataset_id: null,
  created_at: "2026-09-18T00:00:00Z", started_at: null, completed_at: null, stage: "evaluate",
  stage_label: "测试中", error: null, output_path: "/tmp/active",
  parameters: { ...config, policy_label: "Object-Pro" }, log_size: 0,
};

describe("evaluation page", () => {
  afterEach(() => { cleanup(); vi.restoreAllMocks(); });
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.getBootstrap.mockResolvedValue(bootstrap);
    mocks.listEvaluations.mockResolvedValue([]);
    mocks.listOfflineJobs.mockResolvedValue([]);
    mocks.previewEvaluation.mockResolvedValue(preview);
    mocks.enqueueEvaluation.mockResolvedValue({ ...running, status: "QUEUED" });
    mocks.getEvaluationQueue.mockResolvedValue({ jobs: [], waiting_reason: null });
    mocks.getOfflineJob.mockResolvedValue(running);
    mocks.getEvaluation.mockResolvedValue(record);
    mocks.deleteEvaluation.mockResolvedValue({ deleted: record.id });
  });

  it("uses automatic init-state and seed pools by default", async () => {
    render(<EvaluationPage />);
    await screen.findByRole("button", { name: "预览调度" });
    expect((screen.getByLabelText("启用随机环境") as HTMLInputElement).checked).toBe(true);
    expect((screen.getByLabelText("启用随机种子") as HTMLInputElement).checked).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "预览调度" }));
    await waitFor(() => expect(mocks.previewEvaluation).toHaveBeenCalledWith(config));
    expect(await screen.findByText(/调度哈希/)).toBeTruthy();
    expect(screen.getByText(/3 个有限 init states/)).toBeTruthy();
    fireEvent.click(screen.getByLabelText("启用随机环境"));
    expect(screen.queryByText(/调度哈希/)).toBeNull();
  });

  it("maps disabled randomization to one fixed init state and seed", async () => {
    render(<EvaluationPage />);
    await screen.findByRole("button", { name: "预览调度" });
    fireEvent.click(screen.getByLabelText("启用随机环境"));
    fireEvent.click(screen.getByLabelText("启用随机种子"));
    fireEvent.click(screen.getByRole("button", { name: "预览调度" }));
    await waitFor(() => expect(mocks.previewEvaluation).toHaveBeenCalledWith({
      ...config,
      init_state_indices: [0],
      seed_count: 1,
    }));
    expect(screen.getByText("固定使用 init state #0")).toBeTruthy();
    expect(screen.getByText("固定使用 seed 7")).toBeTruthy();
  });

  it("registers a test only after the schedule has been previewed", async () => {
    render(<EvaluationPage />);
    const start = await screen.findByRole("button", { name: "注册测试任务" }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "预览调度" }));
    await waitFor(() => expect(start.disabled).toBe(false));
    fireEvent.click(start);
    await waitFor(() => expect(mocks.enqueueEvaluation).toHaveBeenCalledWith(config, preview.schedule_sha256));
    expect(await screen.findByText("测试监视器 eval-active:QUEUED")).toBeTruthy();
  });

  it("registers independent tests during a running test without replacing its monitor", async () => {
    mocks.getEvaluationQueue.mockResolvedValue({ jobs: [running], waiting_reason: null });
    mocks.enqueueEvaluation.mockImplementation(async (input) => ({ ...running, id: `queued-${input.trials}`,
      status: "QUEUED", parameters: { ...input } }));
    render(<EvaluationPage />);
    await screen.findByText("测试监视器 eval-active:RUNNING");
    fireEvent.click(screen.getByRole("button", { name: "预览调度" }));
    const register = screen.getByRole("button", { name: "注册测试任务" }) as HTMLButtonElement;
    await waitFor(() => expect(register.disabled).toBe(false));
    fireEvent.click(register);
    await screen.findByText("queued-100");
    expect(screen.getByTestId("monitor").textContent).toContain("eval-active:RUNNING");
    fireEvent.change(screen.getByLabelText("仿真次数"), { target: { value: "50" } });
    expect(register.disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "预览调度" }));
    await waitFor(() => expect(register.disabled).toBe(false));
    fireEvent.click(register);
    await screen.findByText("queued-50");
    expect(mocks.enqueueEvaluation.mock.calls.map(([input]) => input.trials)).toEqual([100, 50]);
    expect((screen.getByLabelText("仿真次数") as HTMLInputElement).value).toBe("50");
  });

  it("ignores a preview response that arrives after the form was edited", async () => {
    let resolve!: (value: EvaluationPreview) => void;
    mocks.previewEvaluation.mockImplementation(() => new Promise<EvaluationPreview>((done) => { resolve = done; }));
    render(<EvaluationPage />);
    const button = screen.getByRole("button", { name: "预览调度" }) as HTMLButtonElement;
    await waitFor(() => expect(button.disabled).toBe(false));
    fireEvent.click(button);
    fireEvent.change(screen.getByLabelText("仿真次数"), { target: { value: "50" } });
    await act(async () => resolve(preview));
    expect(screen.queryByText(/调度哈希/)).toBeNull();
    expect((screen.getByRole("button", { name: "注册测试任务" }) as HTMLButtonElement).disabled).toBe(true);
    expect(mocks.enqueueEvaluation).not.toHaveBeenCalled();
  });

  it("restores the queue and cancels only the selected waiting test", async () => {
    const waiting = { ...running, id: "waiting", status: "QUEUED" };
    mocks.getEvaluationQueue.mockResolvedValue({ jobs: [running, waiting], waiting_reason: null });
    mocks.stopEvaluation.mockResolvedValue({ ...waiting, status: "CANCELED" });
    mocks.listEvaluations.mockResolvedValue([{ ...record, id: "waiting", status: "QUEUED" }]);
    render(<EvaluationPage />);
    await screen.findByText("测试监视器 eval-active:RUNNING");
    expect((screen.getByRole("button", { name: "删除" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(within(screen.getByRole("region", { name: "测试队列" })).getByRole("button", { name: "取消排队" }));
    await waitFor(() => expect(mocks.stopEvaluation).toHaveBeenCalledExactlyOnceWith("waiting"));
    expect(screen.getByRole("button", { name: "停止本轮" })).toBeTruthy();
    expect(screen.getByTestId("monitor").textContent).toContain("eval-active:RUNNING");
  });

  it("follows the next running test when polling detects completion without a websocket update", async () => {
    const next = { ...running, id: "next" };
    mocks.getEvaluationQueue.mockResolvedValue({ jobs: [running, { ...next, status: "QUEUED" }], waiting_reason: null });
    mocks.getOfflineJob.mockImplementation(async (id) => id === "next" ? next : running);
    render(<EvaluationPage />);
    await screen.findByText("测试监视器 eval-active:RUNNING");
    mocks.getEvaluationQueue.mockResolvedValue({ jobs: [{ ...running, status: "COMPLETED" }, next], waiting_reason: null });
    await waitFor(() => expect(screen.getByTestId("monitor").textContent).toContain("next:RUNNING"), { timeout: 4000 });
    expect(mocks.listEvaluations.mock.calls.length).toBeGreaterThan(1);
  });

  it("preserves manual queue inspection until following is explicitly restored", async () => {
    const waiting = { ...running, id: "waiting", status: "QUEUED" };
    mocks.getEvaluationQueue.mockResolvedValue({ jobs: [running, waiting], waiting_reason: null });
    mocks.getOfflineJob.mockImplementation(async (id) => id === "waiting" ? waiting : running);
    render(<EvaluationPage />);
    await screen.findByText("测试监视器 eval-active:RUNNING");
    const row = screen.getByText("waiting").closest("article")!;
    fireEvent.click(within(row).getByRole("button", { name: "查看进度" }));
    await screen.findByText("测试监视器 waiting:QUEUED");
    await waitFor(() => expect(mocks.getEvaluationQueue.mock.calls.length).toBeGreaterThan(1), { timeout: 4000 });
    expect(screen.getByTestId("monitor").textContent).toContain("waiting:QUEUED");
    expect(screen.getByRole("button", { name: "跟随当前测试" })).toBeTruthy();
  });

  it("opens a historical test detail with success confidence and grouped metrics", async () => {
    mocks.listEvaluations.mockResolvedValue([record]);
    render(<EvaluationPage />);
    await screen.findByText("eval-history");
    fireEvent.click(screen.getByRole("button", { name: "查看详情" }));
    await waitFor(() => expect(mocks.getEvaluation).toHaveBeenCalledWith(record.id));
    expect(await screen.findByText(/Wilson 95% 置信区间/)).toBeTruthy();
    expect(screen.getByLabelText("成功率 70.0%")).toBeTruthy();
  });

  it("requires two confirmations before permanently deleting a terminal test", async () => {
    mocks.listEvaluations.mockResolvedValue([record]);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.spyOn(window, "prompt").mockReturnValue(record.id);
    render(<EvaluationPage />);
    await screen.findByText("eval-history");
    fireEvent.click(screen.getByRole("button", { name: "删除" }));
    await waitFor(() => expect(mocks.deleteEvaluation).toHaveBeenCalledWith(record.id, record.id));
    expect(window.confirm).toHaveBeenCalledOnce();
    expect(window.prompt).toHaveBeenCalledOnce();
  });
});
