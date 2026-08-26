import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Bootstrap, EvaluationPreview, EvaluationRecord } from "../features/run-control/types";

const mocks = vi.hoisted(() => ({
  getBootstrap: vi.fn(), listEvaluations: vi.fn(), listOfflineJobs: vi.fn(),
  previewEvaluation: vi.fn(), startEvaluation: vi.fn(), getEvaluation: vi.fn(),
  deleteEvaluation: vi.fn(),
}));
vi.mock("../features/run-control/api", () => ({
  ...mocks,
  stopEvaluation: vi.fn(),
}));
vi.mock("../features/evaluation/EvaluationMonitor", () => ({
  EvaluationMonitor: ({ initial }: { initial: { id: string } }) => <div>测试监视器 {initial.id}</div>,
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
  schedule_sha256: "abcdef0123456789", init_state_counts: { "0": 34, "1": 33, "2": 33 },
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

describe("evaluation page", () => {
  afterEach(() => { cleanup(); vi.restoreAllMocks(); });
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.getBootstrap.mockResolvedValue(bootstrap);
    mocks.listEvaluations.mockResolvedValue([]);
    mocks.listOfflineJobs.mockResolvedValue([]);
    mocks.previewEvaluation.mockResolvedValue(preview);
    mocks.startEvaluation.mockResolvedValue({
      id: "eval-active", kind: "evaluation", status: "STARTING", dataset_id: null,
      created_at: "now", started_at: null, completed_at: null, stage: "starting",
      stage_label: "启动测试", error: null, output_path: "/tmp/active",
      parameters: { trials: 100 }, log_size: 0,
    });
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

  it("starts a test only after the schedule has been previewed", async () => {
    render(<EvaluationPage />);
    const start = await screen.findByRole("button", { name: "开始测试" }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "预览调度" }));
    await waitFor(() => expect(start.disabled).toBe(false));
    fireEvent.click(start);
    await waitFor(() => expect(mocks.startEvaluation).toHaveBeenCalledWith(config));
    expect(await screen.findByText("测试监视器 eval-active")).toBeTruthy();
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
