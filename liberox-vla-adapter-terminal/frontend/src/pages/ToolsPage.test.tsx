import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Bootstrap, OfflineJob, TrainingDataset } from "../features/run-control/types";
import type { CriticDefaults, CriticDiagnostic } from "../features/tools/api";

const mocks = vi.hoisted(() => ({
  getBootstrap: vi.fn(), listTrainingDatasets: vi.fn(), getCriticDefaults: vi.fn(),
  listCriticDiagnostics: vi.fn(), getCriticDiagnostic: vi.fn(), startCriticDiagnostic: vi.fn(),
  stopCriticDiagnostic: vi.fn(), deleteCriticDiagnostic: vi.fn(),
}));
vi.mock("../features/run-control/api", () => ({ getBootstrap: mocks.getBootstrap, listTrainingDatasets: mocks.listTrainingDatasets }));
vi.mock("../features/tools/api", () => mocks);
vi.mock("../features/training/JobMonitor", () => ({
  JobMonitor: ({ initial }: { initial: OfflineJob }) => <div data-testid="job-monitor" data-status={initial.status}>日志 {initial.id}</div>,
}));
import { ToolsPage } from "./ToolsPage";

const task = { task_id: "task-a", family_id: "family-a", family_label: "Place bowl", level: "LEVEL1", task_name: "task-a", prompt: "place bowl", init_state_count: 10, init_state_index_min: 0, init_state_index_max: 9 };
const policy = { policy_id: "base", label: "Registered model", kind: "base" as const,
  base_checkpoint: "base", stats_key: "stats", training_step: null, compatibility_sha256: null,
  io: { native_action_horizon: 50, replay_horizon: 50 },
  inference: { open_loop_steps: 5, max_open_loop_steps: 50 } };
const bootstrap = { config: { max_steps: 300, seed: 7 }, model: { policy_id: "base" },
  policy_catalog: [policy], task, task_catalog: [task] } as Bootstrap;
const dataset = { id: "ds-a", name: "Takeover", task_id: "task-a", integrity_status: "HEALTHY" } as TrainingDataset;
const defaults: CriticDefaults = { train_steps: 3000, batch_size: 32,
  reward_sources: ["final", "rynnvalue"], default_reward_source: "final" };
const job = { id: "critic-1", kind: "critic_diagnostic", status: "QUEUED", dataset_id: "ds-a",
  created_at: "2026-10-22T00:00:00Z", started_at: null, completed_at: null, stage: "queued",
  stage_label: "等待前序任务完成", error: null, output_path: "/tmp/critic-1", log_size: 0,
  parameters: { policy_id: "base", policy_label: "Registered model", dataset_name: "Takeover", train_steps: 3000, batch_size: 32 },
} as CriticDiagnostic;

describe("critic diagnostic tools", () => {
  afterEach(() => { cleanup(); vi.restoreAllMocks(); });
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.getBootstrap.mockResolvedValue(bootstrap);
    mocks.listTrainingDatasets.mockResolvedValue([dataset]);
    mocks.getCriticDefaults.mockResolvedValue(defaults);
    mocks.listCriticDiagnostics.mockResolvedValue({ jobs: [] });
    mocks.startCriticDiagnostic.mockResolvedValue(job);
    mocks.getCriticDiagnostic.mockResolvedValue(job);
    mocks.stopCriticDiagnostic.mockResolvedValue({ ...job, status: "CANCELED" });
    mocks.deleteCriticDiagnostic.mockResolvedValue({ deleted: job.id });
  });

  it("submits existing dataset rewards and model-owned execution settings without actor options", async () => {
    render(<ToolsPage />);
    const start = screen.getByRole("button", { name: "开始诊断" }) as HTMLButtonElement;
    await waitFor(() => expect(start.disabled).toBe(false));
    expect(mocks.listTrainingDatasets).toHaveBeenCalledWith("task-a");
    expect((screen.getByLabelText("每次预测执行步数") as HTMLInputElement).value).toBe("5");
    fireEvent.change(screen.getByLabelText("Rollout 控制步数"), { target: { value: "240" } });
    fireEvent.change(screen.getByLabelText("训练步数"), { target: { value: "120" } });
    fireEvent.change(screen.getByLabelText("Batch size"), { target: { value: "16" } });
    fireEvent.change(screen.getByLabelText("初始状态索引"), { target: { value: "3" } });
    fireEvent.click(start);
    await waitFor(() => expect(mocks.startCriticDiagnostic).toHaveBeenCalledWith({
      dataset_id: "ds-a", reward_source: "final", task_id: "task-a", policy_id: "base",
      max_steps: 240, open_loop_steps: 5, seed: 7, init_state_index: 3, train_steps: 120, batch_size: 16,
    }));
    expect(await screen.findByTestId("job-monitor")).toHaveProperty("textContent", "日志 critic-1");
    expect(screen.queryByText(/Actor loss/)).toBeNull();
  });

  it("blocks diagnostics when no sealed dataset reward is available", async () => {
    mocks.getCriticDefaults.mockResolvedValue({ ...defaults, reward_sources: [], default_reward_source: null });
    render(<ToolsPage />);
    await screen.findByText(/请先在数据集页面配置并完成奖励评价/);
    const start = screen.getByRole("button", { name: "开始诊断" }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    fireEvent.click(start);
    expect(mocks.startCriticDiagnostic).not.toHaveBeenCalled();
  });

  it("changes defaults by policy variant and validates against native output rather than eight steps", async () => {
    mocks.getBootstrap.mockResolvedValue({ ...bootstrap, policy_catalog: [policy,
      { ...policy, policy_id: "other", label: "Another variant", inference: { open_loop_steps: 10, max_open_loop_steps: 50 } },
    ] });
    render(<ToolsPage />);
    const start = screen.getByRole("button", { name: "开始诊断" }) as HTMLButtonElement;
    await waitFor(() => expect(start.disabled).toBe(false));
    const execution = screen.getByLabelText("每次预测执行步数") as HTMLInputElement;
    expect(execution.max).toBe("50");
    fireEvent.change(screen.getByLabelText("模型"), { target: { value: "other" } });
    expect(execution.value).toBe("10");
    fireEvent.change(execution, { target: { value: "12" } });
    fireEvent.change(screen.getByLabelText("Batch size"), { target: { value: "24" } });
    expect(execution.value).toBe("12");
    expect(start.disabled).toBe(false);
    fireEvent.change(execution, { target: { value: "51" } });
    expect(start.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("模型"), { target: { value: "base" } });
    expect(execution.value).toBe("5");
  });

  it("only offers healthy datasets belonging to the selected task", async () => {
    mocks.listTrainingDatasets.mockResolvedValue([dataset,
      { ...dataset, id: "wrong", task_id: "other-task", name: "Other task dataset" },
      { ...dataset, id: "broken", integrity_status: "BROKEN", name: "Broken dataset" },
    ]);
    render(<ToolsPage />);
    await screen.findByRole("option", { name: "Takeover" });
    expect(screen.queryByRole("option", { name: "Other task dataset" })).toBeNull();
    expect(screen.queryByRole("option", { name: "Broken dataset" })).toBeNull();
  });

  it("enforces the same training-step and batch-size bounds as the backend", async () => {
    render(<ToolsPage />);
    const start = screen.getByRole("button", { name: "开始诊断" }) as HTMLButtonElement;
    await waitFor(() => expect(start.disabled).toBe(false));
    const train = screen.getByLabelText("训练步数") as HTMLInputElement;
    const batch = screen.getByLabelText("Batch size") as HTMLInputElement;
    expect(train.max).toBe("1000000");
    expect(batch.max).toBe("4096");
    fireEvent.change(train, { target: { value: "1000000" } });
    fireEvent.change(batch, { target: { value: "4096" } });
    expect(start.disabled).toBe(false);
    fireEvent.change(train, { target: { value: "1000001" } });
    expect(start.disabled).toBe(true);
    fireEvent.change(train, { target: { value: "3000" } });
    fireEvent.change(batch, { target: { value: "4097" } });
    expect(start.disabled).toBe(true);
    fireEvent.click(start);
    expect(mocks.startCriticDiagnostic).not.toHaveBeenCalled();
  });

  it("does not apply stale reward defaults after switching dataset", async () => {
    mocks.listTrainingDatasets.mockResolvedValue([dataset, { ...dataset, id: "ds-b", name: "Second dataset" }]);
    let resolveOld!: (value: CriticDefaults) => void;
    mocks.getCriticDefaults.mockImplementation((id) => id === "ds-a"
      ? new Promise<CriticDefaults>((resolve) => { resolveOld = resolve; })
      : Promise.resolve({ ...defaults, reward_sources: ["stage"], default_reward_source: "stage" }));
    render(<ToolsPage />);
    await waitFor(() => expect(mocks.getCriticDefaults).toHaveBeenCalledWith("ds-a"));
    fireEvent.change(screen.getByLabelText("训练数据集"), { target: { value: "ds-b" } });
    await screen.findByRole("option", { name: "Stage" });
    await act(async () => resolveOld(defaults));
    expect((screen.getByLabelText("已有奖励") as HTMLSelectElement).value).toBe("stage");
    expect(screen.queryByRole("option", { name: "RynnValue" })).toBeNull();
  });

  it("restores job logs, critic progress, figures and downloadable numerical results", async () => {
    const completed: CriticDiagnostic = { ...job, status: "COMPLETED", critic_progress: {
      train_step: 3000, train_steps: 3000, q_loss: .03, value_loss: .01,
    }, diagnostic_summary: { rollout_steps: 240 }, artifacts: {
      "qv.png": "/qv.png", "training.png": "/training.png", "qv.csv": "/qv.csv",
      "qv.npz": "/qv.npz", "summary.json": "/summary.json", "critic.pt": "/critic.pt",
      "effective_config.yaml": "/effective_config.yaml",
    } };
    mocks.listCriticDiagnostics.mockResolvedValue({ jobs: [completed] });
    mocks.getCriticDiagnostic.mockResolvedValue(completed);
    render(<ToolsPage />);
    expect(await screen.findByRole("img", { name: "Rollout Q / V" })).toHaveProperty("src", "http://localhost:3000/qv.png");
    expect(screen.getByRole("img", { name: "Critic 训练曲线" })).toBeTruthy();
    expect(screen.getByTestId("job-monitor").textContent).toBe("日志 critic-1");
    expect(screen.getByLabelText("Critic 训练进度").textContent).toContain("3000 / 3000");
    expect(screen.getByRole("link", { name: "Q/V CSV" }).getAttribute("href")).toBe("/qv.csv");
    expect(screen.getByRole("link", { name: "Critic checkpoint" })).toBeTruthy();
    expect(screen.getByText(/不是策略真实回报/)).toBeTruthy();
  });

  it("shows rollout progress without undefined training steps or critic losses", async () => {
    const rollout: CriticDiagnostic = { ...job, status: "RUNNING", critic_progress: {
      phase: "rollout", rollout_step: 120, rollout_steps: 300,
    } };
    mocks.listCriticDiagnostics.mockResolvedValue({ jobs: [rollout] });
    mocks.getCriticDiagnostic.mockResolvedValue(rollout);
    render(<ToolsPage />);
    const progress = await screen.findByLabelText("Critic 训练进度");
    await waitFor(() => expect(progress.textContent).toContain("生成 Rollout"));
    expect(progress.textContent).toContain("120 / 300");
    expect(progress.textContent).not.toContain("undefined");
    expect(within(progress).queryByText("Q loss")).toBeNull();
    expect(within(progress).getByRole("progressbar").getAttribute("max")).toBe("300");
  });

  it("warns when a rollout chunk length was not represented in critic training", async () => {
    const completed: CriticDiagnostic = { ...job, status: "COMPLETED", diagnostic_summary: {
      unsupported_probe_chunk_lengths: [2, 10],
    } };
    mocks.listCriticDiagnostics.mockResolvedValue({ jobs: [completed] });
    mocks.getCriticDiagnostic.mockResolvedValue(completed);
    render(<ToolsPage />);
    const warning = await screen.findByRole("alert");
    expect(warning.textContent).toContain("2、10 步");
    expect(warning.textContent).toContain("Q 值涉及外推，仅供参考");
  });

  it("cancels only the chosen pending diagnostic", async () => {
    mocks.listCriticDiagnostics.mockResolvedValue({ jobs: [job] });
    render(<ToolsPage />);
    fireEvent.click(await screen.findByRole("button", { name: "取消排队" }));
    await waitFor(() => expect(mocks.stopCriticDiagnostic).toHaveBeenCalledExactlyOnceWith("critic-1"));
    await screen.findByText("已取消");
  });

  it("does not let an older detail request restore a canceled task to queued", async () => {
    let resolveDetail!: (value: CriticDiagnostic) => void;
    mocks.listCriticDiagnostics.mockResolvedValue({ jobs: [job] });
    mocks.getCriticDiagnostic.mockImplementation(() => new Promise<CriticDiagnostic>((resolve) => { resolveDetail = resolve; }));
    render(<ToolsPage />);
    await waitFor(() => expect(mocks.getCriticDiagnostic).toHaveBeenCalledWith(job.id));
    fireEvent.click(screen.getByRole("button", { name: "取消排队" }));
    await screen.findByText("已取消");
    await act(async () => resolveDetail(job));
    expect(screen.getByTestId("job-monitor").getAttribute("data-status")).toBe("CANCELED");
    expect(screen.queryByRole("button", { name: "取消排队" })).toBeNull();
  });

  it("requires confirmation before deleting a completed diagnostic and its artifacts", async () => {
    const completed = { ...job, status: "COMPLETED" as const };
    mocks.listCriticDiagnostics.mockResolvedValue({ jobs: [completed] });
    mocks.getCriticDiagnostic.mockResolvedValue(completed);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<ToolsPage />);
    const history = screen.getByRole("region", { name: "诊断任务记录" });
    const remove = await within(history).findByRole("button", { name: "删除" });
    fireEvent.click(remove);
    expect(mocks.deleteCriticDiagnostic).not.toHaveBeenCalled();
    confirm.mockReturnValue(true);
    fireEvent.click(remove);
    await waitFor(() => expect(mocks.deleteCriticDiagnostic).toHaveBeenCalledExactlyOnceWith("critic-1"));
    await screen.findByText("暂无诊断记录。");
    expect(screen.queryByTestId("job-monitor")).toBeNull();
  });
});
