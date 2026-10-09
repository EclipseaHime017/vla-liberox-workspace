import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AnnotationLabPage } from "./AnnotationLabPage";
import * as lab from "../features/annotation-lab/api";
import * as runs from "../features/run-control/api";
import { api } from "../api/client";
import type { Bootstrap, PaginatedRuns, Session } from "../features/run-control/types";

vi.mock("../features/run-control/api", () => ({ getBootstrap: vi.fn(), listDatasetRuns: vi.fn() }));
vi.mock("../api/client", () => ({ api: vi.fn() }));
vi.mock("../features/annotation-lab/api", async (original) => ({ ...await original<typeof lab>(),
  labDefaults: vi.fn(), listExperiments: vi.fn(), startExperiment: vi.fn(), getExperiment: vi.fn(), stopExperiment: vi.fn(), getLabResult: vi.fn(),
}));
const config = { model_id: "Qwen/Qwen3-VL-4B-Instruct", revision: "a".repeat(40), environment: "keyframe-vlm", cameras: ["agentview_image"], coarse_fps: 5, window_seconds: 2 };
const experiment: lab.Experiment = { id: "lab_test", status: "COMPLETED", error: null, created_at: "2026-10-06", run_ids: ["one", "two"], config, progress: null };
const goal: lab.GoalRange = { stage_id: "s1",
  regions: [{ start_step: 0, end_step: 4, status: "confirmed", window_indices: [0] },
            { start_step: 4, end_step: 8, status: "uncertain", window_indices: [0] },
            { start_step: 8, end_step: 20, status: "outside", window_indices: [0] }],
  passes: [{ steps: [0,4,8,12,16,20], labels: ["confirmed", "uncertain", "outside", "outside", "outside"] }],
};
const result: lab.LabResult = { schema_version: 9, mode: "localize", status: "COMPLETED", error: null, action_count: 20, control_hz: 10, success_step: null,
  source: { prompt: "pick the bowl" }, plan: { stages: [
    { id: "s1", label: "Grasp", achieved_when: "lift", lost_when: "drop", initial_state: "unmet", initial_reason: "on table", depends_on: [] },
    { id: "s2", label: "Place", achieved_when: "supported", lost_when: "displaced", initial_state: "unmet", initial_reason: "empty target", depends_on: ["s1"] }], notes: "check" },
  localization: { assigned_goals: 2, total_goals: 2, mode: "forward_regions", windows_completed: 1, windows_total: 1 },
  sampling: { fps: 5, steps: [0,4,8,12,16,20], window_seconds: 2, stride_seconds: 2 },
  goal_ranges: [goal, { ...goal, stage_id: "s2", regions: [
    { start_step: 0, end_step: 12, status: "outside", window_indices: [0] },
    { start_step: 12, end_step: 20, status: "confirmed", window_indices: [0] },
  ] }], calls: [],
};
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(lab.labDefaults).mockResolvedValue({ config, model_available: true, environment_available: true, message: null });
  vi.mocked(lab.listExperiments).mockResolvedValue([]);
  vi.mocked(runs.getBootstrap).mockResolvedValue({ task_catalog: [{ task_id: "LEVEL1::bowl", task_name: "bowl", prompt: "pick the bowl", level: "LEVEL1" }] } as Bootstrap);
  vi.mocked(runs.listDatasetRuns).mockResolvedValue({ items: [{ id: "one", status: "COMPLETED", action_count: 20, task: "pick the bowl", success: true }], page: 1, pages: 1, total: 1 } as PaginatedRuns);
  vi.mocked(api).mockResolvedValue({ artifacts: {} } as Session);
  vi.mocked(lab.getLabResult).mockResolvedValue(result);
  vi.mocked(lab.getExperiment).mockResolvedValue(experiment);
});
afterEach(cleanup);

it("selects one milestone panel, preserves video and exposes contextual evidence", async () => {
  vi.mocked(lab.listExperiments).mockResolvedValue([experiment]);
  vi.mocked(api).mockResolvedValue({ artifacts: { "agentview.mp4": "main.mp4" } } as unknown as Session);
  const { container } = render(<AnnotationLabPage />);
  await screen.findByLabelText("子任务");
  expect(screen.queryByRole("button", { name: "精定位" })).toBeNull();
  expect(screen.queryByRole("button", { name: "候选与复核" })).toBeNull();
  expect(container.querySelectorAll(".lab-goal")).toHaveLength(1);
  expect(within(screen.getByRole("region", { name: "定位区间" })).queryByText(/完成条件|失效条件|必要前置/)).toBeNull();
  expect(container.querySelectorAll(".lab-goal-timeline .lab-region-confirmed")).toHaveLength(1);
  expect(container.querySelectorAll(".lab-goal-timeline .lab-region-uncertain")).toHaveLength(1);
  expect(container.querySelectorAll(".lab-goal-timeline .lab-region-outside")).toHaveLength(1);
  expect(screen.queryByText(/作用域表示阶段归属/)).toBeNull();
  await waitFor(() => expect(container.querySelector("video")).not.toBeNull());
  fireEvent.click(screen.getByRole("button", { name: /时间轴 确定定位区间 帧 0–4/ }));
  expect((await screen.findAllByRole("img"))[0].getAttribute("src")).toContain("/agentview_image/0");
  expect(container.querySelector(".lab-evidence > p")).toBeNull();
  expect(screen.getByText(/前移约 2 秒，不重叠/)).toBeTruthy();
  const video = container.querySelector("video");
  expect(video).not.toBeNull();
  video!.currentTime = 1.25;
  fireEvent.change(screen.getByLabelText("子任务"), { target: { value: "s2" } });
  expect(screen.queryByRole("button", { name: /时间轴 确定定位区间 帧 0–4/ })).toBeNull();
  expect(screen.getByRole("button", { name: /时间轴 确定定位区间 帧 12–20/ })).toBeTruthy();
  expect(screen.queryByText(/必要前置：Grasp/)).toBeNull();
  expect(container.querySelector("video")).toBe(video);
  expect(video!.currentTime).toBe(1.25);
  expect(container.querySelectorAll(".lab-goal")).toHaveLength(1);
  expect(screen.queryAllByRole("img")).toHaveLength(0);
  fireEvent.click(screen.getByRole("button", { name: "任务拆解" }));
  fireEvent.click(screen.getByRole("button", { name: "区间定位" }));
  expect((screen.getByLabelText("子任务") as HTMLSelectElement).value).toBe("s2");
  expect(container.querySelector("video")).toBe(video);
});

it.each([8, 10])("shows visual evidence for schema %i without rebuilding the video", async (schema) => {
  vi.mocked(lab.listExperiments).mockResolvedValue([experiment]);
  vi.mocked(lab.getLabResult).mockResolvedValue({ ...result, schema_version: schema,
    goal_ranges: [{ ...goal, passes: goal.passes.map(item => ({ ...item, reason: "The bowl stays on the stove as the gripper separates." })) }] });
  render(<AnnotationLabPage />);
  fireEvent.click(await screen.findByRole("button", { name: /时间轴 确定定位区间 帧 0–4/ }));
  expect(screen.getByText("The bowl stays on the stove as the gripper separates.")).toBeTruthy();
});

it("preserves selected evidence while the video session is still loading", async () => {
  vi.mocked(lab.listExperiments).mockResolvedValue([experiment]);
  let resolveSession!: (value: Session) => void;
  vi.mocked(api).mockReturnValue(new Promise<Session>(resolve => { resolveSession = resolve; }));
  const { container } = render(<AnnotationLabPage />);
  fireEvent.click(await screen.findByRole("button", { name: /时间轴 确定定位区间 帧 0–4/ }));
  const evidence = container.querySelector(".lab-evidence");
  expect(evidence).not.toBeNull();
  expect(screen.getAllByRole("img")).toHaveLength(goal.passes[0].steps.length);
  await act(async () => resolveSession({ artifacts: {} } as Session));
  expect(container.querySelector(".lab-evidence")).toBe(evidence);
});

it("runs selected recordings only through the independent endpoint", async () => {
  vi.mocked(lab.startExperiment).mockResolvedValue(experiment);
  render(<AnnotationLabPage />);
  fireEvent.click(await screen.findByLabelText("选择 one"));
  fireEvent.click(screen.getByRole("button", { name: "开始独立实验" }));
  await waitFor(() => expect(lab.startExperiment).toHaveBeenCalledWith(["one"], { mode: "localize", coarse_fps: 5, window_seconds: 2, cameras: ["agentview_image"] }));
  expect(await screen.findByLabelText("子任务")).toBeTruthy();
  expect(vi.mocked(api).mock.calls.every(([url]) => url.startsWith("/api/sessions/"))).toBe(true);
});

it("keeps decomposition separate and does not paint unprocessed intervals gray", async () => {
  vi.mocked(lab.listExperiments).mockResolvedValue([experiment]);
  vi.mocked(lab.getLabResult).mockResolvedValue({ ...result, goal_ranges: [{ ...goal,
    regions: [{ start_step: 0, end_step: 20, status: "pending", window_indices: [] }], passes: [] }] });
  const { container } = render(<AnnotationLabPage />);
  await screen.findByLabelText("子任务");
  expect(container.querySelectorAll(".lab-goal-timeline .lab-region-outside")).toHaveLength(0);
  expect(screen.getByRole("button", { name: /时间轴 尚未评价/ }).hasAttribute("disabled")).toBe(true);
  expect(screen.getByText(/空白部分尚未评价/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "任务拆解" }));
  expect(screen.getByText(/初始：未满足 · on table/)).toBeTruthy();
  expect(screen.queryByRole("combobox", { name: "子任务" })).toBeNull();
});

it("does not reinterpret legacy state points as effective action regions", async () => {
  vi.mocked(lab.listExperiments).mockResolvedValue([experiment]);
  vi.mocked(lab.getLabResult).mockResolvedValue({ ...result, schema_version: 6 });
  render(<AnnotationLabPage />);
  expect(await screen.findByText(/旧版结果/)).toBeTruthy();
  expect(screen.queryByLabelText("定位区间时间轴")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "任务拆解" }));
  expect(screen.getByRole("button", { name: "查看拆解输入首帧" })).toBeTruthy();
});

it("does not show a failed missing goal as still running or never completed", async () => {
  vi.mocked(lab.listExperiments).mockResolvedValue([{ ...experiment, status: "FAILED" }]);
  vi.mocked(lab.getLabResult).mockResolvedValue({ ...result, status: "FAILED", goal_ranges: [goal], error: "Invalid boundary JSON" });
  render(<AnnotationLabPage />);
  await screen.findByLabelText("子任务");
  fireEvent.change(screen.getByLabelText("子任务"), { target: { value: "s2" } });
  expect(screen.getByText(/该目标的评价未完成/)).toBeTruthy();
  expect(screen.queryByText("正在定位该目标…")).toBeNull();
});

it("keeps untouched trajectories but replaces the rerun without a history selector", async () => {
  vi.mocked(lab.listExperiments).mockResolvedValue([experiment]);
  vi.mocked(lab.startExperiment).mockResolvedValue({ ...experiment, id: "lab_new", run_ids: ["one"] });
  render(<AnnotationLabPage />);
  await screen.findByLabelText("子任务");
  fireEvent.click(await screen.findByLabelText("选择 one"));
  fireEvent.click(screen.getByRole("button", { name: "开始独立实验" }));
  await waitFor(() => expect(lab.getLabResult).toHaveBeenCalledWith("lab_new", "one"));
  expect(screen.queryByLabelText("实验记录")).toBeNull();
  expect((screen.getByLabelText("结果轨迹") as HTMLSelectElement).options).toHaveLength(3);
  fireEvent.change(screen.getByLabelText("结果轨迹"), { target: { value: "two" } });
  await waitFor(() => expect(lab.getLabResult).toHaveBeenCalledWith("lab_test", "two"));
});

it("blocks unavailable model without hiding current results", async () => {
  vi.mocked(lab.labDefaults).mockResolvedValue({ config, model_available: false, environment_available: true, message: "model missing" });
  vi.mocked(lab.listExperiments).mockResolvedValue([experiment]);
  render(<AnnotationLabPage />);
  await screen.findByText("model missing");
  expect((screen.getByRole("button", { name: "开始独立实验" }) as HTMLButtonElement).disabled).toBe(true);
  expect(await screen.findByLabelText("子任务")).toBeTruthy();
});

it("submits window changes and omits sampling options for plan-only", async () => {
  vi.mocked(lab.startExperiment).mockResolvedValue(experiment);
  render(<AnnotationLabPage />);
  fireEvent.click(await screen.findByLabelText("选择 one"));
  expect((screen.getByLabelText("采样频率（Hz）") as HTMLInputElement).value).toBe("5");
  expect((screen.getByLabelText("观察窗口（秒）") as HTMLInputElement).value).toBe("2");
  fireEvent.change(screen.getByLabelText("采样频率（Hz）"), { target: { value: "10" } });
  fireEvent.click(screen.getByRole("button", { name: "开始独立实验" }));
  await waitFor(() => expect(lab.startExperiment).toHaveBeenCalledWith(["one"], { mode: "localize", coarse_fps: 10, window_seconds: 2, cameras: ["agentview_image"] }));
  fireEvent.change(screen.getByLabelText("运行模式"), { target: { value: "plan_only" } });
  expect(screen.queryByLabelText("采样频率（Hz）")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "开始独立实验" }));
  await waitFor(() => expect(lab.startExperiment).toHaveBeenLastCalledWith(["one"], { mode: "plan_only", cameras: ["agentview_image"] }));
});

it("rejects too-short or invalid observation windows before submission", async () => {
  render(<AnnotationLabPage />);
  fireEvent.click(await screen.findByLabelText("选择 one"));
  fireEvent.change(screen.getByLabelText("观察窗口（秒）"), { target: { value: "0" } });
  expect((screen.getByRole("button", { name: "开始独立实验" }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.change(screen.getByLabelText("观察窗口（秒）"), { target: { value: "0.5" } });
  fireEvent.change(screen.getByLabelText("采样频率（Hz）"), { target: { value: "1" } });
  expect((screen.getByRole("button", { name: "开始独立实验" }) as HTMLButtonElement).disabled).toBe(true);
});

it("shows plan-only requirements and initial evidence without localization failure messages", async () => {
  vi.mocked(lab.listExperiments).mockResolvedValue([experiment]);
  vi.mocked(lab.getLabResult).mockResolvedValue({ ...result, schema_version: 7, mode: "plan_only", goal_ranges: [],
    task_contract: { requirements: [{ id: "r1", instruction_span: "pick the bowl", condition: "bowl held", depends_on: [] }] },
    plan: { stages: [{ ...result.plan!.stages[0], requirement_ids: ["r1"] }], notes: "plan check" },
  });
  render(<AnnotationLabPage />);
  expect(await screen.findByText("任务拆解已完成；本次未运行时间定位。")).toBeTruthy();
  expect(screen.getByText("指令要求")).toBeTruthy();
  expect(screen.getByText("对应要求：bowl held")).toBeTruthy();
  expect(screen.queryByLabelText("子任务")).toBeNull();
  expect(screen.queryByText(/已定位/)).toBeNull();
  expect(screen.queryByText(/粗采样 1 Hz/)).toBeNull();
  expect(screen.queryByText(/正在定位/)).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "查看拆解输入首帧" }));
  expect((await screen.findByRole("img")).getAttribute("src")).toContain("/agentview_image/0");
});

it("keeps a rejected plan and its contract diagnosable without claiming completion", async () => {
  vi.mocked(lab.listExperiments).mockResolvedValue([{ ...experiment, status: "FAILED" }]);
  vi.mocked(lab.getLabResult).mockResolvedValue({ ...result, schema_version: 7, mode: "plan_only", status: "FAILED", plan: null,
    task_contract: { requirements: [{ id: "r1", instruction_span: "close the drawer", condition: "drawer closed", depends_on: [] }] },
    error: "Missing required outcome r1", goal_ranges: [] });
  render(<AnnotationLabPage />);
  expect(await screen.findByText("drawer closed")).toBeTruthy();
  expect(screen.getByText(/未生成通过校验的任务拆解/)).toBeTruthy();
  expect(screen.getByText("Missing required outcome r1")).toBeTruthy();
  expect(screen.queryByText(/任务拆解已完成/)).toBeNull();
  expect(screen.queryByText(/正在定位/)).toBeNull();
});

it("distinguishes member outcomes from a failed batch", async () => {
  const batch: lab.Experiment = { ...experiment, status: "FAILED", run_ids: ["one", "two", "three"], progress: {
    stage: "finished", completed_runs: 2, failed_runs: 1, total_runs: 3, current_run: null, calls: 6,
    elapsed_seconds: 30, estimated_remaining_seconds: null, window_index: 0, window_total: 0, error: null,
    runs: [{ run_id: "one", status: "COMPLETED", error: null }, { run_id: "two", status: "COMPLETED", error: null },
      { run_id: "three", status: "FAILED", error: "bad output" }],
  } };
  vi.mocked(lab.listExperiments).mockResolvedValue([batch]);
  render(<AnnotationLabPage />);
  expect(await screen.findByRole("option", { name: "one · COMPLETED" })).toBeTruthy();
  expect(screen.getByRole("option", { name: "two · COMPLETED" })).toBeTruthy();
  expect(screen.getByRole("option", { name: "three · FAILED" })).toBeTruthy();
  expect(screen.getByText(/批次状态：FAILED/)).toBeTruthy();
  expect(lab.memberStatus(experiment, "one")).toBe("批次 COMPLETED · 单条状态未知");
  expect(lab.memberStatus({ ...batch, progress: { ...batch.progress!, runs: [] } }, "one")).toBe("未处理");
  expect(lab.memberStatus({ ...batch, status: "RUNNING", progress: { ...batch.progress!, runs: [], current_run: "one" } }, "one")).toBe("RUNNING");
});

it("does not install stale refresh results after switching recordings", async () => {
  vi.mocked(lab.listExperiments).mockResolvedValue([experiment]);
  render(<AnnotationLabPage />);
  await screen.findByLabelText("子任务");
  let finish: (value: lab.Experiment) => void = () => {};
  vi.mocked(lab.getExperiment).mockReturnValueOnce(new Promise((resolve) => { finish = resolve; }));
  fireEvent.click(screen.getByRole("button", { name: "刷新过程结果" }));
  vi.mocked(lab.getLabResult).mockResolvedValue({ ...result, source: { prompt: "second recording" } });
  fireEvent.change(screen.getByLabelText("结果轨迹"), { target: { value: "two" } });
  await screen.findByText(/second recording/);
  finish(experiment);
  await waitFor(() => expect((screen.getByLabelText("结果轨迹") as HTMLSelectElement).value).toBe("two"));
  expect(screen.getByText(/second recording/)).toBeTruthy();
});
