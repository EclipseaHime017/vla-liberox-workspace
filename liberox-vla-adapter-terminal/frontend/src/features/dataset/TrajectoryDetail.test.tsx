import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrajectoryDetail } from "./TrajectoryDetail";
import { getStageAnnotation, saveStageAnnotation } from "../run-control/api";
import type { StageAnnotation, TrajectoryDetail as Detail } from "../run-control/types";

vi.mock("../run-control/api", () => ({ getStageAnnotation: vi.fn(), saveStageAnnotation: vi.fn() }));
const original: StageAnnotation = {
  run_id: "episode", status: "missing", action_count: 2, time_seconds: [0, .05, .1],
  success_step: null, success_consecutive_steps: 5, exponent: 2, keyframes: [], scores: [], revision: "rev1",
};
const detail = {
  run: { id: "episode", action_count: 2, success: false, task: "pick bowl" },
  artifacts: { "agentview.mp4": "/main-video.mp4" },
  series: { time_seconds: [0, .05, .1], action_time_seconds: [0, .05], env_action: [Array(7).fill(0), Array(7).fill(0)], done: [false, false, false] },
  evaluation: null, rynnvalue_evaluation: null, robometer_evaluation: null,
} as unknown as Detail;

beforeEach(() => {
  vi.mocked(getStageAnnotation).mockResolvedValue(original);
  vi.mocked(saveStageAnnotation).mockResolvedValue({ ...original, status: "ready", revision: "rev2",
    keyframes: [{ step: 1, kind: "negative" }], scores: [-1, -2, -2] });
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.clearAllMocks(); });

describe("trajectory detail stage integration", () => {
  it("updates the reward plot without clipping or replacing the playing video element", async () => {
    const rendered = render(<TrajectoryDetail detail={detail} onBack={() => {}} />);
    await screen.findByText("尚未保存阶段标注");
    const video = rendered.container.querySelector("video")!;
    Object.defineProperty(video, "duration", { configurable: true, value: .1 });
    fireEvent.loadedMetadata(video);
    fireEvent.click(screen.getByRole("button", { name: "切片 / 标记关键帧" }));
    fireEvent.change(screen.getByRole("slider"), { target: { value: "1" } });
    fireEvent.change(screen.getByLabelText("关键帧类型"), { target: { value: "negative" } });
    fireEvent.click(screen.getByRole("button", { name: "标记当前帧" }));
    fireEvent.click(screen.getByRole("button", { name: "保存关键帧" }));
    const plot = (await screen.findByText("关键帧奖励预览（未评价）")).closest("article")!;
    expect(within(plot).getByText("-2.00")).toBeTruthy();
    expect(rendered.container.querySelector("video")).toBe(video);
    expect(video.getAttribute("src")).toBe("/main-video.mp4");
    expect(video.currentTime).toBeCloseTo(.05);
    await waitFor(() => expect(getStageAnnotation).toHaveBeenCalledTimes(1));
  });

  it("refreshes current dataset plots without exposing versions or losing unsaved marks and video", async () => {
    const onContextChange = vi.fn();
    const version = (id: string, exponent: number, scores: number[]): Detail => ({
      ...detail,
      dataset_context: { dataset_id: "data-a", dataset_name: "Dataset A", version_id: id, source: "stage", config: { stage_exponent: exponent } },
      available_dataset_contexts: [{ dataset_id: "data-a", dataset_name: "Dataset A", reward_version_id: "v2", robometer_version_id: null,
        versions: ["v1", "v2"].map((value) => ({ id: value, evaluator: "stage", status: "COMPLETED", parameters: {}, created_at: "" })) }],
      reward_evaluation: { status: "READY", source: "stage", version_id: id, reward_config: { stage_exponent: exponent }, boundary_steps: [0, 2],
        chunk_lengths: [2], chunk_start_steps: [0], chunk_end_steps: [2], final_reward: [scores[2]],
        time_seconds: [0, .05, .1], stage_scores: scores },
    });
    const rendered = render(<TrajectoryDetail detail={version("v1", 2, [-1, -.5, 0])} onBack={() => {}} onContextChange={onContextChange} />);
    await screen.findByText("尚未保存阶段标注");
    const video = rendered.container.querySelector("video")!;
    Object.defineProperty(video, "duration", { configurable: true, value: .1 });
    fireEvent.loadedMetadata(video);
    fireEvent.click(screen.getByRole("button", { name: "切片 / 标记关键帧" }));
    fireEvent.change(screen.getByRole("slider"), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "标记当前帧" }));
    expect(screen.queryByLabelText("评价版本")).toBeNull();
    expect(screen.queryByText(/v1|v2/)).toBeNull();
    fireEvent.change(screen.getByLabelText("数据来源"), { target: { value: "" } });
    expect(onContextChange).toHaveBeenCalledWith(undefined);
    rendered.rerender(<TrajectoryDetail detail={version("v2", 4, [-1, -.75, 0])} onBack={() => {}} onContextChange={onContextChange} />);
    expect(rendered.container.querySelector("video")).toBe(video);
    expect(video.currentTime).toBeCloseTo(.05);
    expect(screen.getByRole("button", { name: "删除帧 1" })).toBeTruthy();
    expect(screen.getByText("有未保存的修改")).toBeTruthy();
    expect(screen.getByText("插值指数 p 4")).toBeTruthy();
    expect(getStageAnnotation).toHaveBeenCalledTimes(1);
    expect(saveStageAnnotation).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "查看Stage-based Reward逐时间点数据" }));
    const dialog = screen.getByRole("dialog");
    fireEvent.change(within(dialog).getByRole("slider"), { target: { value: "1" } });
    expect(within(dialog).getByText("-0.75000")).toBeTruthy();
    rendered.rerender(<TrajectoryDetail detail={version("v3", 3, [-1, -.625, 0])} onBack={() => {}} onContextChange={onContextChange} />);
    expect(within(screen.getByRole("dialog")).getByText("-0.62500")).toBeTruthy();
    expect(rendered.container.querySelector("video")).toBe(video);
    expect(screen.getByText("有未保存的修改")).toBeTruthy();
  });

  it("shows the default global evaluation and readable parameters without a history selector", async () => {
    vi.mocked(getStageAnnotation).mockResolvedValue({ ...original, status: "ready", scores: [-1, -.5, 0] });
    render(<TrajectoryDetail detail={{ ...detail, global_evaluation: { source: "stage", config: { stage_exponent: 3, gamma: .95 }, origin: "first_evaluation" } }}
      onBack={() => {}} onContextChange={() => {}} />);
    await waitFor(() => expect(getStageAnnotation).toHaveBeenCalled());
    expect((screen.getByLabelText("数据来源") as HTMLSelectElement).value).toBe("");
    expect(screen.getByText("插值指数 p 3")).toBeTruthy();
    expect(screen.getByText("折扣 γ 0.95")).toBeTruthy();
    expect(screen.queryByLabelText("评价版本")).toBeNull();
    expect(screen.queryByText(/stage_exponent=/)).toBeNull();
    expect(screen.queryByText("关键帧奖励预览（未评价）")).toBeNull();
  });

  it("keeps raw Rynn diagnostics but only displays the selected Stage training reward", async () => {
    const rynn = {
      status: "READY", boundary_steps: [0, 2],
      official_outputs: { absolute_temporal_distance_seconds: [[2], [0]], relative_temporal_distance_seconds: [0, -2], absolute_value_entropy_nats: [[.4], [.2]] },
      pbrs_reward: { chunk_start_steps: [0], chunk_end_steps: [2], shape_reward: [2], sparse_reward: [-1], dense_reward: [.2], final_reward: [-.8] },
    } as NonNullable<Detail["rynnvalue_evaluation"]>;
    render(<TrajectoryDetail detail={{ ...detail, rynnvalue_evaluation: rynn,
      global_evaluation: { source: "stage", config: { stage_exponent: 2 } },
      reward_evaluation: { status: "READY", source: "stage", version_id: "internal-id", reward_config: {}, boundary_steps: [0, 2],
        chunk_start_steps: [0], chunk_end_steps: [2], chunk_lengths: [2], final_reward: [0], stage_scores: [-1, -.5, 0] },
    }} onBack={() => {}} />);
    await screen.findByText("尚未保存阶段标注");
    expect(screen.getByText("Stage-based Reward")).toBeTruthy();
    expect(screen.getByText("Stage-based · Final Reward · 宏动作")).toBeTruthy();
    expect(screen.getByText("RynnValue Absolute Remaining Time")).toBeTruthy();
    expect(screen.queryByText(/Reward Components/)).toBeNull();
    expect(screen.queryByText("关键帧奖励预览（未评价）")).toBeNull();
  });

  it("reports an invalid global evaluation instead of substituting a fresh Stage preview", async () => {
    vi.mocked(getStageAnnotation).mockResolvedValue({ ...original, status: "ready", scores: [-1, -.5, 0] });
    render(<TrajectoryDetail detail={{ ...detail, global_evaluation_error: "源数据变化或评价损坏，请手动重新评价覆盖" }} onBack={() => {}} />);
    expect(screen.getByText("源数据变化或评价损坏，请手动重新评价覆盖")).toBeTruthy();
    await waitFor(() => expect(getStageAnnotation).toHaveBeenCalled());
    expect(screen.queryByText("关键帧奖励预览（未评价）")).toBeNull();
  });
});
