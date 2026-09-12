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
    fireEvent.click(screen.getByRole("button", { name: "保存切片并计算奖励" }));
    const plot = (await screen.findByText("Stage-based Reward")).closest("article")!;
    expect(within(plot).getByText("-2.00")).toBeTruthy();
    expect(rendered.container.querySelector("video")).toBe(video);
    expect(video.getAttribute("src")).toBe("/main-video.mp4");
    expect(video.currentTime).toBeCloseTo(.05);
    await waitFor(() => expect(getStageAnnotation).toHaveBeenCalledTimes(1));
  });
});
