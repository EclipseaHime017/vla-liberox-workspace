import { useRef } from "react";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StageAnnotationPanel, stageVideoStep } from "./StageAnnotationPanel";
import { getStageAnnotation, saveStageAnnotation } from "../run-control/api";
import type { StageAnnotation } from "../run-control/types";

vi.mock("../run-control/api", () => ({ getStageAnnotation: vi.fn(), saveStageAnnotation: vi.fn() }));
const onSaved = vi.fn();
const onDirtyChange = vi.fn();
const annotation: StageAnnotation = {
  run_id: "episode", status: "missing", action_count: 10,
  time_seconds: Array.from({ length: 11 }, (_, i) => i / 20), success_step: null,
  success_consecutive_steps: 5, exponent: 2, keyframes: [], scores: [], revision: "original-token",
};

function Fixture() {
  const ref = useRef<HTMLVideoElement>(null);
  return <><video ref={ref} src="/agentview.mp4" data-testid="video" />
    <StageAnnotationPanel runId="episode" videoRef={ref} onSaved={onSaved} onDirtyChange={onDirtyChange} /></>;
}

async function openEditor() {
  render(<Fixture />);
  await waitFor(() => expect(onSaved).toHaveBeenCalled());
  fireEvent.click(screen.getByRole("button", { name: "切片 / 标记关键帧" }));
  const video = screen.getByTestId("video") as HTMLVideoElement;
  Object.defineProperty(video, "duration", { configurable: true, value: 0.5 });
  fireEvent.loadedMetadata(video);
  return video;
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getStageAnnotation).mockResolvedValue(annotation);
  vi.mocked(saveStageAnnotation).mockImplementation(async (_run, body) => ({
    ...annotation, ...body, status: "ready", revision: "saved-token", scores: Array(11).fill(-1),
  }));
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("stage keyframe editor", () => {
  it("uses the actually displayed frame rather than rounding to the next one", () => {
    expect(stageVideoStep(.074, .5, 10)).toBe(1);
    expect(stageVideoStep(.1, .5, 10)).toBe(2);
    expect(stageVideoStep(.5, .5, 10)).toBe(10);
  });
  it("seeks the existing video at integer steps without refetch or source churn", async () => {
    const video = await openEditor();
    const slider = screen.getByRole("slider", { name: "切片进度" });
    fireEvent.change(slider, { target: { value: "3" } });
    expect(video.currentTime).toBeCloseTo(.15);
    fireEvent.click(screen.getByRole("button", { name: "下一帧" }));
    expect(video.currentTime).toBeCloseTo(.2);
    fireEvent.click(screen.getByRole("button", { name: "上一帧" }));
    expect(video.currentTime).toBeCloseTo(.15);
    expect(video.getAttribute("src")).toBe("/agentview.mp4");
    expect(screen.getByTestId("video")).toBe(video);
    expect(getStageAnnotation).toHaveBeenCalledTimes(1);
    expect(saveStageAnnotation).not.toHaveBeenCalled();
    expect(video.play).not.toHaveBeenCalled();
    fireEvent.change(slider, { target: { value: "10" } });
    expect(video.currentTime).toBe(.5);
    expect(screen.getByText(/原录像没有额外的末端帧/)).toBeTruthy();
  });

  it("follows native playback without forcing pause or play", async () => {
    const video = await openEditor();
    video.currentTime = .25;
    fireEvent.timeUpdate(video);
    expect((screen.getByLabelText("Observation step") as HTMLInputElement).value).toBe("5");
    expect(video.pause).not.toHaveBeenCalled();
    expect(video.play).not.toHaveBeenCalled();
  });

  it("captures native playback's current frame even before the next timeupdate", async () => {
    const video = await openEditor();
    fireEvent.change(screen.getByRole("slider"), { target: { value: "2" } });
    video.currentTime = .374;
    fireEvent.click(screen.getByRole("button", { name: "标记当前帧" }));
    expect(screen.getByRole("button", { name: "删除帧 7" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "删除帧 2" })).toBeNull();
  });

  it("adds, replaces and deletes positive/negative marks then saves with revision", async () => {
    await openEditor();
    fireEvent.change(screen.getByRole("slider"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "标记当前帧" }));
    fireEvent.change(screen.getByLabelText("关键帧类型"), { target: { value: "negative" } });
    fireEvent.click(screen.getByRole("button", { name: "更新当前关键帧" }));
    expect(screen.queryByLabelText("插值指数 p")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "保存关键帧" }));
    await waitFor(() => expect(saveStageAnnotation).toHaveBeenCalledWith("episode", {
      keyframes: [{ step: 3, kind: "negative" }], revision: "original-token",
    }));
    await screen.findByText("关键帧已保存");
    expect(onSaved).toHaveBeenLastCalledWith(expect.objectContaining({ scores: Array(11).fill(-1) }));
    fireEvent.click(screen.getByRole("button", { name: "删除帧 3" }));
    expect(screen.queryByRole("button", { name: "删除帧 3" })).toBeNull();
    expect(screen.getByText("有未保存的修改")).toBeTruthy();
  });

  it("permits explicitly saving an empty failed trajectory", async () => {
    await openEditor();
    fireEvent.click(screen.getByRole("button", { name: "保存关键帧" }));
    await waitFor(() => expect(saveStageAnnotation).toHaveBeenCalledWith("episode", {
      keyframes: [], revision: "original-token",
    }));
  });

  it("protects the automatic success anchor and keeps the recorded tail visible", async () => {
    vi.mocked(getStageAnnotation).mockResolvedValue({ ...annotation, success_step: 8 });
    await openEditor();
    fireEvent.click(screen.getByRole("button", { name: "跳转成功帧" }));
    expect((screen.getByRole("button", { name: "标记当前帧" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/step 8（自动锚点/)).toBeTruthy();
    expect((screen.getByRole("slider") as HTMLInputElement).max).toBe("10");
    fireEvent.change(screen.getByRole("slider"), { target: { value: "7" } });
    expect((screen.getByRole("button", { name: "标记当前帧" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("retains unsaved edits after a conflict and requires explicit reload", async () => {
    vi.mocked(saveStageAnnotation).mockRejectedValue(new Error("409: annotation changed"));
    await openEditor();
    fireEvent.change(screen.getByRole("slider"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "标记当前帧" }));
    fireEvent.click(screen.getByRole("button", { name: "保存关键帧" }));
    await screen.findByText(/409: annotation changed/);
    expect(screen.getByRole("button", { name: "删除帧 3" })).toBeTruthy();
    expect(getStageAnnotation).toHaveBeenCalledTimes(1);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    fireEvent.click(screen.getByRole("button", { name: "重新加载标注" }));
    await waitFor(() => expect(getStageAnnotation).toHaveBeenCalledTimes(2));
  });
  it("reports a derivation error without treating saved keyframes as missing", async () => {
    vi.mocked(getStageAnnotation).mockResolvedValue({ ...annotation, status: "ready",
      keyframes: [{ step: 2, kind: "negative" }], derivation_error: "invalid normalization" });
    await openEditor();
    expect(screen.getByText("关键帧已保存")).toBeTruthy();
    expect(screen.getByText(/奖励预览计算失败：invalid normalization/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "删除帧 2" })).toBeTruthy();
    expect(saveStageAnnotation).not.toHaveBeenCalled();
  });
});
