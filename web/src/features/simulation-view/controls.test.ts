import { describe, expect, it } from "vitest";
import {
  canStartDraft,
  selectMainVideoArtifact,
  stepToVideoTime,
  videoTimeToStep,
} from "./controls";

describe("terminal trajectory video", () => {
  it("prefers the human-facing agentview recording", () => {
    expect(selectMainVideoArtifact({
      "vla_views.mp4": "/vla",
      "agentview.mp4": "/main",
    })).toBe("agentview.mp4");
    expect(selectMainVideoArtifact({
      "episode_000_failure_vla_views.mp4": "/vla",
      "episode_000_failure.mp4": "/main",
    })).toBe("episode_000_failure.mp4");
  });

  it("maps timeline states to valid encoded frames", () => {
    expect(stepToVideoTime(50, 300, 20)).toBe(2.5);
    expect(stepToVideoTime(300, 300, 20)).toBe(15);
    expect(videoTimeToStep(2.5, 301, 20)).toBe(50);
    expect(videoTimeToStep(15, 301, 20)).toBe(300);
  });
});

describe("simulation draft", () => {
  it("only enables start after preview readiness", () => {
    expect(canStartDraft(false, false, false)).toBe(false);
    expect(canStartDraft(true, true, false)).toBe(false);
    expect(canStartDraft(true, false, true)).toBe(false);
    expect(canStartDraft(true, false, false)).toBe(true);
  });
});
