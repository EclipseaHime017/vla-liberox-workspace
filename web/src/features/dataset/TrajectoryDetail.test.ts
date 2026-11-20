import { describe, expect, it } from "vitest";
import { actionRewardSeries, observationPotentialSeries } from "./TrajectoryDetail";

describe("trajectory per-action reward", () => {
  it("shows one reward sample per action at the time it was taken", () => {
    const result = actionRewardSeries([-2.5, 1.25], [0, 0.05]);
    expect(result.times).toEqual([0, 0.05]);
    expect(result.values).toEqual([[-2.5], [1.25]]);
    expect(result.sampleLabels).toEqual(["action step 0", "action step 1"]);
  });

  it("rejects rewards that do not match the recorded actions", () => {
    expect(() => actionRewardSeries([-1], [0, 0.05])).toThrow(
      "per-action reward length does not match the recorded actions",
    );
  });
});

describe("RynnValue observation potential", () => {
  it("is the sign-reversed absolute remaining time for every official head", () => {
    expect(observationPotentialSeries([[4.5, 3], [1.25, 0], [0, 0]])).toEqual([
      [-4.5, -3], [-1.25, 0], [0, 0],
    ]);
  });
});
