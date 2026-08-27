import { describe, expect, it } from "vitest";
import { chunkRewardSeries, observationPotentialSeries } from "./TrajectoryDetail";

describe("trajectory chunk reward", () => {
  it("shows one reward sample per chunk at its completion time", () => {
    const result = chunkRewardSeries(
      [0, 3], [3, 5], [-2.5, 1.25], [0, 0.05, 0.1, 0.15, 0.2, 0.25],
    );
    expect(result.times).toEqual([0.15, 0.25]);
    expect(result.values).toEqual([[-2.5], [1.25]]);
    expect(result.sampleLabels).toEqual([
      "chunk 0 · steps [0, 3) · L=3",
      "chunk 1 · steps [3, 5) · L=2",
    ]);
  });

  it("rejects inconsistent chunk metadata", () => {
    expect(() => chunkRewardSeries([0], [3, 5], [-1], [])).toThrow(
      "chunk reward metadata length mismatch",
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
