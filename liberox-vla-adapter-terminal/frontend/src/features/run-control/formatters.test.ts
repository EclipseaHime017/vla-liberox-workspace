import { describe, expect, it } from "vitest";
import { formatComputeDevice } from "./formatters";

describe("compute device display", () => {
  it("shows only the GPU model for CUDA devices", () => {
    expect(formatComputeDevice("cuda:0 (NVIDIA GeForce RTX 5090 Laptop GPU)"))
      .toBe("NVIDIA GeForce RTX 5090 Laptop GPU");
  });

  it("keeps non-CUDA labels unchanged", () => {
    expect(formatComputeDevice("cpu")).toBe("cpu");
  });
});
