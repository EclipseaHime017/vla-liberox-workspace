import { describe, expect, it } from "vitest";
import { formatComputeDevice, runOutcome } from "./formatters";

describe("compute device display", () => {
  it("shows only the GPU model for CUDA devices", () => {
    expect(formatComputeDevice("cuda:0 (NVIDIA GeForce RTX 5090 Laptop GPU)"))
      .toBe("NVIDIA GeForce RTX 5090 Laptop GPU");
  });

  it("keeps non-CUDA labels unchanged", () => {
    expect(formatComputeDevice("cpu")).toBe("cpu");
  });
});

describe("runOutcome", () => {
  it("counts only a confirmed done streak as success", () => {
    expect(runOutcome({ status: "COMPLETED", success: true, success_confirmed: true })).toMatchObject({ label: "成功", confirmed: true });
    expect(runOutcome({ status: "COMPLETED", success: true, success_confirmed: false })).toMatchObject({ label: "未确认", confirmed: false });
    expect(runOutcome({ status: "COMPLETED", success: false, success_confirmed: false }).label).toBe("失败");
    expect(runOutcome({ status: "ERROR", success: true, success_confirmed: true }).label).toBe("错误");
  });
});
