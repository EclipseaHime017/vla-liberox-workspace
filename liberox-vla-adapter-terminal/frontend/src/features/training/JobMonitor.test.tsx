import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { OfflineJob } from "../run-control/types";

let socket: { onmessage: ((event: { data: string }) => void) | null; onerror: (() => void) | null; close: ReturnType<typeof vi.fn> } = {
  onmessage: null, onerror: null, close: vi.fn(),
};
vi.mock("../../api/websocket", () => ({
  jobWebSocket: () => {
    socket = { onmessage: null, onerror: null, close: vi.fn() };
    return socket;
  },
}));
import { JobMonitor } from "./JobMonitor";

const job: OfflineJob = {
  id: "train-1", kind: "training", status: "RUNNING", dataset_id: "ds",
  created_at: "now", started_at: "now", completed_at: null, stage: "train",
  stage_label: "Pixel-IQL", error: null, output_path: "/tmp/output",
  parameters: { critic_warmup_steps: 5 }, log_size: 0,
};

describe("offline job monitor", () => {
  it("restores streamed logs and exposes training metrics", () => {
    render(<JobMonitor initial={job} />);
    act(() => socket.onmessage?.({ data: JSON.stringify({
      type: "job",
      job: { ...job, metrics: {
        step: 8, progress_percent: 40, steps_per_second: 0.5,
        elapsed_seconds: 16, estimated_remaining_seconds: 24,
        estimated_completion_time: "2026-08-24T12:00:00+08:00",
        q_loss: 1.2, value_loss: 0.3, actor_loss: 0.2,
        q_mean: 1, value_mean: 0.8, advantage_mean: 0.2,
        advantage_weight_mean: 2, actor_learning_rate: 1e-4,
        actor_grad_norm: 0.7, cuda_peak_memory_gib: 12.5,
      } },
      logs: { text: '{"level":"INFO","message":"加载 VLA"}\n' },
    }) }));
    expect(screen.getByText("IQL")).toBeTruthy();
    expect(screen.getByText("加载 VLA")).toBeTruthy();
    expect(screen.getByText("Q loss")).toBeTruthy();
    expect(screen.getByText("12.5 GiB")).toBeTruthy();
  });
});
