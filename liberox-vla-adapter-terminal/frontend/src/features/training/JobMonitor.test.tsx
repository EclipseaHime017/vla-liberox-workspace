import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
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
afterEach(cleanup);

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

  it("allows a terminal record to be dismissed", () => {
    const onDismiss = vi.fn();
    render(<JobMonitor initial={{ ...job, status: "COMPLETED" }} onDismiss={onDismiss} />);
    screen.getByRole("button", { name: "关闭记录" }).click();
    expect(onDismiss).toHaveBeenCalledOnce();
  });

  it("uses the latest completion callback without reconnecting the job socket", () => {
    const oldCallback = vi.fn(); const nextCallback = vi.fn();
    const view = render(<JobMonitor initial={job} onUpdate={oldCallback} />);
    const originalSocket = socket;
    view.rerender(<JobMonitor initial={job} onUpdate={nextCallback} />);
    const completed = { ...job, status: "COMPLETED" };
    act(() => socket.onmessage?.({ data: JSON.stringify({ job: completed }) }));
    expect(socket).toBe(originalSocket);
    expect(originalSocket.close).not.toHaveBeenCalled();
    expect(oldCallback).not.toHaveBeenCalled();
    expect(nextCallback).toHaveBeenCalledWith(completed);
  });

  it("shows a global synchronization warning while keeping dataset evaluation completed", () => {
    const onUpdate = vi.fn();
    render(<JobMonitor initial={{ ...job, kind: "annotation" }} onUpdate={onUpdate} onDismiss={() => {}} />);
    const completed: OfflineJob = { ...job, kind: "annotation", status: "COMPLETED",
      stage_label: "评价成功，全局结果同步失败", warning: "数据集结果已保存，但全局评价同步失败：目标目录不可写" };
    act(() => socket.onmessage?.({ data: JSON.stringify({ job: completed }) }));
    expect(screen.getByRole("alert").textContent).toBe(completed.warning);
    expect(screen.getByText("COMPLETED")).toBeTruthy();
    expect(screen.queryByText("FAILED")).toBeNull();
    expect(screen.getByRole("button", { name: "关闭记录" })).toBeTruthy();
    expect(onUpdate).toHaveBeenCalledWith(completed);
  });
});
