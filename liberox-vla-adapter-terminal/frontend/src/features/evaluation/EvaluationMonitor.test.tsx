import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
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
vi.mock("../run-control/api", () => ({ stopEvaluation: vi.fn() }));
import { EvaluationMonitor } from "./EvaluationMonitor";
import { stopEvaluation } from "../run-control/api";

const job: OfflineJob = {
  id: "eval-1", kind: "evaluation", status: "RUNNING", dataset_id: null,
  created_at: "now", started_at: "now", completed_at: null, stage: "evaluate",
  stage_label: "批量测试", error: null, output_path: "/tmp/eval",
  parameters: { trials: 100 }, log_size: 0,
};

describe("evaluation monitor", () => {
  afterEach(cleanup);
  it("ignores a stop response after switching to another monitored test", async () => {
    let resolve!: (value: OfflineJob) => void;
    vi.mocked(stopEvaluation).mockImplementation(() => new Promise((done) => { resolve = done; }));
    const onUpdate = vi.fn();
    const view = render(<EvaluationMonitor initial={job} onUpdate={onUpdate} />);
    fireEvent.click(screen.getByRole("button", { name: "停止测试" }));
    view.rerender(<EvaluationMonitor initial={{ ...job, id: "next" }} onUpdate={onUpdate} />);
    await act(async () => resolve({ ...job, status: "STOPPING" }));
    expect(screen.getByText("RUNNING")).toBeTruthy();
    expect(screen.getByText("next")).toBeTruthy();
    expect(onUpdate).not.toHaveBeenCalled();
  });
  it("accepts polled state updates without losing logs or reconnecting", () => {
    const view = render(<EvaluationMonitor initial={{ ...job, status: "QUEUED" }} />);
    expect(screen.getByRole("button", { name: "取消排队" })).toBeTruthy();
    const connected = socket;
    act(() => connected.onmessage?.({ data: JSON.stringify({ logs: { text: "saved log\n" } }) }));
    view.rerender(<EvaluationMonitor initial={{ ...job, status: "CANCELED" }} />);
    expect(screen.getByText("CANCELED")).toBeTruthy();
    expect(screen.getByText("saved log")).toBeTruthy();
    expect(socket).toBe(connected);
    expect(screen.queryByRole("button", { name: "取消排队" })).toBeNull();
  });

  it("ignores messages from a previous job socket and uses the current callback", () => {
    const oldCallback = vi.fn(), newCallback = vi.fn();
    const view = render(<EvaluationMonitor initial={job} onUpdate={oldCallback} />);
    const previous = socket;
    const next = { ...job, id: "next" };
    view.rerender(<EvaluationMonitor initial={next} onUpdate={newCallback} />);
    act(() => previous.onmessage?.({ data: JSON.stringify({ job }) }));
    expect(oldCallback).not.toHaveBeenCalled();
    expect(newCallback).not.toHaveBeenCalled();
    act(() => socket.onmessage?.({ data: JSON.stringify({ job: next }) }));
    expect(newCallback).toHaveBeenCalledWith(next);
  });
  it("shows streamed schedule progress and live success rate", () => {
    render(<EvaluationMonitor initial={job} />);
    act(() => socket.onmessage?.({ data: JSON.stringify({
      job: { ...job, metrics: {
        attempted_trials: 25, total_trials: 100, successes: 15,
        current_trial: 26,
        init_state_index: 4, seed: 12, measured_control_hz: 19.8,
        elapsed_seconds: 90, estimated_remaining_seconds: 270,
      } },
      logs: { text: '{"level":"INFO","message":"完成回合 25"}\n' },
    }) }));
    expect(screen.getByText("26 / 100")).toBeTruthy();
    expect(screen.getByText(/已完成 25/)).toBeTruthy();
    expect(screen.getByText("60.0%")).toBeTruthy();
    expect(screen.getByText("#4")).toBeTruthy();
    expect(screen.getByText("完成回合 25")).toBeTruthy();
  });
});
