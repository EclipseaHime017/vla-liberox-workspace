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
vi.mock("../run-control/api", () => ({ stopEvaluation: vi.fn() }));
import { EvaluationMonitor } from "./EvaluationMonitor";

const job: OfflineJob = {
  id: "eval-1", kind: "evaluation", status: "RUNNING", dataset_id: null,
  created_at: "now", started_at: "now", completed_at: null, stage: "evaluate",
  stage_label: "批量测试", error: null, output_path: "/tmp/eval",
  parameters: { trials: 100 }, log_size: 0,
};

describe("evaluation monitor", () => {
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
