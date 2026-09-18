import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { getEvaluationQueue, getOfflineJob } from "../run-control/api";
import type { EvaluationQueueState, OfflineJob } from "../run-control/types";
import { useEvaluationQueue } from "./useEvaluationQueue";

vi.mock("../run-control/api", () => ({ getEvaluationQueue: vi.fn(), getOfflineJob: vi.fn(),
  enqueueEvaluation: vi.fn(), stopEvaluation: vi.fn() }));
afterEach(cleanup);

it("does not regress a streamed terminal status when an older queue request resolves", async () => {
  const running: OfflineJob = {
    id: "test", kind: "evaluation", status: "RUNNING", dataset_id: null,
    created_at: "2026-09-18T00:00:00Z", started_at: null, completed_at: null,
    stage: "evaluate", stage_label: "测试中", error: null, output_path: "/tmp/test",
    parameters: { trials: 1 }, log_size: 0,
  };
  const snapshot: EvaluationQueueState = { jobs: [running], waiting_reason: null };
  let resolve!: (value: EvaluationQueueState) => void;
  vi.mocked(getEvaluationQueue).mockResolvedValueOnce(snapshot)
    .mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
  vi.mocked(getOfflineJob).mockResolvedValue(running);
  const onChanged = vi.fn();
  const { result } = renderHook(() => useEvaluationQueue(onChanged, vi.fn()));
  await waitFor(() => expect(result.current.job?.status).toBe("RUNNING"));
  await waitFor(() => expect(getEvaluationQueue).toHaveBeenCalledTimes(2), { timeout: 4000 });
  act(() => result.current.update({ ...running, status: "CANCELED" }));
  await act(async () => resolve(snapshot));
  expect(result.current.job?.status).toBe("CANCELED");
  expect(result.current.queue.jobs[0].status).toBe("CANCELED");
  expect(onChanged).toHaveBeenCalledOnce();
});
