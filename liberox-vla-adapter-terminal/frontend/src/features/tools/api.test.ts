import { afterEach, describe, expect, it, vi } from "vitest";
import {
  deleteCriticDiagnostic, getCriticDefaults, getCriticDiagnostic, listCriticDiagnostics,
  startCriticDiagnostic, stopCriticDiagnostic,
} from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("critic diagnostic API", () => {
  it("uses tool-owned endpoints and confirms artifact deletion by job ID", async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetch);
    await getCriticDefaults("dataset / one");
    await listCriticDiagnostics();
    await getCriticDiagnostic("critic / one");
    await stopCriticDiagnostic("critic / one");
    await deleteCriticDiagnostic("critic / one");
    expect(fetch.mock.calls.map(([url]) => url)).toEqual([
      "/api/tools/critic/defaults?dataset_id=dataset+%2F+one", "/api/tools/critic",
      "/api/tools/critic/critic%20%2F%20one", "/api/tools/critic/critic%20%2F%20one/stop",
      "/api/tools/critic/critic%20%2F%20one",
    ]);
    expect(fetch.mock.calls[3][1].method).toBe("POST");
    expect(fetch.mock.calls[4][1]).toMatchObject({ method: "DELETE", body: JSON.stringify({ confirm_job_id: "critic / one" }) });
  });

  it("sends the selected dataset, reward and frozen-model request without a training-method override", async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: "critic-1" }) });
    vi.stubGlobal("fetch", fetch);
    const body = { dataset_id: "dataset", reward_source: "final", task_id: "task", policy_id: "policy",
      max_steps: 300, open_loop_steps: 10, seed: 7, init_state_index: 0, train_steps: 3000, batch_size: 32 };
    expect(await startCriticDiagnostic(body)).toEqual({ id: "critic-1" });
    expect(fetch).toHaveBeenCalledWith("/api/tools/critic", expect.objectContaining({ method: "POST", body: JSON.stringify(body) }));
  });
});
