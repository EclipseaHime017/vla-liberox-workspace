import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RunTable } from "./RunTable";
import type { Session } from "../run-control/types";

const run = (patch: Partial<Session>): Session => ({
  id: "run", kind: "original", task_id: "task", level: "LEVEL1",
  task_name: "task", task: "pick", parent_session_id: null,
  control_mode: "policy", manual_source: null, policy_id: "base",
  policy_label: "base", policy_base_checkpoint: "base", policy_overlay: null,
  policy_compatibility_sha256: null, manual_translation_gain: null,
  manual_rotation_gain: null, spacemouse_status: null, spacemouse_connected: null,
  spacemouse_stale: null, spacemouse_latency_ms: null, spacemouse_deadman_ms: null,
  status: "COMPLETED", created_at: "2026-08-24T00:00:00Z", max_steps: 300,
  open_loop_steps: 8, current_step: 300, state_count: 301, seed: 7,
  init_state_index: 0, disabled_policy_cameras: [], action_count: 300,
  policy_queries: 38, success: false, error: null, stopped_reason: null,
  measured_control_hz: 20, simulated_duration_seconds: 15, branchable: true,
  legacy: false, managed: true, preparation_phase: null,
  preparation_message: null, countdown_remaining: null, preview_ready: true,
  preparation_timing: {}, artifacts: {}, training_eligible: true,
  training_start_step: 0, training_action_count: 300, training_chunk_count: 38,
  source_type: "inference", outcome: "failure", ...patch,
});

describe("training run catalog", () => {
  it("distinguishes manual and policy-requery suffixes and disables invalid rows", () => {
    const onToggle = vi.fn();
    render(<RunTable selectable onToggle={onToggle} runs={[
      run({ id: "manual", kind: "branch", control_mode: "manual", source_type: "manual", resume_step: 40, training_start_step: 0, training_action_count: 300 }),
      run({ id: "retry", kind: "branch", source_type: "policy_requery", resume_step: 80, training_start_step: 0, training_action_count: 300 }),
      run({ id: "bad", status: "ERROR", source_type: "incomplete", training_eligible: false, ineligible_reason: "运行未正常完成" }),
    ]} />);
    expect(screen.getByText("人工接管")).toBeTruthy();
    expect(screen.getByText("二次推理")).toBeTruthy();
    expect(screen.getByText("接管点 40 · 保留完整轨迹")).toBeTruthy();
    const bad = screen.getByLabelText("选择 bad") as HTMLInputElement;
    expect(bad.disabled).toBe(true);
    fireEvent.click(screen.getByLabelText("选择 manual"));
    expect(onToggle).toHaveBeenCalledWith("manual", true);
  });
});
