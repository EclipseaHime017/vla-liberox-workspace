import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RunConfigForm } from "./RunConfigForm";

describe("policy branch configuration", () => {
  it("locks source context and starts only after the user confirms the execution step count", () => {
    const onBranchOpenLoop = vi.fn();
    const onStartBranch = vi.fn();
    render(<RunConfigForm
      draft={null}
      branchDraft={{
        parent_session_id: "source-session",
        task_prompt: "place the black bowl on the flat stove",
        source_episode: "episode_000 · source-session",
        resume_step: 50,
        target_steps: 300,
        open_loop_steps: 4,
        policy_id: "base",
        policy_label: "Object-Pro base",
      }}
      tasks={[]}
      policies={[]}
      active={false}
      busy={false}
      taskId="task"
      policyId="base"
      maxSteps={300}
      openLoop={8}
      seed={0}
      disabledPolicyCameras={[]}
      onCreate={vi.fn()}
      onStart={vi.fn()}
      onCancel={vi.fn()}
      onStop={vi.fn()}
      onTask={vi.fn()}
      onPolicy={vi.fn()}
      onMaxSteps={vi.fn()}
      onOpenLoop={vi.fn()}
      onSeed={vi.fn()}
      onPolicyCamera={vi.fn()}
      onBranchOpenLoop={onBranchOpenLoop}
      onStartBranch={onStartBranch}
      onCancelBranch={vi.fn()}
    />);

    expect((screen.getByLabelText(/任务（继承源会话）/) as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText(/源 Episode/) as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText(/回溯帧/) as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText(/目标总控制步数/) as HTMLInputElement).disabled).toBe(true);

    const stepInput = screen.getByLabelText(/每次预测执行步数/) as HTMLInputElement;
    expect(stepInput.disabled).toBe(false);
    fireEvent.change(stepInput, { target: { value: "3" } });
    expect(onBranchOpenLoop).toHaveBeenCalledWith(3);

    fireEvent.click(screen.getByRole("button", { name: "开始二次推理" }));
    expect(onStartBranch).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: "创建仿真" })).toBeNull();
  });
});

describe("original simulation configuration", () => {
  it("edits the random seed and policy camera selection in a draft", () => {
    const onSeed = vi.fn();
    const onPolicyCamera = vi.fn();
    render(<RunConfigForm
      draft={{
        id: "draft", task_id: "task", max_steps: 300, open_loop_steps: 8,
        seed: 7, disabled_policy_cameras: [], policy_id: "base", policy_label: "Base",
        preview_status: "READY", preview_revision: 1, preview_ready: true,
        preview_available: true, error: null,
        task: { task_id: "task", level: "LEVEL1", task_name: "task", prompt: "pick", init_state_index: 0 },
      }}
      branchDraft={null}
      tasks={[{ task_id: "task", level: "LEVEL1", task_name: "task", prompt: "pick", init_state_index: 0 }]}
      policies={[{ policy_id: "base", label: "Base", base_checkpoint: "base", stats_key: "stats", kind: "base", training_step: null, compatibility_sha256: null }]}
      active={false} busy={false} taskId="task" policyId="base"
      maxSteps={300} openLoop={8} seed={7} disabledPolicyCameras={[]}
      onCreate={vi.fn()} onStart={vi.fn()} onCancel={vi.fn()} onStop={vi.fn()}
      onTask={vi.fn()} onPolicy={vi.fn()} onMaxSteps={vi.fn()} onOpenLoop={vi.fn()}
      onSeed={onSeed} onPolicyCamera={onPolicyCamera}
      onBranchOpenLoop={vi.fn()} onStartBranch={vi.fn()} onCancelBranch={vi.fn()}
    />);

    fireEvent.change(screen.getByLabelText("随机种子"), { target: { value: "42" } });
    expect(onSeed).toHaveBeenCalledWith(42, false);
    fireEvent.click(screen.getByLabelText("腕部视角"));
    expect(onPolicyCamera).toHaveBeenCalledWith("robot0_eye_in_hand", false);
  });
});
