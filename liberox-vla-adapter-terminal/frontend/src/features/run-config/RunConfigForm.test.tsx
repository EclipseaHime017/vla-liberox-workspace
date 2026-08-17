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
        end_step: 300,
        open_loop_steps: 4,
      }}
      tasks={[]}
      active={false}
      busy={false}
      taskId="task"
      maxSteps={300}
      openLoop={8}
      onCreate={vi.fn()}
      onStart={vi.fn()}
      onCancel={vi.fn()}
      onStop={vi.fn()}
      onTask={vi.fn()}
      onMaxSteps={vi.fn()}
      onOpenLoop={vi.fn()}
      onBranchOpenLoop={onBranchOpenLoop}
      onStartBranch={onStartBranch}
      onCancelBranch={vi.fn()}
    />);

    expect((screen.getByLabelText(/任务（继承源会话）/) as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText(/源 Episode/) as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText(/回溯帧/) as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText(/原轨迹结束步/) as HTMLInputElement).disabled).toBe(true);

    const stepInput = screen.getByLabelText(/每次预测执行步数/) as HTMLInputElement;
    expect(stepInput.disabled).toBe(false);
    fireEvent.change(stepInput, { target: { value: "3" } });
    expect(onBranchOpenLoop).toHaveBeenCalledWith(3);

    fireEvent.click(screen.getByRole("button", { name: "开始二次推理" }));
    expect(onStartBranch).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: "创建仿真" })).toBeNull();
  });
});
