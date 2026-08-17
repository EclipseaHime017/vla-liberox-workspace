import type { Draft, PolicyBranchDraft, TaskInfo } from "../run-control/types";
import { canStartDraft } from "../simulation-view/controls";
import { Button } from "../../components/ui/Button";
import { Input } from "../../components/ui/Input";
import { TaskSelector } from "./TaskSelector";

type Props = {
  draft: Draft | null; branchDraft: PolicyBranchDraft | null;
  tasks: TaskInfo[]; active: boolean; busy: boolean;
  taskId: string; maxSteps: number; openLoop: number;
  onCreate: () => void; onStart: () => void; onCancel: () => void; onStop: () => void;
  onTask: (value: string) => void; onMaxSteps: (value: number, commit: boolean) => void;
  onOpenLoop: (value: number, commit: boolean) => void;
  onBranchOpenLoop: (value: number) => void;
  onStartBranch: () => void; onCancelBranch: () => void;
};

export function RunConfigForm(props: Props) {
  return <div className="new-run">
    {props.branchDraft ? <>
      <div className="branch-config-title">
        <strong>二次推理配置</strong>
        <span>仅可调整执行步数</span>
      </div>
      <label className="locked-field">任务（继承源会话）
        <Input value={props.branchDraft.task_prompt} disabled />
      </label>
      <label className="locked-field">源 Episode
        <Input value={props.branchDraft.source_episode} disabled />
      </label>
      <label className="locked-field">回溯帧
        <Input value={props.branchDraft.resume_step} disabled />
      </label>
      <label className="locked-field">原轨迹结束步
        <Input value={props.branchDraft.end_step} disabled />
      </label>
      <label className="branch-step-field">每次预测执行步数
        <Input
          type="number"
          min={1}
          max={8}
          value={props.branchDraft.open_loop_steps}
          disabled={props.active || props.busy}
          onChange={(event) => props.onBranchOpenLoop(Number(event.target.value))}
        />
      </label>
      <p className="locked-field-note">任务、Episode、回溯帧和轨迹总长度由源会话锁定，二次推理仍从所选帧直接恢复状态。</p>
      <div className="button-row">
        <Button className="primary" disabled={props.active || props.busy || !Number.isInteger(props.branchDraft.open_loop_steps) || props.branchDraft.open_loop_steps < 1 || props.branchDraft.open_loop_steps > 8} onClick={props.onStartBranch}>开始二次推理</Button>
        <Button disabled={props.active || props.busy} onClick={props.onCancelBranch}>取消配置</Button>
      </div>
    </> : !props.draft ? <Button className="primary create-draft" disabled={props.active || props.busy} onClick={props.onCreate}>创建仿真</Button> : <>
      <label>任务场景<TaskSelector tasks={props.tasks} value={props.taskId} disabled={props.active || props.busy} onChange={props.onTask} /></label>
      <label>总控制步数<Input type="number" min={1} max={10000} value={props.maxSteps} disabled={props.active || props.busy} onChange={(event) => props.onMaxSteps(Number(event.target.value), false)} onBlur={() => props.onMaxSteps(props.maxSteps, true)} /></label>
      <label>每次预测执行步数<Input type="number" min={1} max={8} value={props.openLoop} disabled={props.active || props.busy} onChange={(event) => props.onOpenLoop(Number(event.target.value), false)} onBlur={() => props.onOpenLoop(props.openLoop, true)} /></label>
      <div className="button-row">
        <Button className="primary" disabled={!canStartDraft(props.draft.preview_ready, props.active, props.busy)} onClick={props.onStart}>开始仿真</Button>
        <Button disabled={props.active || props.busy} onClick={props.onCancel}>取消草稿</Button>
      </div>
    </>}
    <Button className="danger" disabled={!props.active} onClick={props.onStop}>停止当前仿真</Button>
  </div>;
}
