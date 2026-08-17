import type { Draft, TaskInfo } from "../run-control/types";
import { canStartDraft } from "../simulation-view/controls";
import { Button } from "../../components/ui/Button";
import { Input } from "../../components/ui/Input";
import { TaskSelector } from "./TaskSelector";

type Props = {
  draft: Draft | null; tasks: TaskInfo[]; active: boolean; busy: boolean;
  taskId: string; maxSteps: number; openLoop: number;
  onCreate: () => void; onStart: () => void; onCancel: () => void; onStop: () => void;
  onTask: (value: string) => void; onMaxSteps: (value: number, commit: boolean) => void;
  onOpenLoop: (value: number, commit: boolean) => void;
};

export function RunConfigForm(props: Props) {
  return <div className="new-run">
    {!props.draft ? <Button className="primary create-draft" disabled={props.active || props.busy} onClick={props.onCreate}>创建仿真</Button> : <>
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
