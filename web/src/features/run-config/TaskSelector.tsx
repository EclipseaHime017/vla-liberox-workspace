import type { TaskInfo } from "../run-control/types";
import { Select } from "../../components/ui/Select";
import { familyId, familyLabel, scopeForTask, selectableTasks, taskLevel, type TaskScope } from "./taskHierarchy";

type HierarchyProps = {
  tasks: TaskInfo[]; value: TaskScope; onChange: (value: TaskScope) => void;
  disabled?: boolean; allowAll?: boolean; disableUnavailable?: boolean; labelPrefix?: string;
};

function TaskHierarchy({ tasks, value, onChange, disabled = false, allowAll = false, disableUnavailable = false, labelPrefix = "" }: HierarchyProps) {
  const options = selectableTasks(tasks);
  const families = [...new Map(options.map((task) => [familyId(task), familyLabel(task)])).entries()];
  const familyTasks = options.filter((task) => familyId(task) === value.family_id);
  const levels = [...new Set(familyTasks.map(taskLevel))].sort();
  const prompts = familyTasks.filter((task) => taskLevel(task) === value.level);
  const usable = (task: TaskInfo) => !disableUnavailable || task.available !== false;
  const choose = (family: string, level = "") => {
    if (allowAll) { onChange({ family_id: family, level, task_id: "" }); return; }
    const candidates = options.filter((task) => familyId(task) === family && usable(task));
    const next = candidates.find((task) => taskLevel(task) === (level || value.level)) ?? candidates[0];
    if (next) onChange(scopeForTask(options, next.task_id));
  };
  return <div className="task-hierarchy">
    <label>任务<Select aria-label={`${labelPrefix}任务`} value={value.family_id} disabled={disabled} onChange={(event) => choose(event.target.value)}>
      {allowAll ? <option value="">全部任务</option> : !value.family_id && <option value="">请选择任务</option>}
      {families.map(([id, label]) => <option key={id} value={id} disabled={!allowAll && !options.some((task) => familyId(task) === id && usable(task))}>{label}</option>)}
    </Select></label>
    <label>难度<Select aria-label={`${labelPrefix}难度`} value={value.level} disabled={disabled || !value.family_id} onChange={(event) => choose(value.family_id, event.target.value)}>
      {allowAll ? <option value="">全部难度</option> : !value.level && <option value="">请选择难度</option>}
      {levels.map((level) => <option key={level} value={level} disabled={!allowAll && !familyTasks.some((task) => taskLevel(task) === level && usable(task))}>{level}</option>)}
    </Select></label>
    <label>提示词<Select aria-label={`${labelPrefix}提示词`} value={value.task_id} disabled={disabled || !value.family_id || !value.level} onChange={(event) => onChange({ ...value, task_id: event.target.value })}>
      {allowAll ? <option value="">全部提示词</option> : !value.task_id && <option value="">请选择提示词</option>}
      {prompts.map((task) => <option key={task.task_id} value={task.task_id} disabled={disableUnavailable && task.available === false} title={task.unavailable_reason}>
        {task.prompt}{task.available === false ? "（资源不可用）" : ""}
      </option>)}
    </Select></label>
  </div>;
}

export function TaskSelector({ tasks, value, onChange, disabled = false, disableUnavailable = true, labelPrefix = "" }: {
  tasks: TaskInfo[]; value: string; onChange: (value: string) => void;
  disabled?: boolean; disableUnavailable?: boolean; labelPrefix?: string;
}) {
  return <TaskHierarchy tasks={tasks} value={scopeForTask(tasks, value)} disabled={disabled}
    disableUnavailable={disableUnavailable} labelPrefix={labelPrefix} onChange={(scope) => onChange(scope.task_id)} />;
}

export function TaskFilter(props: Omit<HierarchyProps, "allowAll" | "disableUnavailable">) {
  return <TaskHierarchy {...props} allowAll />;
}
