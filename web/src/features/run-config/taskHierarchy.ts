import type { TaskInfo } from "../run-control/types";

export type TaskScope = { family_id: string; level: string; task_id: string };
export const ALL_TASK_SCOPE: TaskScope = { family_id: "", level: "", task_id: "" };
export const taskLevel = (task: TaskInfo) => task.level || task.task_id.split("::")[0];
export const selectableTasks = (tasks: TaskInfo[]) => tasks.filter((task) => taskLevel(task) !== "LEVEL5");
export const familyId = (task: TaskInfo) => task.family_id || task.task_name || task.task_id;
export const familyLabel = (task: TaskInfo) => task.family_label || task.prompt;

export function scopeForTask(tasks: TaskInfo[], taskId: string): TaskScope {
  const task = tasks.find((item) => item.task_id === taskId);
  return task ? { family_id: familyId(task), level: taskLevel(task), task_id: taskId }
    : { ...ALL_TASK_SCOPE, task_id: taskId };
}

/** Undefined means no restriction; [] means no matching scenes, never 'all'. */
export function taskIdsForScope(tasks: TaskInfo[], scope: TaskScope): string[] | undefined {
  if (scope.task_id) return [scope.task_id];
  if (!scope.family_id && !scope.level) return undefined;
  return selectableTasks(tasks).filter((task) =>
    (!scope.family_id || familyId(task) === scope.family_id) && (!scope.level || taskLevel(task) === scope.level),
  ).map((task) => task.task_id);
}

export function filterByTaskScope<T extends { task_id?: string | null }>(items: T[], tasks: TaskInfo[], scope: TaskScope): T[] {
  const ids = taskIdsForScope(tasks, scope);
  if (ids === undefined) return items;
  const selected = new Set(ids);
  return items.filter((item) => item.task_id != null && selected.has(item.task_id));
}
