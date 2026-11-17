import type { Session } from "./types";

export const ALL_TASKS = "all";

export function filterSessionsByTask(sessions: Session[], taskId: string): Session[] {
  return taskId === ALL_TASKS
    ? sessions
    : sessions.filter((session) => session.task_id === taskId);
}
