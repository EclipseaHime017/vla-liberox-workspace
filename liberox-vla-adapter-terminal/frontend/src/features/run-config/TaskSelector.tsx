import type { TaskInfo } from "../run-control/types";
import { Select } from "../../components/ui/Select";

export function TaskSelector({ tasks, value, disabled, onChange }: { tasks: TaskInfo[]; value: string; disabled: boolean; onChange: (value: string) => void }) {
  return <Select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>{tasks.map((task) => <option key={task.task_id} value={task.task_id}>{task.prompt}</option>)}</Select>;
}
