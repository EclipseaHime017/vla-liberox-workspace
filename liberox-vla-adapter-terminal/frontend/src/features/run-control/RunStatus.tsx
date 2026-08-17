import type { Session } from "./types";

export function RunStatus({ run }: { run: Session }) {
  return <><span className={`status-dot status-${run.status.toLowerCase()}`} /><span><strong>{run.kind === "branch" ? "分支" : "原始"} · {run.id}</strong><small>{run.control_mode}{run.manual_source ? ` / ${run.manual_source}` : ""} · {run.action_count} 步</small></span><em>{run.status}</em></>;
}
