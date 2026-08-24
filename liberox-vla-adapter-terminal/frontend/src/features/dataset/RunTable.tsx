import type { Session } from "../run-control/types";
import { Badge } from "../../components/ui/Badge";

const typeLabel = (run: Session) => {
  if (run.source_type === "manual" || (run.kind === "branch" && run.control_mode === "manual")) return "人工接管";
  if (run.source_type === "policy_requery" || run.kind === "branch") return "二次推理";
  if (run.source_type === "incomplete") return "错误/未完成";
  return "原始推理";
};

export function RunTable({ runs, selectable = false, selected = [], onToggle }: {
  runs: Session[]; selectable?: boolean; selected?: string[];
  onToggle?: (runId: string, checked: boolean) => void;
}) {
  if (!runs.length) return <div className="empty-table">该任务还没有可检索的运行数据。</div>;
  return <div className="table-wrap"><table><thead><tr>{selectable && <th>选择</th>}<th>运行</th><th>任务</th><th>来源</th><th>训练区间</th><th>结果</th><th>时间</th></tr></thead><tbody>{runs.map((run) => <tr key={run.id}>{selectable && <td><input type="checkbox" checked={selected.includes(run.id)} disabled={!run.training_eligible} onChange={(event) => onToggle?.(run.id, event.target.checked)} aria-label={`选择 ${run.id}`} /></td>}<td><code>{run.id}</code>{run.ineligible_reason && <small className="cell-note">{run.ineligible_reason}</small>}</td><td>{run.task ?? run.task_name ?? "未知"}</td><td><Badge tone={run.source_type === "manual" ? "green" : "neutral"}>{typeLabel(run)}</Badge>{run.kind === "branch" && <small className="cell-note">前缀 0–{run.training_start_step ?? 0} · 后缀 {run.training_action_count ?? 0} 步</small>}</td><td>{run.training_action_count ?? run.action_count} 步{run.training_chunk_count != null && <small className="cell-note">{run.training_chunk_count} chunks</small>}</td><td><Badge tone={run.status === "ERROR" ? "red" : run.success ? "green" : "neutral"}>{run.status === "ERROR" ? "错误" : run.success ? "成功" : "失败"}</Badge></td><td>{run.created_at ? new Date(run.created_at).toLocaleString() : "—"}</td></tr>)}</tbody></table></div>;
}
