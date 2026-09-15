import type { Session } from "../run-control/types";
import { Badge } from "../../components/ui/Badge";

const typeLabel = (run: Session) => {
  if (run.source_type === "manual" || (run.kind === "branch" && run.control_mode === "manual")) return "人工接管";
  if (run.source_type === "policy_requery" || run.kind === "branch") return "二次推理";
  if (run.source_type === "incomplete") return "错误/未完成";
  return "原始推理";
};

function EvaluationCell({ evaluation }: { evaluation: Session["rynn_evaluation"] }) {
  const ready = evaluation?.status === "READY";
  const error = evaluation?.status === "ERROR";
  return <td title={evaluation?.error}>
    <Badge tone={ready ? "green" : error ? "red" : "neutral"}>{ready ? "已评价" : error ? "评价异常" : "未评价"}</Badge>
  </td>;
}

export function RunTable({ runs, selectable = false, excludeTests = false, selected = [], onToggle, onOpen }: {
  runs: Session[]; selectable?: boolean; selected?: string[];
  excludeTests?: boolean;
  onToggle?: (runId: string, checked: boolean) => void;
  onOpen?: (runId: string) => void;
}) {
  if (!runs.length) return <div className="empty-table">该任务还没有可检索的运行数据。</div>;
  return <div className="table-wrap"><table><thead><tr>{selectable && <th>选择</th>}<th>运行</th><th>任务</th><th>来源</th><th>用途</th><th>训练区间</th><th>结果</th><th>RynnValue</th><th>Robometer</th><th>时间</th><th /></tr></thead><tbody>{runs.map((run) => <tr key={run.id}>{selectable && <td><input type="checkbox" checked={selected.includes(run.id)} disabled={!run.training_eligible || (excludeTests && run.is_test)} onChange={(event) => onToggle?.(run.id, event.target.checked)} aria-label={`选择 ${run.id}`} /></td>}<td><code>{run.id}</code>{run.ineligible_reason && <small className="cell-note">{run.ineligible_reason}</small>}</td><td>{run.task ?? run.task_name ?? "未知"}</td><td><Badge tone={run.source_type === "manual" ? "green" : "neutral"}>{typeLabel(run)}</Badge>{run.kind === "branch" && <small className="cell-note">接管点 {run.resume_step ?? 0} · 保留完整轨迹</small>}</td><td><Badge tone={run.is_test ? "blue" : "neutral"}>{run.is_test ? "测试" : "训练候选"}</Badge></td><td>{run.training_action_count ?? run.action_count} 步{run.training_chunk_count != null && <small className="cell-note">{run.training_chunk_count} chunks（回放时前缀去重）</small>}</td><td><Badge tone={run.status === "ERROR" ? "red" : run.success ? "green" : "neutral"}>{run.status === "ERROR" ? "错误" : run.success ? "成功" : "失败"}</Badge></td><EvaluationCell evaluation={run.rynn_evaluation} /><EvaluationCell evaluation={run.robometer_evaluation} /><td>{run.created_at ? new Date(run.created_at).toLocaleString() : "—"}</td><td>{onOpen && <button className="table-link" onClick={() => onOpen(run.id)}>详情</button>}</td></tr>)}</tbody></table></div>;
}
