import type { Session } from "../run-control/types";
import { Badge } from "../../components/ui/Badge";

export function RunTable({ runs }: { runs: Session[] }) {
  return <div className="table-wrap"><table><thead><tr><th>运行</th><th>任务</th><th>类型</th><th>步数</th><th>结果</th><th>时间</th></tr></thead><tbody>{runs.map((run) => <tr key={run.id}><td><code>{run.id}</code></td><td>{run.task ?? run.task_name ?? "未知"}</td><td>{run.kind === "branch" ? "分支" : "原始"}</td><td>{run.action_count}/{run.max_steps}</td><td><Badge tone={run.status === "ERROR" ? "red" : run.success ? "green" : "neutral"}>{run.status === "ERROR" ? "错误" : run.success ? "成功" : "未成功"}</Badge></td><td>{run.created_at ? new Date(run.created_at).toLocaleString() : "—"}</td></tr>)}</tbody></table></div>;
}
