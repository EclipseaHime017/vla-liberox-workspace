import { Badge } from "../../components/ui/Badge";
import type { EvaluationQueueState } from "../run-control/types";

const pending = new Set(["QUEUED", "STARTING", "RUNNING", "STOPPING"]);
const labels = { QUEUED: "等待中", STARTING: "启动中", RUNNING: "测试中", STOPPING: "停止中",
  COMPLETED: "已完成", FAILED: "失败", CANCELED: "已取消" };

export function EvaluationQueuePanel({ queue, onInspect, onStop, busyId }: {
  queue: EvaluationQueueState; onInspect: (id: string) => void;
  onStop: (id: string) => void; busyId: string | null;
}) {
  const jobs = queue.jobs.filter((item) => pending.has(item.status));
  return <section className="surface training-queue" aria-label="测试队列">
    <div className="panel-title"><strong>测试队列</strong><span>{jobs.length} 个待完成任务</span></div>
    <div className="training-queue-body">
      <p>按注册顺序串行执行，与训练共享 GPU；取消或停止只影响所选测试，后续任务继续。</p>
      {jobs.length > 0 && queue.waiting_reason && <p role="status">等待资源：{queue.waiting_reason}</p>}
      {jobs.map((item) => <article className="training-queue-item" key={item.id}>
        <details><summary><strong>{String(item.parameters.policy_label ?? item.parameters.policy_id ?? "测试")}</strong>
          <span>{String(item.parameters.trials)} 回合 · {String(item.parameters.max_steps)} 步/回合</span>
          <Badge tone="neutral">{labels[item.status]}</Badge></summary>
          <div className="queue-parameters"><span>{item.id}</span><span>{String(item.parameters.task_prompt ?? item.parameters.task_id ?? "")}</span>
            <span>注册时间：{new Date(item.created_at).toLocaleString()}</span><span>调度：{String(item.parameters.schedule_sha256 ?? "—").slice(0, 20)}</span></div>
        </details>
        <div className="queue-actions"><button onClick={() => onInspect(item.id)}>查看进度</button>
          <button className="danger" disabled={busyId === item.id || item.status === "STOPPING"}
            onClick={() => onStop(item.id)}>{item.status === "QUEUED" ? "取消排队" : "停止本轮"}</button></div>
      </article>)}
      {!jobs.length && <p className="muted">暂无排队测试，预览调度后即可注册。</p>}
    </div>
  </section>;
}
