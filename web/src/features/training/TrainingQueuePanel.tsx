import { Badge } from "../../components/ui/Badge";
import type { TrainingQueueItem, TrainingQueueState } from "../run-control/types";

const pending = new Set(["QUEUED", "STARTING", "RUNNING", "STOPPING"]);
const labels = { QUEUED: "等待中", STARTING: "启动中", RUNNING: "训练中", STOPPING: "停止中",
  COMPLETED: "已完成", FAILED: "失败", CANCELED: "已取消" };

export function TrainingQueuePanel({ queue, onInspect, onStop, busyId }: {
  queue: TrainingQueueState; onInspect: (id: string) => void;
  onStop: (id: string) => void; busyId: string | null;
}) {
  const waiting = queue.jobs.filter((job) => pending.has(job.status));
  const history = queue.jobs.filter((job) => !pending.has(job.status)).slice().reverse();
  const row = (job: TrainingQueueItem) => <article className="training-queue-item" key={job.id}>
    <details>
      <summary><strong>{String(job.parameters.dataset_name ?? job.dataset_id)} · {String(job.parameters.algorithm ?? "iql").toUpperCase()}</strong>
        <span>Batch {String(job.parameters.micro_batch_size ?? "—")} · {String(job.parameters.train_steps ?? "—")} steps · Seed {String(job.parameters.seed ?? "—")}</span>
        <Badge tone={job.status === "COMPLETED" ? "green" : job.status === "FAILED" ? "red" : "neutral"}>{labels[job.status]}</Badge>
      </summary>
      <div className="queue-parameters"><span>{job.id}</span>
        <span>梯度累积：{String(job.parameters.gradient_accumulation_steps ?? "—")}</span>
        <span>注册时间：{new Date(job.created_at).toLocaleString()}</span>
        {job.parameters.algorithm !== "bc" && <span>Discount γ：{String((job.parameters.reward as Record<string, unknown> | undefined)?.gamma ?? "—")}</span>}
      </div>
      {job.error && <p className="job-error">{job.error}</p>}
    </details>
    <div className="queue-actions"><button onClick={() => onInspect(job.id)}>查看进度</button>
      {pending.has(job.status) && <button className="danger" disabled={busyId === job.id || job.status === "STOPPING"}
        onClick={() => onStop(job.id)}>{job.status === "QUEUED" ? "取消排队" : "停止本轮"}</button>}</div>
  </article>;
  return <section className="surface training-queue">
    <details open><summary className="panel-title"><strong>批量训练</strong><span>{waiting.length} 个待完成任务</span></summary>
      <div className="training-queue-body">
        <p>按注册顺序串行执行，各批次独立训练；停止或失败仅影响该任务，后续任务继续。</p>
        {queue.waiting_reason && waiting.length > 0 && <p role="status">等待资源：{queue.waiting_reason}</p>}
        {waiting.length ? waiting.map(row) : <p className="muted">暂无排队任务，可在训练配置中新增。</p>}
        {history.length > 0 && <details className="training-queue-history"><summary>最近训练（{history.length}）</summary>{history.map(row)}</details>}
      </div>
    </details>
  </section>;
}
