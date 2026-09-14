import { useEffect, useMemo, useRef, useState } from "react";
import { jobWebSocket } from "../../api/websocket";
import { stopOfflineJob } from "../run-control/api";
import type { OfflineJob } from "../run-control/types";

const terminal = new Set(["COMPLETED", "FAILED", "CANCELED"]);

function metricNumber(value: unknown, digits = 4) {
  return typeof value === "number" && Number.isFinite(value) ? value.toPrecision(digits) : "—";
}

function duration(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const seconds = Math.max(0, Math.round(value));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
}

function displayLine(line: string) {
  try {
    const parsed = JSON.parse(line) as { time?: string; level?: string; message?: string };
    if (parsed.message) return { time: parsed.time?.slice(11, 19) ?? "", level: parsed.level ?? "INFO", message: parsed.message };
  } catch { /* Training subprocess output is intentionally kept verbatim. */ }
  return { time: "", level: "OUT", message: line };
}

export function JobMonitor({ initial, onUpdate, onDismiss }: {
  initial: OfflineJob;
  onUpdate?: (job: OfflineJob) => void;
  onDismiss?: () => void;
}) {
  const [job, setJob] = useState(initial);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const monitor = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  const updateCallback = useRef(onUpdate);
  updateCallback.current = onUpdate;
  useEffect(() => {
    setJob(initial); setText(""); setError("");
    const socket = jobWebSocket(initial.id);
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as { type: string; job?: OfflineJob; logs?: { text: string } };
      if (payload.job) { setJob(payload.job); updateCallback.current?.(payload.job); }
      if (payload.logs?.text) setText((current) => current + payload.logs!.text);
    };
    socket.onerror = () => setError("任务监视器连接中断，任务仍会在后台继续运行");
    return () => socket.close();
  }, [initial.id]);
  useEffect(() => {
    if (follow.current && monitor.current) monitor.current.scrollTop = monitor.current.scrollHeight;
  }, [text]);
  const lines = useMemo(() => text.split(/\r?\n/).filter(Boolean).map(displayLine), [text]);
  const metric = job.metrics;
  const warmup = Number(job.parameters.critic_warmup_steps ?? 0);
  const phase = job.kind === "annotation" || !metric
    ? job.stage_label
    : Number(metric.step ?? 0) <= warmup ? "Critic warmup / BC" : "IQL";
  return <section className="surface job-monitor-card">
    <div className="panel-title"><strong>任务监视器</strong><span>{job.id}</span></div>
    <div className="job-status-strip">
      <span><b>{job.status}</b>{job.stage_label}</span>
      <span>阶段 <b>{phase}</b></span>
      {metric && <><span>Step <b>{String(metric.step ?? "—")}</b></span><span>进度 <b>{Number(metric.progress_percent ?? 0).toFixed(1)}%</b></span><span>速度 <b>{metricNumber(metric.steps_per_second, 3)} step/s</b></span><span>已用 <b>{duration(metric.elapsed_seconds)}</b></span><span>ETA <b>{duration(metric.estimated_remaining_seconds)}</b></span><span>完成时间 <b>{String(metric.estimated_completion_time ?? "计算中")}</b></span></>}
      {!terminal.has(job.status) && <button className="danger" onClick={() => void stopOfflineJob(job.id).then(setJob).catch((reason) => setError(String(reason)))}>停止任务</button>}
      {terminal.has(job.status) && onDismiss && <button onClick={onDismiss}>关闭记录</button>}
    </div>
    {metric && <div className="job-metric-grid">
      <span><small>Q loss</small><b>{metricNumber(metric.q_loss)}</b></span>
      <span><small>Value loss</small><b>{metricNumber(metric.value_loss)}</b></span>
      <span><small>Actor loss</small><b>{metricNumber(metric.actor_loss)}</b></span>
      <span><small>Q / V / Advantage</small><b>{metricNumber(metric.q_mean)} / {metricNumber(metric.value_mean)} / {metricNumber(metric.advantage_mean)}</b></span>
      <span><small>Advantage weight</small><b>{metricNumber(metric.advantage_weight_mean)}</b></span>
      <span><small>Policy LR</small><b>{metricNumber(metric.actor_learning_rate, 3)}</b></span>
      <span><small>梯度范数</small><b>{metricNumber(metric.actor_grad_norm)}</b></span>
      <span><small>峰值显存</small><b>{metricNumber(metric.cuda_peak_memory_gib, 3)} GiB</b></span>
    </div>}
    {error && <p className="job-warning">{error}</p>}
    {job.warning && <p className="job-warning" role="alert">{job.warning}</p>}
    <div className="serial-monitor job-serial" ref={monitor} onScroll={() => { const node = monitor.current; if (node) follow.current = node.scrollHeight - node.scrollTop - node.clientHeight < 24; }}>
      {lines.length ? lines.map((line, index) => <div key={index}><time>{line.time}</time><span className={line.level === "ERROR" ? "monitor-error" : line.level === "DONE" ? "monitor-ok" : "monitor-info"}>{line.level}</span><p>{line.message}</p></div>) : <p className="empty-log">等待后台进程输出…</p>}
    </div>
    {job.error && <p className="job-error">{job.error}</p>}
  </section>;
}
