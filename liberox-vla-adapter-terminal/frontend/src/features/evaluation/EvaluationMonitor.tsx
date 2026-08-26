import { useEffect, useMemo, useRef, useState } from "react";
import { jobWebSocket } from "../../api/websocket";
import { stopEvaluation } from "../run-control/api";
import type { OfflineJob } from "../run-control/types";

const terminal = new Set(["COMPLETED", "FAILED", "CANCELED"]);

function numeric(source: Record<string, unknown>, ...names: string[]) {
  for (const name of names) {
    const value = source[name];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function duration(value: number | null) {
  if (value == null) return "—";
  const seconds = Math.max(0, Math.round(value));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
}

function displayLine(line: string) {
  try {
    const parsed = JSON.parse(line) as { time?: string; level?: string; message?: string };
    if (parsed.message) return { time: parsed.time?.slice(11, 19) ?? "", level: parsed.level ?? "INFO", message: parsed.message };
  } catch { /* Keep subprocess output verbatim. */ }
  return { time: "", level: "OUT", message: line };
}

export function EvaluationMonitor({ initial, onUpdate, onDismiss }: {
  initial: OfflineJob;
  onUpdate?: (job: OfflineJob) => void;
  onDismiss?: () => void;
}) {
  const [job, setJob] = useState(initial);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const monitor = useRef<HTMLDivElement>(null);
  const follow = useRef(true);

  useEffect(() => {
    setJob(initial); setText(""); setError("");
    const socket = jobWebSocket(initial.id);
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as { job?: OfflineJob; logs?: { text: string } };
      if (payload.job) { setJob(payload.job); onUpdate?.(payload.job); }
      if (payload.logs?.text) setText((current) => current + payload.logs!.text);
    };
    socket.onerror = () => setError("测试监视器连接中断，后台测试不会因此停止");
    return () => socket.close();
  }, [initial.id]);

  useEffect(() => {
    if (follow.current && monitor.current) monitor.current.scrollTop = monitor.current.scrollHeight;
  }, [text]);

  const lines = useMemo(() => text.split(/\r?\n/).filter(Boolean).map(displayLine), [text]);
  const metric = { ...(job.evaluation_summary ?? {}), ...(job.metrics ?? {}) } as Record<string, unknown>;
  const total = numeric(metric, "total_trials", "trials") ?? Number(job.parameters.trials ?? 0);
  const attempted = numeric(metric, "attempted_trials", "completed_trials") ?? 0;
  const currentTrial = numeric(metric, "current_trial") ?? attempted;
  const successes = numeric(metric, "successes", "success_count") ?? 0;
  const successRate = numeric(metric, "success_rate") ?? (attempted > 0 ? successes / attempted : 0);
  const progress = numeric(metric, "progress_percent") ?? (total > 0 ? attempted / total * 100 : 0);
  const elapsed = numeric(metric, "elapsed_seconds", "wall_time_seconds");
  const eta = numeric(metric, "estimated_remaining_seconds", "eta_seconds");
  const hz = numeric(metric, "measured_control_hz", "control_hz");

  return <section className="surface evaluation-monitor-card">
    <div className="panel-title"><strong>测试监视器</strong><span>{job.id}</span></div>
    <div className="evaluation-status-grid">
      <span><small>状态</small><b>{job.status}</b><em>{job.stage_label}</em></span>
      <span><small>回合</small><b>{currentTrial} / {total || "—"}</b><em>已完成 {attempted} · {progress.toFixed(1)}%</em></span>
      <span><small>当前环境</small><b>#{numeric(metric, "init_state_index") ?? "—"}</b><em>seed {numeric(metric, "seed", "current_seed") ?? "—"}</em></span>
      <span><small>成功</small><b>{successes}</b><em>{(successRate * 100).toFixed(1)}%</em></span>
      <span><small>控制频率</small><b>{hz == null ? "—" : `${hz.toFixed(2)} Hz`}</b><em>20 Hz 语义</em></span>
      <span><small>时间</small><b>{duration(elapsed)}</b><em>ETA {duration(eta)}</em></span>
    </div>
    <div className="evaluation-progress"><div style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div>
    <div className="job-status-strip evaluation-job-actions">
      <span><b>{job.stage_label}</b>{String(metric.message ?? "批量测试按冻结调度顺序执行")}</span>
      {!terminal.has(job.status) && <button className="danger" onClick={() => void stopEvaluation(job.id).then((next) => { setJob(next); onUpdate?.(next); }).catch((reason) => setError(String(reason)))}>停止测试</button>}
      {terminal.has(job.status) && onDismiss && <button onClick={onDismiss}>关闭记录</button>}
    </div>
    {error && <p className="job-warning">{error}</p>}
    <div className="serial-monitor evaluation-serial" ref={monitor} onScroll={() => {
      const node = monitor.current;
      if (node) follow.current = node.scrollHeight - node.scrollTop - node.clientHeight < 24;
    }}>
      {lines.length ? lines.map((line, index) => <div key={index}><time>{line.time}</time><span className={line.level === "ERROR" ? "monitor-error" : line.level === "DONE" ? "monitor-ok" : "monitor-info"}>{line.level}</span><p>{line.message}</p></div>) : <p className="empty-log">等待测试进程输出…</p>}
    </div>
    {job.error && <p className="job-error">{job.error}</p>}
  </section>;
}
