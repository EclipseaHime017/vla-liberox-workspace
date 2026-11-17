import { useEffect, useMemo, useRef, useState } from "react";
import { ACTIVE, type Session } from "./types";

const PHASE_LABELS: Record<string, string> = {
  loading_source: "读取轨迹与初始状态",
  loading_model: "加载策略模型",
  warming_controller: "预热 MuJoCo 控制器",
  creating_environment: "创建控制环境",
  restoring_state: "恢复仿真状态",
  preparing_preview: "准备四视角预览",
  countdown: "接管倒计时",
};

export function SessionMonitor({ session }: { session: Session | null }) {
  const [clock, setClock] = useState(Date.now());
  const monitor = useRef<HTMLDivElement | null>(null);
  const followTail = useRef(true);
  useEffect(() => {
    if (session?.preparation_phase !== "loading_model") return;
    const timer = window.setInterval(() => setClock(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, [session?.id, session?.preparation_phase]);
  const events = useMemo(() => {
    if (!session) return [];
    return Object.entries(session.preparation_timing ?? {})
      .filter(([key, value]) => key.endsWith("_at_seconds") && typeof value === "number")
      .map(([key, value]) => ({
        phase: key.slice(0, -"_at_seconds".length),
        seconds: Number(value),
      }))
      .sort((left, right) => left.seconds - right.seconds);
  }, [session]);
  const modelSeconds = session?.preparation_timing?.model_load_seconds;
  const cacheHit = Boolean(session?.preparation_timing?.model_cache_hit);
  const phaseStart = session?.preparation_timing?.loading_model_at_seconds;
  const createdAt = session?.created_at ? Date.parse(session.created_at) : Number.NaN;
  const liveModelSeconds = session?.preparation_phase === "loading_model"
    && typeof phaseStart === "number" && Number.isFinite(createdAt)
    ? Math.max(0, (clock - createdAt) / 1000 - phaseStart)
    : null;
  useEffect(() => {
    followTail.current = true;
    const element = monitor.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [session?.id]);
  useEffect(() => {
    const element = monitor.current;
    if (element && followTail.current) element.scrollTop = element.scrollHeight;
  }, [events.length, liveModelSeconds, modelSeconds, session?.preparation_message, session?.status]);
  if (!session) return null;
  return <div className="panel monitor-panel">
    <div className="panel-title"><h2>运行监视器</h2><span>{session.id}</span></div>
    <div
      className="serial-monitor"
      ref={monitor}
      role="log"
      aria-live="polite"
      onScroll={(event) => {
        const element = event.currentTarget;
        followTail.current = element.scrollHeight - element.scrollTop - element.clientHeight < 8;
      }}
    >
      <div><time>T+0.000s</time><span className="monitor-info">INFO</span><p>会话已创建 · {session.kind === "branch" ? "分支" : "原始仿真"}</p></div>
      {events.map((event) => <div key={event.phase}><time>T+{event.seconds.toFixed(3)}s</time><span className="monitor-info">INFO</span><p>{PHASE_LABELS[event.phase] ?? event.phase}</p></div>)}
      {liveModelSeconds !== null && <div className="monitor-live"><time>{liveModelSeconds.toFixed(1)}s</time><span>WAIT</span><p>模型加载计时中…</p></div>}
      {typeof modelSeconds === "number" && <div><time>{modelSeconds.toFixed(3)}s</time><span className="monitor-ok">DONE</span><p>模型{cacheHit ? "缓存复用" : "首次加载"}完成</p></div>}
      {session.preparation_message && <div className="monitor-live"><time>NOW</time><span>BUSY</span><p>{session.preparation_message}</p></div>}
      {!ACTIVE.has(session.status) && <div><time>END</time><span className={session.status === "ERROR" ? "monitor-error" : "monitor-ok"}>{session.status}</span><p>{session.error ?? `完成 ${session.action_count} 个控制步`}</p></div>}
    </div>
  </div>;
}
