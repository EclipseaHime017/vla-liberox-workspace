import { useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import type { Session } from "../run-control/types";
import { selectMainVideoArtifact, stepToVideoTime } from "../simulation-view/controls";
import { evidenceUrl, type Experiment, type LabResult } from "./api";
import { GoalIntervals } from "./GoalIntervals";
import { TaskPlan } from "./TaskPlan";

export function ResultViewer({ experiment, runId, result }: { experiment: Experiment; runId: string; result: LabResult }) {
  const [session, setSession] = useState<Session | null>(null);
  const [videoError, setVideoError] = useState("");
  const [evidence, setEvidence] = useState<{ title: string; reason: string; steps: number[] } | null>(null);
  const [step, setStep] = useState(0);
  const [tab, setTab] = useState<"plan" | "coarse" | "raw">("coarse");
  const video = useRef<HTMLVideoElement>(null);
  const pendingSeek = useRef<number | null>(null);
  useEffect(() => {
    let current = true;
    api<Session>(`/api/sessions/${encodeURIComponent(runId)}`).then((value) => { if (current) setSession(value); })
      .catch(() => { if (current) setVideoError("原录像不可用，仍可检查模型证据帧。"); });
    return () => { current = false; };
  }, [runId]);
  const name = session ? selectMainVideoArtifact(session.artifacts) : null;
  const planOnly = (result.mode ?? experiment.config.mode) === "plan_only";
  const currentRegions = result.schema_version >= 8;
  const activeTab = tab === "coarse" && (planOnly || !currentRegions) ? "plan" : tab;
  const seek = (next: number) => {
    setStep(next); pendingSeek.current = next;
    if (video.current && Number.isFinite(video.current.duration) && video.current.duration > 0) {
      video.current.pause();
      video.current.currentTime = stepToVideoTime(next, result.action_count, result.action_count / video.current.duration);
      pendingSeek.current = null;
    }
  };
  const inspectFrames = (title: string, reason: string, steps: number[]) => {
    setEvidence({ title, reason, steps }); if (steps.length) seek(steps[0]);
  };
  return <div className="lab-result">
    <p>{result.source.prompt} · {result.status} · 环境成功确认：{result.success_step == null ? "未确认" : `帧 ${result.success_step}`}（仅作检查，不强制补齐目标）</p>
    {!planOnly && !currentRegions && <p role="status">旧版结果：完成状态／边界不等同于动作定位区间。请重新评价；任务拆解与模型原文仍可查看。</p>}
    {planOnly && <p>{result.status === "COMPLETED" ? "任务拆解已完成；本次未运行时间定位。" : "本次仅检查任务拆解，不运行时间定位。"}</p>}
    {!planOnly && currentRegions && result.localization && <p>已定位 {result.localization.assigned_goals}/{result.localization.total_goals} 个子目标 · 窗口 {result.localization.windows_completed}/{result.localization.windows_total}；不生成正式关键帧或奖励。</p>}
    {!planOnly && currentRegions && result.sampling && <p className="muted">{result.sampling.fps} Hz · {result.sampling.window_seconds} 秒正向窗口 · 每 {result.sampling.stride_seconds} 秒前移。颜色表示局部动作区间，不代表完成状态的持续时间。</p>}
    {result.error && <div className="error-banner">{result.error}</div>}
    <div className="lab-result-grid">
      <div className="lab-video">
        {name ? <video ref={video} controls preload="metadata" src={`/api/sessions/${encodeURIComponent(runId)}/artifacts/${name.split("/").map(encodeURIComponent).join("/")}`}
          onError={() => setVideoError("录像读取失败；请查看模型实际使用的 observation 证据帧。")}
          onLoadedMetadata={() => { if (pendingSeek.current != null) seek(pendingSeek.current); }}
          onTimeUpdate={() => { if (video.current && !video.current.seeking && Number.isFinite(video.current.duration) && video.current.duration > 0) setStep(Math.min(result.action_count, Math.floor(video.current.currentTime / video.current.duration * result.action_count + 1e-6))); }} /> : <p>{videoError || "此轨迹无主视角录像"}</p>}
        {name && videoError && <p>{videoError}</p>}
        <label>录像定位 · observation {step} / {result.action_count}<input aria-label="录像定位" type="range" min={0} max={result.action_count || 0} value={step} onChange={(event) => seek(Number(event.target.value))} /></label>
        <div className="lab-tabs"><button disabled={step === 0} onClick={() => seek(step-1)}>上一帧</button><button disabled={step >= result.action_count} onClick={() => seek(step+1)}>下一帧</button></div>
        <p className="muted">录像用于上下文参考；证据帧对应模型实际读取的 observation step。</p>
      </div>
      <div><div className="lab-tabs">{([["plan", "任务拆解"], ...(!planOnly && currentRegions ? [["coarse", "区间定位"]] : []), ["raw", "模型原文"]] as Array<["plan" | "coarse" | "raw", string]>).map(([id, label]) =>
        <button key={id} className={activeTab === id ? "primary" : ""} onClick={() => { setTab(id); setEvidence(null); }}>{label}</button>)}</div>
        {activeTab === "plan" && <TaskPlan result={result} inspectInitial={() => inspectFrames("拆解输入首帧", "只使用 observation 0，不读取后续结果。", [0])} />}
        {!planOnly && currentRegions && <div hidden={activeTab !== "coarse"}><GoalIntervals goals={result.goal_ranges} stages={result.plan?.stages ?? []} count={result.action_count} hz={result.control_hz}
          pending={result.status === "RUNNING"} inspect={inspectFrames} onSelection={() => setEvidence(null)} /></div>}
        {tab === "raw" && result.calls.map((call) => <details key={`${call.name}-${call.attempt}`}><summary>{call.name} · 尝试 {call.attempt + 1} · {call.seconds.toFixed(1)} s · {call.valid ? "结构有效" : "结构错误"}</summary>
          <p>输入 observation steps：{call.steps.join(", ") || "仅任务指令"}</p>
          {!!call.steps.length && <button onClick={() => inspectFrames(call.name, "模型实际采样输入（保留输入顺序）", call.steps)}>查看输入帧</button>}
          <pre>{call.prompt}</pre><pre>{call.raw_response}</pre>{call.error && <p>{call.error}</p>}</details>)}
      </div>
    </div>
    {evidence && <section className="lab-evidence"><strong>{evidence.title}</strong>{evidence.reason && <p>{evidence.reason}</p>}
      <div>{[...new Set(evidence.steps)].map((frame) => <div key={frame}><button onClick={() => seek(frame)}>帧 {frame} · {(frame / result.control_hz).toFixed(2)} s</button>
        {experiment.config.cameras.map((camera) => <figure key={camera}><img loading="lazy" src={evidenceUrl(experiment.id, runId, camera, frame)} alt={`${camera} observation ${frame}`} />
          <figcaption>{camera === "agentview_image" ? "主视角" : "手腕视角"}</figcaption></figure>)}</div>)}</div>
    </section>}
  </div>;
}
