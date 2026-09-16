import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { getStageAnnotation, saveStageAnnotation } from "../run-control/api";
import type { StageAnnotation, StageKeyframe } from "../run-control/types";
import { stepToVideoTime } from "../simulation-view/controls";

export function stageVideoStep(time: number, duration: number, actionCount: number) {
  if (!Number.isFinite(duration) || duration <= 0 || actionCount <= 0) return 0;
  // A displayed video frame covers [i / fps, (i+1) / fps), not a rounded midpoint.
  return Math.max(0, Math.min(actionCount, Math.floor(time / duration * actionCount + 1e-6)));
}

export function StageAnnotationPanel({ runId, videoRef, onSaved, onDirtyChange }: {
  runId: string; videoRef: RefObject<HTMLVideoElement | null>;
  onSaved: (annotation: StageAnnotation) => void; onDirtyChange: (dirty: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const [annotation, setAnnotation] = useState<StageAnnotation | null>(null);
  const [keyframes, setKeyframes] = useState<StageKeyframe[]>([]);
  const [step, setStep] = useState(0);
  const [kind, setKind] = useState<StageKeyframe["kind"]>("positive");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [videoReady, setVideoReady] = useState(false);
  const requestId = useRef(0);
  const dirty = annotation != null && JSON.stringify(keyframes) !== JSON.stringify(annotation.keyframes);

  const accept = useCallback((next: StageAnnotation) => {
    setAnnotation(next); setKeyframes(next.keyframes);
    onSaved(next);
  }, [onSaved]);
  const load = useCallback(async () => {
    const request = ++requestId.current;
    setLoading(true); setError("");
    try {
      const next = await getStageAnnotation(runId);
      if (request === requestId.current) accept(next);
    } catch (reason) {
      if (request === requestId.current) setError(String(reason));
    } finally {
      if (request === requestId.current) setLoading(false);
    }
  }, [accept, runId]);
  useEffect(() => { void load(); return () => { requestId.current += 1; }; }, [load]);
  useEffect(() => { onDirtyChange(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => {
    if (!dirty) return;
    const preventUnload = (event: BeforeUnloadEvent) => { event.preventDefault(); };
    window.addEventListener("beforeunload", preventUnload);
    return () => window.removeEventListener("beforeunload", preventUnload);
  }, [dirty]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !annotation) return;
    const update = () => {
      const ready = Number.isFinite(video.duration) && video.duration > 0;
      setVideoReady(ready);
      if (ready && !video.seeking) {
        // Existing recordings have N action frames and N+1 observation states.
        // Derive encoding FPS from duration, not the physical 20 Hz time axis.
        setStep(stageVideoStep(video.currentTime, video.duration, annotation.action_count));
      }
    };
    update();
    video.addEventListener("loadedmetadata", update);
    video.addEventListener("durationchange", update);
    video.addEventListener("timeupdate", update);
    video.addEventListener("seeked", update);
    video.addEventListener("pause", update);
    return () => {
      video.removeEventListener("loadedmetadata", update);
      video.removeEventListener("durationchange", update);
      video.removeEventListener("timeupdate", update);
      video.removeEventListener("seeked", update);
      video.removeEventListener("pause", update);
    };
  }, [annotation, videoRef]);

  const seek = (nextStep: number) => {
    if (!annotation) return;
    const selected = Math.max(0, Math.min(annotation.action_count, Math.round(nextStep)));
    setStep(selected);
    const existing = keyframes.find((item) => item.step === selected);
    if (existing) setKind(existing.kind);
    const video = videoRef.current;
    if (video && videoReady) {
      video.pause();
      video.currentTime = stepToVideoTime(selected, annotation.action_count,
        annotation.action_count / video.duration);
    }
  };
  const save = async () => {
    if (!annotation) return;
    setSaving(true); setError("");
    try {
      accept(await saveStageAnnotation(runId, {
        keyframes, revision: annotation.revision,
      }));
    } catch (reason) { setError(String(reason)); }
    finally { setSaving(false); }
  };
  const close = () => {
    if (dirty && !window.confirm("切片标记尚未保存，是否丢弃本次修改？")) return;
    if (annotation) setKeyframes(annotation.keyframes);
    setOpen(false);
  };
  const reload = () => {
    if (dirty && !window.confirm("重新加载会丢弃未保存的切片标记，是否继续？")) return;
    void load();
  };
  const selected = keyframes.find((item) => item.step === step);
  const canMark = annotation != null && step > 0
    && (annotation.success_step == null || step < annotation.success_step);
  const putKeyframe = () => {
    if (!annotation) return;
    const video = videoRef.current;
    // Native playback's timeupdate is throttled. Read its current time at the
    // user's click so a keyframe never captures an older React cursor value.
    const currentStep = video && videoReady
      ? stageVideoStep(video.currentTime, video.duration, annotation.action_count) : step;
    if (currentStep <= 0 || (annotation.success_step != null && currentStep >= annotation.success_step)) return;
    video?.pause(); setStep(currentStep);
    setKeyframes((current) => [...current.filter((item) => item.step !== currentStep), { step: currentStep, kind }]
      .sort((left, right) => left.step - right.step));
  };

  return <section className="stage-annotation" aria-label="关键帧切片">
    <div className="stage-toolbar"><button onClick={() => open ? close() : setOpen(true)} disabled={saving}>
      {open ? "收起切片" : "切片 / 标记关键帧"}
    </button><span role="status">{loading ? "读取切片标记…" : dirty ? "有未保存的修改"
      : annotation?.status === "ready" ? "关键帧已保存" : annotation?.status === "stale"
        ? "原数据或标注规则已变化，请检查后重新保存" : "尚未保存阶段标注"}</span></div>
    {annotation?.derivation_error && <p className="error-banner">关键帧已保存，奖励预览计算失败：{annotation.derivation_error}。请检查评价配置，标记无需重新保存。</p>}
    {open && <div className="stage-editor">
      {(error || annotation?.error) && <div className="error-banner"><span>{error || annotation?.error}</span>
        <button onClick={reload} disabled={loading || saving}>重新加载标注</button></div>}
      {loading ? <p>正在读取时间轴与标记，不加载评价模型。</p> : annotation && <>
        <p className="field-hint">仅标记原始 observation step，不裁剪轨迹。首帧、接管前缀与成功后的记录均保留。</p>
        {!videoReady && <p role="status">{videoRef.current ? "等待视频元数据，加载后可逐帧定位。" : "没有主视角视频，不能进行可视化切片。"}</p>}
        <fieldset disabled={saving || !videoReady} className="stage-controls">
          <label>切片进度<input aria-label="切片进度" type="range" min={0} max={annotation.action_count} step={1}
            value={step} onChange={(event) => seek(Number(event.target.value))} /></label>
          <div className="stage-markers" aria-label="已标记时间点">{keyframes.filter((item) => item.step >= 0 && item.step <= annotation.action_count).map((item, index) => <button key={`${item.step}-${index}`}
            className={`stage-marker ${item.kind}`} style={{ left: `${100 * item.step / Math.max(1, annotation.action_count)}%` }}
            title={`${item.kind} · step ${item.step}`} aria-label={`跳转 ${item.kind} 帧 ${item.step}`} onClick={() => seek(item.step)} />)}
            {annotation.success_step != null && <button className="stage-marker success"
              style={{ left: `${100 * annotation.success_step / Math.max(1, annotation.action_count)}%` }}
              title={`success · step ${annotation.success_step}`} aria-label="跳转成功帧" onClick={() => seek(annotation.success_step!)} />}</div>
          <div className="stage-toolbar"><button onClick={() => seek(step - 1)} disabled={step === 0}>上一帧</button>
            <label>Observation step<input aria-label="Observation step" type="number" min={0} max={annotation.action_count} step={1}
              value={step} onChange={(event) => seek(Number(event.target.value))} /></label>
            <button onClick={() => seek(step + 1)} disabled={step === annotation.action_count}>下一帧</button>
            <strong>{(annotation.time_seconds[step] ?? 0).toFixed(3)} s / {(annotation.time_seconds.at(-1) ?? 0).toFixed(3)} s</strong></div>
          {step === annotation.action_count && <p className="field-hint">末端 observation 为 step {step}；原录像没有额外的末端帧，画面停在最后一个编码帧。</p>}
          <div className="stage-toolbar"><label>关键帧类型<select value={kind} onChange={(event) => setKind(event.target.value as StageKeyframe["kind"])}>
            <option value="positive">Positive · 阶段提升</option><option value="negative">Negative · 阶段下降</option>
          </select></label><button onClick={putKeyframe} disabled={!canMark}>{selected ? "更新当前关键帧" : "标记当前帧"}</button>
            {selected && <button onClick={() => setKeyframes((current) => current.filter((item) => item.step !== step))}>删除当前标记</button>}</div>
        </fieldset>
        <div className="stage-toolbar">
          <span>连续 {annotation.success_consecutive_steps} 步成功确认：{annotation.success_step == null ? "未确认成功" : `step ${annotation.success_step}（自动锚点，不计入 positive 数量）`}</span></div>
        <ul className="stage-keyframes">{keyframes.map((item, index) => <li key={`${item.step}-${index}`}>
          <button className={`stage-keyframe ${item.kind}`} disabled={saving || !videoReady} onClick={() => { setKind(item.kind); seek(item.step); }}>
            {item.kind} · step {item.step} · {annotation.time_seconds[item.step] == null ? "超出当前轨迹" : `${annotation.time_seconds[item.step].toFixed(3)} s`}</button>
          <button disabled={saving} aria-label={`删除帧 ${item.step}`} onClick={() => setKeyframes((current) => current.filter((entry) => entry.step !== item.step))}>删除</button>
        </li>)}</ul>
        {!keyframes.length && <p className="field-hint">尚无手动关键帧。失败轨迹可保存空标注，阶段奖励保持 -1。</p>}
        <div className="stage-toolbar"><button className="primary" onClick={() => void save()}
          disabled={saving || (!dirty && annotation.status === "ready")}>{saving ? "正在保存关键帧…" : "保存关键帧"}</button></div>
      </>}
    </div>}
  </section>;
}
