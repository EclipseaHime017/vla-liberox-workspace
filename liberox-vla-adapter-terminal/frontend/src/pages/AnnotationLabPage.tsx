import { useEffect, useRef, useState } from "react";
import { getBootstrap, listDatasetRuns } from "../features/run-control/api";
import type { PaginatedRuns, TaskInfo } from "../features/run-control/types";
import { TaskFilter } from "../features/run-config/TaskSelector";
import { ALL_TASK_SCOPE, taskIdsForScope, type TaskScope } from "../features/run-config/taskHierarchy";
import { getExperiment, getLabResult, isActive, memberStatus, labDefaults, listExperiments, startExperiment, stopExperiment,
  type Experiment, type LabDefaults, type LabMode, type LabResult } from "../features/annotation-lab/api";
import { ResultViewer } from "../features/annotation-lab/ResultViewer";
import "../features/annotation-lab/annotation-lab.css";

const stageNames: Record<string, string> = { initializing: "初始化", loading_model: "加载模型", reading_recording: "读取完整轨迹",
  forward_regions: "正向局部区间定位",
  task_requirements: "提取完整任务要求",
  task_decomposition: "结合初始画面拆解里程碑", recording_complete: "轨迹处理完成", finished: "完成", stopped: "停止" };
const duration = (seconds: number | null | undefined) => seconds == null ? "估算中" : `${Math.floor(seconds / 60)} 分 ${Math.round(seconds % 60)} 秒`;

export function AnnotationLabPage() {
  const [defaults, setDefaults] = useState<LabDefaults | null>(null);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [scope, setScope] = useState<TaskScope>(ALL_TASK_SCOPE);
  const [runs, setRuns] = useState<PaginatedRuns | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(5);
  const [selected, setSelected] = useState<string[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [runId, setRunId] = useState("");
  const [result, setResult] = useState<LabResult | null>(null);
  const [resultKey, setResultKey] = useState("");
  const [fps, setFps] = useState(5);
  const [windowSeconds, setWindowSeconds] = useState(2);
  const [mode, setMode] = useState<LabMode>("localize");
  const [cameras, setCameras] = useState("both");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [resultLoading, setResultLoading] = useState(false);
  const identity = `${experiment?.id ?? ""}/${runId}`;
  const currentIdentity = useRef(identity);
  currentIdentity.current = identity;
  const resultRequest = useRef(0);
  const interacted = useRef(false);
  useEffect(() => {
    let current = true;
    void labDefaults().then((value) => { if (current) { setDefaults(value); setMode(value.config.mode ?? "localize"); setFps(value.config.coarse_fps ?? 5); setWindowSeconds(value.config.window_seconds ?? 2); setCameras(value.config.cameras.length === 2 ? "both" : value.config.cameras[0]); } }).catch((reason) => { if (current) setError(String(reason)); });
    void getBootstrap().then((value) => { if (current) setTasks(value.task_catalog); }).catch((reason) => { if (current) setError(String(reason)); });
    void listExperiments().then((value) => { if (current && !interacted.current) { setExperiments(value); if (value[0]) { setExperiment(value[0]); setRunId(value[0].run_ids[0]); } } }).catch((reason) => { if (current) setError(String(reason)); });
    return () => { current = false; };
  }, []);
  useEffect(() => {
    if (!tasks.length) return;
    let current = true; setLoading(true);
    listDatasetRuns(scope.task_id, page, pageSize, scope.task_id ? undefined : taskIdsForScope(tasks, scope))
      .then((value) => { if (current) { setRuns(value); setPage(value.page); } }).catch((reason) => { if (current) setError(String(reason)); })
      .finally(() => { if (current) setLoading(false); });
    return () => { current = false; };
  }, [scope, page, pageSize, tasks]);
  useEffect(() => {
    if (!experiment || !isActive(experiment.status)) return;
    let current = true, inFlight = false;
    const timer = window.setInterval(() => {
      if (inFlight) return;
      inFlight = true;
      getExperiment(experiment.id).then((value) => {
        if (!current) return;
        setExperiment(value); setExperiments((items) => items.map((item) => item.id === value.id ? value : item));
      }).catch((reason) => { if (current) setError(String(reason)); }).finally(() => { inFlight = false; });
    }, 2000);
    return () => { current = false; window.clearInterval(timer); };
  }, [experiment?.id, experiment ? isActive(experiment.status) : false]);
  const loadResult = async (id: string, run: string) => {
    const key = `${id}/${run}`, request = ++resultRequest.current;
    setResultLoading(true);
    try {
      const value = await getLabResult(id, run);
      if (currentIdentity.current === key && resultRequest.current === request) { setResult(value); setResultKey(key); }
    } finally {
      if (currentIdentity.current === key && resultRequest.current === request) setResultLoading(false);
    }
  };
  // Fetch evidence only for the selected recording, not on every progress heartbeat.
  useEffect(() => {
    if (!experiment || !runId) return;
    let current = true;
    void loadResult(experiment.id, runId).catch((reason) => { if (current) setError(String(reason)); });
    return () => { current = false; resultRequest.current += 1; };
  }, [experiment?.id, experiment?.status, runId]);
  const action = async (callback: () => Promise<void>) => {
    setBusy(true); setError(""); try { await callback(); } catch (reason) { setError(String(reason)); } finally { setBusy(false); }
  };
  const ready = defaults?.model_available && defaults.environment_available;
  const progress = experiment?.progress;
  return <div className="content-page annotation-lab">
    {error && <div className="error-banner" role="alert">{error}</div>}
    <section className="surface"><div className="panel-title"><strong>自动关键帧实验</strong><span>独立结果 · 不写入标注与奖励</span></div>
      <div className="lab-body"><p>任务拆解 → 局部连续观察 → 正向区间定位。不运行反向复核、作用域划分或精定位，不生成正式关键帧或奖励。</p>
        <p className="muted">每条轨迹只保留最新一次实验；重新运行会删除该轨迹的旧实验结果。</p>
        <p className="muted">{defaults ? `${defaults.config.model_id} · ${defaults.config.revision.slice(0, 12)} · ${defaults.config.environment}` : "正在检查模型配置…"}</p>
        {defaults?.message && <p role="status">{defaults.message}</p>}
        <TaskFilter tasks={tasks} value={scope} onChange={(value) => { setScope(value); setPage(1); setSelected([]); setRuns(null); }} labelPrefix="实验" />
      </div>
      <div className="table-wrap"><table><thead><tr><th>选择</th><th>轨迹</th><th>提示词</th><th>来源</th><th>结果</th><th>步数</th></tr></thead><tbody>
        {runs?.items.map((run) => <tr key={run.id}><td><input type="checkbox" aria-label={`选择 ${run.id}`} checked={selected.includes(run.id)} disabled={!run.action_count || !["COMPLETED", "ERROR"].includes(run.status)}
          onChange={(event) => setSelected((values) => event.target.checked ? [...values, run.id] : values.filter((id) => id !== run.id))} /></td>
          <td>{run.id}</td><td>{run.task}</td><td>{run.control_mode}</td><td>{run.success ? "成功" : "失败"}</td><td>{run.action_count}</td></tr>)}
      </tbody></table></div>
      <div className="table-pagination"><span>{loading ? "加载中…" : `共 ${runs?.total ?? 0} 条`} · 已选 {selected.length} 条</span>
        <button disabled={!selected.length} onClick={() => setSelected([])}>清空选择</button>
        <label>每页<select value={pageSize} onChange={(event) => { setPageSize(Number(event.target.value)); setPage(1); }}>{[5, 10, 20, 50].map((size) => <option key={size}>{size}</option>)}</select></label>
        <button disabled={loading || page <= 1} onClick={() => setPage(page-1)}>上一页</button><span>{page} / {runs?.pages || 1}</span>
        <button disabled={loading || page >= (runs?.pages || 1)} onClick={() => setPage(page+1)}>下一页</button></div>
      <div className="lab-options"><label>运行模式<select aria-label="运行模式" value={mode} onChange={(event) => setMode(event.target.value as LabMode)}>
        <option value="plan_only">只拆解任务</option><option value="localize">拆解与定位</option></select></label>
        {mode === "localize" && <><label>采样频率（Hz）<input type="number" min={0.25} max={10} step={0.25} value={fps} onChange={(event) => setFps(Number(event.target.value))} /></label>
          <label>观察窗口（秒）<input type="number" min={0.5} max={4} step={0.5} value={windowSeconds} onChange={(event) => setWindowSeconds(Number(event.target.value))} /></label>
          <span className="muted">每窗约 {Math.floor(fps*windowSeconds)+1} 个时刻 · {cameras === "both" ? 2*(Math.floor(fps*windowSeconds)+1) : Math.floor(fps*windowSeconds)+1} 张图 · 前移约 {windowSeconds} 秒，不重叠</span></>}
        <label>输入视角<select value={cameras} onChange={(event) => setCameras(event.target.value)}><option value="both">主视角 + 手腕</option><option value="agentview_image">仅主视角</option><option value="wrist_image">仅手腕</option></select></label>
        <button className="primary" disabled={!ready || busy || !selected.length || selected.length > 100 || (mode === "localize" && (!Number.isFinite(fps) || fps < 0.25 || fps > 10 || !Number.isFinite(windowSeconds) || windowSeconds < 0.5 || windowSeconds > 4 || fps*windowSeconds < 2))} onClick={() => void action(async () => {
          interacted.current = true;
          const next = await startExperiment(selected, { mode, ...(mode === "localize" ? { coarse_fps: fps, window_seconds: windowSeconds } : {}), cameras: cameras === "both" ? ["agentview_image", "wrist_image"] : [cameras] });
          setExperiment(next); setRunId(next.run_ids[0]);
          setExperiments((items) => [next, ...items.map((item) => ({ ...item, run_ids: item.run_ids.filter((id) => !next.run_ids.includes(id)) })).filter((item) => item.run_ids.length)]);
        })}>开始独立实验</button>
      </div>
    </section>
    <section className="surface"><div className="panel-title"><strong>当前过程结果</strong><span>与仿真／训练共用资源队列</span></div>
      <div className="lab-body"><div className="lab-options"><label>结果轨迹<select aria-label="结果轨迹" value={runId} onChange={(event) => {
        interacted.current = true;
        const next = experiments.find((item) => item.run_ids.includes(event.target.value)) ?? null; setExperiment(next); setRunId(event.target.value);
      }}><option value="" disabled>暂无结果</option>{experiments.flatMap((item) => item.run_ids.map((id) => <option key={id} value={id}>{id} · {memberStatus(item, id)}</option>))}</select></label>
        {experiment && isActive(experiment.status) && <button disabled={busy || experiment.status === "STOPPING"} onClick={() => void action(async () => {
          const key = identity, next = await stopExperiment(experiment.id);
          setExperiments(items => items.map(item => item.id === next.id ? next : item));
          if (currentIdentity.current === key) setExperiment(next);
        })}>停止实验</button>}
        {experiment && <button disabled={busy} onClick={() => void action(async () => {
          const key = identity, next = await getExperiment(experiment.id);
          setExperiments(items => items.map(item => item.id === next.id ? next : item));
          if (currentIdentity.current !== key) return;
          setExperiment(next); await loadResult(next.id, runId);
        })}>刷新过程结果</button>}
      </div>
      {experiment && <p role="status">批次状态：{experiment.status} · {stageNames[progress?.stage ?? ""] ?? "等待资源"} · 完成 {progress?.completed_runs ?? 0} / {experiment.run_ids.length} · 失败 {progress?.failed_runs ?? 0}
        {progress && <> · 当前阶段进度 {progress.window_index}/{progress.window_total} · 模型调用 {progress.calls} · 耗时 {duration(progress.elapsed_seconds)} · 剩余 {duration(progress.estimated_remaining_seconds)}</>}</p>}
      {(experiment?.error || progress?.error) && <div className="error-banner">{experiment?.error || progress?.error}</div>}
      {result && experiment && resultKey === identity ? <ResultViewer key={identity} experiment={experiment} runId={runId} result={result} /> : <p>{resultLoading ? "加载结果…" : "尚无结果。运行时可刷新查看已经完成的步骤。"}</p>}
      </div>
    </section>
  </div>;
}
