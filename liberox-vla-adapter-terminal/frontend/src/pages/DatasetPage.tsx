import { useEffect, useMemo, useRef, useState } from "react";
import {
  createTrainingDataset, datasetExportUrl, deriveTrainingDataset,
  deleteTrainingDataset, evaluateTrajectories, getBootstrap, getDatasetSummary,
  getTrajectoryDetail, listDatasetRuns, listOfflineJobs, listTrainingDatasets,
  previewTrainingDataset,
  setTrajectoryTestLabel,
} from "../features/run-control/api";
import type {
  Bootstrap, DatasetPreview, DatasetSelection, DatasetSummary, OfflineJob,
  PaginatedRuns, Session, TrainingDataset, TrajectoryDetail as TrajectoryDetailData,
} from "../features/run-control/types";
import { SuccessRateChart } from "../features/metrics/SuccessRateChart";
import { RunTable } from "../features/dataset/RunTable";
import { JobMonitor } from "../features/training/JobMonitor";
import { Badge } from "../components/ui/Badge";
import { ApiError } from "../api/client";
import { TrajectoryDetail } from "../features/dataset/TrajectoryDetail";
import { FrozenDatasetCard } from "../features/dataset/FrozenDatasetCard";
import { TaskFilter } from "../features/run-config/TaskSelector";
import { ALL_TASK_SCOPE, filterByTaskScope, scopeForTask, taskIdsForScope, type TaskScope } from "../features/run-config/taskHierarchy";

const sources = ["inference", "manual", "policy_requery"] as const;
const outcomes = ["success", "failure"] as const;
const sourceLabel: Record<(typeof sources)[number], string> = {
  inference: "原始推理", manual: "人工接管", policy_requery: "二次推理",
};
const outcomeLabel = { success: "成功", failure: "失败" } as const;

function initialSelection(): DatasetSelection {
  return {
    mode: "random", size: 1, seed: 7, order: "newest",
    source_types: [...sources], outcomes: [...outcomes], run_ids: [],
    quotas: sources.flatMap((source_type) => outcomes.map((outcome) => ({
      source_type, outcome, count: 0, order: "random" as const,
    }))),
  };
}

export function DatasetPage() {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [taskScope, setTaskScope] = useState<TaskScope>(ALL_TASK_SCOPE);
  const taskId = taskScope.task_id;
  const [runs, setRuns] = useState<Session[]>([]);
  const [runPage, setRunPage] = useState<PaginatedRuns | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(5);
  const [section, setSection] = useState<"evaluation" | "package">("evaluation");
  const [evaluationSelection, setEvaluationSelection] = useState<string[]>([]);
  const [batchOverwrite, setBatchOverwrite] = useState(false);
  const [evaluators, setEvaluators] = useState<Array<"rynnvalue" | "robometer">>(["rynnvalue"]);
  const [detail, setDetail] = useState<TrajectoryDetailData | null>(null);
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [selection, setSelection] = useState<DatasetSelection>(initialSelection);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [builder, setBuilder] = useState(false);
  const [name, setName] = useState("");
  const [parentId, setParentId] = useState<string | null>(null);
  const [validationFraction, setValidationFraction] = useState(0.2);
  const [splitSeed, setSplitSeed] = useState(7);
  const [successSteps, setSuccessSteps] = useState(5);
  const [includePostSuccess, setIncludePostSuccess] = useState(true);
  const [annotationJob, setAnnotationJob] = useState<OfflineJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [runsLoading, setRunsLoading] = useState(false);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [error, setError] = useState("");
  const detailRequest = useRef(0);
  const listRequest = useRef(0);
  const requestedContext = useRef<{ datasetId?: string }>({});
  const completedJobs = useRef(new Set<string>());
  const globalPoll = useRef({ runId: "", attempts: 0 });

  const refresh = async (nextTask = taskId, nextPage = page, nextPageSize = pageSize) => {
    if (!bootstrap) return;
    const request = ++listRequest.current;
    const ids = taskIdsForScope(bootstrap.task_catalog, taskScope);
    setRunsLoading(true); setDatasetsLoading(true);
    const runsRequest = (nextTask ? listDatasetRuns(nextTask, nextPage, nextPageSize) : listDatasetRuns("", nextPage, nextPageSize, ids))
      .then((nextRuns) => {
        if (request !== listRequest.current) return;
        setRunPage(nextRuns); setRuns(nextRuns.items);
        if (nextRuns.page !== nextPage) setPage(nextRuns.page);
      })
      .finally(() => { if (request === listRequest.current) setRunsLoading(false); });
    const metadataRequest = Promise.all([nextTask ? listTrainingDatasets(nextTask) : listTrainingDatasets(undefined, ids), listOfflineJobs()])
      .then(([nextDatasets, jobs]) => {
        if (request !== listRequest.current) return;
        setDatasets(nextDatasets);
        const active = jobs.find((job) => ["annotation", "trajectory_evaluation"].includes(job.kind) && ["STARTING", "RUNNING", "STOPPING"].includes(job.status));
        if (active) setAnnotationJob(active);
      })
      .finally(() => { if (request === listRequest.current) setDatasetsLoading(false); });
    const settled = await Promise.allSettled([runsRequest, metadataRequest]);
    const rejected = settled.find((result): result is PromiseRejectedResult => result.status === "rejected");
    if (rejected && request === listRequest.current) throw rejected.reason;
  };

  useEffect(() => {
    void Promise.all([getDatasetSummary(), getBootstrap()])
      .then(([nextSummary, nextBootstrap]) => {
        setSummary(nextSummary); setBootstrap(nextBootstrap);
        setTaskScope(scopeForTask(nextBootstrap.task_catalog, nextBootstrap.task.task_id));
      })
      .catch((reason) => setError(String(reason)));
  }, []);
  useEffect(() => {
    if (bootstrap) void refresh(taskId, page, pageSize).catch((reason) => setError(String(reason)));
    return () => { listRequest.current += 1; };
  }, [taskScope, page, pageSize, bootstrap]);
  useEffect(() => {
    if (!detail?.global_evaluation_pending) {
      globalPoll.current = { runId: "", attempts: 0 }; return;
    }
    if (busy) return;
    const runId = detail.run.id;
    const pollKey = JSON.stringify([runId, detail.dataset_context?.dataset_id]);
    if (globalPoll.current.runId !== pollKey) globalPoll.current = { runId: pollKey, attempts: 0 };
    if (globalPoll.current.attempts >= 30) {
      setError("全局评价准备时间较长，请稍后重新打开详情；已显示的视频和标记不受影响。"); return;
    }
    let current = true;
    const timer = window.setTimeout(() => {
      const request = ++detailRequest.current;
      globalPoll.current.attempts += 1;
      const fetchDetail = detail.dataset_context
        ? getTrajectoryDetail(runId, detail.dataset_context.dataset_id) : getTrajectoryDetail(runId);
      void fetchDetail.then((next) => {
        if (current && request === detailRequest.current) setDetail(next);
      }).catch((reason) => { if (current && request === detailRequest.current) setError(String(reason)); });
    }, 1000);
    return () => { current = false; window.clearTimeout(timer); };
  }, [detail, busy]);

  const taskStats = useMemo(() => {
    const groups = filterByTaskScope(summary?.tasks ?? [], bootstrap?.task_catalog ?? [], taskScope);
    const runs = groups.reduce((total, group) => total + group.runs, 0);
    const successes = groups.reduce((total, group) => total + group.successes, 0);
    return { runs, success_rate: runs ? successes / runs : 0 };
  }, [summary, bootstrap, taskScope]);
  const robometerCapability = bootstrap?.evaluation_capabilities?.robometer;
  const eligible = runs.filter((run) => run.training_eligible);
  const eligibleCount = runPage?.eligible_count ?? eligible.length;
  const selectedSet = useMemo(() => new Set(selection.run_ids), [selection.run_ids]);
  const patchSelection = (patch: Partial<DatasetSelection>) => {
    setSelection((current) => ({ ...current, ...patch })); setPreview(null);
  };
  const toggleSource = (value: (typeof sources)[number]) => patchSelection({
    source_types: selection.source_types.includes(value)
      ? selection.source_types.filter((item) => item !== value)
      : [...selection.source_types, value],
  });
  const toggleOutcome = (value: (typeof outcomes)[number]) => patchSelection({
    outcomes: selection.outcomes.includes(value)
      ? selection.outcomes.filter((item) => item !== value)
      : [...selection.outcomes, value],
  });
  const openBuilder = (parent?: TrainingDataset) => {
    setBuilder(true); setPreview(null); setError("");
    if (parent) {
      setTaskScope(scopeForTask(bootstrap?.task_catalog ?? [], parent.task_id)); setPage(1);
      setParentId(parent.id); setName(`${parent.name} · 派生`);
      setValidationFraction(parent.validation_fraction ?? 0.2);
      setSplitSeed(parent.split_seed ?? 7); setSuccessSteps(parent.success_consecutive_steps ?? 5);
      setIncludePostSuccess(parent.include_post_success ?? true);
      setSelection({ ...initialSelection(), mode: "manual", run_ids: parent.members.map((item) => item.run_id) });
    } else {
      setParentId(null); setName(`训练数据集 ${new Date().toLocaleString()}`);
      setIncludePostSuccess(true);
      setSelection(initialSelection());
    }
  };
  const makePreview = async () => {
    setBusy(true); setError("");
    try { setPreview(await previewTrainingDataset(taskId, selection)); }
    catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const saveDataset = async () => {
    if (!preview) return;
    setBusy(true); setError("");
    const body = {
      name, selection, validation_fraction: validationFraction,
      split_seed: splitSeed, success_consecutive_steps: successSteps,
      include_post_success: includePostSuccess,
    };
    try {
      if (parentId) await deriveTrainingDataset(parentId, body);
      else await createTrainingDataset({ ...body, task_id: taskId });
      setBuilder(false); setPreview(null); await refresh();
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const evaluate = async (runIds: string[] | null, overwrite: boolean) => {
    if (!taskId) return;
    if (!evaluators.length) { setError("请至少选择一个评价器"); return; }
    setBusy(true); setError("");
    try {
      const result = await evaluateTrajectories({
        task_id: taskId, run_ids: runIds, overwrite, evaluators,
      });
      if (result.job) setAnnotationJob(result.job);
      else window.alert(result.message ?? `已跳过 ${result.skipped_count} 条已有评价`);
      setEvaluationSelection([]);
      await refresh();
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const openDetail = async (runId: string, datasetId?: string) => {
    const request = ++detailRequest.current;
    setBusy(true); setError("");
    try {
      const next = await getTrajectoryDetail(runId, datasetId);
      if (request === detailRequest.current) {
        requestedContext.current = { datasetId };
        setDetail(next);
      }
    }
    catch (reason) { setError(String(reason)); }
    finally { if (request === detailRequest.current) setBusy(false); }
  };
  const removeDataset = async (dataset: TrainingDataset) => {
    if (!window.confirm(`删除数据集“${dataset.name}”？\n\n将删除该数据集清单和专属标注目录，但不会删除源轨迹或全局共享奖励缓存。`)) return;
    setBusy(true); setError("");
    try {
      try {
        await deleteTrainingDataset(dataset.id);
      } catch (reason) {
        const detail = reason instanceof ApiError && typeof reason.detail === "object" && reason.detail
          ? reason.detail as { code?: string; training_jobs?: string[] }
          : null;
        if (detail?.code !== "DATASET_HAS_TRAINING") throw reason;
        const jobs = detail.training_jobs?.join("、") || "已有训练任务";
        const confirmed = window.confirm(
          `该数据集已被训练记录引用：${jobs}\n\n` +
          "继续删除只会移除数据集和专属标注；训练输出、checkpoint 与 policy overlay 会保留，并标记其源数据集已删除。是否继续？",
        );
        if (!confirmed) return;
        await deleteTrainingDataset(dataset.id, true);
      }
      if (annotationJob?.dataset_id === dataset.id) setAnnotationJob(null);
      await refresh();
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };

  const updateJob = (job: OfflineJob) => {
    setAnnotationJob(job);
    if (!["COMPLETED", "FAILED", "CANCELED"].includes(job.status) || completedJobs.current.has(job.id)) return;
    completedJobs.current.add(job.id);
    void refresh().catch((reason) => setError(String(reason)));
    if (detail && !busy && job.status === "COMPLETED") {
      const context = requestedContext.current;
      void openDetail(detail.run.id, context.datasetId);
    }
  };
  const monitor = annotationJob && <JobMonitor initial={annotationJob} onDismiss={() => setAnnotationJob(null)} onUpdate={updateJob} />;

  if (detail) return <>
    {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
    <TrajectoryDetail detail={detail} contextLoading={busy}
      onContextChange={(datasetId) => void openDetail(detail.run.id, datasetId)}
      onBack={() => { detailRequest.current += 1; setDetail(null); setBusy(false); }} onSetTest={async (isTest) => {
    await setTrajectoryTestLabel(detail.run.id, isTest);
    setDetail((current) => current ? { ...current, run: { ...current.run, is_test: isTest } } : current);
    await refresh();
  }} />{monitor}</>;

  return <section className="content-page">
    <div className="page-heading"><p className="eyebrow">DATASET CATALOG</p><h1>数据集</h1><p>打包轨迹并配置评价，重新评价即可更新当前结果，无需重复标记关键帧。测试标签数据不会进入新的训练包。</p></div>
    {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
    {summary && <>
      <div className="dataset-overview surface"><SuccessRateChart rate={taskStats?.success_rate ?? 0} /><div className="dataset-stats"><div><strong>{taskStats?.runs ?? 0}</strong><span>当前任务运行</span></div><div><strong>{eligibleCount}</strong><span>可训练轨迹</span></div><div><strong>{runPage?.rynn_evaluated_count ?? 0} / {runPage?.robometer_evaluated_count ?? 0}</strong><span>RynnValue / Robometer</span></div><div><strong>{datasets.length}</strong><span>冻结数据集</span></div></div></div>
      <div className="dataset-section-tabs surface"><button className={section === "evaluation" ? "active" : ""} onClick={() => { setSection("evaluation"); setBuilder(false); }}>轨迹评价</button><button className={section === "package" ? "active" : ""} onClick={() => setSection("package")}>打包训练数据集</button></div>
      <div className="surface dataset-browser">
        <div className="dataset-toolbar">
          <TaskFilter tasks={bootstrap?.task_catalog ?? []} value={taskScope} onChange={(value) => {
            setTaskScope(value); setPage(1); setBuilder(false); setEvaluationSelection([]); setRunPage(null); setRuns([]);
          }} labelPrefix="数据" />
          <span>{runPage?.total ?? 0} 条记录</span>
          {section === "evaluation" ? <div className="evaluation-actions">
            <label><input type="checkbox" checked={evaluators.includes("rynnvalue")} onChange={(event) => setEvaluators((current) => event.target.checked ? [...current, "rynnvalue"] : current.filter((value) => value !== "rynnvalue"))} />RynnValue</label>
            <label title={robometerCapability?.reason ?? undefined}><input type="checkbox" disabled={robometerCapability?.available === false} checked={evaluators.includes("robometer")} onChange={(event) => setEvaluators((current) => event.target.checked ? [...current, "robometer"] : current.filter((value) => value !== "robometer"))} />Robometer{robometerCapability?.available === false ? "（未配置）" : ""}</label>
            <label><input type="checkbox" checked={batchOverwrite} onChange={(event) => setBatchOverwrite(event.target.checked)} />覆盖已有评价</label>
            <button disabled={busy || !taskId || !evaluators.length} onClick={() => void evaluate(null, batchOverwrite)}>批量评价</button><button className="primary" disabled={busy || !taskId || !evaluationSelection.length || !evaluators.length} onClick={() => void evaluate(evaluationSelection, batchOverwrite)}>{batchOverwrite ? "评价所选（覆盖已有）" : "评价所选（仅补缺失）"}</button>
          </div> : <><button className="primary" disabled={!taskId} onClick={() => openBuilder()}>创建训练数据集</button><a className="export-button" href={taskId ? datasetExportUrl(taskId) : undefined} download aria-disabled={!taskId}>导出任务 ZIP</a></>}
          {!taskId && <span className="task-scope-note">请选择具体提示词后进行评价、打包或导出。</span>}
        </div>
        {runsLoading && !runPage ? <div className="empty-table">正在加载轨迹索引…</div> : <RunTable runs={runs} selectable={section === "evaluation" || (builder && selection.mode === "manual")} excludeTests={section === "package"} selected={section === "evaluation" ? evaluationSelection : selection.run_ids} onToggle={(runId, checked) => section === "evaluation" ? setEvaluationSelection((current) => checked ? [...current, runId] : current.filter((value) => value !== runId)) : patchSelection({ run_ids: checked ? [...selection.run_ids, runId] : selection.run_ids.filter((value) => value !== runId) })} onOpen={(runId) => void openDetail(runId)} />}
        <div className="table-pagination"><button disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>上一页</button><span>第 {runPage?.page ?? page} / {runPage?.pages ?? 1} 页</span><button disabled={page >= (runPage?.pages ?? 1)} onClick={() => setPage((value) => value + 1)}>下一页</button><label>每页<select value={pageSize} onChange={(event) => { setPageSize(Number(event.target.value)); setPage(1); }}>{[5, 10, 20, 50].map((value) => <option key={value} value={value}>{value}</option>)}</select></label></div>
      </div>

      {section === "package" && builder && <div className="surface dataset-builder">
        <div className="panel-title"><strong>{parentId ? "调整成员并另存数据集" : "创建训练数据集"}</strong><button onClick={() => setBuilder(false)}>取消</button></div>
        <div className="builder-grid">
          <label>名称<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <label>成功后动作<select value={String(includePostSuccess)} onChange={(event) => setIncludePostSuccess(event.target.value === "true")}><option value="true">保留</option><option value="false">截断</option></select></label>
          <label>选择方式<select value={selection.mode} onChange={(event) => patchSelection({ mode: event.target.value as DatasetSelection["mode"] })}><option value="random">随机选择</option><option value="sequential">顺序选择</option><option value="rule">分类配额</option><option value="manual">手动选择</option></select></label>
          {selection.mode !== "manual" && selection.mode !== "rule" && <label>轨迹数量 M<input type="number" min={1} max={eligibleCount} value={selection.size ?? 1} onChange={(event) => patchSelection({ size: Number(event.target.value) })} /></label>}
          {(selection.mode === "random" || selection.mode === "rule") && <label>随机种子<input type="number" min={0} value={selection.seed} onChange={(event) => patchSelection({ seed: Number(event.target.value) })} /></label>}
          {selection.mode === "sequential" && <label>时间顺序<select value={selection.order} onChange={(event) => patchSelection({ order: event.target.value as "oldest" | "newest" })}><option value="newest">最新优先</option><option value="oldest">最早优先</option></select></label>}
          <fieldset><legend>来源过滤</legend>{sources.map((value) => <label key={value}><input type="checkbox" checked={selection.source_types.includes(value)} onChange={() => toggleSource(value)} />{sourceLabel[value]}</label>)}</fieldset>
          <fieldset><legend>结果过滤</legend>{outcomes.map((value) => <label key={value}><input type="checkbox" checked={selection.outcomes.includes(value)} onChange={() => toggleOutcome(value)} />{outcomeLabel[value]}</label>)}</fieldset>
          {selection.mode === "manual" && <div className="manual-selection-note">已手动选择 <strong>{selectedSet.size}</strong> 条。请在上方轨迹表中勾选；不可训练记录已禁用。</div>}
          {selection.mode === "rule" && <div className="quota-grid">{selection.quotas.map((quota, index) => <div key={`${quota.source_type}:${quota.outcome}`}><span>{sourceLabel[quota.source_type]} · {outcomeLabel[quota.outcome]}</span><input type="number" min={0} value={quota.count} onChange={(event) => { const quotas = [...selection.quotas]; quotas[index] = { ...quota, count: Number(event.target.value) }; patchSelection({ quotas }); }} /><select value={quota.order} onChange={(event) => { const quotas = [...selection.quotas]; quotas[index] = { ...quota, order: event.target.value as typeof quota.order }; patchSelection({ quotas }); }}><option value="random">随机</option><option value="newest">最新</option><option value="oldest">最早</option></select></div>)}</div>}
          <details><summary>数据准备高级设置</summary><div className="advanced-grid"><label>验证集比例<input type="number" min={0} max={0.9} step={0.05} value={validationFraction} onChange={(event) => setValidationFraction(Number(event.target.value))} /></label><label>划分种子<input type="number" min={0} value={splitSeed} onChange={(event) => setSplitSeed(Number(event.target.value))} /></label><label>连续成功阈值<input type="number" min={1} max={100} value={successSteps} onChange={(event) => setSuccessSteps(Number(event.target.value))} /></label></div></details>
        </div>
        <div className="builder-actions"><button onClick={() => void makePreview()} disabled={busy}>预览选择结果</button><button className="primary" onClick={() => void saveDataset()} disabled={busy || !preview || !name.trim()}>冻结数据集</button></div>
        {preview && <div className="selection-preview"><strong>{preview.selected_count} 条轨迹</strong><span>{preview.action_count} actions</span><span>{preview.chunk_count} chunks</span>{Object.entries(preview.categories).map(([key, count]) => <Badge key={key}>{key} · {count}</Badge>)}</div>}
      </div>}

      {section === "package" && <div className="surface frozen-datasets">
        <div className="panel-title"><strong>已打包数据集</strong><span>{datasets.length}</span></div>
        {datasetsLoading && !datasets.length ? <div className="empty-table">正在加载冻结数据集…</div> : datasets.length ? datasets.map((dataset) => <FrozenDatasetCard key={dataset.id} dataset={dataset} disabled={busy}
          robometerUnavailable={robometerCapability?.available === false ? robometerCapability.reason ?? "Robometer 环境未配置" : undefined}
          onRemove={() => void removeDataset(dataset)} onDerive={() => openBuilder(dataset)} onRefresh={refresh}
          onJob={setAnnotationJob} onError={setError} onOpen={(runId, datasetId) => void openDetail(runId, datasetId)} />) : <div className="empty-table">尚未创建训练数据集。</div>}
      </div>}
      {monitor}
      <div className="path-card"><span>数据根目录</span><code>{summary.dataset_root}</code><span>目录索引</span><code>{summary.catalog}</code></div>
    </>}
  </section>;
}
