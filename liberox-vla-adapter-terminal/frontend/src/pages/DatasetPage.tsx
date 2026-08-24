import { useEffect, useMemo, useState } from "react";
import {
  annotateTrainingDataset, createTrainingDataset, datasetExportUrl, deriveTrainingDataset,
  deleteTrainingDataset, getBootstrap, getDatasetSummary, listDatasetRuns, listOfflineJobs,
  listTrainingDatasets, previewTrainingDataset, verifyTrainingDataset,
} from "../features/run-control/api";
import type {
  Bootstrap, DatasetPreview, DatasetSelection, DatasetSummary, OfflineJob,
  Session, TrainingDataset,
} from "../features/run-control/types";
import { SuccessRateChart } from "../features/metrics/SuccessRateChart";
import { RunTable } from "../features/dataset/RunTable";
import { JobMonitor } from "../features/training/JobMonitor";
import { Badge } from "../components/ui/Badge";

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
  const [taskId, setTaskId] = useState("");
  const [runs, setRuns] = useState<Session[]>([]);
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [selection, setSelection] = useState<DatasetSelection>(initialSelection);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [builder, setBuilder] = useState(false);
  const [name, setName] = useState("");
  const [parentId, setParentId] = useState<string | null>(null);
  const [validationFraction, setValidationFraction] = useState(0.2);
  const [splitSeed, setSplitSeed] = useState(7);
  const [successSteps, setSuccessSteps] = useState(5);
  const [annotationJob, setAnnotationJob] = useState<OfflineJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = async (nextTask = taskId) => {
    if (!nextTask) return;
    const [nextRuns, nextDatasets, jobs] = await Promise.all([
      listDatasetRuns(nextTask), listTrainingDatasets(nextTask), listOfflineJobs(),
    ]);
    setRuns(nextRuns); setDatasets(nextDatasets);
    const active = jobs.find((job) => job.kind === "annotation" && ["STARTING", "RUNNING", "STOPPING"].includes(job.status));
    if (active) setAnnotationJob(active);
  };

  useEffect(() => {
    void Promise.all([getDatasetSummary(), getBootstrap()])
      .then(([nextSummary, nextBootstrap]) => {
        setSummary(nextSummary); setBootstrap(nextBootstrap);
        setTaskId(nextBootstrap.task.task_id);
        return refresh(nextBootstrap.task.task_id);
      })
      .catch((reason) => setError(String(reason)));
  }, []);
  useEffect(() => { if (taskId) void refresh(taskId).catch((reason) => setError(String(reason))); }, [taskId]);

  const taskStats = summary?.tasks.find((task) => task.task_id === taskId);
  const eligible = runs.filter((run) => run.training_eligible);
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
      setParentId(parent.id); setName(`${parent.name} · 派生`);
      setValidationFraction(parent.validation_fraction ?? 0.2);
      setSplitSeed(parent.split_seed ?? 7); setSuccessSteps(parent.success_consecutive_steps ?? 5);
      setSelection({ ...initialSelection(), mode: "manual", run_ids: parent.members.map((item) => item.run_id) });
    } else {
      setParentId(null); setName(`训练数据集 ${new Date().toLocaleString()}`);
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
    };
    try {
      if (parentId) await deriveTrainingDataset(parentId, body);
      else await createTrainingDataset({ ...body, task_id: taskId });
      setBuilder(false); setPreview(null); await refresh();
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const annotate = async (dataset: TrainingDataset) => {
    setBusy(true); setError("");
    try { setAnnotationJob(await annotateTrainingDataset(dataset.id)); await refresh(); }
    catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const removeDataset = async (dataset: TrainingDataset) => {
    const action = dataset.annotation_status === "NOT_STARTED" ? "取消冻结" : "删除数据集";
    if (!window.confirm(`${action}“${dataset.name}”？\n\n将删除该数据集清单和专属标注目录，但不会删除源轨迹或全局共享奖励缓存。`)) return;
    setBusy(true); setError("");
    try { await deleteTrainingDataset(dataset.id); await refresh(); }
    catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };

  return <section className="content-page">
    <div className="page-heading"><p className="eyebrow">DATASET CATALOG</p><h1>数据集</h1><p>按控制来源检索轨迹，冻结可复现的数据集版本，并在独立 RynnValue 环境中完成奖励标注。</p></div>
    {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
    {summary && <>
      <div className="dataset-overview surface"><SuccessRateChart rate={taskStats?.success_rate ?? 0} /><div className="dataset-stats"><div><strong>{taskStats?.runs ?? 0}</strong><span>当前任务运行</span></div><div><strong>{eligible.length}</strong><span>可训练轨迹</span></div><div><strong>{datasets.length}</strong><span>冻结数据集</span></div><div><strong>{datasets.filter((item) => item.annotation_status === "READY").length}</strong><span>已完成标注</span></div></div></div>
      <div className="surface dataset-browser">
        <div className="dataset-toolbar">
          <label>任务<select value={taskId} onChange={(event) => { setTaskId(event.target.value); setBuilder(false); }}>{bootstrap?.task_catalog.map((task) => <option value={task.task_id} key={task.task_id}>{task.prompt}</option>)}</select></label>
          <span>{runs.length} 条记录</span>
          <button className="primary" onClick={() => openBuilder()}>创建训练数据集</button>
          <a className="export-button" href={taskId ? datasetExportUrl(taskId) : undefined} download aria-disabled={!taskId}>导出任务 ZIP</a>
        </div>
        <RunTable runs={runs} selectable={builder && selection.mode === "manual"} selected={selection.run_ids} onToggle={(runId, checked) => patchSelection({ run_ids: checked ? [...selection.run_ids, runId] : selection.run_ids.filter((value) => value !== runId) })} />
      </div>

      {builder && <div className="surface dataset-builder">
        <div className="panel-title"><strong>{parentId ? "派生数据集版本" : "创建不可变训练数据集"}</strong><button onClick={() => setBuilder(false)}>取消</button></div>
        <div className="builder-grid">
          <label>名称<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <label>选择方式<select value={selection.mode} onChange={(event) => patchSelection({ mode: event.target.value as DatasetSelection["mode"] })}><option value="random">随机选择</option><option value="sequential">顺序选择</option><option value="rule">分类配额</option><option value="manual">手动选择</option></select></label>
          {selection.mode !== "manual" && selection.mode !== "rule" && <label>轨迹数量 M<input type="number" min={1} max={eligible.length} value={selection.size ?? 1} onChange={(event) => patchSelection({ size: Number(event.target.value) })} /></label>}
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

      <div className="surface frozen-datasets">
        <div className="panel-title"><strong>冻结数据集版本</strong><span>{datasets.length}</span></div>
        {datasets.length ? datasets.map((dataset) => <article key={dataset.id} className="dataset-card">
          <div><h2>{dataset.name}</h2><code>{dataset.id}</code><p>{dataset.member_count} 条 · 预计 {dataset.action_count} actions · {dataset.chunk_count} chunks</p></div>
          <div className="dataset-badges"><Badge tone={dataset.integrity_status === "HEALTHY" ? "green" : "red"}>{dataset.integrity_status}</Badge><Badge tone={dataset.annotation_status === "READY" ? "green" : dataset.annotation_status === "ERROR" ? "red" : "neutral"}>{dataset.annotation_status}</Badge></div>
          <div className="dataset-card-actions"><button className="danger" disabled={busy || dataset.annotation_status === "RUNNING"} onClick={() => void removeDataset(dataset)}>{dataset.annotation_status === "NOT_STARTED" ? "取消冻结" : "删除数据集"}</button><button onClick={() => void verifyTrainingDataset(dataset.id).then(() => refresh()).catch((reason) => setError(String(reason)))}>验证完整性</button><button onClick={() => openBuilder(dataset)}>派生版本</button><button className="primary" disabled={busy || dataset.integrity_status !== "HEALTHY" || dataset.annotation_status === "RUNNING"} onClick={() => void annotate(dataset)}>{dataset.annotation_status === "READY" ? "重新标注" : "开始标注"}</button></div>
          {dataset.integrity_error && <p className="dataset-integrity-error">{dataset.integrity_error}</p>}
        </article>) : <div className="empty-table">尚未创建训练数据集。</div>}
      </div>
      {annotationJob && <JobMonitor initial={annotationJob} onUpdate={(job) => { setAnnotationJob(job); if (["COMPLETED", "FAILED", "CANCELED"].includes(job.status)) void refresh(); }} />}
      <div className="path-card"><span>数据根目录</span><code>{summary.dataset_root}</code><span>目录索引</span><code>{summary.catalog}</code></div>
    </>}
  </section>;
}
