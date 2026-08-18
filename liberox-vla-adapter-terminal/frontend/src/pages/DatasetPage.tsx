import { useEffect, useState } from "react";
import { datasetExportUrl, getBootstrap, getDatasetSummary, listDatasetRuns } from "../features/run-control/api";
import type { Bootstrap, DatasetSummary, Session } from "../features/run-control/types";
import { SuccessRateChart } from "../features/metrics/SuccessRateChart";
import { RunTable } from "../features/dataset/RunTable";

export function DatasetPage() {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [taskId, setTaskId] = useState("");
  const [runs, setRuns] = useState<Session[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    void Promise.all([getDatasetSummary(), getBootstrap()])
      .then(([nextSummary, nextBootstrap]) => {
        setSummary(nextSummary);
        setBootstrap(nextBootstrap);
        setTaskId(nextBootstrap.task.task_id);
      })
      .catch((reason) => setError(String(reason)));
  }, []);
  useEffect(() => {
    if (!taskId) return;
    void listDatasetRuns(taskId).then(setRuns).catch((reason) => setError(String(reason)));
  }, [taskId]);
  const taskStats = summary?.tasks.find((task) => task.task_id === taskId);
  return <section className="content-page">
    <div className="page-heading"><p className="eyebrow">DATASET CATALOG</p><h1>数据集</h1><p>按任务检索运行记录，并导出可供 offline RL 后训练拆解的数据包。</p></div>
    {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
    {summary && <>
      <div className="dataset-overview surface"><SuccessRateChart rate={taskStats?.success_rate ?? 0} /><div className="dataset-stats"><div><strong>{taskStats?.runs ?? 0}</strong><span>当前任务运行</span></div><div><strong>{taskStats?.successes ?? 0}</strong><span>当前任务成功</span></div><div><strong>{summary.errors}</strong><span>全部错误</span></div><div><strong>{summary.legacy_indexed}</strong><span>旧数据（只读）</span></div></div></div>
      <div className="surface dataset-browser">
        <div className="dataset-toolbar">
          <label>任务<select value={taskId} onChange={(event) => setTaskId(event.target.value)}>{bootstrap?.task_catalog.map((task) => <option value={task.task_id} key={task.task_id}>{task.prompt}</option>)}</select></label>
          <span>{runs.length} 条记录</span>
          <a className="export-button" href={taskId ? datasetExportUrl(taskId) : undefined} download aria-disabled={!taskId}>导出当前任务 ZIP</a>
        </div>
        <RunTable runs={runs} />
        <p className="export-note">ZIP 保留 run/episode 目录结构，包含运行清单、配置、结果、trajectory CSV、agentview.mp4、VLA 双视角 vla_views.mp4 和 DATA_FORMAT.md；NPZ、observation 与图表不导出。</p>
      </div>
      <div className="surface task-summary"><h2>按任务统计</h2>{summary.tasks.map((task) => <button key={task.task_id} className={task.task_id === taskId ? "selected-task" : ""} onClick={() => setTaskId(task.task_id)}><span>{task.task_name}</span><strong>{task.successes}/{task.runs} · {(task.success_rate * 100).toFixed(1)}%</strong></button>)}</div>
      <div className="path-card"><span>数据根目录</span><code>{summary.dataset_root}</code><span>目录索引</span><code>{summary.catalog}</code></div>
    </>}
  </section>;
}
