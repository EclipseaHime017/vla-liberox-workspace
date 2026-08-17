import { useEffect, useState } from "react";
import { getDatasetSummary } from "../features/run-control/api";
import type { DatasetSummary } from "../features/run-control/types";
import { SuccessRateChart } from "../features/metrics/SuccessRateChart";

export function DatasetPage() {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  useEffect(() => { void getDatasetSummary().then(setSummary); }, []);
  return <section className="content-page"><div className="page-heading"><p className="eyebrow">DATASET CATALOG</p><h1>数据集</h1><p>SQLite 负责索引，run 目录仍是可移植的事实来源。</p></div>{summary && <><div className="dataset-overview surface"><SuccessRateChart rate={summary.success_rate} /><div className="dataset-stats"><div><strong>{summary.runs}</strong><span>运行</span></div><div><strong>{summary.successes}</strong><span>成功</span></div><div><strong>{summary.errors}</strong><span>错误</span></div><div><strong>{summary.legacy_indexed}</strong><span>旧数据（只读）</span></div></div></div><div className="surface task-summary"><h2>按任务统计</h2>{summary.tasks.map((task) => <div key={task.task_id}><span>{task.task_name}</span><strong>{task.successes}/{task.runs} · {(task.success_rate * 100).toFixed(1)}%</strong></div>)}</div><div className="path-card"><span>数据根目录</span><code>{summary.dataset_root}</code><span>目录索引</span><code>{summary.catalog}</code></div></>}</section>;
}
