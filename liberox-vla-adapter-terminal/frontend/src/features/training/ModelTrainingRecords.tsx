import { useEffect, useId, useRef, useState } from "react";
import { getModelTrainingConfig } from "../run-control/api";
import type { OfflineJob, TrainingConfiguration } from "../run-control/types";
import { Badge } from "../../components/ui/Badge";

function TrainingRecord({ policyId, job }: { policyId: string; job: OfflineJob }) {
  const configId = useId();
  const [expanded, setExpanded] = useState(false);
  const [config, setConfig] = useState<TrainingConfiguration | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const request = useRef(0);
  useEffect(() => () => { request.current += 1; }, []);
  const load = async () => {
    if (config || loading) return;
    const token = ++request.current;
    setLoading(true); setError("");
    try {
      const result = await getModelTrainingConfig(policyId, job.id);
      if (request.current === token) setConfig(result);
    } catch (reason) { if (request.current === token) setError(String(reason)); }
    finally { if (request.current === token) setLoading(false); }
  };
  const toggleConfig = () => {
    setExpanded(!expanded);
    if (!expanded) void load();
  };
  const download = () => {
    if (!config) return;
    const url = URL.createObjectURL(new Blob([config.text], { type: "text/yaml;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url; link.download = `${job.id}_${config.filename}`; link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };
  const parameters = job.parameters;
  const model = parameters.model as Record<string, unknown> | undefined;
  const reward = parameters.reward as Record<string, unknown> | undefined;
  const method = String(parameters.algorithm ?? "—").toUpperCase();
  const title = `${method} · ${String(parameters.dataset_name ?? job.dataset_id ?? "数据集未知")}`;
  return <article className="model-training-record" aria-label={title}>
    <header className="model-training-record-head">
      <div><strong>{title}</strong>
        <span>生成当前模型</span></div>
      <Badge tone={job.status === "COMPLETED" ? "green" : job.status === "FAILED" ? "red" : "neutral"}>{job.status}</Badge>
    </header>
    <dl className="model-training-summary">
      <div className="model-training-field-wide"><dt>任务</dt><dd>{String(parameters.task_id ?? "—")}</dd></div>
      <div className="model-training-field-wide"><dt>基础模型</dt><dd>{String(model?.base_id ?? "—")}</dd></div>
      <div><dt>训练步数</dt><dd>{String(parameters.train_steps ?? "—")}</dd></div>
      <div><dt>Batch size</dt><dd>{String(parameters.micro_batch_size ?? "—")}</dd></div>
      <div><dt>梯度累积</dt><dd>{String(parameters.gradient_accumulation_steps ?? "—")}</dd></div>
      <div><dt>Seed</dt><dd>{String(parameters.seed ?? "—")}</dd></div>
      <div className="model-training-field-wide"><dt>奖励</dt><dd>{method === "BC" ? "无需奖励" : String(reward?.source ?? "未记录")}</dd></div>
    </dl>
    <div className="model-training-meta">
      <time dateTime={job.created_at}>{job.created_at ? new Date(job.created_at).toLocaleString() : "—"}</time>
      <code>{job.id}</code>
    </div>
    <button className="model-training-toggle" aria-expanded={expanded} aria-controls={configId} onClick={toggleConfig}>
      {expanded ? "收起训练 YAML" : "查看训练 YAML"}<span aria-hidden="true">{expanded ? "⌃" : "⌄"}</span>
    </button>
    {expanded && <div id={configId} className="model-training-yaml">
      {loading && <p role="status">加载 YAML…</p>}
      {error && <div className="model-training-config-error"><p role="alert">{error}</p><button onClick={() => void load()}>重新加载</button></div>}
      {config && <><div className="model-training-yaml-head"><code>{config.filename}</code><button onClick={download}>下载 YAML</button></div>
        <pre>{config.text}</pre></>}
    </div>}
  </article>;
}

export function ModelTrainingRecords({ policyId, jobs }: { policyId: string; jobs: OfflineJob[] }) {
  return <section className="model-training-history"><h3>训练记录</h3>{jobs.length
    ? <div className="model-training-records">{jobs.map((job) => <TrainingRecord key={`${policyId}:${job.id}`} policyId={policyId} job={job} />)}</div>
    : <div className="empty-table">没有关联的本机训练记录。</div>}</section>;
}
