import { useEffect, useState } from "react";
import { copyModel, deleteModel, getModel, listModels, renameModel } from "../features/run-control/api";
import type { PolicyDetail, PolicyInfo } from "../features/run-control/types";
import { Badge } from "../components/ui/Badge";

export function ModelsPage() {
  const [models, setModels] = useState<PolicyInfo[]>([]);
  const [selected, setSelected] = useState("base");
  const [detail, setDetail] = useState<PolicyDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const refresh = async (policyId = selected) => {
    const values = await listModels();
    setModels(values);
    const target = values.some((value) => value.policy_id === policyId) ? policyId : values[0]?.policy_id;
    if (target) { setSelected(target); setDetail(await getModel(target)); }
  };
  useEffect(() => { void refresh().catch((reason) => setError(String(reason))); }, []);
  const choose = async (id: string) => {
    setSelected(id); setBusy(true); setError("");
    try { setDetail(await getModel(id)); } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const rename = async () => {
    if (!detail || detail.kind === "base") return;
    const label = window.prompt("新的模型显示名称", detail.label)?.trim();
    if (!label) return;
    setBusy(true);
    try { setDetail(await renameModel(detail.policy_id, label)); await refresh(detail.policy_id); }
    catch (reason) { setError(String(reason)); } finally { setBusy(false); }
  };
  const duplicate = async () => {
    if (!detail || detail.kind === "base") return;
    const label = window.prompt("复制模型的名称", `${detail.label} · 副本`)?.trim();
    if (!label) return;
    setBusy(true);
    try { const copied = await copyModel(detail.policy_id, label); await refresh(copied.policy_id); }
    catch (reason) { setError(String(reason)); } finally { setBusy(false); }
  };
  const remove = async () => {
    if (!detail || detail.kind === "base") return;
    const confirmation = window.prompt(`永久删除模型 ${detail.policy_id}\n请输入模型 ID 确认：`);
    if (confirmation !== detail.policy_id) return;
    setBusy(true);
    try { await deleteModel(detail.policy_id); await refresh("base"); }
    catch (reason) { setError(String(reason)); } finally { setBusy(false); }
  };
  return <section className="content-page">
    <div className="page-heading"><p className="eyebrow">MODEL REGISTRY</p><h1>模型</h1><p>浏览基础模型与训练 overlay，管理显示名称、副本和训练来源。</p></div>
    {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
    <div className="models-layout">
      <aside className="surface model-list"><div className="panel-title"><strong>模型库</strong><span>{models.length}</span></div>{models.map((model) => <button key={model.policy_id} className={selected === model.policy_id ? "active" : ""} onClick={() => void choose(model.policy_id)}><span><strong>{model.label}</strong><code>{model.policy_id}</code></span><Badge tone={model.kind === "base" ? "neutral" : "green"}>{model.kind === "base" ? "BASE" : (model.algorithm ?? "iql").toUpperCase()}</Badge></button>)}</aside>
      <main className="surface model-detail">{busy && !detail ? <div className="empty-table">加载模型信息…</div> : detail ? <>
        <div className="model-detail-head"><div><p className="eyebrow">{detail.kind}</p><h2>{detail.label}</h2><code>{detail.policy_id}</code></div><div>{detail.kind !== "base" && <><button disabled={busy} onClick={() => void rename()}>重命名</button><button disabled={busy} onClick={() => void duplicate()}>复制模型</button><button className="danger" disabled={busy} onClick={() => void remove()}>删除</button></>}</div></div>
        <dl className="model-properties"><dt>基础 checkpoint</dt><dd>{detail.base_checkpoint}</dd><dt>Stats key</dt><dd>{detail.stats_key}</dd><dt>训练 step</dt><dd>{detail.training_step ?? "—"}</dd><dt>兼容性哈希</dt><dd><code>{detail.compatibility_sha256 ?? "基础模型"}</code></dd></dl>
        {detail.components.length > 0 && <section><h3>可训练组件</h3><div className="component-list">{detail.components.map((component) => <div key={component.name}><strong>{component.name}</strong><span>{(component.size_bytes / 1024 / 1024).toFixed(2)} MiB</span><code>{component.sha256.slice(0, 16)}…</code></div>)}</div></section>}
        <section><h3>训练记录</h3>{detail.training_records.length ? <div className="model-training-records">{detail.training_records.map((job) => <article key={job.id}><div><strong>{job.id}</strong><Badge tone={job.status === "COMPLETED" ? "green" : job.status === "FAILED" ? "red" : "neutral"}>{job.status}</Badge></div><p>数据集 {job.dataset_id ?? "—"} · {job.created_at ? new Date(job.created_at).toLocaleString() : "—"}</p><code>{String(job.output_path ?? "")}</code></article>)}</div> : <div className="empty-table">没有匹配的本机训练记录；从其他设备复制的 overlay 仍可正常推理。</div>}</section>
      </> : <div className="empty-table">模型库为空</div>}</main>
    </div>
  </section>;
}
