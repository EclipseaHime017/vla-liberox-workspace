import { useEffect, useMemo, useState } from "react";
import {
  getBootstrap, getTensorBoard, getTrainingDefaults, listOfflineJobs,
  listTrainingDatasets, startTensorBoard, startTraining,
} from "../features/run-control/api";
import type {
  Bootstrap, OfflineJob, TensorBoardStatus, TrainingDataset, TrainingDefaults,
} from "../features/run-control/types";
import { JobMonitor } from "../features/training/JobMonitor";
import { Badge } from "../components/ui/Badge";

const basicFields = [
  ["train_steps", "训练步数", 1],
  ["critic_warmup_steps", "Critic warmup", 1],
  ["micro_batch_size", "Micro batch size", 1],
  ["gradient_accumulation_steps", "梯度累积", 1],
  ["checkpoint_interval", "Checkpoint 间隔", 1],
  ["seed", "随机种子", 1],
] as const;
const advancedFields = [
  ["critic_lr", "Critic LR", "any"], ["value_lr", "Value LR", "any"],
  ["critic_weight_decay", "Critic weight decay", "any"],
  ["critic_max_grad_norm", "Critic 梯度裁剪", "any"],
  ["value_weight_decay", "Value weight decay", "any"],
  ["value_max_grad_norm", "Value 梯度裁剪", "any"],
  ["policy_peak_lr", "Policy peak LR", "any"], ["policy_final_lr", "Policy final LR", "any"],
  ["expectile", "Expectile", "any"], ["beta", "Advantage beta", "any"],
  ["max_advantage_weight", "最大 advantage weight", "any"], ["target_tau", "Target tau", "any"],
  ["console_interval_steps", "日志间隔", 1], ["flush_seconds", "TensorBoard flush", "any"],
] as const;
const activeJobStates = new Set(["STARTING", "RUNNING", "STOPPING"]);

export function TrainingPage() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [defaults, setDefaults] = useState<TrainingDefaults | null>(null);
  const [taskId, setTaskId] = useState("");
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [parameters, setParameters] = useState<Record<string, number | string | boolean | null>>({});
  const [job, setJob] = useState<OfflineJob | null>(null);
  const [tensorboard, setTensorboard] = useState<TensorBoardStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    void Promise.all([getBootstrap(), getTrainingDefaults(), listOfflineJobs(), getTensorBoard()])
      .then(([nextBootstrap, nextDefaults, jobs, board]) => {
        setBootstrap(nextBootstrap); setDefaults(nextDefaults); setTaskId(nextBootstrap.task.task_id);
        setParameters({ ...nextDefaults.basic, ...nextDefaults.advanced, ...nextDefaults.monitoring, resume_checkpoint: null });
        setTensorboard(board);
        const active = jobs.find(
          (item) => item.kind === "training" && activeJobStates.has(item.status),
        );
        if (active) setJob(active);
      }).catch((reason) => setError(String(reason)));
  }, []);
  useEffect(() => {
    if (!taskId) return;
    void listTrainingDatasets(taskId).then((values) => {
      const ready = values.filter((item) => item.annotation_status === "READY" && item.integrity_status === "HEALTHY");
      setDatasets(ready);
      setDatasetId((current) => ready.some((item) => item.id === current) ? current : ready[0]?.id ?? "");
    }).catch((reason) => setError(String(reason)));
  }, [taskId]);
  useEffect(() => {
    const timer = window.setInterval(() => {
      void getTensorBoard().then(setTensorboard).catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, []);
  const dataset = useMemo(() => datasets.find((item) => item.id === datasetId) ?? null, [datasets, datasetId]);
  useEffect(() => {
    if (!datasetId) return;
    void getTrainingDefaults(datasetId).then((next) => {
      setDefaults(next);
      setParameters((current) => ({
        ...current,
        resume_checkpoint: next.checkpoints.some(
          (checkpoint) => checkpoint.path === current.resume_checkpoint
        ) ? current.resume_checkpoint : null,
      }));
    }).catch((reason) => setError(String(reason)));
  }, [datasetId]);
  const patchParameter = (name: string, value: number | string | boolean | null) => setParameters((current) => ({ ...current, [name]: value }));
  const begin = async () => {
    if (!datasetId) return;
    setBusy(true); setError("");
    try {
      const next = await startTraining(datasetId, parameters);
      setJob(next);
      try { setTensorboard(await startTensorBoard()); } catch (reason) { setError(`训练已启动，但 TensorBoard 启动失败：${String(reason)}`); }
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const startBoard = async () => {
    setBusy(true); setError("");
    try { setTensorboard(await startTensorBoard()); }
    catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };

  return <section className="content-page training-page">
    <div className="page-heading"><p className="eyebrow">OFFLINE RL TRAINING</p><h1>RynnValue + Pixel-IQL</h1><p>选择已冻结并完成标注的单任务数据集，在独立进程中后训练 VLA-Adapter action head 与 proprio projector。</p></div>
    {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
    <div className="training-layout">
      <section className="surface training-config">
        <div className="panel-title"><strong>训练配置</strong><span>{defaults?.environments.training ?? "vla-liberox"}</span></div>
        <div className="training-form">
          <label>任务<select value={taskId} onChange={(event) => setTaskId(event.target.value)}>{bootstrap?.task_catalog.map((task) => <option key={task.task_id} value={task.task_id}>{task.prompt}</option>)}</select></label>
          <label>已标注数据集<select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">请选择</option>{datasets.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.member_count} 条</option>)}</select></label>
          {dataset && <div className="training-dataset-summary"><strong>{dataset.member_count} 条轨迹</strong><span>{dataset.action_count} actions</span><span>{dataset.chunk_count} chunks</span><Badge tone="green">完整性正常</Badge><p>训练固定使用该版本全部成员。若需改变 M，请在数据集页面派生并重新标注。</p></div>}
          <div className="parameter-grid">{basicFields.map(([name, label, step]) => <label key={name}>{label}<input type="number" min={name === "critic_warmup_steps" || name === "seed" ? 0 : 1} step={step} value={String(parameters[name] ?? "")} onChange={(event) => patchParameter(name, Number(event.target.value))} /></label>)}</div>
          {defaults?.checkpoints.length ? <label>断点恢复<select value={String(parameters.resume_checkpoint ?? "")} onChange={(event) => patchParameter("resume_checkpoint", event.target.value || null)}><option value="">不恢复</option>{defaults.checkpoints.map((checkpoint) => <option value={checkpoint.path} key={checkpoint.path}>{checkpoint.label}</option>)}</select></label> : null}
          <details><summary>高级 IQL 参数</summary><div className="parameter-grid advanced-parameters">
            <label>Critic optimizer<select value={String(parameters.critic_optimizer ?? "adam")} onChange={(event) => patchParameter("critic_optimizer", event.target.value)}><option value="adam">Adam</option><option value="adamw">AdamW</option></select></label>
            <label>Value optimizer<select value={String(parameters.value_optimizer ?? "adam")} onChange={(event) => patchParameter("value_optimizer", event.target.value)}><option value="adam">Adam</option><option value="adamw">AdamW</option></select></label>
            {advancedFields.map(([name, label, step]) => <label key={name}>{label}<input type="number" min={0} step={step} value={String(parameters[name] ?? "")} onChange={(event) => patchParameter(name, Number(event.target.value))} /></label>)}
          </div></details>
          <details><summary>训练监控</summary><div className="parameter-grid advanced-parameters">
            <label className="training-toggle"><input type="checkbox" checked={Boolean(parameters.tensorboard)} onChange={(event) => patchParameter("tensorboard", event.target.checked)} />写入 TensorBoard</label>
            <label className="training-toggle"><input type="checkbox" checked={Boolean(parameters.wandb_enabled)} onChange={(event) => patchParameter("wandb_enabled", event.target.checked)} />启用 W&amp;B</label>
            <label>W&amp;B mode<select value={String(parameters.wandb_mode ?? "online")} onChange={(event) => patchParameter("wandb_mode", event.target.value)}><option value="online">online</option><option value="offline">offline</option><option value="disabled">disabled</option></select></label>
            <label>W&amp;B project<input value={String(parameters.wandb_project ?? "")} onChange={(event) => patchParameter("wandb_project", event.target.value)} /></label>
            <label>W&amp;B entity<input value={String(parameters.wandb_entity ?? "")} onChange={(event) => patchParameter("wandb_entity", event.target.value || null)} /></label>
            <label>Run name<input value={String(parameters.wandb_run_name ?? "")} onChange={(event) => patchParameter("wandb_run_name", event.target.value || null)} /></label>
            <label>Group<input value={String(parameters.wandb_group ?? "")} onChange={(event) => patchParameter("wandb_group", event.target.value || null)} /></label>
            <label>Tags（逗号分隔）<input value={String(parameters.wandb_tags ?? "")} onChange={(event) => patchParameter("wandb_tags", event.target.value)} /></label>
            <label>W&amp;B 日志间隔<input type="number" min={1} step={1} value={String(parameters.wandb_log_interval_steps ?? "")} onChange={(event) => patchParameter("wandb_log_interval_steps", Number(event.target.value))} /></label>
          </div></details>
          {defaults && <div className="fixed-parameters"><h2>固定兼容项</h2>{Object.entries(defaults.fixed).map(([name, value]) => <span key={name}><b>{name}</b>{String(value)}</span>)}</div>}
          <button className="primary start-training" disabled={busy || !datasetId || Boolean(job && activeJobStates.has(job.status))} onClick={() => void begin()}>开始训练</button>
        </div>
      </section>
      <section className="surface tensorboard-card">
        <div className="panel-title"><strong>TensorBoard</strong><span>{tensorboard?.running ? "ONLINE" : "OFFLINE"}</span></div>
        <div><span className={`tensorboard-dot ${tensorboard?.running ? "online" : ""}`} /><strong>{tensorboard?.url ?? "http://127.0.0.1:6006/"}</strong><p>集中查看所有平台训练的 loss、Q/V、advantage、动作误差、吞吐与显存曲线。</p>{tensorboard?.running ? <a className="export-button" href={tensorboard.url} target="_blank" rel="noreferrer">打开 TensorBoard</a> : <button onClick={() => void startBoard()} disabled={busy}>启动 TensorBoard</button>}</div>
      </section>
    </div>
    {job && <>
      <JobMonitor initial={job} onUpdate={setJob} onDismiss={() => setJob(null)} />
      {job.training_summary && <section className="surface training-result"><div className="panel-title"><strong>训练结果</strong><span>{job.status}</span></div><div><span>Overlay</span><code>{String(job.training_summary.policy_overlay ?? "尚未发布")}</code><span>数据哈希</span><code>{String(job.training_summary.dataset_sha256 ?? "—")}</code></div></section>}
    </>}
  </section>;
}
