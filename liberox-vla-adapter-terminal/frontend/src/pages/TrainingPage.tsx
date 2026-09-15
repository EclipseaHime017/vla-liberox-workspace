import { useEffect, useMemo, useRef, useState } from "react";
import {
  getBootstrap, getTensorBoard, getTrainingDefaults, listOfflineJobs,
  listTrainingDatasets, startTensorBoard, startTraining,
} from "../features/run-control/api";
import type {
  Bootstrap, OfflineJob, TensorBoardStatus, TrainingDataset, TrainingDefaults, TrainingRewardSource,
} from "../features/run-control/types";
import { JobMonitor } from "../features/training/JobMonitor";
import { Badge } from "../components/ui/Badge";
import { FrozenDatasetCard } from "../features/dataset/FrozenDatasetCard";
import { rewardSourceLabels } from "../features/dataset/rewardVersions";

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
  const [rewardSource, setRewardSource] = useState<TrainingRewardSource>("rynnvalue");
  const [rewardRevision, setRewardRevision] = useState(0);
  const [showDatasetConfig, setShowDatasetConfig] = useState(false);
  const [parameters, setParameters] = useState<Record<string, number | string | boolean | null>>({});
  const [job, setJob] = useState<OfflineJob | null>(null);
  const [tensorboard, setTensorboard] = useState<TensorBoardStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [defaultsLoading, setDefaultsLoading] = useState(false);
  const [defaultsKey, setDefaultsKey] = useState("");
  const [error, setError] = useState("");
  const [defaultsError, setDefaultsError] = useState("");
  const taskIdRef = useRef(taskId);
  const refreshRequest = useRef(0);
  taskIdRef.current = taskId;
  const requestedDefaultsKey = JSON.stringify([datasetId, rewardSource, rewardRevision]);

  useEffect(() => {
    void Promise.all([getBootstrap(), getTrainingDefaults(), listOfflineJobs(), getTensorBoard()])
      .then(([nextBootstrap, nextDefaults, jobs, board]) => {
        setBootstrap(nextBootstrap); setDefaults(nextDefaults); setTaskId(nextBootstrap.task.task_id);
        const { reward_rynnvalue: legacyRynn, ...advanced } = nextDefaults.advanced;
        const configuredSource = advanced.reward_source ?? (legacyRynn === false ? "sparse" : "rynnvalue");
        setRewardSource(configuredSource === "stage" || configuredSource === "sparse" ? configuredSource : "rynnvalue");
        setParameters({ ...nextDefaults.basic, ...advanced, ...nextDefaults.monitoring,
          reward_source: advanced.reward_source ?? (legacyRynn === false ? "sparse" : "rynnvalue"),
          reward_stage_exponent: advanced.reward_stage_exponent ?? 2, resume_checkpoint: null });
        setTensorboard(board);
        const active = jobs.find(
          (item) => ["training", "annotation"].includes(item.kind) && activeJobStates.has(item.status),
        );
        if (active) setJob(active);
      }).catch((reason) => setError(String(reason)));
  }, []);
  useEffect(() => {
    if (!taskId) return;
    let current = true;
    setDatasetsLoading(true); setDatasets([]); setDatasetId("");
    void listTrainingDatasets(taskId).then((values) => {
      if (current) setDatasets(values.filter((item) => item.integrity_status === "HEALTHY"));
    }).catch((reason) => { if (current) setError(String(reason)); })
      .finally(() => { if (current) setDatasetsLoading(false); });
    return () => { current = false; };
  }, [taskId]);
  const availableDatasets = datasets;
  useEffect(() => {
    setDatasetId((current) => availableDatasets.some((item) => item.id === current)
      ? current : availableDatasets.find((item) => item.reward_version_id || item.annotation_status === "READY")?.id ?? availableDatasets[0]?.id ?? "");
  }, [availableDatasets]);
  useEffect(() => {
    const timer = window.setInterval(() => {
      void getTensorBoard().then(setTensorboard).catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, []);
  const dataset = useMemo(() => datasets.find((item) => item.id === datasetId) ?? null, [datasets, datasetId]);
  useEffect(() => {
    setDefaultsError("");
    setDefaultsKey("");
    if (!datasetId) { setDefaultsLoading(false); return; }
    let current = true;
    setDefaultsLoading(true);
    void getTrainingDefaults(datasetId, rewardSource).then((next) => {
      if (!current) return;
      setDefaults(next);
      setDefaultsKey(requestedDefaultsKey);
      setParameters((current) => {
        const pinned = Object.fromEntries(Object.entries(next.advanced)
          .filter(([key]) => key.startsWith("reward_") && key !== "reward_rynnvalue"));
        const unrelated = Object.fromEntries(Object.entries(current).filter(([key]) => !key.startsWith("reward_")));
        return { ...unrelated, ...pinned,
          reward_source: rewardSource,
          reward_version_id: next.reward_version?.id ?? null,
          resume_checkpoint: next.checkpoints.some(
            (checkpoint) => checkpoint.path === current.resume_checkpoint
          ) ? current.resume_checkpoint : null,
        };
      });
    }).catch((reason) => { if (current) setDefaultsError(String(reason)); })
      .finally(() => { if (current) setDefaultsLoading(false); });
    return () => { current = false; };
  }, [datasetId, rewardSource, rewardRevision]);
  const rewardReady = Boolean(datasetId && defaultsKey === requestedDefaultsKey && (
    defaults?.reward_availability?.ready ?? (defaults?.reward_version
      && defaults.reward_version.evaluator === rewardSource && !defaults.reward_version.legacy
      && ["READY", "COMPLETED"].includes(defaults.reward_version.status))
  ));
  useEffect(() => {
    if (defaultsLoading || !defaults?.reward_availability?.pending) return;
    const timer = window.setTimeout(() => setRewardRevision((value) => value + 1), 1000);
    return () => window.clearTimeout(timer);
  }, [defaultsLoading, defaults]);
  const patchParameter = (name: string, value: number | string | boolean | null) => setParameters((current) => ({ ...current, [name]: value }));
  const refreshDataset = async () => {
    const request = ++refreshRequest.current;
    setDefaultsKey("");
    const requestedTaskId = taskId;
    const values = await listTrainingDatasets(requestedTaskId);
    if (taskIdRef.current !== requestedTaskId || request !== refreshRequest.current) return;
    setDatasets(values.filter((item) => item.integrity_status === "HEALTHY"));
    setRewardRevision((value) => value + 1);
  };
  const updateJob = (next: OfflineJob) => {
    setJob(next);
    if (next.kind === "annotation" && !activeJobStates.has(next.status) && job?.status !== next.status) {
      void refreshDataset().catch((reason) => setError(String(reason)));
    }
  };
  const begin = async () => {
    if (busy || datasetsLoading || defaultsLoading || !rewardReady || !availableDatasets.some((item) => item.id === datasetId)) return;
    setBusy(true); setError("");
    try {
      const next = await startTraining(datasetId, { ...parameters, reward_source: rewardSource });
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
    <div className="page-heading"><p className="eyebrow">OFFLINE RL TRAINING</p><h1>VLA-Adapter + Pixel-IQL</h1><p>选择已评价的数据集，后训练 action head 与 proprio projector。Discount ratio 与 cumulative reward 可为本次训练单独调整。</p></div>
    {(error || defaultsError) && <div className="error-banner"><span>{error || defaultsError}</span><button onClick={() => { setError(""); setDefaultsError(""); }}>关闭</button></div>}
    <div className="training-layout">
      <section className="surface training-config">
        <div className="panel-title"><strong>训练配置</strong><span>{defaults?.environments.training ?? "vla-liberox"}</span></div>
        <div className="training-form">
          <label>任务<select value={taskId} onChange={(event) => setTaskId(event.target.value)}>{bootstrap?.task_catalog.map((task) => <option key={task.task_id} value={task.task_id}>{task.prompt}</option>)}</select></label>
          <label>冻结数据集<select value={datasetId} disabled={datasetsLoading} onChange={(event) => setDatasetId(event.target.value)}><option value="">{datasetsLoading ? "加载数据集…" : "请选择"}</option>{availableDatasets.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.member_count} 条</option>)}</select></label>
          <label>Reward 来源<select value={rewardSource} disabled={busy || datasetsLoading} onChange={(event) => {
            const source = event.target.value as TrainingRewardSource;
            setRewardSource(source);
            setParameters((current) => ({ ...current, reward_source: source, reward_version_id: null }));
          }}><option value="sparse">Sparse</option><option value="rynnvalue">RynnValue</option><option value="stage">Stage-based</option></select></label>
          {dataset && <div className="training-dataset-summary"><strong>{dataset.member_count} 条轨迹</strong><span>{dataset.action_count} actions</span><span>{dataset.chunk_count} chunks</span><Badge tone="green">完整性正常</Badge>
            <p>{defaultsLoading || defaultsKey !== requestedDefaultsKey ? "正在读取评价结果…" : rewardReady
              ? `当前训练奖励：${rewardSourceLabels[rewardSource]}${defaults?.reward_availability?.origin === "global" ? " · 使用逐轨迹全局结果" : ""}`
              : `请先完成 ${rewardSourceLabels[rewardSource]} 评价，或确保每条轨迹已有同类型全局结果。`}</p>
            {!defaultsLoading && defaultsKey === requestedDefaultsKey && defaults?.reward_availability?.message &&
              <p className="error-banner">{defaults.reward_availability.message}</p>}
            {!defaultsLoading && defaultsKey === requestedDefaultsKey && !rewardReady && defaults?.reward_availability?.errors?.map((item) =>
              <p key={item.run_id} className="error-banner">{item.run_id}：{item.error}</p>)}
            <p>新数据集默认继承同类型全局评价，已有标签可直接训练，无需再次评价。只有数据集重新评价后才使用专属结果；修改 γ / cumulative reward 只影响本次训练。</p>
            <button aria-expanded={showDatasetConfig} onClick={() => setShowDatasetConfig((value) => !value)}>配置数据集评价</button></div>}
          {dataset && showDatasetConfig && <FrozenDatasetCard key={dataset.id} dataset={dataset} initialExpanded
            disabled={busy || Boolean(job && activeJobStates.has(job.status))}
            robometerUnavailable={bootstrap?.evaluation_capabilities?.robometer?.available === false
              ? bootstrap.evaluation_capabilities.robometer.reason ?? "Robometer 不可用" : undefined}
            onRefresh={refreshDataset} onJob={setJob} onError={setError} />}
          <div className="parameter-grid">{basicFields.map(([name, label, step]) => <label key={name}>{label}<input type="number" min={name === "critic_warmup_steps" || name === "seed" ? 0 : 1} step={step} value={String(parameters[name] ?? "")} onChange={(event) => patchParameter(name, Number(event.target.value))} /></label>)}</div>
          {defaults?.checkpoints.length ? <label>断点恢复<select value={String(parameters.resume_checkpoint ?? "")} onChange={(event) => patchParameter("resume_checkpoint", event.target.value || null)}><option value="">不恢复</option>{defaults.checkpoints.map((checkpoint) => <option value={checkpoint.path} key={checkpoint.path}>{checkpoint.label}</option>)}</select></label> : null}
          <details><summary>高级 IQL 参数</summary><div className="parameter-grid advanced-parameters">
            <label>Discount ratio γ<input type="number" min={0} max={1} step="any" value={String(parameters.reward_gamma ?? "")} disabled={defaultsLoading || !rewardReady}
              onChange={(event) => patchParameter("reward_gamma", Number(event.target.value))} /></label>
            <label>cumulative reward<select disabled={defaultsLoading || !rewardReady} value={String(Boolean(parameters.reward_accumulate_primitive_steps))}
              onChange={(event) => patchParameter("reward_accumulate_primitive_steps", event.target.value === "true")}><option value="false">Off</option><option value="true">On</option></select></label>
            <label>Critic optimizer<select value={String(parameters.critic_optimizer ?? "adamw")} onChange={(event) => patchParameter("critic_optimizer", event.target.value)}><option value="adam">Adam</option><option value="adamw">AdamW</option></select></label>
            <label>Value optimizer<select value={String(parameters.value_optimizer ?? "adamw")} onChange={(event) => patchParameter("value_optimizer", event.target.value)}><option value="adam">Adam</option><option value="adamw">AdamW</option></select></label>
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
          <button className="primary start-training" disabled={busy || datasetsLoading || defaultsLoading || !rewardReady || !availableDatasets.some((item) => item.id === datasetId) || Boolean(job && activeJobStates.has(job.status))} onClick={() => void begin()}>开始训练</button>
        </div>
      </section>
      <section className="surface tensorboard-card">
        <div className="panel-title"><strong>TensorBoard</strong><span>{tensorboard?.running ? "ONLINE" : "OFFLINE"}</span></div>
        <div><span className={`tensorboard-dot ${tensorboard?.running ? "online" : ""}`} /><strong>{tensorboard?.url ?? "http://127.0.0.1:6006/"}</strong><p>集中查看所有平台训练的 loss、Q/V、advantage、动作误差、吞吐与显存曲线。</p>{tensorboard?.running ? <a className="export-button" href={tensorboard.url} target="_blank" rel="noreferrer">打开 TensorBoard</a> : <button onClick={() => void startBoard()} disabled={busy}>启动 TensorBoard</button>}</div>
      </section>
    </div>
    {job && <>
      <JobMonitor initial={job} onUpdate={updateJob} onDismiss={() => setJob(null)} />
      {job.training_summary && <section className="surface training-result"><div className="panel-title"><strong>训练结果</strong><span>{job.status}</span></div><div><span>Overlay</span><code>{String(job.training_summary.policy_overlay ?? "尚未发布")}</code><span>数据哈希</span><code>{String(job.training_summary.dataset_sha256 ?? "—")}</code></div></section>}
    </>}
  </section>;
}
