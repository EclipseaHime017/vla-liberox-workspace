import { useEffect, useMemo, useRef, useState } from "react";
import { TaskFilter } from "../features/run-config/TaskSelector";
import { ALL_TASK_SCOPE, scopeForTask, taskIdsForScope, type TaskScope } from "../features/run-config/taskHierarchy";
import {
  getBootstrap, getTensorBoard, getTrainingDefaults, listOfflineJobs,
  listTrainingDatasets, startTensorBoard, enqueueTraining, getTrainingQueue, getOfflineJob, stopOfflineJob,
} from "../features/run-control/api";
import type {
  Bootstrap, OfflineJob, TensorBoardStatus, TrainingDataset, TrainingDefaults, TrainingQueueState,
} from "../features/run-control/types";
import { JobMonitor } from "../features/training/JobMonitor";
import { TrainingQueuePanel } from "../features/training/TrainingQueuePanel";
import { ModelConfigPanel, modelConfigurationError } from "../features/training/ModelConfigPanel";
import { Badge } from "../components/ui/Badge";
import { FrozenDatasetCard } from "../features/dataset/FrozenDatasetCard";
import { rewardSourceLabels } from "../features/dataset/rewardVersions";

const basicFields = [
  ["train_steps", "训练步数", 1],
  ["critic_warmup_steps", "Critic warmup", 1],
  ["actor_lr_warmup_steps", "Policy LR warmup", 1],
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
  const [taskScope, setTaskScope] = useState<TaskScope>(ALL_TASK_SCOPE);
  const taskId = taskScope.task_id;
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [algorithm, setAlgorithm] = useState<"iql" | "bc">("iql");
  const isBC = algorithm === "bc";
  const [modelFamily, setModelFamily] = useState("vla_adapter");
  const appliedSelection = useRef({ family: "vla_adapter", algorithm: "iql" });
  const [rewardSource, setRewardSource] = useState<"final" | "rynnvalue">("final");
  const [rewardRevision, setRewardRevision] = useState(0);
  const [showDatasetConfig, setShowDatasetConfig] = useState(false);
  const [parameters, setParameters] = useState<Record<string, number | string | boolean | null>>({});
  const [job, setJob] = useState<OfflineJob | null>(null);
  const [queue, setQueue] = useState<TrainingQueueState>({ jobs: [], waiting_reason: null });
  const [stoppingId, setStoppingId] = useState<string | null>(null);
  const jobRef = useRef(job);
  jobRef.current = job;
  const [followQueue, setFollowQueue] = useState(true);
  const followQueueRef = useRef(true);
  const selectionRequest = useRef(0);
  const [tensorboard, setTensorboard] = useState<TensorBoardStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [defaultsLoading, setDefaultsLoading] = useState(false);
  const [defaultsKey, setDefaultsKey] = useState("");
  const [error, setError] = useState("");
  const [defaultsError, setDefaultsError] = useState("");
  const taskIdRef = useRef(JSON.stringify(taskScope));
  const refreshRequest = useRef(0);
  taskIdRef.current = JSON.stringify(taskScope);
  const requestedDefaultsKey = JSON.stringify([datasetId, rewardSource, rewardRevision, algorithm, modelFamily]);

  useEffect(() => {
    void Promise.all([getBootstrap(), getTrainingDefaults(), listOfflineJobs(), getTensorBoard()])
      .then(([nextBootstrap, nextDefaults, jobs, board]) => {
        setBootstrap(nextBootstrap); setDefaults(nextDefaults); setTaskScope(scopeForTask(nextBootstrap.task_catalog, nextBootstrap.task.task_id));
        const { reward_rynnvalue: legacyRynn, ...advanced } = nextDefaults.advanced;
        setParameters({ ...nextDefaults.basic, ...advanced, ...nextDefaults.monitoring, ...nextDefaults.model,
          reward_source: rewardSource,
          reward_stage_exponent: advanced.reward_stage_exponent ?? 2, resume_checkpoint: null });
        setTensorboard(board);
        const active = jobs.find(
          (item) => ["training", "annotation"].includes(item.kind) && activeJobStates.has(item.status),
        );
        if (active && followQueueRef.current) setJob(active);
      }).catch((reason) => setError(String(reason)));
  }, []);
  useEffect(() => {
    if (!bootstrap) return;
    let current = true;
    setDatasetsLoading(true); setDatasets([]); setDatasetId("");
    void (taskId ? listTrainingDatasets(taskId) : listTrainingDatasets(undefined, taskIdsForScope(bootstrap.task_catalog, taskScope))).then((values) => {
      if (current) setDatasets(values.filter((item) => item.integrity_status === "HEALTHY"));
    }).catch((reason) => { if (current) setError(String(reason)); })
      .finally(() => { if (current) setDatasetsLoading(false); });
    return () => { current = false; };
  }, [taskScope, bootstrap]);
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
  useEffect(() => {
    let current = true;
    let timer: number;
    const refresh = async () => {
      try {
        const next = await getTrainingQueue();
        if (!current) return;
        setQueue(next);
        const active = next.jobs.find((item) => activeJobStates.has(item.status));
        const selected = jobRef.current;
        const latestSelected = next.jobs.find((item) => item.id === selected?.id);
        if (selected && latestSelected && latestSelected.status !== selected.status) {
          setJob({ ...selected, ...latestSelected });
        }
        if (followQueueRef.current && active && active.id !== selected?.id && (!selected || !activeJobStates.has(latestSelected?.status ?? selected.status))) {
          const request = selectionRequest.current;
          const running = await getOfflineJob(active.id);
          if (current && followQueueRef.current && request === selectionRequest.current) setJob(running);
        }
      } catch (reason) { if (current) setError(`训练队列读取失败：${String(reason)}`); }
      finally { if (current) timer = window.setTimeout(() => void refresh(), 3000); }
    };
    void refresh();
    return () => { current = false; window.clearTimeout(timer); };
  }, []);
  const dataset = useMemo(() => datasets.find((item) => item.id === datasetId) ?? null, [datasets, datasetId]);
  useEffect(() => {
    setDefaultsError("");
    setDefaultsKey("");
    if (!bootstrap) { setDefaultsLoading(false); return; }
    let current = true;
    setDefaultsLoading(true);
    void getTrainingDefaults(datasetId || undefined, isBC ? undefined : rewardSource, algorithm, modelFamily).then((next) => {
      if (!current) return;
      setDefaults(next);
      setDefaultsKey(requestedDefaultsKey);
      const selectionChanged = appliedSelection.current.family !== modelFamily || appliedSelection.current.algorithm !== algorithm;
      const sameModel = appliedSelection.current.family === modelFamily;
      appliedSelection.current = { family: modelFamily, algorithm };
      setParameters((current) => {
        if (selectionChanged) {
          const modelEdits = sameModel ? Object.fromEntries(Object.entries(current).filter(([key]) => key.startsWith("model_"))) : {};
          current = { ...next.basic, ...next.advanced, ...next.monitoring, ...next.model, ...modelEdits, model_family: modelFamily };
        }
        const pinned = Object.fromEntries(Object.entries(next.advanced)
          .filter(([key]) => key.startsWith("reward_") && key !== "reward_rynnvalue"));
        const unrelated = { ...next.model, ...Object.fromEntries(Object.entries(current).filter(([key]) => !key.startsWith("reward_"))) };
        if (isBC) return { ...unrelated, actor_lr_warmup_steps: unrelated.actor_lr_warmup_steps
          ?? next.basic.actor_lr_warmup_steps ?? next.basic.critic_warmup_steps ?? 0, resume_checkpoint: null };
        if ((rewardSource === "final" && pinned.reward_fusion_mode === "multiplicative")
          || next.reward_editable_parameters?.includes("reward_accumulate_primitive_steps") === false) {
          pinned.reward_accumulate_primitive_steps = false;
        }
        return { ...unrelated, ...pinned,
          reward_source: rewardSource,
          reward_version_id: next.reward_version?.id ?? null,
          resume_checkpoint: null,
        };
      });
    }).catch((reason) => { if (current) setDefaultsError(String(reason)); })
      .finally(() => { if (current) setDefaultsLoading(false); });
    return () => { current = false; };
  }, [datasetId, rewardSource, rewardRevision, algorithm, modelFamily, bootstrap]);
  const rewardReady = Boolean(datasetId && defaultsKey === requestedDefaultsKey && (
    defaults?.reward_availability?.ready ?? (defaults?.reward_version
      && defaults.reward_version.evaluator === rewardSource && !defaults.reward_version.legacy
      && ["READY", "COMPLETED"].includes(defaults.reward_version.status))
  ));
  const cumulativeAllowed = (rewardSource !== "final" || parameters.reward_fusion_mode !== "multiplicative")
    && defaults?.reward_editable_parameters?.includes("reward_accumulate_primitive_steps") !== false;
  useEffect(() => {
    if (isBC || defaultsLoading || !defaults?.reward_availability?.pending) return;
    const timer = window.setTimeout(() => setRewardRevision((value) => value + 1), 1000);
    return () => window.clearTimeout(timer);
  }, [defaultsLoading, defaults, isBC]);
  const trainingReady = Boolean(datasetId && !defaultsError && !modelConfigurationError(parameters)
    && defaultsKey === requestedDefaultsKey && (isBC || rewardReady));
  const patchParameter = (name: string, value: number | string | boolean | null) => setParameters((current) => ({ ...current, [name]: value }));
  const refreshDataset = async () => {
    const request = ++refreshRequest.current;
    setDefaultsKey("");
    const requestedScope = JSON.stringify(taskScope);
    const values = await (taskId ? listTrainingDatasets(taskId) : listTrainingDatasets(undefined, taskIdsForScope(bootstrap?.task_catalog ?? [], taskScope)));
    if (taskIdRef.current !== requestedScope || request !== refreshRequest.current) return;
    setDatasets(values.filter((item) => item.integrity_status === "HEALTHY"));
    setRewardRevision((value) => value + 1);
  };
  const updateJob = (next: OfflineJob) => {
    if (jobRef.current?.id !== next.id) return;
    setJob(next);
    setQueue((current) => ({ ...current, jobs: current.jobs.map((item) => item.id === next.id ? next : item) }));
    if (next.kind === "annotation" && !activeJobStates.has(next.status) && job?.status !== next.status) {
      void refreshDataset().catch((reason) => setError(String(reason)));
    }
  };
  const begin = async () => {
    if (busy || datasetsLoading || defaultsLoading || !trainingReady || !availableDatasets.some((item) => item.id === datasetId)) return;
    setBusy(true); setError("");
    try {
      const bcParameters = Object.fromEntries(Object.entries(parameters).filter(([key]) =>
        !key.startsWith("reward_") && !key.startsWith("critic_") && !key.startsWith("value_")
        && !["expectile", "beta", "max_advantage_weight", "target_tau"].includes(key)));
      const next = await enqueueTraining(datasetId, isBC
        ? { ...bcParameters, algorithm, resume_checkpoint: null }
        : { ...parameters, algorithm, resume_checkpoint: null, reward_source: rewardSource,
          reward_accumulate_primitive_steps: cumulativeAllowed ? Boolean(parameters.reward_accumulate_primitive_steps) : false });
      setQueue((current) => ({ ...current, jobs: [...current.jobs.filter((item) => item.id !== next.id), next] }));
      if (followQueueRef.current && (!jobRef.current || !activeJobStates.has(jobRef.current.status))) setJob(next);
      try { setTensorboard(await startTensorBoard()); } catch (reason) { setError(`训练已注册，但 TensorBoard 启动失败：${String(reason)}`); }
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const inspectJob = async (id: string) => {
    followQueueRef.current = false; setFollowQueue(false);
    const request = ++selectionRequest.current;
    try {
      const selected = await getOfflineJob(id);
      if (request === selectionRequest.current) setJob(selected);
    } catch (reason) { if (request === selectionRequest.current) setError(String(reason)); }
  };
  const stopJob = async (id: string) => {
    setStoppingId(id);
    try {
      const next = await stopOfflineJob(id);
      setQueue((current) => ({ ...current, jobs: current.jobs.map((item) => item.id === id ? next : item) }));
      if (jobRef.current?.id === id) setJob(next);
    } catch (reason) { setError(String(reason)); }
    finally { setStoppingId(null); }
  };
  const startBoard = async () => {
    setBusy(true); setError("");
    try { setTensorboard(await startTensorBoard()); }
    catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };

  return <section className="content-page training-page">
    <div className="page-heading"><p className="eyebrow">POLICY TRAINING</p><h1>策略后训练</h1><p>按模型、训练方法、参数配置的顺序设置；数据筛选在数据集页面完成。</p></div>
    {(error || defaultsError) && <div className="error-banner"><span>{error || defaultsError}</span><button onClick={() => { setError(""); setDefaultsError(""); }}>关闭</button></div>}
    <div className="training-layout">
      <section className="surface training-config">
        <div className="panel-title"><strong>训练配置</strong><span>{defaults?.environments.training ?? "vla-liberox"}</span></div>
        <div className="training-form">
          <label>基础模型<select value={modelFamily} onChange={(event) => setModelFamily(event.target.value)}>
            {(defaults?.models ?? []).map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
          </select></label>
          <label>训练方法<select value={algorithm} onChange={(event) => { setAlgorithm(event.target.value as "iql" | "bc"); setShowDatasetConfig(false); }}>
            <option value="iql">IQL · 奖励加权后训练</option><option value="bc">BC · 等权行为克隆</option>
          </select></label>
          <ModelConfigPanel models={defaults?.models} parameters={parameters} onChange={patchParameter} />
          <TaskFilter tasks={bootstrap?.task_catalog ?? []} value={taskScope} onChange={setTaskScope} labelPrefix="训练" />
          <label>冻结数据集<select value={datasetId} disabled={datasetsLoading} onChange={(event) => setDatasetId(event.target.value)}><option value="">{datasetsLoading ? "加载数据集…" : "请选择"}</option>{availableDatasets.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.member_count} 条</option>)}</select></label>
          {!isBC && <label>训练奖励<select value={rewardSource} disabled={busy || datasetsLoading || !datasetId}
            onChange={(event) => setRewardSource(event.target.value as "final" | "rynnvalue")}>
            <option value="final">Final Reward</option><option value="rynnvalue">RynnValue</option>
          </select></label>}
          {dataset && <div className="training-dataset-summary"><strong>{dataset.member_count} 条轨迹</strong><span>{dataset.action_count} actions</span><span>{dataset.chunk_count} chunks</span><Badge tone="green">完整性正常</Badge>
            <p>{isBC ? "BC 使用所选数据集的全部有效训练样本，无需奖励评价。" : defaultsLoading || defaultsKey !== requestedDefaultsKey ? "正在读取评价结果…" : rewardReady
              ? `当前训练奖励：${rewardSourceLabels[rewardSource]}${defaults?.reward_availability?.origin === "global" ? " · 使用逐轨迹全局结果" : ""}`
              : `请先完成 ${rewardSourceLabels[rewardSource]} 评价，或确保每条轨迹已有同类型全局结果。`}</p>
            {!isBC && !defaultsLoading && defaultsKey === requestedDefaultsKey && defaults?.reward_availability?.message &&
              <p className="error-banner">{defaults.reward_availability.message}</p>}
            {!isBC && !defaultsLoading && defaultsKey === requestedDefaultsKey && !rewardReady && defaults?.reward_availability?.errors?.map((item) =>
              <p key={item.run_id} className="error-banner">{item.run_id}：{item.error}</p>)}
            {!isBC && <><p>按所选奖励读取数据集评价，缺少时继承同类型全局结果；不会回退到其他奖励。修改 γ 或可用的 cumulative reward 只影响本次训练。</p>
            <button aria-expanded={showDatasetConfig} onClick={() => setShowDatasetConfig((value) => !value)}>配置数据集评价</button></>}</div>}
          {!isBC && dataset && showDatasetConfig && <FrozenDatasetCard key={dataset.id} dataset={dataset} initialExpanded
            disabled={busy || Boolean(job && activeJobStates.has(job.status))}
            robometerUnavailable={bootstrap?.evaluation_capabilities?.robometer?.available === false
              ? bootstrap.evaluation_capabilities.robometer.reason ?? "Robometer 不可用" : undefined}
            onRefresh={refreshDataset} onJob={setJob} onError={setError} />}
          <div className="parameter-grid">{basicFields.filter(([name]) => !isBC || name !== "critic_warmup_steps").map(([name, label, step]) => <label key={name}>{label}<input type="number" min={name.endsWith("warmup_steps") || name === "seed" ? 0 : 1} step={step} value={String(parameters[name] ?? "")} onChange={(event) => patchParameter(name, Number(event.target.value))} /></label>)}</div>
          <details><summary>{isBC ? "高级 BC 参数" : "高级 IQL 参数"}</summary><div className="parameter-grid advanced-parameters">
            {!isBC && <><label>Discount ratio γ<input type="number" min={0} max={1} step="any" value={String(parameters.reward_gamma ?? "")} disabled={defaultsLoading || !rewardReady}
              onChange={(event) => patchParameter("reward_gamma", Number(event.target.value))} /></label>
            {cumulativeAllowed && <label>cumulative reward<select disabled={defaultsLoading || !rewardReady} value={String(Boolean(parameters.reward_accumulate_primitive_steps))}
              onChange={(event) => patchParameter("reward_accumulate_primitive_steps", event.target.value === "true")}><option value="false">Off</option><option value="true">On</option></select></label>}
            <label>Critic optimizer<select value={String(parameters.critic_optimizer ?? "adamw")} onChange={(event) => patchParameter("critic_optimizer", event.target.value)}><option value="adam">Adam</option><option value="adamw">AdamW</option></select></label>
            <label>Value optimizer<select value={String(parameters.value_optimizer ?? "adamw")} onChange={(event) => patchParameter("value_optimizer", event.target.value)}><option value="adam">Adam</option><option value="adamw">AdamW</option></select></label></>}
            {advancedFields.filter(([name]) => !isBC || ["policy_peak_lr", "policy_final_lr", "console_interval_steps", "flush_seconds"].includes(name)).map(([name, label, step]) => <label key={name}>{label}<input type="number" min={0} step={step} value={String(parameters[name] ?? "")} onChange={(event) => patchParameter(name, Number(event.target.value))} /></label>)}
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
          <button className="primary start-training" disabled={busy || datasetsLoading || defaultsLoading || !trainingReady || !availableDatasets.some((item) => item.id === datasetId)} onClick={() => void begin()}>注册训练任务</button>
        </div>
      </section>
      <section className="surface tensorboard-card">
        <div className="panel-title"><strong>TensorBoard</strong><span>{tensorboard?.running ? "ONLINE" : "OFFLINE"}</span></div>
        <div><span className={`tensorboard-dot ${tensorboard?.running ? "online" : ""}`} /><strong>{tensorboard?.url ?? "http://127.0.0.1:6006/"}</strong><p>集中查看所有平台训练的 loss、Q/V、advantage、动作误差、吞吐与显存曲线。</p>{tensorboard?.running ? <a className="export-button" href={tensorboard.url} target="_blank" rel="noreferrer">打开 TensorBoard</a> : <button onClick={() => void startBoard()} disabled={busy}>启动 TensorBoard</button>}</div>
      </section>
    </div>
    <TrainingQueuePanel queue={queue} onInspect={(id) => void inspectJob(id)} onStop={(id) => void stopJob(id)} busyId={stoppingId} />
    {!followQueue && <button className="follow-training" onClick={() => {
      selectionRequest.current += 1;
      followQueueRef.current = true; setFollowQueue(true);
    }}>跟随当前训练</button>}
    {job && <>
      <JobMonitor initial={job} onUpdate={updateJob} onDismiss={() => setJob(null)} />
      {job.training_summary && <section className="surface training-result"><div className="panel-title"><strong>训练结果</strong><span>{job.status}</span></div><div><span>Overlay</span><code>{String(job.training_summary.policy_overlay ?? "尚未发布")}</code><span>数据哈希</span><code>{String(job.training_summary.dataset_sha256 ?? "—")}</code></div></section>}
    </>}
  </section>;
}
