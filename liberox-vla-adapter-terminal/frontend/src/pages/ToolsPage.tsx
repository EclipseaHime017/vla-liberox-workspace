import { useEffect, useRef, useState } from "react";
import { Badge } from "../components/ui/Badge";
import { Select } from "../components/ui/Select";
import { PolicySelector } from "../features/run-config/PolicySelector";
import { TaskSelector } from "../features/run-config/TaskSelector";
import { getBootstrap, listTrainingDatasets } from "../features/run-control/api";
import type { Bootstrap, OfflineJob, TrainingDataset } from "../features/run-control/types";
import { JobMonitor } from "../features/training/JobMonitor";
import {
  deleteCriticDiagnostic, getCriticDefaults, getCriticDiagnostic, listCriticDiagnostics,
  startCriticDiagnostic, stopCriticDiagnostic, type CriticDefaults, type CriticDiagnostic,
} from "../features/tools/api";
import "../styles/tools.css";

const pending = new Set(["QUEUED", "STARTING", "RUNNING", "STOPPING"]);
const statuses: Record<OfflineJob["status"], string> = {
  QUEUED: "排队中", STARTING: "启动中", RUNNING: "运行中", STOPPING: "停止中",
  COMPLETED: "已完成", FAILED: "失败", CANCELED: "已取消",
};
const rewards: Record<string, string> = {
  sparse: "Sparse", stage: "Stage", rynnvalue: "RynnValue", final: "Final Reward",
};
const downloads = [
  ["qv.csv", "Q/V CSV"], ["qv.npz", "Q/V NPZ"], ["summary.json", "结果摘要"],
  ["effective_config.yaml", "配置 YAML"], ["critic.pt", "Critic checkpoint"],
] as const;
const validInteger = (value: number, min: number, max = Number.MAX_SAFE_INTEGER) => Number.isInteger(value) && value >= min && value <= max;
const loss = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? "—" : value.toPrecision(5);
const mergeJob = (current: CriticDiagnostic | undefined, next: CriticDiagnostic): CriticDiagnostic => {
  if (current?.id !== next.id) return next;
  if ((!pending.has(current.status) && pending.has(next.status))
    || (current.status === "STOPPING" && ["QUEUED", "STARTING", "RUNNING"].includes(next.status))) return current;
  return { ...current, ...next };
};

function DiagnosticResult({ job }: { job: CriticDiagnostic }) {
  const artifacts = job.artifacts ?? {};
  const progress = job.critic_progress;
  const rolloutPhase = progress?.phase === "rollout" || progress?.phase === "rollout_complete";
  const progressStep = rolloutPhase ? progress?.rollout_step : progress?.train_step;
  const progressSteps = rolloutPhase ? progress?.rollout_steps : progress?.train_steps;
  const progressLabel = progress?.phase === "rollout" ? "生成 Rollout"
    : progress?.phase === "rollout_complete" ? "Rollout 已完成"
      : progress?.phase === "completed" ? "诊断已完成" : "拟合 Critic";
  const details = job.diagnostic_summary ?? {};
  const unsupportedLengths = Array.isArray(details.unsupported_probe_chunk_lengths)
    ? details.unsupported_probe_chunk_lengths.filter((value): value is number => typeof value === "number") : [];
  const counts = details.counts as Record<string, unknown> | undefined;
  const rewardConfig = details.reward_config as Record<string, unknown> | undefined;
  const summary = [
    ["训练步数", details.train_steps], ["Batch size", details.batch_size],
    ["Critic 动作长度", details.critic_horizon], ["折扣 γ", rewardConfig?.gamma],
    ["训练集 chunk", counts?.train], ["验证集 chunk", counts?.validation], ["Rollout chunk", counts?.rollout],
  ].filter(([, value]) => typeof value === "string" || typeof value === "number");
  return <section className="surface tool-results" aria-label="Critic 诊断结果">
    <div className="panel-title"><strong>诊断结果</strong><span>{job.id}</span></div>
    <div className="tool-result-body">
      {progress && <div className="tool-progress" aria-label="Critic 训练进度">
        <span>{progressLabel} {progressStep != null && progressSteps != null && <b>{progressStep} / {progressSteps}</b>}</span>
        {!rolloutPhase && <><span>Q loss <b>{loss(progress.q_loss)}</b></span>
          <span>Value loss <b>{loss(progress.value_loss)}</b></span></>}
        {progressStep != null && progressSteps != null && <progress max={progressSteps || 1} value={progressStep} />}
      </div>}
      <p className="muted">Q/V 是基于所选数据集奖励拟合的 expectile critic 估计，不是策略真实回报；新 rollout 仅用于探测，没有奖励参考值。</p>
      {unsupportedLengths.length > 0 && <div className="warning-banner" role="alert">
        新 rollout 包含训练数据未覆盖的 chunk 长度（{unsupportedLengths.join("、")} 步），对应 Q 值涉及外推，仅供参考。
      </div>}
      {summary.length > 0 && <dl className="tool-summary">{summary.map(([key, value]) =>
        <div key={String(key)}><dt>{String(key)}</dt><dd>{String(value)}</dd></div>)}</dl>}
      <div className="tool-plots">
        {([ ["qv.png", "Rollout Q / V"], ["training.png", "Critic 训练曲线"] ] as const).map(([name, title]) =>
          artifacts[name] && <figure key={name}><figcaption>{title}</figcaption>
            <a href={artifacts[name]} target="_blank" rel="noreferrer"><img src={artifacts[name]} alt={title} /></a>
          </figure>)}
      </div>
      {!Object.keys(artifacts).length && <p className="muted">{job.status === "COMPLETED" ? "本次任务没有可用结果文件。" : pending.has(job.status) ? "完成后在此查看 Q/V 曲线与训练曲线。" : "未生成完整结果，请查看任务日志。"}</p>}
      <div className="tool-downloads">{downloads.map(([name, label]) => artifacts[name]
        && <a key={name} href={artifacts[name]} download={name}>{label}</a>)}</div>
    </div>
  </section>;
}

export function ToolsPage() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [taskId, setTaskId] = useState("");
  const [policyId, setPolicyId] = useState("");
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetId, setDatasetId] = useState("");
  const [defaults, setDefaults] = useState<CriticDefaults | null>(null);
  const [defaultsDataset, setDefaultsDataset] = useState("");
  const [rewardSource, setRewardSource] = useState("");
  const [maxSteps, setMaxSteps] = useState(300);
  const [openLoopSteps, setOpenLoopSteps] = useState(1);
  const [seed, setSeed] = useState(0);
  const [initStateIndex, setInitStateIndex] = useState(0);
  const [trainSteps, setTrainSteps] = useState(3000);
  const [batchSize, setBatchSize] = useState(32);
  const [jobs, setJobs] = useState<CriticDiagnostic[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [selected, setSelected] = useState<CriticDiagnostic | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionId, setActionId] = useState("");
  const [error, setError] = useState("");
  const selection = useRef("");
  const deleted = useRef(new Set<string>());
  selection.current = selectedId;

  useEffect(() => {
    let current = true;
    void getBootstrap().then((value) => {
      if (!current) return;
      setBootstrap(value);
      setTaskId(value.task.task_id);
      const policy = value.policy_catalog.find((item) => item.policy_id === value.model.policy_id) ?? value.policy_catalog[0];
      setPolicyId(policy?.policy_id ?? "");
      setOpenLoopSteps(policy?.inference?.open_loop_steps ?? policy?.io?.native_action_horizon ?? 1);
      setMaxSteps(value.config.max_steps);
      setSeed(value.config.seed);
    }).catch((reason) => { if (current) setError(String(reason)); });
    return () => { current = false; };
  }, []);

  useEffect(() => {
    if (!taskId) return;
    let current = true;
    setDatasetsLoading(true); setDatasets([]); setDatasetId("");
    void listTrainingDatasets(taskId).then((values) => {
      if (!current) return;
      const available = values.filter((item) => item.task_id === taskId && item.integrity_status === "HEALTHY");
      setDatasets(available); setDatasetId(available[0]?.id ?? "");
    }).catch((reason) => { if (current) setError(String(reason)); })
      .finally(() => { if (current) setDatasetsLoading(false); });
    return () => { current = false; };
  }, [taskId]);

  useEffect(() => {
    setDefaults(null); setDefaultsDataset(""); setRewardSource("");
    if (!datasetId) return;
    let current = true;
    void getCriticDefaults(datasetId).then((value) => {
      if (!current) return;
      setDefaults(value); setDefaultsDataset(datasetId);
      setTrainSteps(value.train_steps); setBatchSize(value.batch_size);
      setRewardSource(value.default_reward_source && value.reward_sources.includes(value.default_reward_source)
        ? value.default_reward_source : value.reward_sources[0] ?? "");
    }).catch((reason) => { if (current) setError(String(reason)); });
    return () => { current = false; };
  }, [datasetId]);

  useEffect(() => {
    let current = true;
    let timer: number;
    const refresh = async () => {
      try {
        const { jobs: values } = await listCriticDiagnostics();
        if (!current) return;
        const available = values.filter((job) => !deleted.current.has(job.id));
        setJobs((previous) => {
          const byId = new Map(previous.map((job) => [job.id, job]));
          return [...available.map((job) => mergeJob(byId.get(job.id), job)),
            ...previous.filter((job) => pending.has(job.status) && !deleted.current.has(job.id)
              && !available.some((value) => value.id === job.id))];
        });
        if (!selection.current && available.length) {
          const next = available.find((job) => pending.has(job.status)) ?? available[0];
          setSelectedId(next.id); setSelected(next);
        }
      } catch (reason) { if (current) setError(String(reason)); }
      finally { if (current) timer = window.setTimeout(() => void refresh(), 3000); }
    };
    void refresh();
    return () => { current = false; window.clearTimeout(timer); };
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let current = true;
    let timer: number;
    const refresh = async () => {
      try {
        const value = await getCriticDiagnostic(selectedId);
        if (current && !deleted.current.has(value.id)) setSelected((previous) => mergeJob(previous ?? undefined, value));
      } catch (reason) { if (current) setError(String(reason)); }
      finally { if (current) timer = window.setTimeout(() => void refresh(), 3000); }
    };
    void refresh();
    return () => { current = false; window.clearTimeout(timer); };
  }, [selectedId]);

  const task = bootstrap?.task_catalog.find((item) => item.task_id === taskId);
  const policy = bootstrap?.policy_catalog.find((item) => item.policy_id === policyId);
  const maxOpenLoop = policy?.io?.native_action_horizon ?? policy?.inference?.max_open_loop_steps;
  const dataset = datasets.find((item) => item.id === datasetId && item.task_id === taskId);
  const ready = Boolean(task && policy && dataset && defaultsDataset === datasetId
    && defaults?.reward_sources.includes(rewardSource) && !datasetsLoading
    && validInteger(maxSteps, 1, 10000) && validInteger(openLoopSteps, 1, maxOpenLoop)
    && validInteger(seed, 0, 2147483647) && validInteger(initStateIndex, 0, (task.init_state_count || 1) - 1)
    && validInteger(trainSteps, 1, 1000000) && validInteger(batchSize, 1, 4096));

  const changePolicy = (value: string) => {
    setPolicyId(value);
    const next = bootstrap?.policy_catalog.find((item) => item.policy_id === value);
    setOpenLoopSteps(next?.inference?.open_loop_steps ?? next?.io?.native_action_horizon ?? 1);
  };
  const submit = async () => {
    if (!ready || busy) return;
    setBusy(true); setError("");
    try {
      const job = await startCriticDiagnostic({ dataset_id: datasetId, reward_source: rewardSource,
        task_id: taskId, policy_id: policyId, max_steps: maxSteps, open_loop_steps: openLoopSteps,
        seed, init_state_index: initStateIndex, train_steps: trainSteps, batch_size: batchSize });
      setJobs((values) => [job, ...values.filter((item) => item.id !== job.id)]);
      setSelectedId(job.id); setSelected(job);
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const updateJob = (job: OfflineJob) => {
    setJobs((values) => values.map((item) => item.id === job.id ? mergeJob(item, job) : item));
    if (selection.current === job.id) setSelected((value) => value?.id === job.id ? mergeJob(value, job) : value);
  };
  const actOnJob = async (job: CriticDiagnostic, remove = false) => {
    if (remove && !window.confirm(`删除诊断 ${job.id} 及其全部结果？`)) return;
    setActionId(job.id); setError("");
    try {
      if (remove) {
        await deleteCriticDiagnostic(job.id);
        deleted.current.add(job.id);
        setJobs((values) => values.filter((item) => item.id !== job.id));
        if (selection.current === job.id) { setSelectedId(""); setSelected(null); }
      } else updateJob(await stopCriticDiagnostic(job.id));
    } catch (reason) { setError(String(reason)); }
    finally { setActionId(""); }
  };

  return <section className="content-page tools-page">
    <nav className="surface tools-toolbar" aria-label="测试工具"><button className="active" aria-current="page">IQL Critic 诊断</button></nav>
    {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
    <section className="surface tool-builder" aria-label="Critic 诊断配置">
      <div className="panel-title"><strong>IQL Critic 诊断</strong><span>仅训练 Q / V，不更新 Actor</span></div>
      <div className="tool-form">
        <TaskSelector tasks={bootstrap?.task_catalog ?? []} value={taskId} labelPrefix="诊断" disabled={busy}
          onChange={(value) => { setTaskId(value); setDatasetId(""); setInitStateIndex(0); }} />
        <label>模型<PolicySelector policies={bootstrap?.policy_catalog ?? []} value={policyId} disabled={busy} onChange={changePolicy} /></label>
        <label>训练数据集<Select value={datasetId} disabled={busy || datasetsLoading} onChange={(event) => setDatasetId(event.target.value)}>
          {!datasets.length && <option value="">{datasetsLoading ? "加载中…" : "当前任务没有可用数据集"}</option>}
          {datasets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </Select></label>
        <label>已有奖励<Select value={rewardSource} disabled={busy || defaultsDataset !== datasetId || !defaults?.reward_sources.length} onChange={(event) => setRewardSource(event.target.value)}>
          {!defaults?.reward_sources.length && <option value="">无可用奖励</option>}
          {defaults?.reward_sources.map((source) => <option key={source} value={source}>{rewards[source] ?? source}</option>)}
        </Select></label>
        <label>Rollout 控制步数<input type="number" min={1} max={10000} value={maxSteps} onChange={(event) => setMaxSteps(Number(event.target.value))} /></label>
        <label>每次预测执行步数<input type="number" min={1} max={maxOpenLoop} value={openLoopSteps} onChange={(event) => setOpenLoopSteps(Number(event.target.value))} /></label>
        <label>训练步数<input type="number" min={1} max={1000000} value={trainSteps} onChange={(event) => setTrainSteps(Number(event.target.value))} /></label>
        <label>Batch size<input type="number" min={1} max={4096} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} /></label>
        <label>随机种子<input type="number" min={0} max={2147483647} value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label>
        <label>初始状态索引<input type="number" min={0} max={(task?.init_state_count ?? 1) - 1} value={initStateIndex} onChange={(event) => setInitStateIndex(Number(event.target.value))} /></label>
      </div>
      {datasetId && defaultsDataset === datasetId && !defaults?.reward_sources.length
        && <p className="tool-note" role="status">当前数据集没有已封存的可用奖励，请先在数据集页面配置并完成奖励评价。</p>}
      <div className="tool-actions"><p className="muted">使用现有数据集及其已有奖励训练 Critic，再对所选模型的新 rollout 估计 Q/V。任务进入统一工作队列。</p>
        <button className="primary" disabled={!ready || busy} onClick={() => void submit()}>{busy ? "注册中…" : "开始诊断"}</button></div>
    </section>
    {selected && <><DiagnosticResult job={selected} /><JobMonitor initial={selected} onUpdate={updateJob} /></>}
    <section className="surface tool-history" aria-label="诊断任务记录">
      <div className="panel-title"><strong>诊断任务</strong><span>{jobs.filter((job) => pending.has(job.status)).length} 个待完成任务</span></div>
      {!jobs.length ? <p className="tool-empty muted">暂无诊断记录。</p> : <div className="tool-table-scroll"><table><thead><tr>
        <th>任务 / 模型</th><th>数据集</th><th>配置</th><th>状态</th><th>操作</th>
      </tr></thead><tbody>{jobs.map((job) => <tr key={job.id} className={selectedId === job.id ? "selected" : ""}>
        <td><strong>{String(job.parameters.policy_label
          ?? bootstrap?.policy_catalog.find((item) => item.policy_id === job.parameters.policy_id)?.label
          ?? job.parameters.policy_id ?? job.id)}</strong>
          <small>{bootstrap?.task_catalog.find((item) => item.task_id === job.parameters.task_id)?.prompt}</small><small>{job.id}</small></td>
        <td>{String(job.parameters.dataset_name ?? job.dataset_id ?? "—")}</td>
        <td>{String(job.parameters.train_steps ?? "—")} steps · Batch {String(job.parameters.batch_size ?? "—")}</td>
        <td><Badge tone={job.status === "COMPLETED" ? "green" : job.status === "FAILED" ? "red" : "neutral"}>{statuses[job.status]}</Badge></td>
        <td><div className="queue-actions"><button onClick={() => { setSelectedId(job.id); setSelected(job); }}>查看</button>
          {pending.has(job.status) ? <button className="danger" disabled={actionId === job.id || job.status === "STOPPING"} onClick={() => void actOnJob(job)}>{job.status === "QUEUED" ? "取消排队" : "停止"}</button>
            : <button className="danger" disabled={actionId === job.id} onClick={() => void actOnJob(job, true)}>删除</button>}</div></td>
      </tr>)}</tbody></table></div>}
    </section>
  </section>;
}
