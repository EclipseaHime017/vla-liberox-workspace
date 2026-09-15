import { useEffect, useMemo, useState } from "react";
import { TaskFilter, TaskSelector } from "../features/run-config/TaskSelector";
import { ALL_TASK_SCOPE, taskIdsForScope, type TaskScope } from "../features/run-config/taskHierarchy";
import { Badge } from "../components/ui/Badge";
import {
  deleteEvaluation, getBootstrap, getEvaluation, listEvaluations, listOfflineJobs,
  previewEvaluation, startEvaluation,
} from "../features/run-control/api";
import type {
  Bootstrap, EvaluationAggregate, EvaluationBreakdown, EvaluationConfig,
  EvaluationFilters, EvaluationPreview, EvaluationRecord, EvaluationStatus, OfflineJob,
} from "../features/run-control/types";
import { SuccessRateChart } from "../features/metrics/SuccessRateChart";
import { EvaluationMonitor } from "../features/evaluation/EvaluationMonitor";

const activeStatuses = new Set(["STARTING", "RUNNING", "STOPPING"]);
const terminalStatuses = new Set(["COMPLETED", "FAILED", "CANCELED"]);

function duration(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const value = Math.max(0, Math.round(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  return hours ? `${hours} 小时 ${minutes} 分` : `${minutes} 分 ${value % 60} 秒`;
}

function percentage(value: number | null | undefined) {
  return `${((value ?? 0) * 100).toFixed(1)}%`;
}

function badgeTone(status: EvaluationStatus) {
  if (status === "COMPLETED") return "green";
  if (status === "FAILED") return "red";
  return "neutral";
}

function breakdownRows(values: Record<string, EvaluationBreakdown> | null | undefined) {
  return Object.entries(values ?? {}).sort(([left], [right]) => left.localeCompare(right, undefined, { numeric: true }));
}

function EmptyBreakdown({ text }: { text: string }) {
  return <div className="empty-breakdown">{text}</div>;
}

function Breakdown({ title, prefix, values }: {
  title: string; prefix: string; values: Record<string, EvaluationBreakdown> | null | undefined;
}) {
  const rows = breakdownRows(values);
  return <section className="evaluation-breakdown">
    <h2>{title}</h2>
    {rows.length ? rows.map(([key, value]) => <div key={key}>
      <span>{prefix}{key}</span><div><i style={{ width: percentage(value.success_rate) }} /></div>
      <b>{percentage(value.success_rate)}</b><small>{value.successes}/{value.trials}{value.errors ? ` · ${value.errors} error` : ""}</small>
    </div>) : <EmptyBreakdown text="暂无已完成回合" />}
  </section>;
}

function CombinationMatrix({ record }: { record: EvaluationRecord }) {
  const values = record.aggregate?.by_combination ?? {};
  const states = [...new Set(record.schedule.map((item) => item.init_state_index))]
    .sort((left, right) => left - right);
  const seeds = [...new Set(record.schedule.map((item) => item.seed))]
    .sort((left, right) => left - right);
  if (!states.length || !seeds.length) return <EmptyBreakdown text="暂无组合统计" />;
  return <div className="table-wrap combination-matrix"><table>
    <thead><tr><th>Init / Seed</th>{seeds.map((seed) => <th key={seed}>{seed}</th>)}</tr></thead>
    <tbody>{states.map((state) => <tr key={state}><th>Init #{state}</th>{seeds.map((seed) => {
      const value = values[`${state}:${seed}`];
      return <td key={seed} className={value ? "measured" : "pending"} title={value ? `${value.successes}/${value.trials} success · ${value.errors} error` : "尚未执行"}>
        {value ? <><strong>{percentage(value.success_rate)}</strong><small>{value.successes}/{value.trials}</small></> : "—"}
      </td>;
    })}</tr>)}</tbody>
  </table></div>;
}

function EvaluationDetail({ record }: { record: EvaluationRecord }) {
  const aggregate = record.aggregate;
  return <section className="evaluation-detail">
    <div className="surface evaluation-overview">
      <SuccessRateChart rate={aggregate?.success_rate ?? 0} />
      <div className="evaluation-headline-stats">
        <div><strong>{aggregate?.successes ?? 0}</strong><span>成功</span></div>
        <div><strong>{aggregate?.failures ?? 0}</strong><span>失败</span></div>
        <div><strong>{aggregate?.errors ?? 0}</strong><span>错误</span></div>
        <div><strong>{percentage(aggregate?.completion_rate)}</strong><span>完成覆盖率</span></div>
        <p>Wilson 95% 置信区间：{percentage(aggregate?.wilson_lower)} – {percentage(aggregate?.wilson_upper)}。错误回合计入成功率分母。</p>
      </div>
    </div>
    <div className="surface evaluation-context">
      <div className="panel-title"><strong>测试上下文</strong><span>{record.id}</span></div>
      <div><span>任务</span><b>{record.task_prompt || record.task_name}</b><span>策略</span><b>{record.policy_label}</b><span>调度</span><code>{record.schedule_sha256?.slice(0, 16) || "—"}</code><span>成功规则</span><b>done 连续 5 步，回合仍执行满 horizon</b><span>模型加载</span><b>{duration(record.model_load_seconds)}</b><span>总墙钟时间</span><b>{duration(record.wall_time_seconds)}</b></div>
    </div>
    <div className="evaluation-breakdown-grid surface">
      <Breakdown title="不同随机环境成功率" prefix="Init #" values={aggregate?.by_init_state} />
      <Breakdown title="不同随机参数成功率" prefix="Seed " values={aggregate?.by_seed} />
    </div>
    <section className="surface evaluation-combinations">
      <div className="panel-title"><strong>环境 × Seed 组合矩阵</strong><span>{Object.keys(aggregate?.by_combination ?? {}).length} 个已执行组合</span></div>
      <CombinationMatrix record={record} />
    </section>
    <section className="surface evaluation-trials">
      <div className="panel-title"><strong>逐回合关键数值</strong><span>{record.trials?.length ?? 0}</span></div>
      <div className="table-wrap"><table><thead><tr><th>回合</th><th>环境 / seed</th><th>结果</th><th>成功判定</th><th>推理</th><th>控制</th><th>耗时</th></tr></thead><tbody>{record.trials?.map((trial) => <tr key={trial.trial_index}>
        <td>#{trial.trial_index + 1}</td><td>Init #{trial.init_state_index}<small className="cell-note">seed {trial.seed}</small></td>
        <td><Badge tone={trial.error ? "red" : trial.success ? "green" : "neutral"}>{trial.error ? "错误" : trial.success ? "成功" : "失败"}</Badge>{trial.error && <small className="cell-note">{trial.error}</small>}</td>
        <td>{trial.first_success_step == null ? "—" : `step ${trial.first_success_step}`}<small className="cell-note">最大 streak {trial.max_done_streak} · 最终 done {trial.final_done ? "是" : "否"}</small></td>
        <td>{trial.policy_queries} 次<small className="cell-note">均值 {trial.inference_latency_ms == null ? "—" : `${trial.inference_latency_ms.toFixed(1)} ms`}</small></td>
        <td>{trial.measured_control_hz == null ? "—" : `${trial.measured_control_hz.toFixed(2)} Hz`}<small className="cell-note">miss {trial.deadline_misses}</small></td><td>{duration(trial.elapsed_seconds)}</td>
      </tr>)}</tbody></table></div>
    </section>
  </section>;
}

export function EvaluationPage() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [taskId, setTaskId] = useState("");
  const [policyId, setPolicyId] = useState("");
  const [trials, setTrials] = useState(100);
  const [maxSteps, setMaxSteps] = useState(300);
  const [openLoopSteps, setOpenLoopSteps] = useState(8);
  const [realtime, setRealtime] = useState(true);
  const [randomizeInitStates, setRandomizeInitStates] = useState(true);
  const [customStates, setCustomStates] = useState(false);
  const [stateIndices, setStateIndices] = useState<number[]>([]);
  const [fixedInitStateIndex, setFixedInitStateIndex] = useState(0);
  const [baseSeed, setBaseSeed] = useState(7);
  const [randomizeSeeds, setRandomizeSeeds] = useState(true);
  const [customSeedCount, setCustomSeedCount] = useState(false);
  const [seedCount, setSeedCount] = useState(1);
  const [scheduleSeed, setScheduleSeed] = useState(7);
  const [preview, setPreview] = useState<EvaluationPreview | null>(null);
  const [records, setRecords] = useState<EvaluationRecord[]>([]);
  const [selected, setSelected] = useState<EvaluationRecord | null>(null);
  const [job, setJob] = useState<OfflineJob | null>(null);
  const [filters, setFilters] = useState<EvaluationFilters>({});
  const [historyScope, setHistoryScope] = useState<TaskScope>(ALL_TASK_SCOPE);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const task = useMemo(() => bootstrap?.task_catalog.find((item) => item.task_id === taskId) ?? null, [bootstrap, taskId]);
  const availableStates = useMemo(() => task ? Array.from({ length: task.init_state_count }, (_, index) => task.init_state_index_min + index) : [], [task]);
  const config: EvaluationConfig = {
    task_id: taskId, policy_id: policyId, trials, max_steps: maxSteps,
    open_loop_steps: openLoopSteps, realtime,
    init_state_indices: randomizeInitStates ? (customStates ? stateIndices : null) : [fixedInitStateIndex],
    base_seed: baseSeed, seed_count: randomizeSeeds ? (customSeedCount ? seedCount : null) : 1,
    schedule_seed: scheduleSeed,
  };
  const invalidate = (change: () => void) => { change(); setPreview(null); };

  const loadHistory = async (nextFilters = filters) => {
    const next = await listEvaluations({ ...nextFilters, task_id: historyScope.task_id || undefined,
      task_ids: historyScope.task_id ? undefined : taskIdsForScope(bootstrap?.task_catalog ?? [], historyScope) });
    setRecords(next);
    if (selected) {
      const refreshed = next.find((item) => item.id === selected.id);
      if (refreshed) setSelected(await getEvaluation(refreshed.id));
    }
  };

  useEffect(() => {
    void Promise.all([getBootstrap(), listEvaluations(), listOfflineJobs()]).then(([nextBootstrap, nextRecords, jobs]) => {
      setBootstrap(nextBootstrap); setRecords(nextRecords);
      setTaskId(nextBootstrap.task.task_id);
      setPolicyId(nextBootstrap.model.policy_id || nextBootstrap.policy_catalog[0]?.policy_id || "base");
      setMaxSteps(nextBootstrap.config.max_steps); setOpenLoopSteps(nextBootstrap.config.open_loop_steps);
      setBaseSeed(nextBootstrap.config.seed); setScheduleSeed(nextBootstrap.config.seed);
      const nextTask = nextBootstrap.task_catalog.find((item) => item.task_id === nextBootstrap.task.task_id) ?? nextBootstrap.task;
      setStateIndices(Array.from({ length: nextTask.init_state_count }, (_, index) => nextTask.init_state_index_min + index));
      setFixedInitStateIndex(nextTask.init_state_index_min);
      const active = jobs.find((item) => item.kind === "evaluation" && activeStatuses.has(item.status));
      if (active) setJob(active);
    }).catch((reason) => setError(String(reason)));
  }, []);

  const changeTask = (value: string) => invalidate(() => {
    setTaskId(value);
    const next = bootstrap?.task_catalog.find((item) => item.task_id === value);
    setStateIndices(next ? Array.from({ length: next.init_state_count }, (_, index) => next.init_state_index_min + index) : []);
    setFixedInitStateIndex(next?.init_state_index_min ?? 0);
  });
  const toggleState = (value: number) => invalidate(() => setStateIndices((current) => current.includes(value) ? current.filter((item) => item !== value) : [...current, value].sort((a, b) => a - b)));
  const makePreview = async () => {
    setBusy(true); setError("");
    try { setPreview(await previewEvaluation(config)); }
    catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const begin = async () => {
    setBusy(true); setError("");
    try { setJob(await startEvaluation(config)); setPreview(null); await loadHistory(); }
    catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const inspect = async (id: string) => {
    setBusy(true); setError("");
    try { setSelected(await getEvaluation(id)); }
    catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const remove = async (record: EvaluationRecord) => {
    if (!window.confirm(`永久删除测试“${record.id}”及其后台日志？\n\n策略、checkpoint、训练数据集和仿真记录不会被删除。`)) return;
    const confirmation = window.prompt(`二次确认：请输入测试 ID\n${record.id}`, "");
    if (confirmation !== record.id) { if (confirmation !== null) setError("测试 ID 不匹配，未执行删除"); return; }
    setBusy(true); setError("");
    try { await deleteEvaluation(record.id, confirmation); if (selected?.id === record.id) setSelected(null); await loadHistory(); }
    catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };
  const updateJob = (next: OfflineJob) => {
    setJob(next);
    if (terminalStatuses.has(next.status)) void loadHistory().catch((reason) => setError(String(reason)));
  };

  return <section className="content-page evaluation-page">
    <div className="page-heading"><p className="eyebrow">POLICY EVALUATION</p><h1>策略批量测试</h1><p>使用冻结的环境与随机种子调度评价基础 VLA 或训练后的 overlay。测试仅保留关键数值，不写入视频、图像或轨迹。</p></div>
    {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}

    <section className="surface evaluation-builder">
      <div className="panel-title"><strong>创建测试</strong><span>done 连续 5 步确认成功</span></div>
      <div className="evaluation-form">
        <TaskSelector tasks={bootstrap?.task_catalog ?? []} value={taskId} onChange={changeTask} labelPrefix="测试" />
        <label>模型<select aria-label="测试模型" value={policyId} onChange={(event) => invalidate(() => setPolicyId(event.target.value))}>{bootstrap?.policy_catalog.map((policy) => <option key={policy.policy_id} value={policy.policy_id}>{policy.label}{policy.training_step == null ? "" : ` · step ${policy.training_step}`}</option>)}</select></label>
        <label>仿真次数<input aria-label="仿真次数" type="number" min={1} max={1000} value={trials} onChange={(event) => invalidate(() => setTrials(Number(event.target.value)))} /></label>
        <label>每回合控制步数<input aria-label="每回合控制步数" type="number" min={1} max={10000} value={maxSteps} onChange={(event) => invalidate(() => setMaxSteps(Number(event.target.value)))} /></label>
        <label>每次预测执行步数<input aria-label="测试预测执行步数" type="number" min={1} max={8} value={openLoopSteps} onChange={(event) => invalidate(() => setOpenLoopSteps(Number(event.target.value)))} /></label>
        <fieldset className="evaluation-randomization-options">
          <legend>测试随机化</legend>
          <label className="evaluation-randomization-option">
            <input aria-label="启用随机环境" type="checkbox" checked={randomizeInitStates} onChange={(event) => invalidate(() => setRandomizeInitStates(event.target.checked))} />
            <span><b>启用随机环境</b><small>{randomizeInitStates ? `在 ${customStates ? stateIndices.length : task?.init_state_count ?? 0} 个有限 init states 中均衡调度` : `固定使用 init state #${fixedInitStateIndex}`}</small></span>
          </label>
          <label className="evaluation-randomization-option">
            <input aria-label="启用随机种子" type="checkbox" checked={randomizeSeeds} onChange={(event) => invalidate(() => setRandomizeSeeds(event.target.checked))} />
            <span><b>启用随机种子</b><small>{randomizeSeeds ? (customSeedCount ? `使用 ${seedCount} 个连续 seed` : "根据测试规模自动计算 seed 数量") : `固定使用 seed ${baseSeed}`}</small></span>
          </label>
          <p>两项默认启用；调度会在选定环境与 seed 组合之间保持确定性均衡。</p>
        </fieldset>
        <details>
          <summary>高级随机环境与执行设置</summary>
          <div className="evaluation-advanced">
            <fieldset><legend>环境（init_state_index）</legend>{randomizeInitStates ? <><label className="switch-field"><input type="checkbox" checked={customStates} onChange={(event) => invalidate(() => setCustomStates(event.target.checked))} />自定义随机环境池</label>{customStates && <div className="state-index-grid">{availableStates.map((index) => <label key={index}><input type="checkbox" checked={stateIndices.includes(index)} onChange={() => toggleState(index)} />#{index}</label>)}</div>}<small>{customStates ? `已选择 ${stateIndices.length} 个状态` : "自动使用当前任务全部合法状态"}</small></> : <><label>固定 init_state_index<select aria-label="固定初始状态" value={fixedInitStateIndex} onChange={(event) => invalidate(() => setFixedInitStateIndex(Number(event.target.value)))}>{availableStates.map((index) => <option key={index} value={index}>#{index}</option>)}</select></label><small>随机环境已关闭，每个回合使用同一个 benchmark 初始状态。</small></>}</fieldset>
            <fieldset><legend>参数（seed）</legend><label>基础 seed<input type="number" min={0} value={baseSeed} onChange={(event) => invalidate(() => setBaseSeed(Number(event.target.value)))} /></label>{randomizeSeeds ? <><label className="switch-field"><input type="checkbox" checked={customSeedCount} onChange={(event) => invalidate(() => setCustomSeedCount(event.target.checked))} />自定义 seed 数量</label>{customSeedCount && <label>seed 数量<input aria-label="seed 数量" type="number" min={1} max={1000} value={seedCount} onChange={(event) => invalidate(() => setSeedCount(Number(event.target.value)))} /></label>}<small>seed 池从基础 seed 连续递增；未自定义时按测试规模自动计算。</small></> : <small>随机种子已关闭，全部回合固定使用基础 seed。</small>}</fieldset>
            <fieldset><legend>执行顺序与速度</legend><label>调度 seed<input type="number" min={0} value={scheduleSeed} onChange={(event) => invalidate(() => setScheduleSeed(Number(event.target.value)))} /></label><label className="switch-field"><input type="checkbox" checked={realtime} onChange={(event) => invalidate(() => setRealtime(event.target.checked))} />严格 20 Hz 墙钟限速</label><small>关闭限速只加快墙钟速度，不改变 20 Hz 模拟时间语义。</small></fieldset>
          </div>
        </details>
      </div>
      <div className="evaluation-builder-actions"><button onClick={() => void makePreview()} disabled={busy || !taskId || !policyId || (randomizeInitStates && customStates && !stateIndices.length)}>预览调度</button><button className="primary" onClick={() => void begin()} disabled={busy || !preview || Boolean(job && activeStatuses.has(job.status))}>开始测试</button></div>
      {preview && <div className="evaluation-preview">
        <div><strong>{preview.schedule.length || trials}</strong><span>回合</span></div><div><strong>{Object.keys(preview.init_state_counts ?? {}).length}</strong><span>环境数</span></div><div><strong>{Object.keys(preview.seed_counts ?? {}).length}</strong><span>Seed 数</span></div><div><strong>{duration(preview.estimated_duration_seconds)}</strong><span>预计墙钟时间</span></div>
        <p>调度哈希 <code>{preview.schedule_sha256?.slice(0, 20)}</code> · 每个状态、seed 与组合的分配次数差不超过 1。</p>
        <div className="preview-schedule">{preview.schedule.slice(0, 12).map((item) => <span key={item.trial_index}>#{item.trial_index + 1} · init {item.init_state_index} · seed {item.seed}</span>)}</div>
      </div>}
    </section>

    {job && <EvaluationMonitor initial={job} onUpdate={updateJob} onDismiss={() => setJob(null)} />}

    <section className="surface evaluation-history">
      <div className="panel-title"><strong>测试历史</strong><span>{records.length}</span></div>
      <div className="evaluation-filters">
        <TaskFilter tasks={bootstrap?.task_catalog ?? []} value={historyScope} onChange={setHistoryScope} labelPrefix="测试记录" />
        <label>模型<select value={filters.policy_id ?? ""} onChange={(event) => setFilters((current) => ({ ...current, policy_id: event.target.value || undefined }))}><option value="">全部模型</option>{bootstrap?.policy_catalog.map((item) => <option key={item.policy_id} value={item.policy_id}>{item.label}</option>)}</select></label>
        <label>状态<select value={filters.status ?? ""} onChange={(event) => setFilters((current) => ({ ...current, status: event.target.value as EvaluationStatus || undefined }))}><option value="">全部状态</option>{["STARTING", "RUNNING", "STOPPING", "COMPLETED", "FAILED", "CANCELED"].map((status) => <option key={status}>{status}</option>)}</select></label>
        <label>开始日期<input type="date" value={filters.date_from ?? ""} onChange={(event) => setFilters((current) => ({ ...current, date_from: event.target.value || undefined }))} /></label><label>结束日期<input type="date" value={filters.date_to ?? ""} onChange={(event) => setFilters((current) => ({ ...current, date_to: event.target.value || undefined }))} /></label>
        <button onClick={() => void loadHistory().catch((reason) => setError(String(reason)))} disabled={busy}>检索</button>
      </div>
      <div className="table-wrap"><table><thead><tr><th>时间 / ID</th><th>任务</th><th>策略</th><th>规模</th><th>成功率</th><th>覆盖率</th><th>状态 / 耗时</th><th>操作</th></tr></thead><tbody>{records.map((record) => <tr key={record.id} className={selected?.id === record.id ? "selected-row" : ""}>
        <td>{record.created_at ? new Date(record.created_at).toLocaleString() : "—"}<small className="cell-note">{record.id}</small></td><td>{record.task_prompt || record.task_name}</td><td>{record.policy_label}</td><td>{record.aggregate?.attempted_trials ?? 0}/{record.config?.trials ?? record.aggregate?.total_trials ?? 0}</td><td><strong>{percentage(record.aggregate?.success_rate)}</strong><small className="cell-note">{record.aggregate?.successes ?? 0} 成功 · {record.aggregate?.errors ?? 0} error</small></td><td>{percentage(record.aggregate?.completion_rate)}</td><td><Badge tone={badgeTone(record.status)}>{record.status}</Badge><small className="cell-note">{duration(record.wall_time_seconds)}</small></td>
        <td><div className="history-actions"><button onClick={() => void inspect(record.id)}>查看详情</button><button className="danger" disabled={busy || activeStatuses.has(record.status)} onClick={() => void remove(record)}>删除</button></div></td>
      </tr>)}</tbody></table>{!records.length && <div className="empty-table">尚无符合条件的策略测试。</div>}</div>
    </section>
    {selected && <EvaluationDetail record={selected} />}
  </section>;
}
