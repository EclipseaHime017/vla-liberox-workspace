import { useEffect, useState } from "react";
import {
  annotateTrainingDataset, getDatasetRewardConfig,
  listTrainingDatasetMembers, verifyTrainingDataset, updateDatasetTrainingOptions,
} from "../run-control/api";
import type {
  OfflineJob, PaginatedRuns, RewardParameters, RewardSource, TrainingDataset, EvaluationOperation,
} from "../run-control/types";
import { Badge } from "../../components/ui/Badge";
import { RunTable } from "./RunTable";
import { evaluationOperationLabels, rewardParameterLabels, rewardSourceLabels } from "./rewardVersions";
const successful = (status: string) => ["COMPLETED", "READY"].includes(status);

export function FrozenDatasetCard({ dataset, disabled, robometerUnavailable, onRemove, onDerive, initialExpanded = false,
  onRefresh, onJob, onError, onOpen }: {
  dataset: TrainingDataset; disabled: boolean; robometerUnavailable?: string;
  onRemove?: () => void; onDerive?: () => void; initialExpanded?: boolean; onRefresh: () => Promise<void>;
  onJob: (job: OfflineJob) => void; onError: (error: string) => void;
  onOpen?: (runId: string, datasetId: string) => void;
}) {
  const [expanded, setExpanded] = useState(initialExpanded);
  const [showMembers, setShowMembers] = useState(false);
  const [configs, setConfigs] = useState<Partial<Record<RewardSource | EvaluationOperation, RewardParameters>>>({});
  const [configReady, setConfigReady] = useState(false);
  const [source, setSource] = useState<EvaluationOperation>("final");
  const [busy, setBusy] = useState(false);
  const [includePostSuccess, setIncludePostSuccess] = useState(dataset.include_post_success ?? true);
  useEffect(() => { setIncludePostSuccess(dataset.include_post_success ?? true); },
    [dataset.id, dataset.include_post_success]);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(5);
  const [members, setMembers] = useState<PaginatedRuns | null>(null);
  const [membersLoading, setMembersLoading] = useState(false);
  const versions = dataset.evaluation_versions ?? [];
  const evaluationRevision = JSON.stringify([dataset.evaluation_version_ids,
    dataset.reward_version_id, dataset.robometer_version_id,
    versions.map(({ id, status, completed_at }) => [id, status, completed_at])]);
  const currentFor = (evaluator: RewardSource) => {
    const id = dataset.evaluation_version_ids
      ? dataset.evaluation_version_ids[evaluator]
      : evaluator === "robometer" ? dataset.robometer_version_id : dataset.reward_version_id;
    const selected = versions.find((item) => item.id === id && item.evaluator === evaluator && successful(item.status));
    return selected ?? (dataset.evaluation_version_ids ? undefined
      : [...versions].reverse().find((item) => item.evaluator === evaluator && successful(item.status)));
  };
  const parameters = configs[source] ?? {};
  const currentEvaluation = currentFor(source === "all" ? "final" : source);
  const latestAttempt = [...versions].reverse().find((item) => item.evaluator === (source === "all" ? "final" : source));
  const finalMode = source === "final" || source === "all";
  const multiplicative = finalMode && parameters.fusion_mode === "multiplicative";
  const taskRunning = dataset.annotation_status === "RUNNING"
    || versions.some((version) => ["STARTING", "RUNNING", "STOPPING"].includes(version.status));
  const blocked = disabled || busy || taskRunning;

  useEffect(() => {
    if (!expanded || configReady) return;
    let current = true;
    setBusy(true);
    void getDatasetRewardConfig(dataset.id).then((defaults) => {
      if (!current) return;
      const next = { ...defaults };
      for (const evaluator of Object.keys(rewardSourceLabels) as RewardSource[]) {
        const version = currentFor(evaluator);
        next[evaluator] = { ...next[evaluator], ...version?.parameters,
          checkpoint: defaults[evaluator]?.checkpoint, revision: defaults[evaluator]?.revision,
          prefix_frames: defaults[evaluator]?.prefix_frames, force_model: false, overwrite_global: false };
      }
      next.all = { ...next.all, ...next.final,
        max_frames: next.rynnvalue.max_frames, batch_size: next.rynnvalue.batch_size,
        sampling_hz: next.robometer.sampling_hz, robometer_batch_size: next.robometer.batch_size,
        force_model: true };
      for (const kind of ["final", "all"] as const) {
        if (next[kind].fusion_mode === "multiplicative") next[kind].accumulate_primitive_steps = false;
      }
      setConfigs(next); setConfigReady(true);
    }).catch((error) => { if (current) onError(String(error)); })
      .finally(() => { if (current) setBusy(false); });
    return () => { current = false; };
    // Defaults initialize the form once; background polling must not overwrite edits.
  }, [expanded, configReady, dataset.id]);

  useEffect(() => {
    if (!showMembers) return;
    let current = true;
    setMembersLoading(true);
    void listTrainingDatasetMembers(dataset.id, page, pageSize).then((next) => {
      if (current) { setMembers(next); if (next.page !== page) setPage(next.page); }
    }).catch((error) => { if (current) onError(String(error)); })
      .finally(() => { if (current) setMembersLoading(false); });
    return () => { current = false; };
  }, [showMembers, dataset.id, page, pageSize, evaluationRevision]);

  const patch = (values: Partial<RewardParameters>) => setConfigs((current) => ({
    ...current, [source]: { ...current[source], ...values },
  }));
  const run = async (operation: () => Promise<void>) => {
    setBusy(true);
    try { await operation(); } catch (error) { onError(String(error)); }
    finally { setBusy(false); }
  };
  const generate = () => run(async () => {
    // Only send parameters meaningful for this evaluator; display metadata is read-only.
    const request: RewardParameters & { source: EvaluationOperation } = { source, overwrite_global: parameters.overwrite_global ?? false };
    if (source !== "robometer") {
      request.gamma = parameters.gamma;
      request.accumulate_primitive_steps = multiplicative ? false : parameters.accumulate_primitive_steps;
    }
    if (finalMode) {
      request.stage_exponent = parameters.stage_exponent;
      request.alpha = parameters.alpha ?? 0;
      request.fusion_mode = parameters.fusion_mode ?? "additive";
      request.shaping_weight = parameters.shaping_weight;
    }
    if (source === "rynnvalue" || source === "all") {
      request.shaping_weight = parameters.shaping_weight;
      request.max_frames = parameters.max_frames;
    }
    if (source === "rynnvalue" || source === "robometer" || source === "all") {
      request.batch_size = parameters.batch_size;
      request.force_model = source === "all" || (parameters.force_model ?? false);
    }
    if (source === "robometer" || source === "all") request.sampling_hz = parameters.sampling_hz;
    if (source === "all") request.robometer_batch_size = parameters.robometer_batch_size;
    onJob(await annotateTrainingDataset(dataset.id, request));
    await onRefresh();
  });
  const numeric = (key: keyof RewardParameters, label: string, min: number, step: number | string, max?: number) => (
    <label>{label}<input type="number" min={min} max={max} step={step}
      value={String(parameters[key] ?? "")} onChange={(event) => patch({ [key]: Number(event.target.value) })} /></label>
  );

  return <article className="dataset-card">
    <div className="dataset-card-description"><h2>{dataset.name}</h2><p>{dataset.member_count} 条轨迹 · {dataset.action_count} actions · {dataset.chunk_count} chunks</p>
      <p>成功后动作：{dataset.include_post_success === false ? "截断" : "保留"}</p>
      <p>{(["final", "rynnvalue", "robometer"] as RewardSource[]).map((evaluator) =>
        `${rewardSourceLabels[evaluator]}：${currentFor(evaluator) ? "数据集专属" : "继承全局"}`).join(" · ")}</p></div>
    <div className="dataset-badges"><Badge tone={dataset.integrity_status === "HEALTHY" ? "green" : "red"}>{dataset.integrity_status}</Badge>
      <Badge tone={dataset.annotation_status === "READY" ? "green" : dataset.annotation_status === "ERROR" ? "red" : "neutral"}>{dataset.annotation_status === "NOT_STARTED" ? "全局继承" : dataset.annotation_status}</Badge></div>
    <div className="dataset-card-actions">
      {onRemove && <button className="danger" disabled={blocked} onClick={onRemove}>删除数据集</button>}
      <button disabled={blocked} onClick={() => void run(async () => { await verifyTrainingDataset(dataset.id); await onRefresh(); })}>验证完整性</button>
      {onDerive && <button disabled={blocked} onClick={onDerive}>调整成员并另存</button>}
      {onOpen && <button aria-expanded={showMembers} onClick={() => setShowMembers((value) => !value)}>成员</button>}
      <button aria-expanded={expanded} aria-controls={`reward-config-${dataset.id}`} onClick={() => setExpanded((value) => !value)}>配置</button>
    </div>
    {dataset.integrity_error && <p className="dataset-integrity-error">{dataset.integrity_error}</p>}
    {expanded && <section id={`reward-config-${dataset.id}`} className="dataset-reward-config" aria-label={`${dataset.name} 评价配置`}>
      <fieldset className="dataset-sampling-controls" aria-label="训练取样配置" disabled={disabled || busy}>
        <label>成功后动作<select value={String(includePostSuccess)} onChange={(event) => setIncludePostSuccess(event.target.value === "true")}>
          <option value="true">保留</option><option value="false">截断</option>
        </select></label>
        <button disabled={includePostSuccess === (dataset.include_post_success ?? true)}
          onClick={() => void run(async () => { await updateDatasetTrainingOptions(dataset.id, includePostSuccess); await onRefresh(); })}>保存</button>
      </fieldset>
      {!configReady ? <p role="status">{busy ? "正在加载评价配置…" : "配置加载失败，请收起后重试。"}</p> : <>
        <div className="evaluation-config-heading"><div><h3>评价配置</h3><p>默认复用各轨迹的全局评价，无需为新数据集再次评价。重新评价只覆盖当前数据集的所选类型。</p></div>
          <Badge tone={currentEvaluation ? "green" : undefined}>{currentEvaluation ? "数据集专属评价" : "默认继承全局评价"}</Badge></div>
        <fieldset disabled={blocked} className={`parameter-grid ${source === "final" ? "final-reward-parameters" : ""}`}>
          <label>评价类型<select value={source} onChange={(event) => setSource(event.target.value as EvaluationOperation)}>
            {Object.entries(evaluationOperationLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          {finalMode && <><label>形式<select value={parameters.fusion_mode ?? "additive"} onChange={(event) => patch({ fusion_mode: event.target.value as "additive" | "multiplicative", ...(event.target.value === "multiplicative" ? { accumulate_primitive_steps: false } : {}) })}><option value="additive">相加</option><option value="multiplicative">相乘</option></select></label>
            {numeric("stage_exponent", "Stage 指数 p", 1, "any")}
            {parameters.fusion_mode !== "multiplicative" && numeric("alpha", "Stage 系数 α", 0, "any", 1)}
            {numeric("shaping_weight", "塑形系数 κ", 0, "any")}</>}
          <label>同步覆盖全局评价<select value={String(parameters.overwrite_global ?? false)} onChange={(event) => patch({ overwrite_global: event.target.value === "true" })}><option value="false">Off</option><option value="true">On</option></select></label>
          {source !== "robometer" && <>{numeric("gamma", "Discount γ", 0, "any", 1)}
            {!multiplicative && <label>cumulative reward<select value={String(parameters.accumulate_primitive_steps ?? false)} onChange={(event) => patch({ accumulate_primitive_steps: event.target.value === "true" })}><option value="false">Off</option><option value="true">On</option></select></label>}</>}
          {(source === "rynnvalue" || source === "all") && numeric("max_frames", "RynnValue max_frames", 2, 1, 64)}
          {source === "rynnvalue" && numeric("shaping_weight", "Shape reward 系数 κ", 0, "any")}
          {(source === "robometer" || source === "all") && <>{numeric("sampling_hz", "Robometer fps", .01, "any", 20)}<label>前缀帧数<input value={4} disabled readOnly /></label></>}
          {source === "all" && <>{numeric("batch_size", "RynnValue batch size", 1, 1)}{numeric("robometer_batch_size", "Robometer batch size", 1, 1)}</>}
          {(source === "rynnvalue" || source === "robometer") && <>{numeric("batch_size", "评价 batch size", 1, 1)}
            <label>重新运行模型<select value={String(parameters.force_model ?? false)} onChange={(event) => patch({ force_model: event.target.value === "true" })}><option value="false">Off · 复用兼容结果</option><option value="true">On · 重新计算模型输出</option></select></label></>}
        </fieldset>
        {(source === "rynnvalue" || source === "robometer") && <div className="evaluation-model-summary"><span>评价模型</span><strong title={parameters.revision ? `固定 revision：${parameters.revision}` : undefined}>{parameters.checkpoint ?? "未配置"}</strong><Badge>已锁定</Badge></div>}
        {source === "all" && (["rynnvalue", "robometer"] as const).map((model) => <div key={model} className="evaluation-model-summary"><span>{rewardSourceLabels[model]}</span><strong title={`固定 revision：${configs[model]?.revision ?? "未配置"}`}>{configs[model]?.checkpoint ?? "未配置"}</strong><Badge>已锁定</Badge></div>)}
        {currentEvaluation && <div className="evaluation-current-summary"><span>当前结果配置</span><div className="reward-parameter-chips">{rewardParameterLabels(currentEvaluation.parameters).map((label) => <span key={label}>{label}</span>)}</div></div>}
        <p className="field-hint">首次评价会保存该类型的全局结果；同步覆盖仅更新同类型的全局结果。数据集缺少某类型结果时，每条轨迹使用对应的全局结果。</p>
        <p className="field-hint">{source === "final" ? "复用已有模型输出与最新关键帧重算，不运行模型。缺少所需输入或 Stage 分数大于 0 时停止。"
          : source === "robometer" ? "仅用于诊断与对比，不改变当前训练奖励。"
            : source === "all" ? "顺序重新运行 RynnValue、Robometer，并生成 Final Reward；全部成功后替换对应结果。"
              : "已有兼容模型输出会复用；奖励参数变化只重新计算奖励。"}</p>
        {(source === "robometer" || source === "all") && robometerUnavailable && <p className="error-banner">{robometerUnavailable}</p>}
        {latestAttempt?.error && <p className="error-banner">上次评价失败：{latestAttempt.error}</p>}
        <div className="evaluation-config-actions"><button className="primary" disabled={blocked || dataset.integrity_status !== "HEALTHY" || ((source === "robometer" || source === "all") && Boolean(robometerUnavailable))} onClick={() => void generate()}>{taskRunning ? "正在评价…" : currentEvaluation ? "重新评价" : "开始评价"}</button></div>
      </>}
    </section>}
    {showMembers && <section className="dataset-member-browser" aria-label={`${dataset.name} 成员`}>
      <div className="panel-title"><strong>数据集成员</strong><span>详情显示当前数据集的评价结果</span></div>
      {membersLoading ? <p role="status">正在加载成员…</p> : members && <RunTable runs={members.items} onOpen={(runId) => onOpen?.(runId, dataset.id)} />}
      <div className="table-pagination"><button disabled={page <= 1 || membersLoading} onClick={() => setPage((value) => value - 1)}>上一页</button><span>第 {members?.page ?? page} / {members?.pages ?? 1} 页</span>
        <button disabled={page >= (members?.pages ?? 1) || membersLoading} onClick={() => setPage((value) => value + 1)}>下一页</button><label>每页<select value={pageSize} onChange={(event) => { setPageSize(Number(event.target.value)); setPage(1); }}>{[5, 10, 20, 50].map((size) => <option key={size} value={size}>{size}</option>)}</select></label></div>
    </section>}
  </article>;
}
