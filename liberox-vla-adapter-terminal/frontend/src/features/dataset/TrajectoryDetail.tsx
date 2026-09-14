import { useEffect, useMemo, useRef, useState } from "react";
import type { StageAnnotation, TrajectoryDetail as Detail } from "../run-control/types";
import { Badge } from "../../components/ui/Badge";
import { Dialog } from "../../components/ui/Dialog";
import { StageAnnotationPanel } from "./StageAnnotationPanel";
import { selectMainVideoArtifact } from "../simulation-view/controls";
import { rewardParameterLabels, rewardSourceLabels } from "./rewardVersions";

const colors = ["#3775e8", "#e06c4f", "#3c9a70", "#9b63d4", "#c38b27", "#3c94a6", "#d14f86"];

type Plot = {
  title: string; unit: string; times: number[]; labels: string[]; values: number[][];
  interpolation?: "linear" | "step"; sampleLabels?: string[];
  xLabel?: string; yLabel?: string;
};

type ChartGeometry = {
  width: number; height: number; left: number; right: number; top: number; bottom: number;
};

export function chunkRewardSeries(
  starts: number[], ends: number[], rewards: number[], timeSeconds: number[],
) {
  if (starts.length !== ends.length || starts.length !== rewards.length) {
    throw new Error("chunk reward metadata length mismatch");
  }
  const times: number[] = [];
  const values: number[][] = [];
  const sampleLabels: string[] = [];
  starts.forEach((start, chunk) => {
    const end = ends[chunk];
    if (!Number.isInteger(start) || !Number.isInteger(end) || end <= start) {
      throw new Error(`invalid chunk interval [${start}, ${end})`);
    }
    times.push(timeSeconds[end] ?? end / 20);
    values.push([rewards[chunk]]);
    sampleLabels.push(`chunk ${chunk} · steps [${start}, ${end}) · L=${end - start}`);
  });
  return { times, values, sampleLabels };
}

export function observationPotentialSeries(remainingTimeSeconds: number[][]) {
  return remainingTimeSeconds.map((heads) => heads.map((seconds) => (
    seconds === 0 ? 0 : -seconds
  )));
}

function interpolate(times: number[], values: number[], targets: number[]) {
  return targets.map((target) => {
    if (!times.length) return Number.NaN;
    if (target <= times[0]) return values[0];
    if (target >= times[times.length - 1]) return values[values.length - 1];
    let right = 1;
    while (right < times.length && times[right] < target) right += 1;
    const left = right - 1;
    const ratio = (target - times[left]) / (times[right] - times[left]);
    return values[left] + ratio * (values[right] - values[left]);
  });
}

function pearson(left: number[], right: number[]) {
  if (left.length !== right.length || left.length < 2) return null;
  const lm = left.reduce((sum, value) => sum + value, 0) / left.length;
  const rm = right.reduce((sum, value) => sum + value, 0) / right.length;
  let numerator = 0; let ld = 0; let rd = 0;
  left.forEach((value, index) => {
    const a = value - lm; const b = right[index] - rm;
    numerator += a * b; ld += a * a; rd += b * b;
  });
  return ld <= 1e-12 || rd <= 1e-12 ? null : numerator / Math.sqrt(ld * rd);
}

function bounds(values: number[]) {
  let low = Number.POSITIVE_INFINITY; let high = Number.NEGATIVE_INFINITY;
  values.forEach((value) => { if (Number.isFinite(value)) { low = Math.min(low, value); high = Math.max(high, value); } });
  if (!Number.isFinite(low)) return [0, 1] as const;
  return [low, high === low ? low + 1 : high] as const;
}

function pathData(
  times: number[], values: number[], geometry: ChartGeometry,
  range = bounds(values), timeRange = bounds(times),
  interpolation: "linear" | "step" = "linear",
) {
  if (!values.length) return "";
  const [low, high] = range;
  const [firstTime, lastTime] = timeRange;
  const span = high - low;
  const timeSpan = lastTime - firstTime;
  const innerWidth = geometry.width - geometry.left - geometry.right;
  const innerHeight = geometry.height - geometry.top - geometry.bottom;
  const coordinates = values.map((value, index) => {
    const time = times[index] ?? index;
    const x = geometry.left + (time - firstTime) / timeSpan * innerWidth;
    const y = geometry.top + innerHeight - (value - low) / span * innerHeight;
    return [x, y] as const;
  });
  let result = `M ${coordinates[0][0].toFixed(2)} ${coordinates[0][1].toFixed(2)}`;
  for (let index = 1; index < coordinates.length; index += 1) {
    const [x, y] = coordinates[index];
    if (interpolation === "step") {
      result += ` H ${x.toFixed(2)} V ${y.toFixed(2)}`;
    } else {
      result += ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
    }
  }
  return result;
}

function tickValues([low, high]: readonly [number, number], count = 5) {
  return Array.from({ length: count }, (_, index) => low + (high - low) * index / (count - 1));
}

function tickLabel(value: number) {
  const absolute = Math.abs(value);
  if (absolute >= 1000 || (absolute > 0 && absolute < 0.01)) return value.toExponential(1);
  return value.toFixed(absolute >= 10 ? 1 : 2);
}

function Chart({ plot, large = false, cursorIndex }: {
  plot: Plot; large?: boolean; cursorIndex?: number;
}) {
  const geometry: ChartGeometry = large
    ? { width: 940, height: 330, left: 72, right: 22, top: 18, bottom: 48 }
    : { width: 680, height: 220, left: 58, right: 16, top: 12, bottom: 40 };
  const valueRange = bounds(plot.values.flat());
  const timeRange = bounds(plot.times);
  const xTicks = tickValues(timeRange);
  const yTicks = tickValues(valueRange);
  const innerWidth = geometry.width - geometry.left - geometry.right;
  const innerHeight = geometry.height - geometry.top - geometry.bottom;
  const x = (value: number) => geometry.left
    + (value - timeRange[0]) / (timeRange[1] - timeRange[0]) * innerWidth;
  const y = (value: number) => geometry.top + innerHeight
    - (value - valueRange[0]) / (valueRange[1] - valueRange[0]) * innerHeight;
  const cursorX = cursorIndex == null ? null : x(plot.times[cursorIndex] ?? cursorIndex);
  return <svg viewBox={`0 0 ${geometry.width} ${geometry.height}`} preserveAspectRatio="xMidYMid meet" aria-label={plot.title}>
    {yTicks.map((value) => <g key={`y-${value}`}><line x1={geometry.left} x2={geometry.width - geometry.right} y1={y(value)} y2={y(value)} className="plot-grid" /><text x={geometry.left - 8} y={y(value) + 4} textAnchor="end" className="plot-tick">{tickLabel(value)}</text></g>)}
    {xTicks.map((value) => <g key={`x-${value}`}><line x1={x(value)} x2={x(value)} y1={geometry.top} y2={geometry.top + innerHeight} className="plot-grid" /><text x={x(value)} y={geometry.top + innerHeight + 18} textAnchor="middle" className="plot-tick">{tickLabel(value)}</text></g>)}
    <line x1={geometry.left} x2={geometry.left} y1={geometry.top} y2={geometry.top + innerHeight} className="plot-axis" />
    <line x1={geometry.left} x2={geometry.width - geometry.right} y1={geometry.top + innerHeight} y2={geometry.top + innerHeight} className="plot-axis" />
    {plot.labels.map((label, index) => <path key={label} d={pathData(plot.times, plot.values.map((row) => row[index]), geometry, valueRange, timeRange, plot.interpolation)} fill="none" stroke={colors[index % colors.length]} strokeWidth="2" vectorEffect="non-scaling-stroke" />)}
    {cursorX != null && <line x1={cursorX} x2={cursorX} y1={geometry.top} y2={geometry.top + innerHeight} stroke="#111" strokeWidth="1" vectorEffect="non-scaling-stroke" />}
    <text x={geometry.left + innerWidth / 2} y={geometry.height - 5} textAnchor="middle" className="plot-axis-label">{plot.xLabel ?? "时间 [s]"}</text>
    <text transform={`translate(14 ${geometry.top + innerHeight / 2}) rotate(-90)`} textAnchor="middle" className="plot-axis-label">{plot.yLabel ?? plot.unit}</text>
  </svg>;
}

function PlotCard({ plot, onOpen }: { plot: Plot; onOpen: () => void }) {
  return <article className="trajectory-plot">
    <div className="trajectory-plot-title"><strong>{plot.title}</strong><span>{plot.unit} · 点击查看逐时间点数据</span></div>
    <button className="trajectory-plot-open" onClick={onOpen} aria-label={`查看${plot.title}逐时间点数据`}>
      <Chart plot={plot} />
      <div className="plot-legend">{plot.labels.map((label, index) => <span key={label}><i style={{ background: colors[index % colors.length] }} />{label}</span>)}</div>
    </button>
  </article>;
}

function PlotInspector({ plot, onClose }: { plot: Plot; onClose: () => void }) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing || plot.values.length < 2) return;
    const timer = window.setInterval(() => setIndex((value) => value + 1 >= plot.values.length ? 0 : value + 1), 100);
    return () => window.clearInterval(timer);
  }, [playing, plot.values.length]);
  const current = plot.values[index] ?? [];
  return <Dialog title={`${plot.title} · 逐时间点预览`} actions={<><button onClick={() => setPlaying((value) => !value)}>{playing ? "暂停" : "自动播放"}</button><button className="primary" onClick={onClose}>关闭</button></>}>
    <div className="plot-inspector-chart"><Chart plot={plot} large cursorIndex={index} /></div>
    <input className="plot-time-slider" type="range" min={0} max={Math.max(0, plot.values.length - 1)} value={index} onChange={(event) => { setPlaying(false); setIndex(Number(event.target.value)); }} />
    <div className="plot-sample"><strong>{(plot.times[index] ?? index).toFixed(3)} s</strong>{plot.sampleLabels?.[index] && <span>{plot.sampleLabels[index]}</span>}{plot.labels.map((label, line) => <span key={label}><i style={{ background: colors[line % colors.length] }} />{label}: <b>{Number(current[line] ?? 0).toFixed(5)}</b> {plot.unit}</span>)}</div>
  </Dialog>;
}

export function TrajectoryDetail({ detail, onBack, onSetTest, onContextChange, contextLoading = false }: {
  detail: Detail; onBack: () => void;
  onSetTest?: (isTest: boolean) => Promise<void>;
  onContextChange?: (datasetId?: string) => void;
  contextLoading?: boolean;
}) {
  const [openedTitle, setOpenedTitle] = useState<string | null>(null);
  const [isTest, setIsTest] = useState(Boolean(detail.run.is_test));
  const [labelBusy, setLabelBusy] = useState(false);
  const [labelError, setLabelError] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stageAnnotation, setStageAnnotation] = useState<StageAnnotation | null>(null);
  const [stageDirty, setStageDirty] = useState(false);
  const plots = useMemo<Plot[]>(() => {
    const values: Plot[] = [
      { title: "VLA 环境 action", unit: "normalized command [-]", times: detail.series.action_time_seconds, labels: ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"], values: detail.series.env_action },
    ];
    const reward = detail.reward_evaluation;
    if (reward?.source === "stage" && reward.stage_scores?.length) {
      const times = reward.time_seconds ?? detail.series.time_seconds;
      values.push({ title: "Stage-based Reward", unit: "reward [-]", times,
        labels: ["direct stage reward"], values: reward.stage_scores.map((score) => [score]),
        sampleLabels: times.map((_, index) => `observation step ${reward.observation_steps?.[index] ?? index}`) });
    } else if (!detail.dataset_context && !detail.global_evaluation && !detail.global_evaluation_pending
      && !detail.global_evaluation_error && !reward && !detail.rynnvalue_evaluation
      && !detail.evaluation && stageAnnotation?.status === "ready" && stageAnnotation.scores.length) {
      values.push({
        title: "关键帧奖励预览（未评价）", unit: "reward [-]", times: stageAnnotation.time_seconds,
        labels: ["direct stage reward"], values: stageAnnotation.scores.map((score) => [score]),
        sampleLabels: stageAnnotation.time_seconds.map((_, step) => `observation step ${step}`),
      });
    }
    if (reward && reward.source !== "rynnvalue") {
      const series = chunkRewardSeries(reward.chunk_start_steps, reward.chunk_end_steps, reward.final_reward, detail.series.time_seconds);
      values.push({ title: `${rewardSourceLabels[reward.source]} · Final Reward · ${reward.reward_config.accumulate_primitive_steps ? "逐步累计" : "宏动作"}`,
        unit: "reward [-]", ...series, labels: ["final reward"], interpolation: "step" });
    }
    const evaluation = detail.rynnvalue_evaluation ?? detail.evaluation;
    if (evaluation) {
      const official = evaluation.official_outputs;
      const boundaryTimes = evaluation.boundary_steps.map(
        (step) => detail.series.time_seconds[step] ?? step / 20,
      );
      const absoluteHeadCount = official.absolute_temporal_distance_seconds[0]?.length ?? 0;
      const absoluteHeadLabels = Array.from(
        { length: absoluteHeadCount },
        (_, index) => absoluteHeadCount === 1 ? "remaining time" : `remaining time head ${index}`,
      );
      const potentialHeadLabels = Array.from(
        { length: absoluteHeadCount },
        (_, index) => absoluteHeadCount === 1
          ? "observation potential Φ(s)"
          : `observation potential Φ${index}(s)`,
      );
      values.push(
        {
          title: "RynnValue Absolute Remaining Time", unit: "s",
          times: boundaryTimes, labels: absoluteHeadLabels,
          values: official.absolute_temporal_distance_seconds,
        },
        {
          title: "RynnValue Observation Potential", unit: "s",
          times: boundaryTimes, labels: potentialHeadLabels,
          values: observationPotentialSeries(official.absolute_temporal_distance_seconds),
        },
        {
          title: "RynnValue Relative Remaining Time", unit: "s",
          times: boundaryTimes, labels: ["relative remaining time"],
          values: official.relative_temporal_distance_seconds.map((value) => [value]),
        },
        {
          title: "RynnValue Entropy", unit: "nats",
          times: boundaryTimes,
          labels: absoluteHeadLabels.map((label) => `${label} entropy`),
          values: official.absolute_value_entropy_nats,
        },
      );
      // Raw model diagnostics may coexist with another selected reward source.
      // Only the current source supplies the displayed training reward.
      if (!reward || reward.source === "rynnvalue") {
        const sparseReward = evaluation.pbrs_reward.sparse_reward;
        if (sparseReward.length !== evaluation.pbrs_reward.shape_reward.length
          || sparseReward.length !== evaluation.pbrs_reward.dense_reward.length
          || sparseReward.length !== evaluation.pbrs_reward.final_reward.length) {
          throw new Error("reward component length mismatch");
        }
        const shapeSeries = chunkRewardSeries(evaluation.pbrs_reward.chunk_start_steps,
          evaluation.pbrs_reward.chunk_end_steps, evaluation.pbrs_reward.shape_reward, detail.series.time_seconds);
        values.push({
          title: `Reward Components · ${evaluation.pbrs_reward.accumulate_primitive_steps
            ? "逐步累计" : "宏动作"}`,
          unit: "reward [-]",
          times: shapeSeries.times,
          labels: ["sparse reward", "dense reward (κ × shape)", "final reward"],
          values: sparseReward.map((sparse, index) => [
            sparse, evaluation.pbrs_reward.dense_reward[index],
            evaluation.pbrs_reward.final_reward[index],
          ]),
          sampleLabels: shapeSeries.sampleLabels,
          interpolation: "step",
        });
      }
    }
    const robometer = detail.robometer_evaluation;
    if (robometer) {
      values.push(
        { title: "Robometer Progress · 模型原始输出", unit: "probability [-]",
          times: robometer.time_seconds, labels: ["progress_pred"],
          values: robometer.progress_pred.map((value) => [value]) },
        { title: "Robometer Success Probability · 模型原始输出", unit: "probability [-]",
          times: robometer.time_seconds, labels: ["success_probs"],
          values: robometer.success_probs.map((value) => [value]) },
        { title: "环境 done", unit: "boolean [-]", times: detail.series.time_seconds,
          labels: ["done"], values: detail.series.done.map((value) => [value ? 1 : 0]),
          interpolation: "step" },
      );
    }
    if (evaluation && robometer) {
      const distance = evaluation.official_outputs.absolute_temporal_distance_seconds
        .map((heads) => heads[0]);
      const d0 = distance[0];
      if (distance.length >= 2 && Number.isFinite(d0) && d0 > 1e-6) {
        const boundaryTimes = evaluation.boundary_steps.map(
          (step) => detail.series.time_seconds[step] ?? step / 20,
        );
        const progress = distance.map((value) => Math.max(0, Math.min(1, 1 - value / d0)));
        const aligned = interpolate(boundaryTimes, progress, robometer.time_seconds);
        values.push({
          title: "RynnValue / Robometer · UI 派生归一化对比", unit: "normalized progress [-]",
          times: robometer.time_seconds, labels: ["Robometer progress", "Rynn normalized progress"],
          values: robometer.progress_pred.map((value, index) => [value, aligned[index]]),
        });
      }
    }
    return values;
  }, [detail, stageAnnotation]);
  const opened = plots.find((plot) => plot.title === openedTitle);
  const videoName = selectMainVideoArtifact(detail.artifacts);
  const video = videoName ? detail.artifacts[videoName] : null;
  const comparison = useMemo(() => {
    const rynn = detail.rynnvalue_evaluation ?? detail.evaluation;
    const robo = detail.robometer_evaluation;
    if (!rynn || !robo) return null;
    const distance = rynn.official_outputs.absolute_temporal_distance_seconds.map((row) => row[0]);
    const d0 = distance[0];
    if (distance.length < 2 || robo.progress_pred.length < 2 || !Number.isFinite(d0) || d0 <= 1e-6) {
      return { available: false as const, reason: "初始距离过小或样本不足，无法计算相关性" };
    }
    const times = rynn.boundary_steps.map((step) => detail.series.time_seconds[step] ?? step / 20);
    const normalized = distance.map((value) => Math.max(0, Math.min(1, 1 - value / d0)));
    const aligned = interpolate(times, normalized, robo.time_seconds);
    const correlation = pearson(aligned, robo.progress_pred);
    return {
      available: correlation != null, correlation,
      reason: correlation == null ? "至少一条序列没有方差，无法计算相关性" : null,
      rynnEnd: aligned[aligned.length - 1], roboEnd: robo.progress_pred.at(-1),
      successEnd: robo.success_probs.at(-1), environmentSuccess: detail.run.success,
    };
  }, [detail]);
  const evaluationSource = detail.dataset_context?.source ?? detail.global_evaluation?.source;
  return <section className="content-page trajectory-detail-page">
    <div className="page-heading detail-heading"><div><p className="eyebrow">TRAJECTORY DETAIL</p><h1>轨迹 {detail.run.id}</h1><p>{detail.run.task} · {detail.run.action_count} steps</p></div><button onClick={() => {
      if (!stageDirty || window.confirm("切片标记尚未保存，是否丢弃并返回数据集？")) onBack();
    }}>返回数据集</button></div>
    <div className="detail-summary surface"><Badge tone={detail.run.success ? "green" : "neutral"}>{detail.run.success ? "成功" : "失败"}</Badge><span>{detail.run.source_type ?? detail.run.control_mode}</span><span>{detail.run.created_at ? new Date(detail.run.created_at).toLocaleString() : "—"}</span><label className="test-label-switch"><input type="checkbox" checked={isTest} disabled={labelBusy || !onSetTest} onChange={async (event) => { const next = event.target.checked; setLabelBusy(true); setLabelError(""); try { await onSetTest?.(next); setIsTest(next); } catch (reason) { setLabelError(String(reason)); } finally { setLabelBusy(false); } }} /><span><b>测试标签</b><small>启用后不进入新训练数据集，任务级批量评价默认跳过</small></span></label></div>
    {onContextChange && <div className="surface detail-evaluation-context">
      <label>数据来源<select disabled={contextLoading} value={detail.dataset_context?.dataset_id ?? ""} onChange={(event) => onContextChange(event.target.value || undefined)}>
        <option value="">全局轨迹评价</option>{detail.available_dataset_contexts?.map((context) => <option key={context.dataset_id} value={context.dataset_id}>{context.dataset_name}</option>)}
      </select></label>
      <div><strong>{contextLoading || detail.global_evaluation_pending ? "正在更新评价图表…" : detail.dataset_context
        ? `${detail.dataset_context.dataset_name} · 当前评价` : "全局轨迹评价"}</strong>
        <div className="reward-parameter-chips">{evaluationSource && <span>{rewardSourceLabels[evaluationSource]}</span>}
          {rewardParameterLabels(detail.dataset_context?.config ?? detail.global_evaluation?.config).map((label) => <span key={label}>{label}</span>)}</div>
        <small>关键帧保存在原轨迹中；在数据集配置中重新评价即可更新奖励。</small></div>
    </div>}
    {!detail.dataset_context && detail.global_evaluation_error && <p className="error-banner">{detail.global_evaluation_error}</p>}
    {labelError && <div className="error-banner"><span>{labelError}</span><button onClick={() => setLabelError("")}>关闭</button></div>}
    <article className="surface trajectory-video"><h2>结果视频</h2>{video ? <video ref={videoRef} controls preload="metadata" src={video} /> : <div className="empty-table">没有可用结果视频</div>}
      <StageAnnotationPanel key={detail.run.id} runId={detail.run.id} videoRef={videoRef} onSaved={setStageAnnotation} onDirtyChange={setStageDirty} />
    </article>
    {comparison && <article className="surface detail-summary"><strong>RynnValue / Robometer 对比</strong>{comparison.available ? <><span>Pearson r = {comparison.correlation?.toFixed(4)}</span><span>终点进度：Rynn {comparison.rynnEnd?.toFixed(3)} / Robometer {comparison.roboEnd?.toFixed(3)}</span><span>Robometer 成功概率 {comparison.successEnd?.toFixed(3)}</span><span>环境成功：{comparison.environmentSuccess ? "是" : "否"}</span></> : <span>{comparison.reason}</span>}</article>}
    <div className="trajectory-plots">{plots.map((plot) => <PlotCard key={plot.title} plot={plot} onOpen={() => setOpenedTitle(plot.title)} />)}</div>
    {opened && <PlotInspector key={opened.title} plot={opened} onClose={() => setOpenedTitle(null)} />}
  </section>;
}
