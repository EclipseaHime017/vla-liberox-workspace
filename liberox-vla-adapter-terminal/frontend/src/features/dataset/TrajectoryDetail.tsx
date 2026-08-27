import { useEffect, useMemo, useState } from "react";
import type { TrajectoryDetail as Detail } from "../run-control/types";
import { Badge } from "../../components/ui/Badge";
import { Dialog } from "../../components/ui/Dialog";

const colors = ["#3775e8", "#e06c4f", "#3c9a70", "#9b63d4", "#c38b27", "#3c94a6", "#d14f86"];

type Plot = {
  title: string; unit: string; times: number[]; labels: string[]; values: number[][];
  interpolation?: "linear" | "step"; sampleLabels?: string[];
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

function bounds(values: number[]) {
  let low = Number.POSITIVE_INFINITY; let high = Number.NEGATIVE_INFINITY;
  values.forEach((value) => { if (Number.isFinite(value)) { low = Math.min(low, value); high = Math.max(high, value); } });
  if (!Number.isFinite(low)) return [0, 1] as const;
  return [low, high === low ? low + 1 : high] as const;
}

function pathData(
  values: number[], width = 640, height = 170, range = bounds(values),
  interpolation: "linear" | "step" = "linear",
) {
  if (!values.length) return "";
  const [low, high] = range;
  const span = high - low;
  const coordinates = values.map((value, index) => {
    const x = values.length === 1 ? 0 : index / (values.length - 1) * width;
    const y = height - (value - low) / span * height;
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

function PlotCard({ plot, onOpen }: { plot: Plot; onOpen: () => void }) {
  const range = bounds(plot.values.flat());
  return <article className="trajectory-plot">
    <div className="trajectory-plot-title"><strong>{plot.title}</strong><span>{plot.unit} · 点击查看逐chunk数据</span></div>
    <button className="trajectory-plot-open" onClick={onOpen} aria-label={`查看${plot.title}逐chunk数据`}>
      <svg viewBox="0 0 640 170" preserveAspectRatio="none" aria-label={plot.title}>
        {plot.labels.map((label, index) => <path key={label} d={pathData(plot.values.map((row) => row[index]), 640, 170, range, plot.interpolation)} fill="none" stroke={colors[index % colors.length]} strokeWidth="2" vectorEffect="non-scaling-stroke" />)}
      </svg>
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
  const range = bounds(plot.values.flat());
  return <Dialog title={`${plot.title} · 逐时间点预览`} actions={<><button onClick={() => setPlaying((value) => !value)}>{playing ? "暂停" : "自动播放"}</button><button className="primary" onClick={onClose}>关闭</button></>}>
    <div className="plot-inspector-chart"><svg viewBox="0 0 900 280" preserveAspectRatio="none">
      {plot.labels.map((label, line) => <path key={label} d={pathData(plot.values.map((row) => row[line]), 900, 280, range, plot.interpolation)} fill="none" stroke={colors[line % colors.length]} strokeWidth="2" vectorEffect="non-scaling-stroke" />)}
      <line x1={plot.values.length <= 1 ? 0 : index / (plot.values.length - 1) * 900} x2={plot.values.length <= 1 ? 0 : index / (plot.values.length - 1) * 900} y1="0" y2="280" stroke="#111" strokeWidth="1" vectorEffect="non-scaling-stroke" />
    </svg></div>
    <input className="plot-time-slider" type="range" min={0} max={Math.max(0, plot.values.length - 1)} value={index} onChange={(event) => { setPlaying(false); setIndex(Number(event.target.value)); }} />
    <div className="plot-sample"><strong>{(plot.times[index] ?? index).toFixed(3)} s</strong>{plot.sampleLabels?.[index] && <span>{plot.sampleLabels[index]}</span>}{plot.labels.map((label, line) => <span key={label}><i style={{ background: colors[line % colors.length] }} />{label}: <b>{Number(current[line] ?? 0).toFixed(5)}</b> {plot.unit}</span>)}</div>
  </Dialog>;
}

export function TrajectoryDetail({ detail, onBack }: { detail: Detail; onBack: () => void }) {
  const [opened, setOpened] = useState<Plot | null>(null);
  const plots = useMemo<Plot[]>(() => {
    const values: Plot[] = [
      { title: "VLA 环境 action", unit: "normalized command [-]", times: detail.series.action_time_seconds, labels: ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"], values: detail.series.env_action },
      { title: "末端位置", unit: "m", times: detail.series.time_seconds, labels: ["X", "Y", "Z"], values: detail.series.eef_position },
      { title: "末端轴角", unit: "rad", times: detail.series.time_seconds, labels: ["Rx", "Ry", "Rz"], values: detail.series.eef_axis_angle },
    ];
    const evaluation = detail.evaluation;
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
      const rewardSeries = (reward: number[]) => chunkRewardSeries(
        evaluation.pbrs_reward.chunk_start_steps,
        evaluation.pbrs_reward.chunk_end_steps,
        reward,
        detail.series.time_seconds,
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
        {
          title: "RynnValue Shape Reward", unit: "reward [-]", labels: ["shape reward"],
          ...rewardSeries(evaluation.pbrs_reward.shape_reward),
        },
        {
          title: "Final Reward", unit: "reward [-]", labels: ["final reward"],
          ...rewardSeries(evaluation.pbrs_reward.final_reward),
        },
      );
    }
    return values;
  }, [detail]);
  const video = Object.entries(detail.artifacts).find(([name]) => name.endsWith("/agentview.mp4"))?.[1]
    ?? Object.entries(detail.artifacts).find(([name]) => name.endsWith(".mp4"))?.[1];
  return <section className="content-page trajectory-detail-page">
    <div className="page-heading detail-heading"><div><p className="eyebrow">TRAJECTORY DETAIL</p><h1>轨迹 {detail.run.id}</h1><p>{detail.run.task} · {detail.run.action_count} steps</p></div><button onClick={onBack}>返回数据集</button></div>
    <div className="detail-summary surface"><Badge tone={detail.run.success ? "green" : "neutral"}>{detail.run.success ? "成功" : "失败"}</Badge><span>{detail.run.source_type ?? detail.run.control_mode}</span><span>{detail.run.created_at ? new Date(detail.run.created_at).toLocaleString() : "—"}</span></div>
    <article className="surface trajectory-video"><h2>结果视频</h2>{video ? <video controls preload="metadata" src={video} /> : <div className="empty-table">没有可用结果视频</div>}</article>
    <div className="trajectory-plots">{plots.map((plot) => <PlotCard key={plot.title} plot={plot} onOpen={() => setOpened(plot)} />)}</div>
    {opened && <PlotInspector plot={opened} onClose={() => setOpened(null)} />}
  </section>;
}
