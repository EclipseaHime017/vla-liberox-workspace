import { useMemo, useRef, useState, type MouseEvent } from "react";
import type { ReturnData } from "./api";

/** Index of the query whose chunk is executing at ``step``: the last query at or before it. */
export function queryIndexAt(queries: { step: number }[], step: number): number {
  let low = 0, high = queries.length - 1, found = 0;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (queries[middle].step <= step) { found = middle; low = middle + 1; } else high = middle - 1;
  }
  return found;
}

/** Sample counts as probability densities over the shared bins. */
export function densities(counts: number[], edges: number[], samples: number): number[] {
  return counts.map((count, index) => count / (samples * (edges[index + 1] - edges[index])));
}

const CHOSEN = "#d9534f";
const OTHER = "#9aa4b2";
const REFERENCE = "#222";
const format = (value: number) => Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2);

type Frame = { width: number; height: number; left: number; right: number; top: number; bottom: number };
const histogramFrame: Frame = { width: 480, height: 240, left: 44, right: 12, top: 12, bottom: 30 };
const timelineFrame: Frame = { width: 480, height: 200, left: 44, right: 12, top: 12, bottom: 30 };
const scale = (value: number, low: number, high: number, start: number, length: number) =>
  start + (high === low ? 0 : (value - low) / (high - low)) * length;

function XAxis({ frame, low, high, label }: { frame: Frame; low: number; high: number; label: string }) {
  const width = frame.width - frame.left - frame.right;
  const base = frame.height - frame.bottom;
  return <g className="return-axis">
    <line x1={frame.left} x2={frame.left + width} y1={base} y2={base} />
    {[0, 1, 2, 3, 4].map((tick) => {
      const value = low + (high - low) * tick / 4;
      const x = frame.left + width * tick / 4;
      return <g key={tick}><line x1={x} x2={x} y1={base} y2={base + 4} /><text x={x} y={base + 16} textAnchor="middle">{format(value)}</text></g>;
    })}
    <text x={frame.left + width} y={frame.height - 2} textAnchor="end">{label}</text>
  </g>;
}

/** Rollout video with each query's candidate return distributions, following the playback time. */
export function ReturnReplay({ data, video }: { data: ReturnData; video: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [step, setStep] = useState(0);
  const low = data.edges[0], high = data.edges[data.edges.length - 1];
  const duration = Math.max(data.action_count / data.fps, 1e-6);
  const peak = useMemo(() => Math.max(1e-9, ...data.queries.flatMap((query) =>
    query.histograms.flatMap((counts) => densities(counts, data.edges, data.samples)))), [data]);
  if (!data.queries.length) return <p className="muted">这次 rollout 没有策略查询。</p>;
  const index = queryIndexAt(data.queries, step);
  const query = data.queries[index];
  const reference = data.reference.values?.[index];
  const sync = () => {
    const current = videoRef.current;
    if (current) setStep(Math.min(data.action_count, Math.round(current.currentTime * data.fps)));
  };
  const seek = (event: MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const plot = (timelineFrame.width - timelineFrame.left - timelineFrame.right) / timelineFrame.width * box.width;
    const offset = (event.clientX - box.left - timelineFrame.left / timelineFrame.width * box.width) / plot;
    const time = Math.max(0, Math.min(1, offset)) * duration;
    if (videoRef.current) videoRef.current.currentTime = time;
    setStep(Math.round(time * data.fps));
  };

  const h = histogramFrame, plotWidth = h.width - h.left - h.right, plotHeight = h.height - h.top - h.bottom;
  const x = (value: number) => scale(value, low, high, h.left, plotWidth);
  const y = (density: number) => h.top + plotHeight - Math.min(density / peak, 1) * plotHeight;
  const outline = (values: number[]) => values.flatMap((value, bin) =>
    [`${x(data.edges[bin])},${y(value)}`, `${x(data.edges[bin + 1])},${y(value)}`]).join(" ");
  const chosen = densities(query.histograms[query.chosen], data.edges, data.samples);

  const t = timelineFrame, lineWidth = t.width - t.left - t.right, lineHeight = t.height - t.top - t.bottom;
  const tx = (time: number) => scale(time, 0, duration, t.left, lineWidth);
  const ty = (value: number) => t.top + lineHeight - (high === low ? 0 : (value - low) / (high - low)) * lineHeight;
  const times = data.queries.map((item) => item.step / data.fps);
  const line = (values: number[]) => values.map((value, item) => `${tx(times[item])},${ty(value)}`).join(" ");
  const pick = (key: "q10" | "median" | "q90") => data.queries.map((item) => item[key][item.chosen]);
  const band = [...line(pick("q10")).split(" "), ...line(pick("q90")).split(" ").reverse()].join(" ");

  return <div className="return-replay">
    <video ref={videoRef} controls preload="metadata" src={video} onTimeUpdate={sync} onSeeked={sync} />
    <div className="return-charts">
      <p className="return-status" aria-live="polite">
        查询 <b>{index + 1} / {data.queries.length}</b> · 第 <b>{query.step}</b> 步 · 选中候选 <b>#{query.chosen + 1}</b> ·
        选择分数 <b>{format(query.scores[query.chosen])}</b> · 分布均值 <b>{format(query.mean[query.chosen])}</b>
        （10–90%：{format(query.q10[query.chosen])} ~ {format(query.q90[query.chosen])}）
        {reference != null && <> · 实际回报 <b>{format(reference)}</b></>}
      </p>
      <svg viewBox={`0 0 ${h.width} ${h.height}`} role="img" aria-label="当前查询的候选回报分布">
        {query.histograms.map((counts, candidate) => candidate !== query.chosen
          && <polyline key={candidate} points={outline(densities(counts, data.edges, data.samples))}
            fill="none" stroke={OTHER} strokeWidth={1} opacity={0.75} />)}
        {chosen.map((value, bin) => value > 0 && <rect key={bin} x={x(data.edges[bin])} y={y(value)}
          width={Math.max(0, x(data.edges[bin + 1]) - x(data.edges[bin]))} height={h.top + plotHeight - y(value)}
          fill={CHOSEN} opacity={0.6} />)}
        <line x1={x(query.scores[query.chosen])} x2={x(query.scores[query.chosen])} y1={h.top} y2={h.top + plotHeight}
          stroke={CHOSEN} strokeDasharray="2 3" />
        {reference != null && <line x1={x(reference)} x2={x(reference)} y1={h.top} y2={h.top + plotHeight}
          stroke={REFERENCE} strokeDasharray="6 4" />}
        <XAxis frame={h} low={low} high={high} label="回报" />
      </svg>
      <svg viewBox={`0 0 ${t.width} ${t.height}`} role="img" aria-label="选中块回报分布随时间变化" onClick={seek}>
        <polygon points={band} fill={CHOSEN} opacity={0.18} />
        <polyline points={line(pick("median"))} fill="none" stroke={CHOSEN} strokeWidth={1.6} />
        {data.reference.values && <polyline points={line(data.reference.values)} fill="none" stroke={REFERENCE}
          strokeDasharray="6 4" strokeWidth={1.2} />}
        <line x1={tx(step / data.fps)} x2={tx(step / data.fps)} y1={t.top} y2={t.top + lineHeight} stroke="#555" />
        <text x={t.left - 6} y={t.top + 8} textAnchor="end" className="return-tick">{format(high)}</text>
        <text x={t.left - 6} y={t.top + lineHeight} textAnchor="end" className="return-tick">{format(low)}</text>
        <XAxis frame={t} low={0} high={duration} label="时间 (s)" />
      </svg>
      <ul className="return-legend">
        <li><i style={{ background: CHOSEN }} />选中块的回报分布</li>
        <li><i style={{ background: OTHER }} />其余候选</li>
        <li><i className="dotted" style={{ borderColor: CHOSEN }} />选择分数（单次单步采样）</li>
        {data.reference.available && <li><i className="dashed" style={{ borderColor: REFERENCE }} />实际折扣回报</li>}
      </ul>
    </div>
  </div>;
}
