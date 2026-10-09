import { useState } from "react";
import type { GoalRange, Milestone, RegionStatus } from "./api";

const names: Record<RegionStatus, string> = {
  confirmed: "确定定位区间", uncertain: "待定定位区间", outside: "非定位区间", pending: "尚未评价",
};

export function GoalIntervals({ goals, stages, count, hz, pending, inspect, onSelection }: {
  goals: GoalRange[]; stages: Milestone[]; count: number; hz: number; pending: boolean;
  inspect: (title: string, reason: string, steps: number[]) => void; onSelection: () => void;
}) {
  const [selected, setSelected] = useState("");
  const [selectedWindow, setSelectedWindow] = useState<number | null>(null);
  const stage = stages.find((item) => item.id === selected) ?? stages[0];
  const goal = goals.find((item) => item.stage_id === stage?.id);
  const range = (lo: number, hi: number) => `帧 ${lo}–${hi} · ${(lo/hz).toFixed(2)}–${(hi/hz).toFixed(2)} s`;
  if (!stage) return <p>暂无待定位子任务，请检查任务拆解原文。</p>;
  const showWindow = (index: number) => {
    const item = goal!.passes[index];
    setSelectedWindow(index);
    inspect(`${stage.label} · 窗口 ${index+1}`, item.reason ?? "", item.steps);
  };
  return <>
    <label className="lab-goal-selector">子任务<select aria-label="子任务" value={stage.id} onChange={(event) => {
      setSelected(event.target.value); setSelectedWindow(null); onSelection();
    }}>{stages.map((item, index) => <option key={item.id} value={item.id}>{index+1}. {item.label}</option>)}</select></label>
    <section className="lab-goal" aria-label="定位区间">
      {!goal ? <p>{pending ? "正在定位该目标…" : "该目标的评价未完成；请检查任务错误与模型原文。"}</p> : <>
        <div className="lab-region-legend">{(["confirmed", "uncertain", "outside"] as const).map(status =>
          <span key={status}><i className={`lab-region-${status}`} />{names[status]}</span>)}</div>
        <div className="lab-goal-timeline" aria-label="定位区间时间轴">
          {goal.regions.map(region => <button key={region.start_step} className={`lab-region-${region.status}`}
            style={{ left: `${100*region.start_step/Math.max(1,count)}%`, width: `${100*(region.end_step-region.start_step)/Math.max(1,count)}%` }}
            title={`${names[region.status]} · ${range(region.start_step,region.end_step)}`}
            aria-label={`时间轴 ${names[region.status]} ${range(region.start_step,region.end_step)}`}
            disabled={!region.window_indices.length} onClick={() => showWindow(region.window_indices[0])} />)}
        </div>
        <div className="lab-time-axis"><span>0 s</span><span>{(count/hz).toFixed(2)} s</span></div>
        {goal.regions.some(r => r.status === "pending") && <p className="muted">空白部分尚未评价，不代表非定位区间。</p>}
        <div className="lab-region-list">{goal.regions.filter(r => r.status === "confirmed" || r.status === "uncertain").map(region =>
          <div key={region.start_step} className={`lab-region-card lab-region-${region.status}`}>
            <strong>{range(region.start_step,region.end_step)}</strong>
            <div className="lab-window-links">{region.window_indices.map(index =>
              <button key={index} aria-pressed={selectedWindow === index} onClick={() => showWindow(index)}>
                窗口 {index+1}
              </button>)}</div>
          </div>)}</div>
        {!goal.regions.some(r => r.status === "confirmed" || r.status === "uncertain") &&
          <p className="muted">{goal.regions.some(r => r.status === "pending") ? "已处理部分暂无定位区间。" : "暂无定位区间。"}</p>}
      </>}
    </section>
  </>;
}
