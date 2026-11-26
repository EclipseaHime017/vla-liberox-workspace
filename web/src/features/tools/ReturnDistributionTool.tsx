import { useEffect, useState } from "react";
import { Badge } from "../../components/ui/Badge";
import { PolicySelector } from "../run-config/PolicySelector";
import { TaskSelector } from "../run-config/TaskSelector";
import type { Bootstrap } from "../run-control/types";
import { JobMonitor } from "../training/JobMonitor";
import {
  deleteReturnDistribution, getReturnData, getReturnDefaults, getReturnDistribution, listReturnDistributions,
  startReturnDistribution, stopReturnDistribution, type ReturnData, type ReturnDefaults, type ReturnDistribution,
} from "./api";
import { ReturnReplay } from "./ReturnReplay";
import { pendingStatuses, statusLabels, useToolJobs, validInteger } from "./useToolJobs";

const endpoints = { list: () => listReturnDistributions(), get: (id: string) => getReturnDistribution(id),
  stop: (id: string) => stopReturnDistribution(id), remove: (id: string) => deleteReturnDistribution(id) };
const downloads = [
  ["returns.mp4", "回报分布视频"], ["returns.npz", "分布 NPZ"], ["returns.json", "回放数据 JSON"],
  ["summary.json", "结果摘要"], ["effective_config.yaml", "配置 YAML"],
] as const;
const sources: Record<string, string> = { sparse: "Sparse", stage: "Stage", rynnvalue: "RynnValue", final: "Final Reward" };

function ReturnResult({ job }: { job: ReturnDistribution }) {
  const artifacts = job.artifacts ?? {};
  const progress = job.return_progress;
  const details = job.return_summary ?? {};
  const [data, setData] = useState<{ url: string; value: ReturnData } | null>(null);
  const [error, setError] = useState("");
  const url = artifacts["returns.json"];
  useEffect(() => {
    if (!url) return;
    let current = true;
    void getReturnData(url).then((value) => { if (current) setData({ url, value }); })
      .catch((reason) => { if (current) setError(String(reason)); });
    return () => { current = false; };
  }, [url]);
  const recipe = details.reward_config as Record<string, unknown> | undefined;
  const bounds = details.return_bounds as number[] | null | undefined;
  const fraction = details.chosen_best_mean_fraction;
  const summary = [
    ["策略查询", details.queries], ["每次候选块", details.candidates],
    ["每个分布采样", (details.analysis as Record<string, unknown> | undefined)?.return_samples],
    ["折扣 γ", details.discount], ["奖励", recipe ? sources[String(recipe.source ?? "rynnvalue")] ?? recipe.source : undefined],
    ["回报范围", Array.isArray(bounds) ? bounds.map((value) => value.toFixed(1)).join(" ~ ") : bounds === null ? "不裁剪" : undefined],
    ["环境成功", typeof details.success === "boolean" ? details.success ? "是" : "否" : undefined],
    ["选中均值最高块的比例", typeof fraction === "number" ? `${(fraction * 100).toFixed(0)}%` : undefined],
  ].filter(([, value]) => typeof value === "string" || typeof value === "number");
  const phase = progress?.phase === "completed" ? "分析已完成" : progress?.phase === "rollout_complete" ? "Rollout 已完成，正在采样分布" : "生成 Rollout";
  const replay = data && data.url === url ? data.value : null;
  return <section className="surface tool-results" aria-label="回报分布结果">
    <div className="panel-title"><strong>回报分布</strong><span>{job.id}</span></div>
    <div className="tool-result-body">
      {progress && <div className="tool-progress" aria-label="回报分布进度">
        <span>{phase} {progress.rollout_step != null && progress.rollout_steps != null && <b>{progress.rollout_step} / {progress.rollout_steps}</b>}</span>
        {progress.rollout_step != null && progress.rollout_steps != null && <progress max={progress.rollout_steps || 1} value={progress.rollout_step} />}
      </div>}
      <p className="muted">直方图是该模型导出的 VF critic 对（查询时的观测，候选动作块）的回报分布估计，每个分布由噪声样本经完整 flow 积分得到。推理时每个候选只做一次单步采样，取分数最高的执行；该分数不是分布的均值。</p>
      {replay && (replay.reference.available
        ? <p className="muted">黑色虚线是这条轨迹从该查询起实际得到的折扣回报，只是一个样本。</p>
        : <p className="tool-note" role="status">未画实际回报：{replay.reference.reason}</p>)}
      {error && <div className="error-banner" role="alert"><span>{error}</span></div>}
      {summary.length > 0 && <dl className="tool-summary">{summary.map(([key, value]) =>
        <div key={String(key)}><dt>{String(key)}</dt><dd>{String(value)}</dd></div>)}</dl>}
      {replay && artifacts["rollout.mp4"] && <ReturnReplay data={replay} video={artifacts["rollout.mp4"]} />}
      {!Object.keys(artifacts).length && <p className="muted">{job.status === "COMPLETED" ? "本次任务没有可用结果文件。" : pendingStatuses.has(job.status) ? "完成后在此回放视频与回报分布。" : "未生成完整结果，请查看任务日志。"}</p>}
      <div className="tool-downloads">{downloads.map(([name, label]) => artifacts[name]
        && <a key={name} href={artifacts[name]} download={name}>{label}</a>)}</div>
    </div>
  </section>;
}

/** Roll out a Value Flows model and show how its critic's return distributions evolve. */
export function ReturnDistributionTool({ bootstrap }: { bootstrap: Bootstrap | null }) {
  const [taskId, setTaskId] = useState("");
  const [policyId, setPolicyId] = useState("");
  const [defaults, setDefaults] = useState<ReturnDefaults | null>(null);
  const [maxSteps, setMaxSteps] = useState(300);
  const [openLoopSteps, setOpenLoopSteps] = useState(1);
  const [seed, setSeed] = useState(0);
  const [initStateIndex, setInitStateIndex] = useState(0);
  const [samples, setSamples] = useState(256);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const { jobs, selectedId, selected, actionId, select, add, update, act } = useToolJobs<ReturnDistribution>(endpoints, setError);

  useEffect(() => {
    if (!bootstrap) return;
    setTaskId(bootstrap.task.task_id); setMaxSteps(bootstrap.config.max_steps); setSeed(bootstrap.config.seed);
  }, [bootstrap]);
  useEffect(() => {
    let current = true;
    void getReturnDefaults().then((value) => { if (current) { setDefaults(value); setSamples(value.return_samples); } })
      .catch((reason) => { if (current) setError(String(reason)); });
    return () => { current = false; };
  }, []);
  const policies = (bootstrap?.policy_catalog ?? []).filter((item) => defaults?.policy_ids.includes(item.policy_id));
  const changePolicy = (value: string) => {
    setPolicyId(value);
    const next = policies.find((item) => item.policy_id === value);
    setOpenLoopSteps(next?.inference?.open_loop_steps ?? next?.io?.native_action_horizon ?? 1);
  };
  useEffect(() => {
    // Default to the first eligible model once both the catalog and the tool defaults arrive.
    if (policies.length && !policies.some((item) => item.policy_id === policyId)) changePolicy(policies[0].policy_id);
  }, [bootstrap, defaults]);

  const task = bootstrap?.task_catalog.find((item) => item.task_id === taskId);
  const policy = policies.find((item) => item.policy_id === policyId);
  const maxOpenLoop = policy?.io?.native_action_horizon ?? policy?.inference?.max_open_loop_steps;
  const ready = Boolean(task && policy && validInteger(maxSteps, 1, 10000) && validInteger(openLoopSteps, 1, maxOpenLoop)
    && validInteger(seed, 0, 2147483647) && validInteger(initStateIndex, 0, (task.init_state_count || 1) - 1)
    && validInteger(samples, 16, 4096));
  const submit = async () => {
    if (!ready || busy) return;
    setBusy(true); setError("");
    try {
      add(await startReturnDistribution({ task_id: taskId, policy_id: policyId, max_steps: maxSteps,
        open_loop_steps: openLoopSteps, seed, init_state_index: initStateIndex, return_samples: samples }));
    } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  };

  return <>
    {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
    <section className="surface tool-builder" aria-label="回报分布配置">
      <div className="panel-title"><strong>VF 回报分布</strong><span>用模型导出的 critic，不训练</span></div>
      <div className="tool-form">
        <TaskSelector tasks={bootstrap?.task_catalog ?? []} value={taskId} labelPrefix="回报分布" disabled={busy}
          onChange={(value) => { setTaskId(value); setInitStateIndex(0); }} />
        <label>模型<PolicySelector policies={policies} value={policyId} disabled={busy || !policies.length} onChange={changePolicy} /></label>
        <label>Rollout 控制步数<input type="number" min={1} max={10000} value={maxSteps} onChange={(event) => setMaxSteps(Number(event.target.value))} /></label>
        <label>每次预测执行步数<input type="number" min={1} max={maxOpenLoop} value={openLoopSteps} onChange={(event) => setOpenLoopSteps(Number(event.target.value))} /></label>
        <label>随机种子<input type="number" min={0} max={2147483647} value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label>
        <label>初始状态索引<input type="number" min={0} max={(task?.init_state_count ?? 1) - 1} value={initStateIndex} onChange={(event) => setInitStateIndex(Number(event.target.value))} /></label>
        <label>每个分布采样数<input type="number" min={16} max={4096} value={samples} onChange={(event) => setSamples(Number(event.target.value))} /></label>
      </div>
      {defaults && !policies.length && <p className="tool-note" role="status">没有带 Value Flows critic 的已训练模型，请先用 VF 方法训练并导出。</p>}
      <div className="tool-actions"><p className="muted">先用所选模型跑一条仿真，记录每次查询采样的全部候选块和 critic 分数；再用模型导出的 critic 采样每个候选的回报分布，生成随视频同步的回放和并排视频。Rollout 阶段需要 GPU，任务进入统一工作队列。</p>
        <button className="primary" disabled={!ready || busy} onClick={() => void submit()}>{busy ? "注册中…" : "开始分析"}</button></div>
    </section>
    {selected && <><ReturnResult job={selected} /><JobMonitor initial={selected} onUpdate={update} /></>}
    <section className="surface tool-history" aria-label="回报分布任务记录">
      <div className="panel-title"><strong>回报分布任务</strong><span>{jobs.filter((job) => pendingStatuses.has(job.status)).length} 个待完成任务</span></div>
      {!jobs.length ? <p className="tool-empty muted">暂无回报分布记录。</p> : <div className="tool-table-scroll"><table><thead><tr>
        <th>任务 / 模型</th><th>配置</th><th>状态</th><th>操作</th>
      </tr></thead><tbody>{jobs.map((job) => <tr key={job.id} className={selectedId === job.id ? "selected" : ""}>
        <td><strong>{String(job.parameters.policy_label ?? job.parameters.policy_id ?? job.id)}</strong>
          <small>{bootstrap?.task_catalog.find((item) => item.task_id === job.parameters.task_id)?.prompt}</small><small>{job.id}</small></td>
        <td>{String(job.parameters.max_steps ?? "—")} steps · {String(job.parameters.return_samples ?? "—")} 样本 / 分布</td>
        <td><Badge tone={job.status === "COMPLETED" ? "green" : job.status === "FAILED" ? "red" : "neutral"}>{statusLabels[job.status]}</Badge></td>
        <td><div className="queue-actions"><button onClick={() => select(job)}>查看</button>
          {pendingStatuses.has(job.status) ? <button className="danger" disabled={actionId === job.id || job.status === "STOPPING"} onClick={() => void act(job)}>{job.status === "QUEUED" ? "取消排队" : "停止"}</button>
            : <button className="danger" disabled={actionId === job.id} onClick={() => void act(job, `删除回报分布 ${job.id} 及其全部结果？`)}>删除</button>}</div></td>
      </tr>)}</tbody></table></div>}
    </section>
  </>;
}
