import type { LabResult } from "./api";

const states: Record<string, string> = { met: "已满足", unmet: "未满足", unknown: "不确定" };

export function TaskPlan({ result, inspectInitial }: { result: LabResult; inspectInitial: () => void }) {
  const requirements = result.task_contract?.requirements ?? [];
  const plan = result.plan;
  return <section className="lab-plan">
    {!!requirements.length && <>
      <strong>指令要求</strong>
      <ol className="lab-stages">{requirements.map((item) => <li key={item.id}>
        <strong>{item.condition}</strong><p>原文：{item.instruction_span}</p>
        {!!item.depends_on.length && <p>前置要求：{item.depends_on.map(id => requirements.find(r => r.id === id)?.condition ?? id).join("、")}</p>}
      </li>)}</ol>
    </>}
    {plan ? <>
      <strong>任务拆解 · {plan.stages.length} 个里程碑</strong>
      <ol className="lab-stages">{plan.stages.map((stage) => <li key={stage.id}>
        <strong>{stage.label}</strong><p>达成：{stage.achieved_when}</p><p>失效：{stage.lost_when}</p>
        {stage.initial_state && <p>初始：{states[stage.initial_state]} · {stage.initial_reason}</p>}
        {!!stage.depends_on?.length && <p>前置里程碑：{stage.depends_on.map(id => plan.stages.find(s => s.id === id)?.label ?? id).join("、")}</p>}
        {stage.requirement_ids && <p>对应要求：{stage.requirement_ids.length ? stage.requirement_ids.map(id => requirements.find(r => r.id === id)?.condition ?? id).join("、") : "辅助动作"}</p>}
      </li>)}</ol>
      <p>{plan.notes}</p>{plan.initial_scene && <p>初始场景：{plan.initial_scene}</p>}
      {result.schema_version >= 6 && <button onClick={inspectInitial}>查看拆解输入首帧</button>}
    </> : <p>{result.status === "RUNNING" ? "正在生成任务拆解…" : "未生成通过校验的任务拆解，请检查错误与模型原文。"}</p>}
  </section>;
}
