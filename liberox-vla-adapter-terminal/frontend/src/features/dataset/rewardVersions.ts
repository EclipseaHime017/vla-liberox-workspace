import type { EvaluationOperation, RewardParameters, RewardSource } from "../run-control/types";

export const rewardSourceLabels: Record<RewardSource, string> = {
  sparse: "Sparse", stage: "Stage-based", rynnvalue: "RynnValue", robometer: "Robometer", final: "Final Reward",
};
export const evaluationOperationLabels: Record<EvaluationOperation, string> = {
  final: "Final Reward", rynnvalue: "RynnValue", robometer: "Robometer", all: "All",
};

export function rewardParameterLabels(parameters: RewardParameters = {}): string[] {
  const labels: string[] = [];
  if (parameters.fusion_mode) labels.push(parameters.fusion_mode === "multiplicative" ? "相乘" : "相加");
  if (parameters.alpha != null && parameters.fusion_mode !== "multiplicative") labels.push(`Stage 系数 α ${parameters.alpha}`);
  if (parameters.stage_exponent != null) labels.push(`插值指数 p ${parameters.stage_exponent}`);
  if (parameters.shaping_weight != null) labels.push(`塑形系数 κ ${parameters.shaping_weight}`);
  if (parameters.gamma != null) labels.push(`折扣 γ ${parameters.gamma}`);
  if (parameters.accumulate_primitive_steps != null) labels.push(`累计奖励 ${parameters.accumulate_primitive_steps ? "On" : "Off"}`);
  if (parameters.max_frames != null) labels.push(`最多 ${parameters.max_frames} 帧`);
  if (parameters.sampling_hz != null) labels.push(`采样 ${parameters.sampling_hz} fps`);
  if (parameters.batch_size != null) labels.push(`批大小 ${parameters.batch_size}`);
  return labels;
}
