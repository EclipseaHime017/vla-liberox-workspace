/** Shared parameter names of the critic trained alongside the actor (expectile Q/V). */
const CRITIC_KEYS = new Set(["expectile", "beta", "max_advantage_weight", "target_tau"]);

export const isCriticParameter = (name: string) =>
  name.startsWith("critic_") || name.startsWith("value_") || CRITIC_KEYS.has(name);

/** Method flags recorded with a training job; records before 0.9 carry only the method id and reward. */
export function recordedMethod(parameters: Record<string, unknown>) {
  const recorded = "critic" in parameters;
  return {
    label: String(parameters.algorithm ?? "").toUpperCase(),
    critic: recorded ? parameters.critic != null : parameters.reward != null,
    rewards: recorded ? Boolean(parameters.requires_rewards) : parameters.reward != null,
  };
}
