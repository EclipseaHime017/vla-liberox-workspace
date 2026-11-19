import type { TrainingMethod } from "../run-control/types";

/** Critic form fields declared by any method in the catalog (its `critic_keys`). */
export const sharedCriticKeys = (methods: TrainingMethod[]) =>
  new Set(methods.flatMap((method) => method.critic_keys));

/** A shared critic field applies only to methods that declare it; other fields always apply. */
export const acceptsField = (name: string, method: TrainingMethod | undefined, shared: Set<string>) =>
  !shared.has(name) || Boolean(method?.critic_keys.includes(name));

/** Method flags recorded with a training job; records before 0.9 carry only the method id and reward. */
export function recordedMethod(parameters: Record<string, unknown>) {
  const recorded = "critic" in parameters;
  return {
    label: String(parameters.algorithm ?? "").toUpperCase(),
    critic: recorded ? parameters.critic != null : parameters.reward != null,
    rewards: recorded ? Boolean(parameters.requires_rewards) : parameters.reward != null,
  };
}
