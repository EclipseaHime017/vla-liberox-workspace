import type { PolicyInfo, PolicyOption, PolicyScope, Session } from "../run-control/types";

export const ALL_POLICY_SCOPE: PolicyScope = { policy_family: "", policy_id: "" };
export const UNKNOWN_FAMILY = "__unknown__";
export const UNKNOWN_POLICY = "__unknown__";

export function filterByPolicyScope<T extends Pick<Session, "policy_id" | "policy_family">>(values: T[], scope: PolicyScope): T[] {
  return values.filter((item) => (!scope.policy_family || (item.policy_family || UNKNOWN_FAMILY) === scope.policy_family)
    && (!scope.policy_id || (item.policy_id || UNKNOWN_POLICY) === scope.policy_id));
}

export function policyOptions(models: PolicyInfo[], records: PolicyOption[] = []): PolicyOption[] {
  const choices = new Map<string, PolicyOption>();
  for (const item of [...records, ...models.map((model) => ({
    policy_id: model.policy_id, label: model.label, family: model.family ?? model.model_config?.family ?? UNKNOWN_FAMILY,
    family_label: model.family_label,
  }))]) choices.set(`${item.family}:${item.policy_id}`, item);
  return [...choices.values()];
}

export function recordedPolicyOptions(sessions: Session[]): PolicyOption[] {
  return sessions.map((run) => ({
    policy_id: run.policy_id || UNKNOWN_POLICY, label: run.policy_label || run.policy_id || "未记录模型",
    family: run.policy_family || UNKNOWN_FAMILY,
  }));
}

export function policyScopeParameters(scope: PolicyScope): Partial<PolicyScope> {
  return Object.fromEntries(Object.entries(scope).filter(([, value]) => value));
}
