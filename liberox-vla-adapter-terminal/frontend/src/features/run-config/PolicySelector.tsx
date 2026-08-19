import type { PolicyInfo } from "../run-control/types";

export function PolicySelector({
  policies, value, disabled, onChange,
}: {
  policies: PolicyInfo[];
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  return <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
    {policies.map((policy) => <option key={policy.policy_id} value={policy.policy_id}>
      {policy.label}
    </option>)}
  </select>;
}
