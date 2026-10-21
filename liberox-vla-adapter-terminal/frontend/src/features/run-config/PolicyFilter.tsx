import type { PolicyOption, PolicyScope } from "../run-control/types";
import { Select } from "../../components/ui/Select";
import { UNKNOWN_FAMILY } from "./policyScope";

export function PolicyFilter({ options, value, onChange, disabled = false, labelPrefix = "" }: {
  options: PolicyOption[]; value: PolicyScope; onChange: (value: PolicyScope) => void;
  disabled?: boolean; labelPrefix?: string;
}) {
  const families = new Map(options.map((item) => [item.family,
    item.family === UNKNOWN_FAMILY ? "未知家族" : item.family_label || item.family]));
  // Registered labels also name families whose older records have no label.
  for (const item of options) if (item.family_label) families.set(item.family, item.family_label);
  return <div className="policy-filter">
    <label>模型家族<Select aria-label={`${labelPrefix}模型家族`} value={value.policy_family} disabled={disabled}
      onChange={(event) => onChange({ policy_family: event.target.value, policy_id: "" })}>
      <option value="">全部家族</option>
      {[...families].map(([id, label]) => <option key={id} value={id}>{label}</option>)}
    </Select></label>
    <label>具体模型<Select aria-label={`${labelPrefix}具体模型`} value={value.policy_id} disabled={disabled || !value.policy_family}
      onChange={(event) => onChange({ ...value, policy_id: event.target.value })}>
      <option value="">全部模型</option>
      {options.filter((item) => item.family === value.policy_family).map((item) =>
        <option key={item.policy_id} value={item.policy_id}>{item.label}</option>)}
    </Select></label>
  </div>;
}
