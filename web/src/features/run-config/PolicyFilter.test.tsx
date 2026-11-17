import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { PolicyFilter } from "./PolicyFilter";
import { filterByPolicyScope, policyOptions, recordedPolicyOptions } from "./policyScope";
import type { PolicyInfo, Session } from "../run-control/types";

afterEach(cleanup);

it("keeps family and concrete model in one shared row and resets model when family changes", () => {
  const onChange = vi.fn();
  const { container } = render(<PolicyFilter options={[
    { family: "future", family_label: "Future VLA", policy_id: "x", label: "Custom model" },
    { family: "another", policy_id: "y", label: "Other model" },
  ]} value={{ policy_family: "future", policy_id: "x" }} onChange={onChange} />);
  expect(container.querySelector(".policy-filter")?.querySelectorAll("select")).toHaveLength(2);
  expect(screen.queryByRole("option", { name: "Other model" })).toBeNull();
  fireEvent.change(screen.getByLabelText("模型家族"), { target: { value: "another" } });
  expect(onChange).toHaveBeenCalledWith({ policy_family: "another", policy_id: "" });
});

it("retains deleted models and does not guess historical family from the model name", () => {
  const records = [
    { policy_id: "removed", policy_family: "custom", policy_label: "Historical model" },
    { policy_id: "pi05-unregistered", policy_family: null },
  ] as Session[];
  const choices = policyOptions([{ family: "future", family_label: "Future", policy_id: "new", label: "New" } as PolicyInfo], recordedPolicyOptions(records));
  expect(choices.find((item) => item.policy_id === "removed")?.family).toBe("custom");
  expect(choices.find((item) => item.policy_id === "pi05-unregistered")?.family).toBe("__unknown__");
  expect(filterByPolicyScope(records, { policy_family: "custom", policy_id: "removed" })).toEqual([records[0]]);
  expect(filterByPolicyScope(records, { policy_family: "future", policy_id: "removed" })).toEqual([]);
});
