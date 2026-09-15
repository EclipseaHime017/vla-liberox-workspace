import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { useState } from "react";
import { TaskFilter, TaskSelector } from "./TaskSelector";
import { ALL_TASK_SCOPE, filterByTaskScope, scopeForTask, taskIdsForScope } from "./taskHierarchy";
import type { TaskInfo } from "../run-control/types";

afterEach(cleanup);

const bowl = { family_id: "bowl", family_label: "place the bowl on the stove", task_name: "bowl", init_state_count: 10, init_state_index_min: 0, init_state_index_max: 9 };
const tasks: TaskInfo[] = [
  { ...bowl, task_id: "LEVEL1::bowl", level: "LEVEL1", prompt: "place the black bowl on the flat stove" },
  { ...bowl, task_id: "LEVEL2::bowl", level: "LEVEL2", prompt: "place the black bowl on the flat stove" },
  { ...bowl, task_id: "LEVEL4::cyan", level: "LEVEL4", prompt: "place the cyan bowl on the stove" },
  { ...bowl, task_id: "LEVEL4::grey", level: "LEVEL4", prompt: "place the grey bowl on the stove" },
  { ...bowl, task_id: "LEVEL5::cyan::L5-1", level: "LEVEL5", prompt: "lay down the aqua bowl" },
  { ...bowl, family_id: "drawer", family_label: "open the drawer", task_id: "LEVEL1::drawer", level: "LEVEL1", prompt: "open the top drawer" },
  { ...bowl, task_id: "LEVEL3::missing", level: "LEVEL3", prompt: "missing scene", available: false },
];

it("resolves purpose → difficulty → exact prompt with three selectors and excludes LEVEL5", () => {
  function Form() {
    const [value, setValue] = useState(tasks[0].task_id);
    return <TaskSelector tasks={tasks} value={value} onChange={setValue} />;
  }
  render(<Form />);
  expect(screen.getAllByRole("combobox")).toHaveLength(3);
  expect(screen.queryByRole("option", { name: "LEVEL5" })).toBeNull();
  expect((screen.getByRole("option", { name: "LEVEL3" }) as HTMLOptionElement).disabled).toBe(true);
  fireEvent.change(screen.getByLabelText("难度"), { target: { value: "LEVEL4" } });
  const prompt = screen.getByLabelText("提示词") as HTMLSelectElement;
  expect(prompt.value).toBe("LEVEL4::cyan");
  expect(prompt.options).toHaveLength(2);
  fireEvent.change(prompt, { target: { value: "LEVEL4::grey" } });
  expect(prompt.value).toBe("LEVEL4::grey");
  fireEvent.change(screen.getByLabelText("任务"), { target: { value: "drawer" } });
  expect((screen.getByLabelText("难度") as HTMLSelectElement).value).toBe("LEVEL1");
  expect(prompt.value).toBe("LEVEL1::drawer");
});

it("resets child filters and includes missing assets for historical retrieval", () => {
  const change = vi.fn();
  render(<TaskFilter tasks={tasks} value={{ family_id: "bowl", level: "LEVEL3", task_id: "" }} onChange={change} />);
  expect((screen.getByRole("option", { name: /资源不可用/ }) as HTMLOptionElement).disabled).toBe(false);
  fireEvent.change(screen.getByLabelText("任务"), { target: { value: "drawer" } });
  expect(change).toHaveBeenLastCalledWith({ family_id: "drawer", level: "", task_id: "" });
  fireEvent.change(screen.getByLabelText("难度"), { target: { value: "LEVEL4" } });
  expect(change).toHaveBeenLastCalledWith({ family_id: "bowl", level: "LEVEL4", task_id: "" });
});

it("filters all, a task family, a level or one exact scene without conflating identical prompts", () => {
  expect(taskIdsForScope(tasks, ALL_TASK_SCOPE)).toBeUndefined();
  expect(taskIdsForScope(tasks, { family_id: "unknown", level: "", task_id: "" })).toEqual([]);
  const scope = { family_id: "bowl", level: "LEVEL4", task_id: "" };
  expect(taskIdsForScope(tasks, scope)).toEqual(["LEVEL4::cyan", "LEVEL4::grey"]);
  expect(filterByTaskScope(tasks, tasks, { family_id: "bowl", level: "", task_id: "" })).toHaveLength(5);
  expect(filterByTaskScope(tasks, tasks, scopeForTask(tasks, "LEVEL2::bowl"))).toEqual([tasks[1]]);
});

it("locks the entire scene context while running", () => {
  render(<TaskSelector tasks={tasks} value={tasks[0].task_id} disabled onChange={vi.fn()} />);
  expect(screen.getAllByRole("combobox").every((select) => (select as HTMLSelectElement).disabled)).toBe(true);
});
