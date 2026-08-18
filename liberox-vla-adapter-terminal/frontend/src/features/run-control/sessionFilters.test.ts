import { describe, expect, it } from "vitest";
import type { Session } from "./types";
import { ALL_TASKS, filterSessionsByTask } from "./sessionFilters";

describe("session task retrieval", () => {
  const sessions = [
    { id: "a", task_id: "task-a" },
    { id: "b", task_id: "task-b" },
    { id: "legacy", task_id: null },
  ] as Session[];

  it("returns all indexed sessions for the all-tasks option", () => {
    expect(filterSessionsByTask(sessions, ALL_TASKS).map((item) => item.id))
      .toEqual(["a", "b", "legacy"]);
  });

  it("returns only the requested task", () => {
    expect(filterSessionsByTask(sessions, "task-b").map((item) => item.id))
      .toEqual(["b"]);
  });
});
