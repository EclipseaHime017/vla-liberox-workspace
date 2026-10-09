import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { Sidebar } from "./Sidebar";

afterEach(cleanup);

it("keeps task navigation without a separate work queue page", () => {
  const onPage = vi.fn();
  render(<Sidebar page="collect" onPage={onPage} />);
  expect(screen.queryByRole("button", { name: /工作队列/ })).toBeNull();
  expect(screen.getAllByRole("button")).toHaveLength(8);
  fireEvent.click(screen.getByRole("button", { name: /标注实验/ }));
  expect(onPage).toHaveBeenLastCalledWith("annotation-lab");
  fireEvent.click(screen.getByRole("button", { name: /训练/ }));
  expect(onPage).toHaveBeenLastCalledWith("training");
  fireEvent.click(screen.getByRole("button", { name: /测试/ }));
  expect(onPage).toHaveBeenLastCalledWith("evaluation");
});
