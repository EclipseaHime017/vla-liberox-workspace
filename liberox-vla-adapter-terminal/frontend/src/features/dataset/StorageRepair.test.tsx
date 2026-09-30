import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StorageRepair } from "./StorageRepair";
import { repairDatasetStorage } from "../run-control/api";

vi.mock("../run-control/api", () => ({ repairDatasetStorage: vi.fn() }));
const unchanged = {
  status: "UNCHANGED", moved: 0, already_current: 2, skipped: [],
  layout: "undated" as const, removed_date_dirs: 0, retained_date_dirs: [],
  message: "已是无日期目录，无需修改",
};
beforeEach(() => { vi.resetAllMocks(); });
afterEach(cleanup);

function setup() {
  const onBusyChange = vi.fn();
  const onRepaired = vi.fn().mockResolvedValue(undefined);
  render(<StorageRepair disabled={false} onBusyChange={onBusyChange} onRepaired={onRepaired} />);
  fireEvent.click(screen.getByRole("button", { name: "一键修复存储目录" }));
  return { onBusyChange, onRepaired };
}

describe("storage repair confirmation", () => {
  it("does not scan or modify storage until confirmed and allows canceling", () => {
    setup();
    expect(screen.getByText(/开始后自动检测当前存储布局/)).toBeTruthy();
    expect(repairDatasetStorage).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "取消" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(repairDatasetStorage).not.toHaveBeenCalled();
  });

  it("reports unchanged storage without triggering a history refresh", async () => {
    vi.mocked(repairDatasetStorage).mockResolvedValue(unchanged);
    const { onBusyChange, onRepaired } = setup();
    fireEvent.click(screen.getByRole("button", { name: "开始检测并修复" }));
    await screen.findByText(unchanged.message);
    expect(onRepaired).not.toHaveBeenCalled();
    expect(onBusyChange.mock.calls).toEqual([[true], [false]]);
    fireEvent.click(screen.getByRole("button", { name: "关闭" }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("reports empty-directory cleanup and preserved nonempty directories", async () => {
    vi.mocked(repairDatasetStorage).mockResolvedValue({ ...unchanged, status: "COMPLETED",
      layout: "mixed", removed_date_dirs: 5, retained_date_dirs: ["task/2026-01-02"],
      skipped: ["task/2026-01-02/notes.txt"], message: "已清理遗留空日期目录，数据无需迁移" });
    const { onRepaired } = setup();
    fireEvent.click(screen.getByRole("button", { name: "开始检测并修复" }));
    await screen.findByText(/已清理 5 个空日期目录/);
    expect(screen.getByText("保留 1 个非空日期目录")).toBeTruthy();
    expect(screen.getByText("task/2026-01-02/notes.txt")).toBeTruthy();
    expect(onRepaired).not.toHaveBeenCalled();
  });

  it("shows running state and blocks repeated submission until completion", async () => {
    let finish!: (value: typeof unchanged) => void;
    vi.mocked(repairDatasetStorage).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    setup();
    fireEvent.click(screen.getByRole("button", { name: "开始检测并修复" }));
    expect(screen.getByRole("status").textContent).toContain("正在检测当前存储布局");
    const button = screen.getByRole("button", { name: "正在检测并修复…" }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect((screen.getByRole("button", { name: "取消" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(button);
    expect(repairDatasetStorage).toHaveBeenCalledOnce();
    finish(unchanged);
    await screen.findByText(unchanged.message);
  });

  it("keeps failures visible and releases the busy state", async () => {
    vi.mocked(repairDatasetStorage).mockRejectedValue(new Error("请先停止仿真"));
    const { onBusyChange, onRepaired } = setup();
    fireEvent.click(screen.getByRole("button", { name: "开始检测并修复" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("请先停止仿真"));
    expect(onBusyChange.mock.calls).toEqual([[true], [false]]);
    expect(onRepaired).not.toHaveBeenCalled();
    expect((screen.getByRole("button", { name: "取消" }) as HTMLButtonElement).disabled).toBe(false);
  });
});
