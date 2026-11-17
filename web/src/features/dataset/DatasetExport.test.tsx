import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { DatasetExportButton, DatasetExportResult, useDatasetExport } from "./DatasetExport";
import * as api from "../run-control/api";
import type { DatasetExportStatus } from "../run-control/types";

vi.mock("../run-control/api", () => ({ getDatasetExport: vi.fn(), startDatasetExport: vi.fn() }));
function DatasetExport({ datasetId, disabled, enabled = true }: { datasetId: string; disabled: boolean; enabled?: boolean }) {
  const controller = useDatasetExport(datasetId, enabled);
  return <>
    <DatasetExportButton controller={controller} disabled={disabled} />
    <DatasetExportResult controller={controller} />
  </>;
}
const running: DatasetExportStatus = { id: "export", dataset_id: "ds_one", status: "RUNNING",
  stage: "复制轨迹", completed_runs: 1, total_runs: 2, current_file: "trajectory_observations.npz",
  output_path: null, error: null };
const completed: DatasetExportStatus = { ...running, status: "COMPLETED", completed_runs: 2,
  output_path: "/workspace/dataset-exports/export/runs" };
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.getDatasetExport).mockResolvedValue(null);
  vi.mocked(api.startDatasetExport).mockResolvedValue(running);
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

it("exports the selected dataset and displays the destination without reloading the page", async () => {
  render(<DatasetExport datasetId="ds_one" disabled={false} />);
  await act(async () => {});
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "导出数据集" })); });
  expect(api.startDatasetExport).toHaveBeenCalledExactlyOnceWith("ds_one");
  expect(screen.getByRole("status").textContent).toContain("1/2 条");
  expect(screen.getByRole("status").textContent).toContain("trajectory_observations.npz");
  expect((screen.getByRole("button") as HTMLButtonElement).disabled).toBe(true);
});

it("restores a completed export and allows another fresh copy", async () => {
  vi.mocked(api.getDatasetExport).mockResolvedValue(completed);
  render(<DatasetExport datasetId="ds_one" disabled={false} />);
  await act(async () => {});
  expect((screen.getByLabelText("导出目录") as HTMLInputElement).value).toBe(completed.output_path);
  expect(screen.getByLabelText("导出目录").title).toBe(completed.output_path);
  expect((screen.getByRole("button", { name: "导出数据集" }) as HTMLButtonElement).disabled).toBe(false);
});

it("recovers from repeated identical polling errors and stops polling on completion", async () => {
  vi.useFakeTimers();
  vi.mocked(api.getDatasetExport).mockResolvedValueOnce(running)
    .mockRejectedValueOnce(new Error("network")).mockRejectedValueOnce(new Error("network"))
    .mockResolvedValue(completed);
  render(<DatasetExport datasetId="ds_one" disabled={false} />);
  await act(async () => {});
  await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
  expect(screen.getByRole("alert").textContent).toContain("network");
  await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
  expect(screen.getByRole("status").textContent).toContain("导出完成");
  expect(screen.queryByRole("alert")).toBeNull();
  const calls = vi.mocked(api.getDatasetExport).mock.calls.length;
  await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
  expect(api.getDatasetExport).toHaveBeenCalledTimes(calls);
});

it("retries an initial failed status request", async () => {
  vi.useFakeTimers();
  vi.mocked(api.getDatasetExport).mockRejectedValueOnce(new Error("network")).mockResolvedValue(completed);
  render(<DatasetExport datasetId="ds_one" disabled={false} />);
  await act(async () => {});
  await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
  expect(screen.getByRole("status").textContent).toContain("导出完成");
});

it("ignores stale initial status after starting a new export", async () => {
  let resolve!: (value: DatasetExportStatus) => void;
  vi.mocked(api.getDatasetExport).mockReturnValue(new Promise((done) => { resolve = done; }));
  render(<DatasetExport datasetId="ds_one" disabled={false} />);
  await act(async () => { fireEvent.click(screen.getByRole("button")); });
  await act(async () => { resolve(completed); });
  expect(screen.getByRole("status").textContent).toContain("复制轨迹");
});

it("shows failure details and permits retry", async () => {
  vi.mocked(api.startDatasetExport).mockRejectedValue(new Error("文件校验失败"));
  render(<DatasetExport datasetId="ds_one" disabled={false} />);
  await act(async () => { fireEvent.click(screen.getByRole("button")); });
  expect(screen.getByRole("alert").textContent).toContain("文件校验失败");
  expect((screen.getByRole("button") as HTMLButtonElement).disabled).toBe(false);
});

it("copies the entire path and confirms only after clipboard success", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { clipboard: { writeText } });
  vi.mocked(api.getDatasetExport).mockResolvedValue(completed);
  render(<DatasetExport datasetId="ds_one" disabled={false} />);
  await act(async () => {});
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "复制路径" })); });
  expect(writeText).toHaveBeenCalledExactlyOnceWith(completed.output_path);
  expect(screen.getByRole("button", { name: "已复制" })).toBeTruthy();
  expect(api.startDatasetExport).not.toHaveBeenCalled();
});

it.each(["missing", "rejected"])("supports manual copy when clipboard is %s", async (kind) => {
  vi.stubGlobal("navigator", kind === "missing" ? {} : {
    clipboard: { writeText: vi.fn().mockRejectedValue(new Error("denied")) },
  });
  vi.mocked(api.getDatasetExport).mockResolvedValue(completed);
  render(<DatasetExport datasetId="ds_one" disabled={false} />);
  await act(async () => {});
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "复制路径" })); });
  const path = screen.getByLabelText("导出目录") as HTMLInputElement;
  expect(document.activeElement).toBe(path);
  expect(path.selectionStart).toBe(0);
  expect(path.selectionEnd).toBe(path.value.length);
  expect(screen.getByText("路径已选中，请手动复制")).toBeTruthy();
  expect(screen.getByText("导出完成")).toBeTruthy();
  expect(screen.queryByRole("button", { name: "已复制" })).toBeNull();
});

it("does not request export status when the feature is disabled", async () => {
  render(<DatasetExport datasetId="ds_one" disabled={false} enabled={false} />);
  await act(async () => {});
  expect(api.getDatasetExport).not.toHaveBeenCalled();
});

it("does not offer copy when the output path is missing", async () => {
  vi.mocked(api.getDatasetExport).mockResolvedValue({ ...completed, output_path: null });
  render(<DatasetExport datasetId="ds_one" disabled={false} />);
  await act(async () => {});
  expect(screen.queryByRole("button", { name: "复制路径" })).toBeNull();
  expect(screen.getByText("导出目录不可用")).toBeTruthy();
});

it("resets copy feedback when the dataset changes", async () => {
  vi.stubGlobal("navigator", { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
  vi.mocked(api.getDatasetExport).mockResolvedValue(completed);
  const { rerender } = render(<DatasetExport datasetId="ds_one" disabled={false} />);
  await act(async () => {});
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "复制路径" })); });
  vi.mocked(api.getDatasetExport).mockResolvedValue({ ...completed, id: "export_two", output_path: "/new/runs" });
  rerender(<DatasetExport datasetId="ds_two" disabled={false} />);
  await act(async () => {});
  expect(screen.queryByRole("button", { name: "已复制" })).toBeNull();
  expect((screen.getByLabelText("导出目录") as HTMLInputElement).value).toBe("/new/runs");
});
