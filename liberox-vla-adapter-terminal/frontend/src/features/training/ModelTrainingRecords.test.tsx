import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { getModelTrainingConfig } from "../run-control/api";
import type { OfflineJob, TrainingConfiguration } from "../run-control/types";
import { ModelTrainingRecords } from "./ModelTrainingRecords";

vi.mock("../run-control/api", () => ({ getModelTrainingConfig: vi.fn() }));
afterEach(() => { cleanup(); vi.resetAllMocks(); });

const job = (id: string): OfflineJob => ({
  id, kind: "training", status: "COMPLETED", dataset_id: "dataset", created_at: "2026-10-20T00:00:00Z",
  started_at: null, completed_at: null, stage: "", stage_label: "", error: null, output_path: "", log_size: 0,
  model_relation: "output", parameters: {
    algorithm: "bc", dataset_name: id, task_id: "place the black bowl on the flat stove",
    model: { base_id: "registered-model" }, train_steps: 32, micro_batch_size: 4,
    gradient_accumulation_steps: 8, seed: 17, reward: null,
  },
});
const yaml = { filename: "effective_config.yaml", text: "training:\n  seed: 17\n" };

it("keeps the empty state without a card or YAML request", () => {
  render(<ModelTrainingRecords policyId="base" jobs={[]} />);
  expect(screen.getByText("没有关联的本机训练记录。")).toBeTruthy();
  expect(screen.queryByRole("article")).toBeNull();
  expect(screen.queryByRole("button")).toBeNull();
  expect(getModelTrainingConfig).not.toHaveBeenCalled();
});

it.each([
  ["bc", { source: null }, "无需奖励"],
  ["iql", undefined, "未记录"],
  ["iql", { source: "final_reward" }, "final_reward"],
])("distinguishes unused rewards from missing reward metadata for %s", (algorithm, reward, label) => {
  const record = job("first");
  record.parameters = { ...record.parameters, algorithm, reward };
  render(<ModelTrainingRecords policyId="trained" jobs={[record]} />);
  expect(screen.getByText("奖励").nextElementSibling?.textContent).toBe(label);
});

it("shows separate summaries, loads only the selected record and caches collapsed YAML", async () => {
  vi.mocked(getModelTrainingConfig).mockResolvedValue(yaml);
  render(<ModelTrainingRecords policyId="trained" jobs={[job("first"), job("second")]} />);
  const first = within(screen.getByRole("article", { name: "BC · first" }));
  const second = within(screen.getByRole("article", { name: "BC · second" }));
  expect(second.getByText("Batch size").nextElementSibling?.textContent).toBe("4");
  expect(second.getByText("梯度累积").nextElementSibling?.textContent).toBe("8");
  expect(second.getByText("无需奖励")).toBeTruthy();
  expect(getModelTrainingConfig).not.toHaveBeenCalled();
  const toggle = second.getByRole("button", { name: "查看训练 YAML" });
  expect(toggle.getAttribute("aria-expanded")).toBe("false");
  fireEvent.click(toggle);
  await second.findByText(yaml.filename);
  expect(toggle.getAttribute("aria-expanded")).toBe("true");
  expect(document.getElementById(toggle.getAttribute("aria-controls")!)?.querySelector("pre")?.textContent).toBe(yaml.text);
  expect(getModelTrainingConfig).toHaveBeenCalledWith("trained", "second");
  expect(first.queryByText(yaml.filename)).toBeNull();
  fireEvent.click(toggle);
  expect(second.queryByText(yaml.filename)).toBeNull();
  expect(second.getByText("registered-model")).toBeTruthy();
  fireEvent.click(toggle);
  expect(second.getByText(yaml.filename)).toBeTruthy();
  expect(getModelTrainingConfig).toHaveBeenCalledTimes(1);
});

it("allows folding a pending request without loading twice or reopening it on completion", async () => {
  let resolveConfig: (value: TrainingConfiguration) => void = () => {};
  vi.mocked(getModelTrainingConfig).mockImplementation(() => new Promise((resolve) => { resolveConfig = resolve; }));
  render(<ModelTrainingRecords policyId="trained" jobs={[job("first")]} />);
  const toggle = screen.getByRole("button", { name: "查看训练 YAML" });
  fireEvent.click(toggle);
  expect(screen.getByRole("status").textContent).toBe("加载 YAML…");
  fireEvent.click(toggle);
  fireEvent.click(toggle);
  expect(getModelTrainingConfig).toHaveBeenCalledTimes(1);
  fireEvent.click(toggle);
  await act(async () => { resolveConfig(yaml); });
  expect(toggle.getAttribute("aria-expanded")).toBe("false");
  expect(screen.queryByText(yaml.filename)).toBeNull();
  fireEvent.click(toggle);
  expect(screen.getByText(yaml.filename)).toBeTruthy();
  expect(getModelTrainingConfig).toHaveBeenCalledTimes(1);
});

it("keeps a config error inside its card and supports retry", async () => {
  vi.mocked(getModelTrainingConfig).mockRejectedValueOnce(new Error("Configuration not found")).mockResolvedValueOnce(yaml);
  render(<ModelTrainingRecords policyId="trained" jobs={[job("first"), job("second")]} />);
  const first = within(screen.getByRole("article", { name: "BC · first" }));
  const second = within(screen.getByRole("article", { name: "BC · second" }));
  fireEvent.click(first.getByRole("button", { name: "查看训练 YAML" }));
  expect((await first.findByRole("alert")).textContent).toBe("Error: Configuration not found");
  expect(second.queryByRole("alert")).toBeNull();
  fireEvent.click(first.getByRole("button", { name: "重新加载" }));
  await first.findByText(yaml.filename);
  await waitFor(() => expect(first.queryByRole("alert")).toBeNull());
  expect(getModelTrainingConfig).toHaveBeenCalledTimes(2);
});
